"""Dataset Explorer to summarize datasets stored in S3 buckets."""

from collections import defaultdict
from pathlib import Path

import pandas as pd
from sdv.metadata import Metadata

from sdgym.datasets import BUCKET, _get_available_datasets, _validate_modality, load_dataset


class DatasetExplorer:
    """``DatasetExplorer`` class.

    This class provides utilities to analyze datasets hosted on S3 by loading
    their metadata and data, computing schema and data summaries, and optionally
    saving the results as a CSV file.

    Args:
        s3_url (str, optional):
            The base S3 bucket URL containing the datasets. Defaults to `s3://sdv-datasets-public`.
        aws_access_key_id (str, optional):
            AWS access key ID for authentication. Defaults to ``None``.
        aws_secret_access_key (str, optional):
            AWS secret access key for authentication. Defaults to ``Non``.
    """

    def __init__(self, s3_url=BUCKET, aws_access_key_id=None, aws_secret_access_key=None):
        self.s3_url = s3_url
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key

    @staticmethod
    def _get_max_schema_branch(relationships):
        """Compute the maximum number of child tables branching from any parent table.

        Args:
            relationships (list[dict]):
                A list of relationship dictionaries describing parent-child table relationships.

        Returns:
            int:
                The maximum number of children linked to a single parent table.
        """
        branch_counts = defaultdict(set)
        for rel in relationships:
            parent = rel['parent_table_name']
            child = rel['child_table_name']
            branch_counts[parent].add(child)

        return max((len(children) for children in branch_counts.values()), default=0)

    @staticmethod
    def _get_max_depth(metadata):
        """Calculate the maximum depth of a metadata.

        Args:
            metadata (sdv.metadata.Metadata):
                The SDV Metadata object representing the dataset.

        Returns:
            int:
                The maximum schema depth (i.e., the longest parent-child relationship chain).
        """
        child_map = metadata._get_child_map()
        parent_map = metadata._get_parent_map()
        if not child_map:
            return 1

        def dfs(table):
            if table not in child_map or not child_map[table]:
                return 1

            return 1 + max(dfs(child) for child in child_map[table])

        parent_map = metadata._get_parent_map()
        root_tables = [table for table in child_map.keys() if table not in parent_map]
        if not root_tables:
            return 1

        return max(dfs(root) for root in root_tables)

    @staticmethod
    def _summarize_metadata_columns(metadata):
        """Summarize column-level details from a dataset’s metadata.

        Args:
            metadata (sdv.metadata.Metadata):
                The SDV Metadata object containing table and column information.

        Returns:
            dict:
                A dictionary summarizing total and per-type column counts across all tables.
        """
        results = {
            'Total_Num_Columns': 0,
            'Total_Num_Columns_Categorical': 0,
            'Total_Num_Columns_Numerical': 0,
            'Total_Num_Columns_Datetime': 0,
            'Total_Num_Columns_PII': 0,
            'Total_Num_Columns_ID_NonKey': 0,
            'Max_Num_Columns_Per_Table': 0,
        }
        for table_name, table in metadata.tables.items():
            num_cols = len(table.columns)
            results['Total_Num_Columns'] += num_cols
            results['Max_Num_Columns_Per_Table'] = max(
                results['Max_Num_Columns_Per_Table'], num_cols
            )
            for column_name, column in table.columns.items():
                sdtype = column['sdtype']
                if sdtype in ['categorical', 'boolean']:
                    results['Total_Num_Columns_Categorical'] += 1
                elif sdtype in ['numerical']:
                    results['Total_Num_Columns_Numerical'] += 1
                elif sdtype in ['datetime']:
                    results['Total_Num_Columns_Datetime'] += 1
                elif sdtype in ['id'] and column_name != table.primary_key:
                    results['Total_Num_Columns_ID_NonKey'] += 1
                elif column_name == table.primary_key:
                    continue
                else:
                    results['Total_Num_Columns_PII'] += 1

        return results

    @staticmethod
    def get_metadata_summary(metadata):
        """Summarize schema-level information from dataset metadata.

        Args:
            metadata (dict or Metadata):
                The dataset metadata as a dictionary or SDV Metadata object.

        Returns:
            dict:
                A dictionary containing aggregated schema statistics such as number of
                relationships, schema depth, branching factor, and column-type counts.
        """
        if isinstance(metadata, dict):
            metadata = Metadata.load_from_dict(metadata)

        metadata_summary = DatasetExplorer._summarize_metadata_columns(metadata)
        total_relationships = len(metadata.relationships)
        metadata_summary.update({
            'Num_Relationships': total_relationships,
            'Max_Schema_Depth': DatasetExplorer._get_max_depth(metadata),
            'Max_Schema_Branch': DatasetExplorer._get_max_schema_branch(metadata.relationships),
        })
        return metadata_summary

    @staticmethod
    def get_data_summary(data):
        """Summarize record-level information from dataset tables.

        Args:
            data (dict[str, pd.DataFrame] or pd.DataFrame):
                The dataset data, either as a dictionary of table DataFrames or a single DataFrame.

        Returns:
            dict:
                A dictionary summarizing total number of rows and maximum table size.
        """
        data_dict = data if isinstance(data, dict) else {'dataset': data}
        data_summary = {
            'Total_Num_Rows': 0,
            'Max_Num_Rows_Per_Table': 0,
        }
        for table_name, table in data_dict.items():
            table_num_rows = len(table)
            data_summary['Total_Num_Rows'] += table_num_rows
            data_summary['Max_Num_Rows_Per_Table'] = max(
                data_summary['Max_Num_Rows_Per_Table'], table_num_rows
            )

        return data_summary

    def _load_and_summarize_datasets(self, modality):
        """Load all datasets for the given modality and compute summary statistics.

        Args:
            modality (str):
                The dataset modality to load (e.g., 'single-table' or 'multi-table').

        Returns:
            list[dict]:
                A list of dictionaries, each containing metadata and data summaries
                for an individual dataset.
        """
        results = []
        datasets = _get_available_datasets(
            modality=modality,
            bucket=self.s3_url,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
        )
        for _, dataset_row in datasets.iterrows():
            dataset_name = dataset_row['dataset_name']
            dataset_size_mb = dataset_row['size_MB']
            dataset_num_table = dataset_row['num_tables']
            data, metadata_dict = load_dataset(
                modality,
                dataset=dataset_name,
                bucket=self.s3_url,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
            )

            metadata_stats = DatasetExplorer.get_metadata_summary(metadata_dict)
            data_stats = DatasetExplorer.get_data_summary(data)
            max_schema_depth = metadata_stats.pop('Max_Schema_Depth')
            max_schema_branch = metadata_stats.pop('Max_Schema_Branch')
            num_relationships = metadata_stats.pop('Num_Relationships')
            results.append({
                'Dataset': dataset_name,
                'Datasize_Size_MB': dataset_size_mb,
                'Num_Tables': dataset_num_table,
                **metadata_stats,
                **data_stats,
                'Num_Relationships': num_relationships,
                'Max_Schema_Depth': max_schema_depth,
                'Max_Schema_Branch': max_schema_branch,
            })

        return results

    def _validate_output_filepath(self, output_filepath):
        """Validate that the provided output path has a .csv file extension.

        Args:
            output_filepath (str or None):
                The file path to validate.

        Raises:
            ValueError:
                If the provided path is not None and does not end with '.csv'.
        """
        if output_filepath and not Path(output_filepath).suffix == '.csv':
            raise ValueError(
                f"The 'output_filepath' has to be a .csv file, provided: '{output_filepath}'."
            )

    def summarize_datasets(self, modality, output_filepath=None):
        """Load, summarize, and optionally export dataset statistics for a given modality.

        Args:
            modality (str):
                It must be ``'single_table'``, ``'multi_table'`` or ``'sequential'``.
            output_filepath (str, optional):
                The path to save the summary as a CSV file. If `None`, results are returned only.

        Returns:
            pd.DataFrame:
                A DataFrame containing aggregated dataset summaries including schema and
                data-level statistics.

        Raises:
            ValueError:
                If `output_filepath` is provided and does not have a '.csv' extension.
            ValueError:
                If the modality provided is not `single_table`, `multi_table` or `sequential`.
        """
        self._validate_output_filepath(output_filepath)
        _validate_modality(modality)
        results = self._load_and_summarize_datasets(modality)
        dataset_summary = pd.DataFrame(results)
        if output_filepath:
            dataset_summary.to_csv(output_filepath, index=False)

        return dataset_summary
