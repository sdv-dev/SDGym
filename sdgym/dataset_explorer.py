"""Dataset Explorer to summarize datasets stored in S3 buckets."""

import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sdv.metadata import Metadata

from sdgym.datasets import (
    SDV_DATASETS_PUBLIC_BUCKET,
    _get_available_datasets,
    _load_dataset_with_client,
    _validate_modality,
)
from sdgym.s3 import _get_s3_client, _validate_s3_url

SUMMARY_OUTPUT_COLUMNS = [
    'Dataset',
    'Datasize_Size_MB',
    'Num_Tables',
    'Total_Num_Columns',
    'Total_Num_Columns_Categorical',
    'Total_Num_Columns_Numerical',
    'Total_Num_Columns_Datetime',
    'Total_Num_Columns_PII',
    'Total_Num_Columns_ID_NonKey',
    'Max_Num_Columns_Per_Table',
    'Total_Num_Rows',
    'Max_Num_Rows_Per_Table',
    'Num_Relationships',
    'Max_Schema_Depth',
    'Max_Schema_Branch',
]


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
            AWS secret access key for authentication. Defaults to ``None``.
    """

    def __init__(
        self, s3_url=SDV_DATASETS_PUBLIC_BUCKET, aws_access_key_id=None, aws_secret_access_key=None
    ):
        self.s3_url = s3_url
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self._bucket_name = _validate_s3_url(self.s3_url)
        self.s3_client = (
            _get_s3_client(s3_url, self.aws_access_key_id, self.aws_secret_access_key)
            if self.aws_access_key_id and self.aws_secret_access_key
            else None
        )

    @staticmethod
    def _get_max_schema_branch_factor(relationships):
        """Compute the maximum number of child tables branching from any parent table.

        Args:
            relationships (list[dict]):
                A list of relationship dictionaries describing parent-child table relationships.

        Returns:
            int:
                The maximum number of children linked to a single parent table.
        """
        branch_counts = defaultdict(int)
        for rel in relationships:
            parent = rel['parent_table_name']
            branch_counts[parent] += 1

        return max((value for value in branch_counts.values()), default=0)

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
        if not any(child_map.values()):
            return 1

        def dfs(table):
            if table not in child_map or not child_map[table]:
                return 1

            return 1 + max(dfs(child) for child in child_map[table])

        parent_map = metadata._get_parent_map()
        root_tables = [table for table in child_map.keys() if table not in parent_map]
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
            keys = [table.primary_key, table.sequence_key, table.sequence_index]
            if isinstance(table.alternate_keys, list):
                keys += table.alternate_keys

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
        max_schema_branch_factor = DatasetExplorer._get_max_schema_branch_factor(
            metadata.relationships
        )
        metadata_summary.update({
            'Num_Relationships': total_relationships,
            'Max_Schema_Depth': DatasetExplorer._get_max_depth(metadata),
            'Max_Schema_Branch': max_schema_branch_factor,
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
                The dataset modality to load (e.g., 'single_table' or 'multi_table').

        Returns:
            list[dict]:
                A list of dictionaries, each containing metadata and data summaries
                for an individual dataset.
        """
        results = []

        datasets = _get_available_datasets(
            modality=modality,
            bucket=self._bucket_name,
            s3_client=self.s3_client,
        )
        for _, dataset_row in datasets.iterrows():
            dataset_name = dataset_row['dataset_name']
            dataset_size_mb = dataset_row['size_MB']
            dataset_num_table = dataset_row['num_tables']
            data, metadata_dict = _load_dataset_with_client(
                modality, dataset=dataset_name, bucket=self._bucket_name, s3_client=self.s3_client
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
                If the provided path is not None and does not end with '.csv', or
                if the target file already exists.
        """
        if output_filepath:
            path_obj = Path(output_filepath)
            if path_obj.suffix != '.csv':
                raise ValueError(
                    f"The 'output_filepath' has to be a .csv file, provided: '{output_filepath}'."
                )
            if path_obj.exists():
                raise ValueError(
                    f"The file '{path_obj}' already exists. Please provide a new 'output_filepath'."
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

        if not results:
            warning_msg = (
                f"The provided S3 URL '{self.s3_url}' does not contain any datasets "
                f"of modality '{modality}'."
            )
            warnings.warn(warning_msg, UserWarning)
            dataset_summary = pd.DataFrame(columns=SUMMARY_OUTPUT_COLUMNS)
        else:
            dataset_summary = pd.DataFrame(results)

        if output_filepath:
            dataset_summary.to_csv(output_filepath, index=False)

        return dataset_summary

    def list_datasets(self, modality, output_filepath=None):
        """List available datasets for a modality using metainfo only.

        This is a lightweight alternative to ``summarize_datasets`` that does not load
        the actual data. It reads dataset information from the ``metainfo.yaml`` files
        in the bucket and returns a table equivalent to the legacy
        ``get_available_datasets`` output.

        Args:
            modality (str):
                It must be ``'single_table'``, ``'multi_table'`` or ``'sequential'``.
            output_filepath (str, optional):
                Full path to a ``.csv`` file where the resulting table will be written.
                If not provided, the table is only returned.

        Returns:
            pd.DataFrame:
                A DataFrame with columns: ``['dataset_name', 'size_MB', 'num_tables']``.
        """
        self._validate_output_filepath(output_filepath)
        _validate_modality(modality)

        dataframe = _get_available_datasets(
            modality=modality,
            bucket=self._bucket_name,
            s3_client=self.s3_client,
        )
        if output_filepath:
            dataframe.to_csv(output_filepath, index=False)

        return dataframe
