from collections import defaultdict

import pandas as pd
from sdv.metadata import Metadata

from sdgym.datasets import BUCKET, _validate_modality, load_dataset, _get_available_datasets


class DatasetExplorer:

    def __init__(self, s3_url=BUCKET, aws_access_key_id=None, aws_secret_access_key=None):
        self.s3_url = s3_url
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key

    @staticmethod
    def _get_max_schema_branch(relationships):
        branch_counts = defaultdict(set)
        for rel in relationships:
            parent = rel["parent_table_name"]
            child = rel["child_table_name"]
            branch_counts[parent].add(child)

        return max((len(children) for children in branch_counts.values()), default=0)

    @staticmethod
    def _get_max_depth(metadata):
        child_map = metadata._get_child_map()
        if not child_map:
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
                results['Max_Num_Columns_Per_Table'],
                num_cols
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
    def summarize_metadata(metadata):
        """Return a dict summarizing metadata information."""
        if isinstance(metadata, dict):
            metadata = Metadata.load_from_dict(metadata)

        metadata_summary = DatasetExplorer._summarize_metadata_columns(metadata)
        total_relationships = len(metadata.relationships)
        total_number_of_tables = len(metadata.tables)
        metadata_summary.update({
            'Num_Relationships': total_relationships,
            'Max_Schema_Depth': DatasetExplorer._get_max_depth(metadata),
            'Max_Schema_Branch': DatasetExplorer._get_max_schema_branch(metadata.relationships)
        })
        return metadata_summary

    @staticmethod
    def summarize_data(data):
        """Return a dict summarizing data information."""
        data_dict = data if isinstance(data, dict) else {'dataset': data}
        data_summary = {
            'Total_Num_Rows': 0,
            'Max_Num_Rows_Per_Table': 0,
        }
        for table_name, table in data_dict.items():
            table_num_rows = len(table)
            data_summary['Total_Num_Rows'] += table_num_rows
            data_summary['Max_Num_Rows_Per_Table'] = max(
                data_summary['Max_Num_Rows_Per_Table'],
                table_num_rows
            )

        return data_summary

    @staticmethod
    def _load_and_summarize_datasets(datasets):
        results = []
        for _, dataset_row in datasets.iterrows():
            dataset_name = dataset_row['dataset_name']
            dataset_size_mb = dataset_row['size_MB']
            dataset_num_table = dataset_row['num_tables']
            data, metadata_dict = load_dataset(
                modality,
                dataset=dataset_name
            )

            metadata_stats = DatasetExplorer.summarize_metadata(metadata_dict)
            data_stats = DatasetExplorer.summarize_data(data)
            max_schema_depth = metadata_stats.pop('Max_Schema_Depth')
            max_schema_branch = metadata_stats.pop('Max_Schema_Branch')
            results.append({
                'Dataset': dataset_name,
                'Datasize_Size_MB': dataset_size_mb,
                'Num_Tables': dataset_num_table,
                **metadata_stats,
                **data_stats,
                'Max_Schema_Depth': max_schema_depth,
                'Max_Schema_Branch': max_schema_branch
            })

        return results

    def summarize(self, modality, output_path=None):
        _validate_output_path(output_path)
        datasets = _get_available_datasets(
            modality=modality,
            bucket=self.s3_url,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
        )
        results = _load_and_summarize_datasets(datasets)
        dataset_summary = pd.DataFrame(results)
        if output_path:
            dataset_summary.to_csv(output_path, index=False)

        return dataset_summary
