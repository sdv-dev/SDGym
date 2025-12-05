"""Handlers for managing SDGym benchmark results, supporting both local and S3 storage."""

import io
import operator
import os
from abc import ABC, abstractmethod
from datetime import datetime

import cloudpickle
import pandas as pd
import yaml
from botocore.exceptions import ClientError

from sdgym._dataset_utils import _read_zipped_data

SYNTHESIZER_BASELINE = 'GaussianCopulaSynthesizer'
RESULTS_FOLDER_PREFIX = 'SDGym_results_'
metainfo_PREFIX = 'metainfo'
RESULTS_FILE_PREFIX = 'results'
NUM_DIGITS_DATE = 10
REGEX_SYNTHESIZER_NAME = r'\s*\(\d+\)\s*$'


class ResultsHandler(ABC):
    """Abstract base class for handling results storage and retrieval."""

    def __init__(self, baseline_synthesizer=SYNTHESIZER_BASELINE):
        self.baseline_synthesizer = baseline_synthesizer or SYNTHESIZER_BASELINE

    @abstractmethod
    def list(self):
        """List all runs in the results directory."""
        pass

    @abstractmethod
    def get_file_path(self, path_parts, end_filename):
        """Validate access to a specific file in the results directory."""
        pass

    @abstractmethod
    def load_synthesizer(self, file_path):
        """Load a synthesizer from a file."""
        pass

    @abstractmethod
    def load_synthetic_data(self, file_path):
        """Load synthetic data from a file."""
        pass

    @abstractmethod
    def _load_yaml_file(self, folder_name, file_name):
        """Load a YAML file from the results folder."""
        pass

    def _validate_folder_name(self, folder_name):
        """Validate that the provided folder name exists in the results directory."""
        all_folders = self.list()
        if folder_name not in all_folders:
            raise ValueError(f"Folder '{folder_name}' does not exist in the results directory.")

    def _compute_wins(self, result):
        synthesizers = result['Synthesizer'].unique()
        datasets = result['Dataset'].unique()
        result['Win'] = 0
        for dataset in datasets:
            score_baseline = result.loc[
                (result['Synthesizer'] == self.baseline_synthesizer)
                & (result['Dataset'] == dataset)
            ]['Quality_Score'].to_numpy()
            if score_baseline.size == 0:
                continue

            for synthesizer in synthesizers:
                loc_synthesizer = (result['Synthesizer'] == synthesizer) & (
                    result['Dataset'] == dataset
                )
                score_synthesizer = result.loc[loc_synthesizer]['Quality_Score'].to_numpy()
                result.loc[loc_synthesizer, 'Win'] = (score_synthesizer > score_baseline).astype(
                    int
                )

    def _get_summarize_table(self, folder_to_results, folder_infos):
        """Create a summary table from the results."""
        columns = []
        for folder, results in folder_to_results.items():
            date_str = folder_infos[folder]['date']
            date_obj = datetime.strptime(date_str, '%m_%d_%Y')
            column_name = (
                f'{date_str}'
                f' - # datasets: {folder_infos[folder]["# datasets"]}'
                f' - sdgym version: {folder_infos[folder]["sdgym_version"]}'
            )
            results = results.loc[results['Synthesizer'] != self.baseline_synthesizer]
            column_data = results.groupby(['Synthesizer'])['Win'].sum()
            columns.append((date_obj, column_name, column_data))

        columns.sort(key=operator.itemgetter(0))
        summarized_results = pd.DataFrame()
        for _, column_name, column_data in reversed(columns):
            summarized_results[column_name] = column_data

        summarized_results = summarized_results.fillna('-')
        summarized_results = summarized_results.reset_index()
        summarized_results = summarized_results.rename(columns={'index': 'Synthesizer'})

        return summarized_results

    def _get_column_name_infos(self, folder_to_results):
        folder_to_info = {}
        for folder, results in folder_to_results.items():
            yaml_files = self._get_results_files(folder, prefix=metainfo_PREFIX, suffix='.yaml')
            if not yaml_files:
                continue

            metainfo_info = self._load_yaml_file(folder, yaml_files[0])
            baseline_mask = results['Synthesizer'] == self.baseline_synthesizer
            if baseline_mask.any():
                num_datasets = results.loc[baseline_mask, 'Dataset'].nunique()
            else:
                num_datasets = results['Dataset'].nunique()
            folder_to_info[folder] = {
                'date': metainfo_info.get('starting_date')[:NUM_DIGITS_DATE],
                'sdgym_version': metainfo_info.get('sdgym_version'),
                '# datasets': num_datasets,
            }

        return folder_to_info

    def _process_results(self, results):
        """Process results to ensure they are unique and each dataset has all synthesizers."""
        aggregated_results = pd.concat(results, ignore_index=True)
        aggregated_results['Synthesizer'] = (
            aggregated_results['Synthesizer']
            .astype(str)
            .str.replace(REGEX_SYNTHESIZER_NAME, '', regex=True)
            .str.strip()
        )
        aggregated_results = aggregated_results.drop_duplicates(
            subset=['Dataset', 'Synthesizer'], keep='first'
        )
        all_synthesizers = aggregated_results['Synthesizer'].unique()
        dataset_synth_counts = aggregated_results.groupby('Dataset')['Synthesizer'].nunique()
        valid_datasets = dataset_synth_counts[dataset_synth_counts == len(all_synthesizers)].index
        filtered_results = aggregated_results[aggregated_results['Dataset'].isin(valid_datasets)]
        if filtered_results.empty:
            raise ValueError(
                'There is no dataset that has been run by all synthesizers. Cannot '
                'summarize results.'
            )

        filtered_results = filtered_results.sort_values(by=['Dataset', 'Synthesizer'])
        return filtered_results.reset_index(drop=True)

    def summarize(self, folder_name):
        """Summarize the results in the specified folder."""
        all_folders = [f for f in self.list() if f.startswith(RESULTS_FOLDER_PREFIX)]
        if folder_name not in all_folders:
            raise ValueError(f'Folder "{folder_name}" does not exist in the results directory.')

        date = pd.to_datetime(folder_name[-NUM_DIGITS_DATE:], format='%m_%d_%Y')
        folder_to_results = {}
        for folder in all_folders:
            folder_date = pd.to_datetime(folder[len(RESULTS_FOLDER_PREFIX) :], format='%m_%d_%Y')
            if folder_date > date:
                continue

            result_filenames = self._get_results_files(
                folder, prefix=RESULTS_FILE_PREFIX, suffix='.csv'
            )
            if not result_filenames:
                continue

            results = self._get_results(folder, result_filenames)
            if not results:
                continue

            aggregated_results = self._process_results(results)
            self._compute_wins(aggregated_results)
            folder_to_results[folder] = aggregated_results
            folder_infos = self._get_column_name_infos(folder_to_results)

        summarized_table = self._get_summarize_table(folder_to_results, folder_infos)

        return summarized_table, folder_to_results[folder_name]

    def load_results(self, results_folder_name):
        """Load and aggregate all the results CSV files from the specified results folder.

        Args:
            results_folder_name (str):
                The name of the results folder to load results from.

        Returns:
            pd.DataFrame:
                A DataFrame containing the results of the specified folder.
        """
        self._validate_folder_name(results_folder_name)
        result_filenames = self._get_results_files(
            results_folder_name, prefix=RESULTS_FILE_PREFIX, suffix='.csv'
        )

        return pd.concat(
            self._get_results(results_folder_name, result_filenames),
            ignore_index=True,
        )

    def load_metainfo(self, results_folder_name):
        """Load and aggregate all the metainfo YAML files from the specified results folder.

        Args:
            results_folder_name (str):
                The name of the results folder to load metainfo from.

        Returns:
            dict:
                A dictionary containing the metainfo of the specified folder.
        """
        self._validate_folder_name(results_folder_name)
        yaml_files = self._get_results_files(
            results_folder_name, prefix=metainfo_PREFIX, suffix='.yaml'
        )
        results = {}
        for yaml_file in yaml_files:
            metainfo = self._load_yaml_file(results_folder_name, yaml_file)
            run_id = metainfo.pop('run_id', None)
            results[run_id] = metainfo

        return results

    def all_runs_complete(self, folder_name):
        """Check if all runs in the specified folder are complete."""
        yaml_files = self._get_results_files(folder_name, prefix=metainfo_PREFIX, suffix='.yaml')
        if not yaml_files:
            return False

        for yaml_file in yaml_files:
            metainfo_info = self._load_yaml_file(folder_name, yaml_file)
            if metainfo_info.get('completed_date') is None:
                return False

        return True


class LocalResultsHandler(ResultsHandler):
    """Results handler for local filesystem."""

    def __init__(self, base_path, baseline_synthesizer=SYNTHESIZER_BASELINE):
        super().__init__(baseline_synthesizer=baseline_synthesizer)
        self.base_path = base_path

    def list(self):
        """List all runs in the local filesystem."""
        return [
            d for d in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, d))
        ]

    def get_file_path(self, path_parts, end_filename):
        """Validate access to a specific file in the local filesystem."""
        full_path = os.path.join(self.base_path, *path_parts)
        if not os.path.exists(full_path):
            raise ValueError(f'Path does not exist: {full_path}')

        if not os.path.isfile(os.path.join(full_path, end_filename)):
            raise ValueError(f'File does not exist: {end_filename}')

        return os.path.join(*path_parts, end_filename)

    def load_synthesizer(self, file_path):
        """Load a synthesizer from a pickle file."""
        with open(os.path.join(self.base_path, file_path), 'rb') as f:
            return cloudpickle.load(f)

    def load_synthetic_data(self, file_path):
        """Load synthetic data from a CSV or ZIP file."""
        full_path = os.path.join(self.base_path, file_path)
        if full_path.endswith('.zip'):
            return _read_zipped_data(full_path, modality='multi_table')

        return pd.read_csv(full_path)

    def _get_results_files(self, folder_name, prefix, suffix):
        return [
            f
            for f in os.listdir(os.path.join(self.base_path, folder_name))
            if f.endswith(suffix) and f.startswith(prefix)
        ]

    def _get_results(self, folder_name, file_names):
        return [
            pd.read_csv(os.path.join(self.base_path, folder_name, file_name))
            for file_name in file_names
        ]

    def _load_yaml_file(self, folder_name, file_name):
        file_path = os.path.join(self.base_path, folder_name, file_name)
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)


class S3ResultsHandler(ResultsHandler):
    """Results handler for AWS S3 storage."""

    def __init__(self, path, s3_client, baseline_synthesizer=SYNTHESIZER_BASELINE):
        super().__init__(baseline_synthesizer=baseline_synthesizer)
        self.s3_client = s3_client
        self.bucket_name = path.split('/')[2]
        self.prefix = '/'.join(path.split('/')[3:]).rstrip('/') + '/'

    def list(self):
        """List all runs in the S3 bucket."""
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket_name, Prefix=self.prefix, Delimiter='/'
        )
        return [cp['Prefix'].split('/')[-2] for cp in response.get('CommonPrefixes', [])]

    def get_file_path(self, path_parts, end_filename):
        """Validate access to a specific file in S3."""
        idx_to_structure = {0: 'Folder', 1: 'Dataset', 2: 'Synthesizer'}
        file_path = '/'.join(path_parts + [end_filename])
        previous_s3_key = self.prefix
        for idx in range(len(path_parts)):
            level_name = idx_to_structure[idx]
            current_path = '/'.join(path_parts[: idx + 1]) + '/'
            s3_key = f'{self.prefix}{current_path}'
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name, Prefix=s3_key, MaxKeys=1
            )

            if 'Contents' not in response:
                # If missing, fetch available items under previous level
                parent_response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name, Prefix=previous_s3_key
                )
                available_items = set()
                if 'Contents' in parent_response:
                    for obj in parent_response['Contents']:
                        rel_path = obj['Key'][len(previous_s3_key) :]
                        if '/' in rel_path:
                            folder = rel_path.split('/')[0]
                            if folder:
                                folder = folder[: -NUM_DIGITS_DATE - 1] if idx == 1 else folder
                                available_items.add(folder)

                folder_name = path_parts[idx]
                available_list = ',\n'.join(sorted(available_items)) or 'None'
                if level_name == 'Dataset':
                    folder_name = folder_name[: -NUM_DIGITS_DATE - 1]

                if level_name == 'Folder':
                    raise ValueError(
                        f"The specified run '{folder_name}' does not exist in 'Benchmarks'. "
                        f'The available runs are:\n{available_list}'
                    )
                elif level_name == 'Dataset':
                    run_name = path_parts[0]
                    raise ValueError(
                        f"Dataset '{folder_name}' was not part of the run '{run_name}'. "
                        f'The available datasets for this run are:\n{available_list}'
                    )
                else:
                    run_name = path_parts[0]
                    dataset_name = path_parts[1][: -NUM_DIGITS_DATE - 1]
                    raise ValueError(
                        f"Synthesizer '{folder_name}' was not part of the run '{run_name}' "
                        f"for the dataset '{dataset_name}'. "
                        'The available synthesizers for this run and dataset are'
                        f':\n{available_list}'
                    )

            previous_s3_key = s3_key

        key = f'{self.prefix}{file_path}'
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
        except ClientError as e:
            raise ValueError(
                f'File "{end_filename}" does not exist in S3 path: {self.prefix}{file_path}'
            ) from e

        return file_path

    def load_synthesizer(self, file_path):
        """Load a synthesizer from S3."""
        response = self.s3_client.get_object(
            Bucket=self.bucket_name, Key=f'{self.prefix}{file_path}'
        )
        return cloudpickle.loads(response['Body'].read())

    def load_synthetic_data(self, file_path):
        """Load synthetic data from S3."""
        key = f'{self.prefix}{file_path}'
        response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
        body = response['Body'].read()
        if file_path.endswith('.zip'):
            return _read_zipped_data(io.BytesIO(body), modality='multi_table')

        return pd.read_csv(io.BytesIO(body))

    def _get_results_files(self, folder_name, prefix, suffix):
        s3_prefix = f'{self.prefix}{folder_name}/'
        response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=s3_prefix)
        if 'Contents' not in response:
            return []

        return [
            obj['Key'].split('/')[-1]
            for obj in response['Contents']
            if obj['Key'].startswith(s3_prefix + prefix) and obj['Key'].endswith(suffix)
        ]

    def _get_results(self, folder_name, file_names):
        results = []
        for file_name in file_names:
            s3_key = f'{self.prefix}{folder_name}/{file_name}'
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            result_df = pd.read_csv(io.BytesIO(response['Body'].read()))
            results.append(result_df)

        return results

    def _load_yaml_file(self, folder_name, file_name):
        s3_key = f'{self.prefix}{folder_name}/{file_name}'
        response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
        return yaml.safe_load(response['Body'])
