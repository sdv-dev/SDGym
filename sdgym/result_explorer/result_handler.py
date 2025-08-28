"""Handlers for managing SDGym benchmark results, supporting both local and S3 storage."""

import io
import operator
import os
import pickle
from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd
import yaml
from botocore.exceptions import ClientError

SYNTHESIZER_BASELINE = 'GaussianCopulaSynthesizer'
RESULTS_FOLDER_PREFIX = 'SDGym_results_'
RUN_ID_PREFIX = 'run_'
RESULTS_FILE_PREFIX = 'results_'
NUM_DIGITS_DATE = 10


class ResultsHandler(ABC):
    """Abstract base class for handling results storage and retrieval."""

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

    def _compute_wins(self, result):
        synthesizers = result['Synthesizer'].unique()
        datasets = result['Dataset'].unique()
        result['Win'] = 0
        for dataset in datasets:
            score_baseline = result.loc[
                (result['Synthesizer'] == SYNTHESIZER_BASELINE) & (result['Dataset'] == dataset)
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
            results = results.loc[results['Synthesizer'] != SYNTHESIZER_BASELINE]
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
            yaml_files = self._get_results_files(folder, prefix=RUN_ID_PREFIX, suffix='.yaml')
            if not yaml_files:
                continue

            run_id_info = self._load_yaml_file(folder, yaml_files[0])
            num_datasets = results.loc[
                results['Synthesizer'] == SYNTHESIZER_BASELINE, 'Dataset'
            ].nunique()
            folder_to_info[folder] = {
                'date': run_id_info.get('starting_date')[:NUM_DIGITS_DATE],  # Extract only the YYYY-MM-DD
                'sdgym_version': run_id_info.get('sdgym_version'),
                '# datasets': num_datasets,
            }

        return folder_to_info

    def _process_results(self, results):
        """Process results to ensure they are unique and each dataset has all synthesizers."""
        aggregated_results = pd.concat(results, ignore_index=True)
        aggregated_results = aggregated_results.drop_duplicates(subset=['Dataset', 'Synthesizer'])
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

        date = pd.to_datetime(folder_name[NUM_DIGITS_DATE:], format='%m_%d_%Y')
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

    def all_runs_complete(self, folder_name):
        """Check if all runs in the specified folder are complete."""
        yaml_files = self._get_results_files(folder_name, prefix=RUN_ID_PREFIX, suffix='.yaml')
        if not yaml_files:
            return False

        for yaml_file in yaml_files:
            run_id_info = self._load_yaml_file(folder_name, yaml_file)
            if run_id_info.get('completed_date') is None:
                return False

        return True


class LocalResultsHandler(ResultsHandler):
    """Results handler for local filesystem."""

    def __init__(self, base_path):
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
            return pickle.load(f)

    def load_synthetic_data(self, file_path):
        """Load synthetic data from a CSV file."""
        return pd.read_csv(os.path.join(self.base_path, file_path))

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

    def __init__(self, path, s3_client):
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
            current_path = '/'.join(path_parts[: idx + 1]) + '/'
            s3_key = f'{self.prefix}{current_path}'
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name, Prefix=s3_key, MaxKeys=1
            )
            if 'Contents' not in response:
                level_name = idx_to_structure[idx]
                if level_name == 'Dataset':
                    path_parts[idx] = path_parts[idx][: -NUM_DIGITS_DATE - 1]  # Remove date and '_'
                raise ValueError(
                    f'{level_name} "{path_parts[idx]}" does not exist in S3 path: {previous_s3_key}'
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
        return pickle.loads(response['Body'].read())

    def load_synthetic_data(self, file_path):
        """Load synthetic data from S3."""
        response = self.s3_client.get_object(
            Bucket=self.bucket_name, Key=f'{self.prefix}{file_path}'
        )
        return pd.read_csv(io.BytesIO(response['Body'].read()))

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
            df = pd.read_csv(io.BytesIO(response['Body'].read()))
            results.append(df)

        return results

    def _load_yaml_file(self, folder_name, file_name):
        s3_key = f'{self.prefix}{folder_name}/{file_name}'
        response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
        return yaml.safe_load(response['Body'])
