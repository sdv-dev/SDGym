"""Handlers for managing SDGym benchmark results, supporting both local and S3 storage."""

import io
import os
import pickle
from abc import ABC, abstractmethod

import pandas as pd
from botocore.exceptions import ClientError

SYNTHESIZER_BASELINE = 'GaussianCopulaSynthesizer'


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

    def _get_results_files(self, folder_name):
        return [
            f for f in os.listdir(os.path.join(self.base_path, folder_name))
            if f.endswith('.csv') and f.startswith('results_')
        ]

    def _get_results(self, file_names):
        return [
            pd.read_csv(os.path.join(self.base_path, file_name)) for file_name in file_names
        ]

    def _compute_wins(self, result):
        synthesizers = result['Synthesizer'].unique()
        datasets = result['Dataset'].unique()
        for synthesizer in synthesizers:
            for dataset in datasets:
                loc_synthesizer = (result['Synthesizer'] == synthesizer) & (result['Dataset'] == dataset)
                score_synthesizer = result.loc[loc_synthesizer]['Quality_Score']
                score_baseline = result.loc[(result['Synthesizer'] == SYNTHESIZER_BASELINE) & (result['Dataset'] == dataset)]['Quality_Score']
                result.loc[loc_synthesizer, 'Win'] = score_synthesizer.values > score_baseline.values

    def _get_summarize_table(self, folder_to_results):
        """Create a summary table from the results."""
        summarized_results = pd.DataFrame()
        for folder, results in folder_to_results.items():
            summarized_results[folder] = results.groupby(['Synthesizer'])['Win'].sum()

        summarized_results = summarized_results.fillna('-')
        return summarized_results

    def summarize(self, folder_name):
        """Summarize the results in the specified folder."""
        date = pd.to_datetime(folder_name[-10:])
        other_folders = [
            f for f in os.listdir(self.base_path) if f.startswith(folder_name[:-11])
        ]
        all_folder = other_folders + [folder_name]
        folder_to_results = {}
        for folder in all_folder:
            folder_date = pd.to_datetime(folder[-10:])
            if folder_date > date:
                continue

            result_filenames = self._get_results_files(folder)
            if not result_filenames:
                continue

            results = self._get_results(result_filenames)
            if not results:
                continue

            aggregated_results = pd.concat(results, ignore_index=True)
            aggregated_results = aggregated_results.drop_duplicates(subset=['dataset_name', 'synthesizer_name'])
            self._compute_wins(aggregated_results)
            folder_to_results[folder] = aggregated_results

        summarized_table = self._get_summarize_table(folder_to_results)
        return summarized_table


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
        file_path = '/'.join(path_parts + [end_filename])
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=f'{self.prefix}{file_path}')
        except ClientError as e:
            raise ValueError(f'S3 object does not exist: {file_path}') from e
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
