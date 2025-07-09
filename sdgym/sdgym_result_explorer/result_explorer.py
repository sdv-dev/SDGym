"""SDGym Results Explorer for accessing and managing benchmark results."""

import io
import os
import pickle

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from sdgym.benchmark import DEFAULT_DATASETS, _validate_bucket_access
from sdgym.datasets import get_dataset_paths, load_dataset
from sdgym.s3 import is_s3_path


def _validate_path(path, aws_access_key_id=None, aws_secret_access_key=None):
    """Validates the provided path to ensure it is either a local directory or an S3 bucket."""
    if is_s3_path(path):
        _validate_bucket_access(path, aws_access_key_id, aws_secret_access_key)
        return True
    else:
        if not os.path.isdir(path):
            raise ValueError(f"The provided path '{path}' is not a valid directory or S3 bucket.")

        return False


class SDGymResultsExplorer:
    """Class to explore SDGym benchmark results stored in a local directory or S3 bucket.

    This class provides methods to list runs, load synthesizers, synthetic data, and real data
    used for fitting.

    Args:
        path (str):
            The path to the results directory or S3 bucket.
        aws_access_key_id (str, optional):
            AWS access key ID for S3 access. Defaults to None
        aws_secret_access_key (str, optional):
            AWS secret access key for S3 access. Defaults to None
    """

    def __init__(self, path, aws_access_key_id=None, aws_secret_access_key=None):
        self._is_s3_path = _validate_path(path, aws_access_key_id, aws_secret_access_key)
        self.path = path
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self._s3_client = None
        if self._is_s3_path:
            self._s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )
            self._bucket_name = self.path.split('/')[2]
            self._prefix = '/'.join(self.path.split('/')[3:]).rstrip('/') + '/'

    def list(self):
        """Lists all of the results in the bucket.

        Returns:
            list: List of all run names Eg. ['SDGym_results_04_18_2025', 'SDGym_results_05_24_2025']
        """
        if self._is_s3_path:
            response = self._s3_client.list_objects_v2(
                Bucket=self._bucket_name, Prefix=self._prefix, Delimiter='/'
            )
            return [cp['Prefix'].split('/')[-2] for cp in response.get('CommonPrefixes', [])]

        else:
            return [d for d in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, d))]

    def _validate_access(self, results_folder_name, dataset_name, synthesizer_name, type):
        """Validates the synthesizer access."""
        end_filename = f'{synthesizer_name}_'
        if type == 'synthetic_data':
            end_filename += 'synthetic_data.csv'
        elif type == 'synthesizer':
            end_filename += 'synthesizer.pkl'

        date = results_folder_name.split('_')[-3:]
        date = '_'.join(date)
        file_path = f'{results_folder_name}/{dataset_name}_{date}/{synthesizer_name}/{end_filename}'
        if not self._is_s3_path:
            synthesizer_path = os.path.join(
                self.path, results_folder_name, f'{dataset_name}_{date}', synthesizer_name
            )
            if not os.path.exists(synthesizer_path):
                raise ValueError(
                    f"Synthesizer '{synthesizer_name}' for dataset '{dataset_name}' in"
                    f" run '{results_folder_name}' does not exist."
                )

            if not os.path.isfile(os.path.join(synthesizer_path, end_filename)):
                raise ValueError(
                    f"Synthesizer file '{end_filename}' does not exist in '{synthesizer_path}'."
                )
        else:
            try:
                self._s3_client.head_object(
                    Bucket=self._bucket_name, Key=f'{self._prefix}{file_path}'
                )
            except ClientError as e:
                raise ValueError(
                    f"Synthesizer '{synthesizer_name}' for dataset '{dataset_name}' "
                    f"in run '{results_folder_name}' does not exist."
                ) from e

        return file_path

    def load_synthesizer(
        self,
        results_folder_name='SDGym_run_04_18_2025',
        dataset_name='adult',
        synthesizer_name='ctgan',
    ):
        """Load the synthesizer object.

        Args:
            results_folder_name (str): The name of the folder for the desired run.
            dataset_name (str): The name of the dataset.
            synthesizer_name (str): The name of the synthesizer (eg. ctgan).

        Returns:
            sdgym.synthesizers.BaselineSynthesizer
        """
        file_path = self._validate_access(
            results_folder_name, dataset_name, synthesizer_name, 'synthesizer'
        )
        if self._is_s3_path:
            response = self._s3_client.get_object(
                Bucket=self._bucket_name, Key=f'{self._prefix}{file_path}'
            )
            synthesizer = pickle.loads(response['Body'].read())
        else:
            with open(os.path.join(self.path, file_path), 'rb') as f:
                synthesizer = pickle.load(f)

        return synthesizer

    def load_synthetic_data(
        self,
        results_folder_name='SDGym_run_04_18_2025',
        dataset_name='adult',
        synthesizer_name='ctgan',
    ):
        """Load the synthetic data created by a specific synthesizer for a given run and dataset.

        Args:
            results_folder_name (str): The name of the folder for the desired run.
            dataset_name (str): The name of the dataset.
            synthesizer_name (str): The name of the synthesizer (eg. ctgan).

        Returns:
            pd.DataFrame
        """
        file_path = self._validate_access(
            results_folder_name, dataset_name, synthesizer_name, 'synthetic_data'
        )

        if self._is_s3_path:
            response = self._s3_client.get_object(
                Bucket=self._bucket_name, Key=f'{self._prefix}{file_path}'
            )
            synthetic_data = pd.read_csv(io.BytesIO(response['Body'].read()))
        else:
            synthetic_data = pd.read_csv(os.path.join(self.path, file_path))

        return synthetic_data

    def load_real_data(self, dataset_name):
        """Load the data used for fitting.

        Args:
            dataset_name (str): The name of the dataset.

        Returns:
            pd.DataFrame
        """
        if dataset_name in DEFAULT_DATASETS:
            dataset_path = get_dataset_paths(
                datasets=[dataset_name],
                aws_key=self.aws_access_key_id,
                aws_secret=self.aws_secret_access_key,
            )[0]
        else:
            raise ValueError(
                f"Dataset '{dataset_name}' is not a default dataset. "
                'Please provide a valid dataset name.'
            )

        data, _ = load_dataset(
            'single_table',
            dataset_path,
            aws_key=self.aws_access_key_id,
            aws_secret=self.aws_secret_access_key,
        )

        return data
