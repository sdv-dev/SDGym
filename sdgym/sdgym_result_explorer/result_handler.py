"""Handlers for managing SDGym benchmark results, supporting both local and S3 storage."""

import io
import os
import pickle
from abc import ABC, abstractmethod

import boto3
import pandas as pd
from botocore.exceptions import ClientError


class ResultsHandler(ABC):
    """Abstract base class for handling results storage and retrieval."""

    @abstractmethod
    def list_runs(self):
        """List all runs in the results directory."""
        pass

    @abstractmethod
    def validate_access(self, results_folder_name, dataset_name, synthesizer_name, type):
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

    def list_runs(self):
        """List all runs in the local filesystem."""
        return [
            d for d in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, d))
        ]

    def validate_access(self, path_parts, end_filename):
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


class S3ResultsHandler(ResultsHandler):
    """Results handler for AWS S3 storage."""

    def __init__(self, path, aws_key, aws_secret):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_key,
            aws_secret_access_key=aws_secret,
        )
        self.bucket_name = path.split('/')[2]
        self.prefix = '/'.join(path.split('/')[3:]).rstrip('/') + '/'

    def list_runs(self):
        """List all runs in the S3 bucket."""
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket_name, Prefix=self.prefix, Delimiter='/'
        )
        return [cp['Prefix'].split('/')[-2] for cp in response.get('CommonPrefixes', [])]

    def validate_access(self, path_parts, end_filename):
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
