"""Results writer for SDGym benchmark."""

import io
import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import yaml

from sdgym.s3 import _parse_s3_uri


class ResultsWriter(ABC):
    """Abstract base class for writing results to files."""

    @abstractmethod
    def write_dataframe(self, data, path_key, append=False):
        """Write a DataFrame to a file."""
        pass

    @abstractmethod
    def write_pickle(self, obj, path_key):
        """Write a Python object to a pickle file."""
        pass

    @abstractmethod
    def write_yaml(self, data, file_name, append=False):
        """Write data to a YAML file."""
        pass


class LocalResultsWriter(ResultsWriter):
    """Results writer for local file system."""

    def __init__(self, base_path):
        self.base_path = Path(base_path)

    def write_dataframe(self, data, path_key, append=False):
        """Write a DataFrame to a CSV file."""
        path = self.base_path / path_key
        if path.exists() and append:
            data.to_csv(path, mode='a', index=False, header=False)
        else:
            data.to_csv(path, mode='w', index=False)

    def write_pickle(self, obj, path_key):
        """Write a Python object to a pickle file."""
        path = self.base_path / path_key
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    def write_yaml(self, data, file_name, append=False):
        """Write data to a YAML file."""
        if append:
            path = self.base_path / file_name
            if path.exists():
                with open(path, 'r') as f:
                    run_data = yaml.safe_load(f) or {}
                for key, value in data.items():
                    run_data[key] = value

                data = run_data

        path = self.base_path / file_name
        with open(path, 'w') as f:
            yaml.dump(data, f)


class S3ResultsWriter(ResultsWriter):
    """Results writer for S3."""

    def __init__(self, s3_client):
        self.s3_client = s3_client

    def write_dataframe(self, data, s3_uri, append=False):
        """Write a DataFrame to S3 as a CSV file."""
        bucket, key = _parse_s3_uri(s3_uri)
        if append:
            try:
                response = self.s3_client.get_object(Bucket=bucket, Key=key)
                existing_data = pd.read_csv(io.BytesIO(response['Body'].read()))
                if not existing_data.empty:
                    data = pd.concat([existing_data, data], ignore_index=True)

            except Exception:
                pass  # If the file does not exist, we will create it

        csv_buffer = data.to_csv(index=False).encode()
        self.s3_client.put_object(Body=csv_buffer, Bucket=bucket, Key=key)

    def write_pickle(self, obj, s3_uri):
        """Write a Python object to S3 as a pickle file."""
        bucket, key = _parse_s3_uri(s3_uri)
        buffer = io.BytesIO()
        pickle.dump(obj, buffer)
        buffer.seek(0)
        self.s3_client.put_object(Body=buffer.read(), Bucket=bucket, Key=key)

    def write_yaml(self, data, s3_uri, append=False):
        """Write data to a YAML file in S3."""
        bucket, key = _parse_s3_uri(s3_uri)
        run_data = {}
        if append:
            try:
                response = self.s3_client.get_object(Bucket=bucket, Key=key)
                content = response['Body'].read().decode()
                run_data = yaml.safe_load(content) or {}
            except self.s3_client.exceptions.NoSuchKey:
                pass

        run_data.update(data)
        new_content = yaml.dump(run_data)
        self.s3_client.put_object(Body=new_content.encode(), Bucket=bucket, Key=key)
