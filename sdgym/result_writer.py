"""Results writer for SDGym benchmark."""

import io
import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import yaml

from sdgym.s3 import parse_s3_path


class ResultsWriter(ABC):
    """Abstract base class for writing results to files."""

    @abstractmethod
    def write_dataframe(self, data, file_path, append=False):
        """Write a DataFrame to a file."""
        pass

    @abstractmethod
    def write_pickle(self, obj, file_path):
        """Write a Python object to a pickle file."""
        pass

    @abstractmethod
    def write_yaml(self, data, file_path, append=False):
        """Write data to a YAML file."""
        pass


class LocalResultsWriter(ResultsWriter):
    """Results writer for local file system."""

    def write_dataframe(self, data, file_path, append=False, index=False):
        """Write a DataFrame to a CSV file."""
        file_path = Path(file_path)
        if file_path.exists() and append:
            data.to_csv(file_path, mode='a', index=index, header=False)
        else:
            data.to_csv(file_path, mode='w', index=index, header=True)

    def write_pickle(self, obj, file_path):
        """Write a Python object to a pickle file."""
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)

    def write_yaml(self, data, file_path, append=False):
        """Write data to a YAML file."""
        file_path = Path(file_path)
        if append:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    run_data = yaml.safe_load(f) or {}
                for key, value in data.items():
                    run_data[key] = value

                data = run_data

        with open(file_path, 'w') as f:
            yaml.dump(data, f)


class S3ResultsWriter(ResultsWriter):
    """Results writer for S3."""

    def __init__(self, s3_client):
        self.s3_client = s3_client

    def write_dataframe(self, data, file_path, append=False, index=False):
        """Write a DataFrame to S3 as a CSV file."""
        bucket, key = parse_s3_path(file_path)
        if append:
            try:
                response = self.s3_client.get_object(Bucket=bucket, Key=key)
                existing_data = pd.read_csv(io.BytesIO(response['Body'].read()))
                if not existing_data.empty:
                    data = pd.concat([existing_data, data], ignore_index=True)

            except Exception:
                pass  # If the file does not exist, we will create it

        csv_buffer = data.to_csv(index=index).encode()
        self.s3_client.put_object(Body=csv_buffer, Bucket=bucket, Key=key)

    def write_pickle(self, obj, file_path):
        """Write a Python object to S3 as a pickle file."""
        bucket, key = parse_s3_path(file_path)
        buffer = io.BytesIO()
        pickle.dump(obj, buffer)
        buffer.seek(0)
        self.s3_client.put_object(Body=buffer.read(), Bucket=bucket, Key=key)

    def write_yaml(self, data, file_path, append=False):
        """Write data to a YAML file in S3."""
        bucket, key = parse_s3_path(file_path)
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
