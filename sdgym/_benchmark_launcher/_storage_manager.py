import io

import pandas as pd

from sdgym._benchmark_launcher.utils import resolve_credentials
from sdgym.result_writer import S3ResultsWriter
from sdgym.s3 import _list_s3_bucket_contents, get_s3_client, is_s3_path, parse_s3_path


def _validate_s3_output_destinations(instance_jobs):
    """Validate that all output destinations are S3 paths."""
    for instance_job in instance_jobs:
        output_destination = instance_job['output_destination']
        if not is_s3_path(output_destination):
            raise ValueError(
                f'Only S3 storage is currently supported. Found: {output_destination!r}.'
            )


def _build_s3_uri(output_destination, key):
    """Build a full S3 URI from an output destination and key."""
    bucket_name, _ = parse_s3_path(output_destination)
    return f's3://{bucket_name}/{key}'


class BaseStorageManager:
    """Base class for storage-specific managers."""

    def handles_destination(self, output_destination):
        """Return whether this manager supports the given destination."""
        raise NotImplementedError

    def list_files(self, output_destination):
        """Return the files currently stored under the given destination."""
        raise NotImplementedError

    def get_existing_filenames(self, output_destination):
        """Return the existing filenames for the given destination."""
        raise NotImplementedError

    def file_exists(self, output_destination, key):
        """Return whether the provided key exists in the destination."""
        raise NotImplementedError

    def read_csv(self, output_destination, key):
        """Read a CSV artifact from storage."""
        raise NotImplementedError

    def write_csv(self, result, output_destination, key):
        """Write a CSV artifact to storage."""
        raise NotImplementedError

    def load_results(self, output_destination, result_filename):
        """Load an aggregate results CSV."""
        raise NotImplementedError

    def write_results(self, result, output_destination, result_filename):
        """Write an aggregate results CSV."""
        raise NotImplementedError

    def load_job_result(self, output_destination, key):
        """Load a per-job result CSV if it exists, otherwise return None."""
        raise NotImplementedError

    def update_metainfo(self, output_destination, key, content):
        """Update metainfo for an artifact."""
        raise NotImplementedError

    def delete(self, output_destination, key):
        """Delete an artifact from storage."""
        raise NotImplementedError

    def save_pickle(self, object, filepath):
        """Save a picklable object to storage."""
        raise NotImplementedError


class S3StorageManager(BaseStorageManager):
    """Manage benchmark artifacts stored in S3."""

    def __init__(self, credentials_filepath, instance_jobs):
        _validate_s3_output_destinations(instance_jobs)
        self.credentials_filepath = credentials_filepath
        self._existing_files = {}
        self._writer = None

    def __getstate__(self):
        """Return the picklable state."""
        state = self.__dict__.copy()
        state['_writer'] = None
        return state

    def __setstate__(self, state):
        """Restore the state after unpickling."""
        self.__dict__.update(state)

    def _get_writer(self):
        """Build the results writer lazily."""
        if self._writer is None:
            self._writer = S3ResultsWriter(self._get_client())

        return self._writer

    def handles_destination(self, output_destination):
        """Return whether the destination is an S3 path."""
        return is_s3_path(output_destination)

    def _get_client(self):
        """Build and return the S3 client."""
        credentials = resolve_credentials(self.credentials_filepath)
        aws_credentials = credentials.get('aws', {})
        return get_s3_client(
            aws_access_key_id=aws_credentials.get('aws_access_key_id'),
            aws_secret_access_key=aws_credentials.get('aws_secret_access_key'),
        )

    def _get_s3_resources(self, output_destination):
        """Return the S3 client and bucket name for a destination."""
        if not self.handles_destination(output_destination):
            raise ValueError(
                f'S3StorageManager only supports S3 paths. Found: {output_destination!r}.'
            )

        s3_client = self._get_client()
        bucket_name, _ = parse_s3_path(output_destination)
        return s3_client, bucket_name

    def list_files(self, output_destination):
        """List files under the provided S3 output destination."""
        if not self.handles_destination(output_destination):
            raise ValueError(
                f'S3StorageManager only supports S3 paths. Found: {output_destination!r}.'
            )

        s3_client = self._get_client()
        bucket_name, key_prefix = parse_s3_path(output_destination)
        return _list_s3_bucket_contents(s3_client, bucket_name, key_prefix)

    def get_existing_filenames(self, output_destination):
        """Return the existing filenames for the given destination."""
        return {obj['Key'] for obj in self.list_files(output_destination)}

    def file_exists(self, output_destination, key):
        """Return whether the provided key exists."""
        return key in self.get_existing_filenames(output_destination)

    def read_csv(self, output_destination, filename):
        """Read a CSV artifact from S3."""
        s3_client, bucket_name = self._get_s3_resources(output_destination)
        response = s3_client.get_object(Bucket=bucket_name, Key=filename)
        return pd.read_csv(io.BytesIO(response['Body'].read()))

    def write_csv(self, result, output_destination, filename):
        """Write a CSV artifact to S3."""
        bucket_name, _ = parse_s3_path(output_destination)
        file_path = f's3://{bucket_name}/{filename}'
        self._get_writer().write_dataframe(result, file_path, index=False)

    def load_results(self, output_destination, result_filename):
        """Load an aggregate results CSV."""
        return self.read_csv(output_destination, result_filename)

    def write_results(self, result, output_destination, result_filename):
        """Write an aggregate results CSV."""
        self.write_csv(result, output_destination, result_filename)

    def load_job_result(self, output_destination, filename):
        """Load a per-job result CSV if it exists, otherwise return None."""
        if not self.file_exists(output_destination, filename):
            return None

        return self.read_csv(output_destination, filename)

    def update_metainfo(self, output_destination, filename, content):
        """Update metainfo for an artifact."""
        file_path = _build_s3_uri(output_destination, filename)
        self._get_writer().write_yaml(data=content, file_path=file_path, append=True)

    def delete(self, output_destination, key):
        """Delete an artifact from storage."""
        s3_client, bucket_name = self._get_s3_resources(output_destination)
        s3_client.delete_object(Bucket=bucket_name, Key=key)

    def save_pickle(self, object, filepath):
        """Save a picklable object to S3."""
        bucket_name, key = parse_s3_path(filepath)
        file_path = f's3://{bucket_name}/{key}'
        self._get_writer().write_pickle(object, file_path)
