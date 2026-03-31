from sdgym._benchmark_launcher.utils import resolve_credentials
from sdgym.s3 import _list_s3_bucket_contents, get_s3_client, is_s3_path, parse_s3_path


class BaseStorageManager:
    """Base class for storage-specific managers."""

    def handles_destination(self, output_destination):
        """Return whether this manager supports the given destination."""
        raise NotImplementedError

    def list_files(self, output_destination):
        """Return the files currently stored under the given destination."""
        raise NotImplementedError

    def get_existing_filenames(self, output_destination):
        """Return the existing storage keys under the given destination."""
        raise NotImplementedError


class S3StorageManager(BaseStorageManager):
    """Manage benchmark artifacts stored in S3."""

    def __init__(self, credentials_filepath):
        self.credentials_filepath = credentials_filepath

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
        """Return the existing S3 keys under the provided output destination."""
        return {obj['Key'] for obj in self.list_files(output_destination)}
