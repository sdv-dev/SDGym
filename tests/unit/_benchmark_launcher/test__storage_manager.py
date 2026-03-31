"""Unit tests for the storage manager classes."""

import re
from unittest.mock import Mock, call, patch

import pytest

from sdgym._benchmark_launcher._storage_manager import (
    BaseStorageManager,
    S3StorageManager,
    _validate_s3_output_destinations,
)


@patch('sdgym._benchmark_launcher._storage_manager.is_s3_path')
def test_validate_s3_output_destinations(mock_is_s3_path):
    """Test `_validate_s3_output_destinations` with valid S3 destinations."""
    # Setup
    instance_jobs = [
        {'output_destination': 's3://bucket/path1'},
        {'output_destination': 's3://bucket/path2'},
    ]
    mock_is_s3_path.return_value = True

    # Run
    _validate_s3_output_destinations(instance_jobs)

    # Assert
    assert mock_is_s3_path.call_args_list == [
        call('s3://bucket/path1'),
        call('s3://bucket/path2'),
    ]


@patch('sdgym._benchmark_launcher._storage_manager.is_s3_path')
def test_validate_s3_output_destinations_invalid_path(mock_is_s3_path):
    """Test `_validate_s3_output_destinations` raises an error for invalid paths."""
    # Setup
    instance_jobs = [
        {'output_destination': 's3://bucket/path1'},
        {'output_destination': '/tmp/local-path'},
    ]
    mock_is_s3_path.side_effect = [True, False]
    expected_message = re.escape(
        "Only S3 storage is currently supported. Found: '/tmp/local-path'."
    )

    # Run and Assert
    with pytest.raises(ValueError, match=expected_message):
        _validate_s3_output_destinations(instance_jobs)


class TestBaseStorageManager:
    def test_list_files(self):
        """Test the `list_files` method."""
        # Setup
        storage_manager = BaseStorageManager()

        # Run and Assert
        with pytest.raises(NotImplementedError):
            storage_manager.list_files('s3://bucket/prefix')

    def test_get_existing_filenames(self):
        """Test the `get_existing_filenames` method."""
        # Setup
        storage_manager = BaseStorageManager()

        # Run and Assert
        with pytest.raises(NotImplementedError):
            storage_manager.get_existing_filenames('s3://bucket/prefix')

    def test_handles_destination(self):
        """Test the `handles_destination` method."""
        # Setup
        storage_manager = BaseStorageManager()

        # Run and Assert
        with pytest.raises(NotImplementedError):
            storage_manager.handles_destination('s3://bucket/prefix')


class TestS3StorageManager:
    @patch('sdgym._benchmark_launcher._storage_manager._validate_s3_output_destinations')
    def test__init__(self, mock_validate_s3_output_destinations):
        """Test the `__init__` method."""
        # Setup
        credentials_filepath = 'creds.json'
        instance_jobs = [{'output_destination': 's3://bucket/path'}]

        # Run
        storage_manager = S3StorageManager(credentials_filepath, instance_jobs)

        # Assert
        mock_validate_s3_output_destinations.assert_called_once_with(instance_jobs)
        assert storage_manager.credentials_filepath == credentials_filepath

    @patch('sdgym._benchmark_launcher._storage_manager.resolve_credentials')
    @patch('sdgym._benchmark_launcher._storage_manager.get_s3_client')
    def test_get_client(self, mock_get_s3_client, mock_resolve_credentials):
        """Test the `_get_client` method."""
        # Setup
        storage_manager = S3StorageManager('creds.json', [])
        mock_resolve_credentials.return_value = {
            'aws': {
                'aws_access_key_id': 'AKIA',
                'aws_secret_access_key': 'SECRET',
            }
        }
        mock_client = Mock()
        mock_get_s3_client.return_value = mock_client

        # Run
        result = storage_manager._get_client()

        # Assert
        mock_resolve_credentials.assert_called_once_with('creds.json')
        mock_get_s3_client.assert_called_once_with(
            aws_access_key_id='AKIA',
            aws_secret_access_key='SECRET',
        )
        assert result is mock_client

    @patch('sdgym._benchmark_launcher._storage_manager.is_s3_path')
    def test_handles_destination(self, mock_is_s3_path):
        """Test the `handles_destination` method."""
        # Setup
        storage_manager = S3StorageManager('creds.json', [])
        mock_is_s3_path.return_value = True

        # Run
        result = storage_manager.handles_destination('s3://bucket/prefix')

        # Assert
        mock_is_s3_path.assert_called_once_with('s3://bucket/prefix')
        assert result is True

    @patch('sdgym._benchmark_launcher._storage_manager._list_s3_bucket_contents')
    @patch('sdgym._benchmark_launcher._storage_manager.parse_s3_path')
    def test_list_files(self, mock_parse_s3_path, mock_list_s3_bucket_contents):
        """Test the `list_files` method."""
        # Setup
        storage_manager = S3StorageManager('creds.json', [])
        storage_manager._get_client = Mock(return_value='s3-client')
        storage_manager.handles_destination = Mock(return_value=True)
        mock_parse_s3_path.return_value = ('bucket', 'prefix')
        mock_list_s3_bucket_contents.return_value = [
            {'Key': 'prefix/file1.csv'},
            {'Key': 'prefix/file2.csv'},
        ]

        # Run
        result = storage_manager.list_files('s3://bucket/prefix')

        # Assert
        storage_manager.handles_destination.assert_called_once_with('s3://bucket/prefix')
        storage_manager._get_client.assert_called_once_with()
        mock_parse_s3_path.assert_called_once_with('s3://bucket/prefix')
        mock_list_s3_bucket_contents.assert_called_once_with('s3-client', 'bucket', 'prefix')
        assert result == [
            {'Key': 'prefix/file1.csv'},
            {'Key': 'prefix/file2.csv'},
        ]

    def test_list_files_invalid_destination(self):
        """Test `list_files` raises an error for unsupported destinations."""
        # Setup
        storage_manager = S3StorageManager('creds.json', [])
        storage_manager.handles_destination = Mock(return_value=False)
        expected_message = re.escape(
            "S3StorageManager only supports S3 paths. Found: 'not-a-valid-path'."
        )

        # Run and Assert
        with pytest.raises(ValueError, match=expected_message):
            storage_manager.list_files('not-a-valid-path')

        storage_manager.handles_destination.assert_called_once_with('not-a-valid-path')

    def test_get_existing_filenames(self):
        """Test the `get_existing_filenames` method."""
        # Setup
        storage_manager = S3StorageManager('creds.json', [])
        storage_manager.list_files = Mock(
            return_value=[
                {'Key': 'prefix/file1.csv'},
                {'Key': 'prefix/file2.csv'},
            ]
        )

        # Run
        result = storage_manager.get_existing_filenames('s3://bucket/prefix')

        # Assert
        storage_manager.list_files.assert_called_once_with('s3://bucket/prefix')
        assert result == {'prefix/file1.csv', 'prefix/file2.csv'}
