"""Unit tests for the storage manager classes."""

import re
from unittest.mock import Mock, call, patch

import pandas as pd
import pytest

from sdgym._benchmark_launcher._storage_manager import (
    BaseStorageManager,
    S3StorageManager,
    _build_s3_uri,
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


@patch('sdgym._benchmark_launcher._storage_manager.parse_s3_path')
def test_build_s3_uri(mock_parse_s3_path):
    """Test the `_build_s3_uri` method."""
    # Setup
    mock_parse_s3_path.return_value = ('my-bucket', 'prefix')

    # Run
    result = _build_s3_uri('s3://my-bucket/prefix', 'path/to/file.csv')

    # Assert
    mock_parse_s3_path.assert_called_once_with('s3://my-bucket/prefix')
    assert result == 's3://my-bucket/path/to/file.csv'


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

    def test_file_exists(self):
        """Test the `file_exists` method."""
        # Setup
        storage_manager = BaseStorageManager()

        # Run and Assert
        with pytest.raises(NotImplementedError):
            storage_manager.file_exists('s3://bucket/prefix', 'file.csv')

    def test_read_csv(self):
        """Test the `read_csv` method."""
        # Setup
        storage_manager = BaseStorageManager()

        # Run and Assert
        with pytest.raises(NotImplementedError):
            storage_manager.read_csv('s3://bucket/prefix', 'file.csv')

    def test_write_csv(self):
        """Test the `write_csv` method."""
        # Setup
        storage_manager = BaseStorageManager()

        # Run and Assert
        with pytest.raises(NotImplementedError):
            storage_manager.write_csv(Mock(), 's3://bucket/prefix', 'file.csv')

    def test_load_results(self):
        """Test the `load_results` method."""
        # Setup
        storage_manager = BaseStorageManager()

        # Run and Assert
        with pytest.raises(NotImplementedError):
            storage_manager.load_results('s3://bucket/prefix', 'results.csv')

    def test_write_results(self):
        """Test the `write_results` method."""
        # Setup
        storage_manager = BaseStorageManager()

        # Run and Assert
        with pytest.raises(NotImplementedError):
            storage_manager.write_results(Mock(), 's3://bucket/prefix', 'results.csv')

    def test_load_job_result(self):
        """Test the `load_job_result` method."""
        # Setup
        storage_manager = BaseStorageManager()

        # Run and Assert
        with pytest.raises(NotImplementedError):
            storage_manager.load_job_result('s3://bucket/prefix', 'job_result.csv')

    def test_update_metainfo(self):
        """Test the `update_metainfo` method."""
        # Setup
        storage_manager = BaseStorageManager()

        # Run and Assert
        with pytest.raises(NotImplementedError):
            storage_manager.update_metainfo('s3://bucket/prefix', 'metainfo.yaml', {'a': 1})

    def test_delete(self):
        """Test the `delete` method."""
        # Setup
        storage_manager = BaseStorageManager()

        # Run and Assert
        with pytest.raises(NotImplementedError):
            storage_manager.delete('s3://bucket/prefix', 'file.csv')

    def test_save_pickle(self):
        """Test the `save_pickle` method."""
        # Setup
        storage_manager = BaseStorageManager()

        # Run and Assert
        with pytest.raises(NotImplementedError):
            storage_manager.save_pickle(Mock(), 's3://bucket/file.pkl')


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

    def test_getstate(self):
        """Test the `__getstate__` method."""
        # Setup
        storage_manager = S3StorageManager('creds.json', [])
        storage_manager._writer = Mock()

        # Run
        state = storage_manager.__getstate__()

        # Assert
        assert state['credentials_filepath'] == 'creds.json'
        assert state['_writer'] is None

    def test_setstate(self):
        """Test the `__setstate__` method."""
        # Setup
        storage_manager = S3StorageManager('creds.json', [])
        state = {
            'credentials_filepath': 'other_creds.json',
            '_existing_files': {'a': 1},
            '_writer': None,
        }

        # Run
        storage_manager.__setstate__(state)

        # Assert
        assert storage_manager.credentials_filepath == 'other_creds.json'
        assert storage_manager._existing_files == {'a': 1}
        assert storage_manager._writer is None

    @patch('sdgym._benchmark_launcher._storage_manager.S3ResultsWriter')
    def test_get_writer_builds_writer(self, mock_s3_results_writer):
        """Test the `_get_writer` method builds the writer."""
        # Setup
        storage_manager = S3StorageManager('creds.json', [])
        storage_manager._get_client = Mock(return_value='s3-client')

        # Run
        result = storage_manager._get_writer()

        # Assert
        storage_manager._get_client.assert_called_once_with()
        mock_s3_results_writer.assert_called_once_with('s3-client')
        assert result == mock_s3_results_writer.return_value

    @patch('sdgym._benchmark_launcher._storage_manager.S3ResultsWriter')
    def test_get_writer_returns_existing_writer(self, mock_s3_results_writer):
        """Test the `_get_writer` method when the writer already exists."""
        # Setup
        storage_manager = S3StorageManager('creds.json', [])
        storage_manager._writer = Mock()

        # Run
        result = storage_manager._get_writer()

        # Assert
        mock_s3_results_writer.assert_not_called()
        assert result == storage_manager._writer

    @patch('sdgym._benchmark_launcher._storage_manager.parse_s3_path')
    def test_get_s3_resources(self, mock_parse_s3_path):
        """Test the `_get_s3_resources` method."""
        # Setup
        storage_manager = S3StorageManager('creds.json', [])
        storage_manager.handles_destination = Mock(return_value=True)
        storage_manager._get_client = Mock(return_value='s3-client')
        mock_parse_s3_path.return_value = ('bucket', 'prefix')

        # Run
        result = storage_manager._get_s3_resources('s3://bucket/prefix')

        # Assert
        storage_manager.handles_destination.assert_called_once_with('s3://bucket/prefix')
        storage_manager._get_client.assert_called_once_with()
        mock_parse_s3_path.assert_called_once_with('s3://bucket/prefix')
        assert result == ('s3-client', 'bucket')

    def test_get_s3_resources_invalid_destination(self):
        """Test `_get_s3_resources` raises an error for unsupported destinations."""
        # Setup
        storage_manager = S3StorageManager('creds.json', [])
        storage_manager.handles_destination = Mock(return_value=False)
        expected_message = re.escape(
            "S3StorageManager only supports S3 paths. Found: 'not-a-valid-path'."
        )

        # Run and Assert
        with pytest.raises(ValueError, match=expected_message):
            storage_manager._get_s3_resources('not-a-valid-path')

        storage_manager.handles_destination.assert_called_once_with('not-a-valid-path')

    def test_file_exists(self):
        """Test the `file_exists` method."""
        # Setup
        storage_manager = S3StorageManager('creds.json', [])
        storage_manager.get_existing_filenames = Mock(return_value={'a.csv', 'b.csv'})

        # Run
        result = storage_manager.file_exists('s3://bucket/prefix', 'a.csv')
        result_false = storage_manager.file_exists('s3://bucket/prefix', 'c.csv')

        # Assert
        assert result is True
        assert result_false is False
        assert storage_manager.get_existing_filenames.call_args_list == [
            call('s3://bucket/prefix'),
            call('s3://bucket/prefix'),
        ]

    @patch('sdgym._benchmark_launcher._storage_manager.pd.read_csv')
    def test_read_csv(self, mock_read_csv):
        """Test the `read_csv` method."""
        # Setup
        storage_manager = S3StorageManager('creds.json', [])
        body = Mock()
        body.read.return_value = b'col1,col2\n1,2\n'
        s3_client = Mock()
        s3_client.get_object.return_value = {'Body': body}
        storage_manager._get_s3_resources = Mock(return_value=(s3_client, 'bucket'))
        mock_df = Mock()
        mock_read_csv.return_value = mock_df

        # Run
        result = storage_manager.read_csv('s3://bucket/prefix', 'prefix/file.csv')

        # Assert
        storage_manager._get_s3_resources.assert_called_once_with('s3://bucket/prefix')
        s3_client.get_object.assert_called_once_with(Bucket='bucket', Key='prefix/file.csv')
        mock_read_csv.assert_called_once()
        assert result is mock_df

    @patch('sdgym._benchmark_launcher._storage_manager.parse_s3_path')
    def test_write_csv(self, mock_parse_s3_path):
        """Test the `write_csv` method."""
        # Setup
        storage_manager = S3StorageManager('creds.json', [])
        storage_manager._get_writer = Mock()
        result = pd.DataFrame({'a': [1]})
        mock_parse_s3_path.return_value = ('bucket', 'prefix')

        # Run
        storage_manager.write_csv(result, 's3://bucket/prefix', 'path/file.csv')

        # Assert
        mock_parse_s3_path.assert_called_once_with('s3://bucket/prefix')
        storage_manager._get_writer.assert_called_once_with()
        storage_manager._get_writer.return_value.write_dataframe.assert_called_once_with(
            result,
            's3://bucket/path/file.csv',
            index=False,
        )

    def test_load_results(self):
        """Test the `load_results` method."""
        # Setup
        storage_manager = S3StorageManager('creds.json', [])
        storage_manager.read_csv = Mock(return_value='dataframe')

        # Run
        result = storage_manager.load_results('s3://bucket/prefix', 'results.csv')

        # Assert
        storage_manager.read_csv.assert_called_once_with('s3://bucket/prefix', 'results.csv')
        assert result == 'dataframe'

    def test_write_results(self):
        """Test the `write_results` method."""
        # Setup
        storage_manager = S3StorageManager('creds.json', [])
        storage_manager.write_csv = Mock()
        result = pd.DataFrame({'a': [1]})

        # Run
        storage_manager.write_results(result, 's3://bucket/prefix', 'results.csv')

        # Assert
        storage_manager.write_csv.assert_called_once_with(
            result,
            's3://bucket/prefix',
            'results.csv',
        )

    def test_load_job_result_when_file_exists(self):
        """Test the `load_job_result` method when the file exists."""
        # Setup
        storage_manager = S3StorageManager('creds.json', [])
        storage_manager.file_exists = Mock(return_value=True)
        storage_manager.read_csv = Mock(return_value='job_result_df')

        # Run
        result = storage_manager.load_job_result('s3://bucket/prefix', 'job.csv')

        # Assert
        storage_manager.file_exists.assert_called_once_with('s3://bucket/prefix', 'job.csv')
        storage_manager.read_csv.assert_called_once_with('s3://bucket/prefix', 'job.csv')
        assert result == 'job_result_df'

    def test_load_job_result_when_file_does_not_exist(self):
        """Test the `load_job_result` method when the file does not exist."""
        # Setup
        storage_manager = S3StorageManager('creds.json', [])
        storage_manager.file_exists = Mock(return_value=False)
        storage_manager.read_csv = Mock()

        # Run
        result = storage_manager.load_job_result('s3://bucket/prefix', 'job.csv')

        # Assert
        storage_manager.file_exists.assert_called_once_with('s3://bucket/prefix', 'job.csv')
        storage_manager.read_csv.assert_not_called()
        assert result is None

    @patch('sdgym._benchmark_launcher._storage_manager._build_s3_uri')
    def test_update_metainfo(self, mock_build_s3_uri):
        """Test the `update_metainfo` method."""
        # Setup
        storage_manager = S3StorageManager('creds.json', [])
        storage_manager._get_writer = Mock()
        mock_build_s3_uri.return_value = 's3://bucket/prefix/metainfo.yaml'
        content = {'completed_date': '01_01_2026 10:00:00'}

        # Run
        storage_manager.update_metainfo('s3://bucket/prefix', 'prefix/metainfo.yaml', content)

        # Assert
        mock_build_s3_uri.assert_called_once_with(
            's3://bucket/prefix',
            'prefix/metainfo.yaml',
        )
        storage_manager._get_writer.assert_called_once_with()
        storage_manager._get_writer.return_value.write_yaml.assert_called_once_with(
            data=content,
            file_path='s3://bucket/prefix/metainfo.yaml',
            append=True,
        )

    def test_delete(self):
        """Test the `delete` method."""
        # Setup
        storage_manager = S3StorageManager('creds.json', [])
        s3_client = Mock()
        storage_manager._get_s3_resources = Mock(return_value=(s3_client, 'bucket'))

        # Run
        storage_manager.delete('s3://bucket/prefix', 'prefix/file.csv')

        # Assert
        storage_manager._get_s3_resources.assert_called_once_with('s3://bucket/prefix')
        s3_client.delete_object.assert_called_once_with(
            Bucket='bucket',
            Key='prefix/file.csv',
        )

    @patch('sdgym._benchmark_launcher._storage_manager.parse_s3_path')
    def test_save_pickle(self, mock_parse_s3_path):
        """Test the `save_pickle` method."""
        # Setup
        storage_manager = S3StorageManager('creds.json', [])
        storage_manager._get_writer = Mock()
        mock_parse_s3_path.return_value = ('bucket', 'prefix/file.pkl')
        obj = {'a': 1}

        # Run
        storage_manager.save_pickle(obj, 's3://bucket/prefix/file.pkl')

        # Assert
        mock_parse_s3_path.assert_called_once_with('s3://bucket/prefix/file.pkl')
        storage_manager._get_writer.assert_called_once_with()
        storage_manager._get_writer.return_value.write_pickle.assert_called_once_with(
            obj,
            's3://bucket/prefix/file.pkl',
        )
