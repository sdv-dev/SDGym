import io
import pickle
import re
from unittest.mock import Mock, patch

import botocore
import pandas as pd
import pytest
from botocore.exceptions import NoCredentialsError

from sdgym.s3 import (
    S3_REGION,
    _get_s3_client,
    _list_s3_bucket_contents,
    _load_yaml_metainfo_from_s3,
    _read_data_from_bucket_key,
    _upload_dataframe_to_s3,
    _upload_pickle_to_s3,
    is_s3_path,
    parse_s3_path,
    write_csv,
    write_file,
)


def test_is_s3_path_with_local_dir():
    """Test the ``sdgym.s3.is_s3_path`` function with a local directory.

    If the path is not an s3 path, it should return ``False``.

    Input:
    - path to a local directory

    Output:
    - False
    """
    # setup
    path = 'path/to/local/dir'

    # run
    result = is_s3_path(path)

    # asserts
    assert not result


def test_is_s3_path_with_s3_bucket():
    """Test the ``sdgym.s3.is_s3_path`` function with an s3 directory.

    If the path is an s3 path, it should return ``True``.

    Input:
    - path to an s3 directory

    Output:
    - True
    """
    # setup
    path = 's3://my-bucket/my/path'

    # run
    result = is_s3_path(path)

    # asserts
    assert result


def test_parse_s3_path_bucket_only():
    """Test the ``sdgym.s3.parse_s3_path`` function with an s3 path.

    If the s3 path contains only the bucket name, the returned tuple
    should be ``(bucket_name, '')``.

    Input:
    - path to s3 bucket

    Output:
    - ('my-bucket', '')
    """
    # setup
    expected_bucket_name = 'my-bucket'
    expected_key_prefix = ''
    path = f's3://{expected_bucket_name}/{expected_key_prefix}'

    # run
    bucket_name, key_prefix = parse_s3_path(path)

    # asserts
    assert bucket_name == expected_bucket_name
    assert key_prefix == expected_key_prefix


def test_parse_s3_path_bucket_and_dir_path():
    """Test the `sdgym.s3.parse_s3_path`` function with an s3 path.

    If the s3 path contains the bucket and a sub directory, the returned
    tuple should be ``(bucket_name, subdirectory)``.

    Input:
    - path to s3 directory

    Output:
    - ('my-bucket', 'path/to/dir')
    """
    # setup
    expected_bucket_name = 'my-bucket'
    expected_key_prefix = 'path/to/dir'
    path = f's3://{expected_bucket_name}/{expected_key_prefix}'

    # run
    bucket_name, key_prefix = parse_s3_path(path)

    # asserts
    assert bucket_name == expected_bucket_name
    assert key_prefix == expected_key_prefix


def test_write_file(tmpdir):
    """Test the `sdgym.s3.write_file`` function with a local path.

    If the path is a local path, a file with the correct
    contents should be created at the specified path.

    Input:
    - contents of the local file
    - path to the local file
    - aws_access_key_id is None
    - aws_secret_access_key is None

    Output:
    - None

    Side effects:
    - file creation at the specified path with the given contents
    """
    # setup
    content_str = 'test_content'
    path = f'{tmpdir}/test.txt'

    # run
    write_file(content_str.encode('utf-8'), path, None, None)

    # asserts
    with open(path, 'r') as f:
        assert f.read() == content_str


@patch('sdgym.s3.boto3')
def test_write_file_s3(boto3_mock):
    """Test the `sdgym.s3.write_file`` function with an s3 path.

    If the path is an s3 path, a file with the given contents
    should be created at the specified s3 path.

    Input:
    - contents of the s3 file
    - path to the s3 file location
    - aws_access_key_id for aws authentication
    - aws_secret_access_key for aws authentication

    Output:
    - None

    Side effects:
    - s3 client creation with aws credentials (aws_access_key_id, aws_secret_access_key)
    - s3 method call to create a file in the given bucket with the
      given contents
    """
    # setup
    content_str = 'test_content'
    bucket_name = 'my-bucket'
    key = 'test.txt'
    path = f's3://{bucket_name}/{key}'
    aws_access_key_id = 'my-key'
    aws_secret_access_key = 'my-secret'

    s3_mock = Mock()
    boto3_mock.client.return_value = s3_mock

    # run
    write_file(content_str.encode('utf-8'), path, aws_access_key_id, aws_secret_access_key)

    # asserts
    boto3_mock.client.assert_called_once_with(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=S3_REGION,
    )
    s3_mock.put_object.assert_called_once_with(
        Bucket=bucket_name,
        Key=key,
        Body=content_str.encode('utf-8'),
        ContentEncoding='',
    )


@patch('sdgym.s3.write_file')
def test_write_csv(write_file_mock):
    """Test the ``sdgym.s3.write_csv`` function.

    If ``write_csv`` is called with a DataFrame,
    ``write_file`` should be called with the expected DataFrame
    contents.

    Input:
    - data to be written to the csv file
    - path of the desired csv file
    - aws_access_key_id is None
    - aws_secret_access_key is None

    Output:
    - None

    Side effects:
    - call to write_file with the correct contents and path
    """
    # setup
    data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    path = 'tmp/path'

    # run
    write_csv(data, path, None, None)

    # asserts
    input_data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    expected_content = input_data.to_csv(index=False).encode('utf-8')
    write_file_mock.assert_called_once_with(expected_content, path, None, None)


def test_upload_dataframe_to_s3():
    """Test the `_upload_dataframe_to_s3` function."""
    # Setup
    data = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    s3_client_mock = Mock()
    bucket_name = 'test-bucket'
    key = 'path/to/data.csv'

    # Run
    _upload_dataframe_to_s3(data, s3_client_mock, bucket_name, key)

    # Assert
    s3_client_mock.put_object.assert_called_once()
    call_kwargs = s3_client_mock.put_object.call_args.kwargs
    assert call_kwargs['Bucket'] == bucket_name
    assert call_kwargs['Key'] == key
    body = call_kwargs['Body']
    assert isinstance(body, str)
    csv_buffer = io.StringIO()
    data.to_csv(csv_buffer, index=False)
    expected_csv = csv_buffer.getvalue()
    assert body == expected_csv


@patch('sdgym.s3.LOGGER')
def test_upload_dataframe_to_s3_no_existing_file(logger_mock):
    """Test the `_upload_dataframe_to_s3` function when no existing file is present."""
    # Setup
    data = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    s3_client_mock = Mock()
    bucket_name = 'test-bucket'
    key = 'path/to/data.csv'
    expected_log = f'File {key} does not exist, creating a new one.'
    s3_client_mock.get_object.side_effect = botocore.exceptions.ClientError(
        {'Error': {'Code': 'NoSuchKey'}}, 'GetObject'
    )

    # Run
    _upload_dataframe_to_s3(data, s3_client_mock, bucket_name, key, append=True)

    # Assert
    s3_client_mock.put_object.assert_called_once()
    logger_mock.info.assert_called_once_with(expected_log)


def test_upload_pickle_to_s3():
    """Test the `_upload_pickle_to_s3` function."""
    # Setup
    obj = {'foo': 'bar'}
    s3_client_mock = Mock()
    bucket_name = 'test-bucket'
    key = 'path/to/object.pkl'

    # Run
    _upload_pickle_to_s3(obj, s3_client_mock, bucket_name, key)

    # Assert
    s3_client_mock.put_object.assert_called_once()
    call_kwargs = s3_client_mock.put_object.call_args.kwargs
    assert call_kwargs['Bucket'] == bucket_name
    assert call_kwargs['Key'] == key
    body = call_kwargs['Body']
    assert isinstance(body, io.BytesIO)
    body.seek(0)
    unpickled_obj = pickle.load(body)
    assert unpickled_obj == obj


@patch('sdgym.s3.boto3.client')
def test__get_s3_client_with_credentials(mock_boto_client):
    """Test `_get_s3_client` with a valid S3 bucket."""
    # Setup
    output_destination = 's3://my-bucket/results'
    aws_access_key_id = 'test_access_key'
    aws_secret_access_key = 'test_secret_key'

    mock_s3_client = mock_boto_client.return_value

    # Run
    _get_s3_client(output_destination, aws_access_key_id, aws_secret_access_key)

    # Assert
    mock_boto_client.assert_called_once_with(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=S3_REGION,
    )
    mock_s3_client.head_bucket.assert_called_once_with(Bucket='my-bucket')


def test__get_s3_client_errors():
    """Test `_get_s3_client` raises error for invalid input."""
    # Setup
    output_destination = 's3:/'
    expected_error = re.escape(f'Invalid S3 URL: {output_destination}')

    # Run and Assert
    with pytest.raises(ValueError, match=expected_error):
        _get_s3_client(output_destination)

    with pytest.raises(NoCredentialsError, match='Unable to locate credentials'):
        _get_s3_client('s3://bucket_name/')


def test__read_data_from_bucket_key_reads_body():
    """Test that the function reads data from S3 object body."""
    # Setup
    s3_client = Mock()
    bucket = 'test-bucket'
    key = 'path/to/object.yaml'
    s3_client.get_object.return_value = {'Body': Mock(read=lambda: b'data-bytes')}

    # Run
    result = _read_data_from_bucket_key(s3_client, bucket, key)

    # Assert
    s3_client.get_object.assert_called_once_with(Bucket=bucket, Key=key)
    assert result == b'data-bytes'


def test__list_s3_bucket_contents_multiple_pages():
    """Test that paginator combines multiple pages of S3 contents."""
    # Setup
    s3_client = Mock()
    paginator = Mock()
    s3_client.get_paginator.return_value = paginator

    paginator.paginate.return_value = [
        {'Contents': [{'Key': 'file1'}, {'Key': 'file2'}]},
        {'Contents': [{'Key': 'file3'}]},
    ]

    # Run
    result = _list_s3_bucket_contents(s3_client, 'bucket', 'prefix/')

    # Assert
    s3_client.get_paginator.assert_called_once_with('list_objects_v2')
    paginator.paginate.assert_called_once_with(Bucket='bucket', Prefix='prefix/')
    assert result == [{'Key': 'file1'}, {'Key': 'file2'}, {'Key': 'file3'}]


def test__list_s3_bucket_contents_empty():
    """Test when paginator returns pages without Contents."""
    # Setup
    s3_client = Mock()
    paginator = Mock()
    s3_client.get_paginator.return_value = paginator
    paginator.paginate.return_value = [{'NoContents': True}, {}]

    # Run
    result = _list_s3_bucket_contents(s3_client, 'bucket', 'prefix/')

    # Assert
    assert result == []


@patch('sdgym.s3._read_data_from_bucket_key')
@patch('sdgym.s3.yaml.safe_load')
def test_load_yaml_metainfo_from_s3_parses_yaml(yaml_mock, read_mock):
    """Test that YAML data is read and parsed correctly."""
    # Setup
    s3_client = Mock()
    bucket = 'bucket'
    key = 'meta.yaml'

    read_mock.return_value = b'dataset-size-mb: 10\nnum-tables: 2'
    yaml_mock.return_value = {'dataset-size-mb': 10, 'num-tables': 2}

    # Run
    result = _load_yaml_metainfo_from_s3(s3_client, bucket, key)

    # Assert
    read_mock.assert_called_once_with(s3_client, bucket, key)
    yaml_mock.assert_called_once_with(b'dataset-size-mb: 10\nnum-tables: 2')
    assert result == {'dataset-size-mb': 10, 'num-tables': 2}


@patch('sdgym.s3._read_data_from_bucket_key')
@patch('sdgym.s3.yaml.safe_load', return_value=None)
def test_load_yaml_metainfo_from_s3_empty_yaml(yaml_mock, read_mock):
    """Test that None or empty YAML returns an empty dict."""
    # Setup
    s3_client = Mock()
    read_mock.return_value = b''

    # Run
    result = _load_yaml_metainfo_from_s3(s3_client, 'bucket', 'empty.yaml')

    # Assert
    read_mock.assert_called_once()
    yaml_mock.assert_called_once()
    assert result == {}
