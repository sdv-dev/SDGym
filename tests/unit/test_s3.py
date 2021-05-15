from unittest.mock import Mock, patch

import pandas as pd

from sdgym.s3 import is_s3_path, parse_s3_path, write_csv, write_file


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
    - aws_key is None
    - aws_secret is None

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
    - aws_key for aws authentication
    - aws_secret for aws authentication

    Output:
    - None

    Side effects:
    - s3 client creation with aws credentials (aws_key, aws_secret)
    - s3 method call to create a file in the given bucket with the
      given contents
    """
    # setup
    content_str = 'test_content'
    bucket_name = 'my-bucket'
    key = 'test.txt'
    path = f's3://{bucket_name}/{key}'
    aws_key = 'my-key'
    aws_secret = 'my-secret'

    s3_mock = Mock()
    boto3_mock.client.return_value = s3_mock

    # run
    write_file(content_str.encode('utf-8'), path, aws_key, aws_secret)

    # asserts
    boto3_mock.client.assert_called_once_with(
        's3',
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret
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
    - aws_key is None
    - aws_secret is None

    Output:
    - None

    Side effects:
    - call to write_file with the correct contents and path
    """
    # setup
    data = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    path = 'tmp/path'

    # run
    write_csv(data, path, None, None)

    # asserts
    expected_content = 'col1,col2\n1,3\n2,4\n'
    write_file_mock.assert_called_once_with(
        expected_content.encode('utf-8'),
        path,
        None,
        None
    )
