import io
from unittest.mock import Mock, patch
from zipfile import ZipFile

import botocore

from sdgym.datasets import download_dataset


class AnyConfigWith:
    """AnyConfigWith matches any s3 config with the specified signature version."""
    def __init__(self, signature_version):
        self.signature_version = signature_version

    def __eq__(self, other):
        return self.signature_version == other.signature_version


@patch('sdgym.s3.boto3')
def test_download_dataset_public_bucket(boto3_mock, tmpdir):
    """Test the ``sdv.datasets.download_dataset`` method. It calls `download_dataset`
    with a dataset in a public bucket, and does not pass in any aws credentials.

    Setup:
    The boto3 library for s3 access is patched, and mocks are created for the
    s3 bucket and dataset zipfile. The tmpdir fixture is used for the expected
    file creation.

    Input:
    - dataset name
    - datasets path
    - bucket

    Output:
    - n/a

    Side effects:
    - s3 client creation
    - s3 method call to the bucket
    - file creation for dataset in datasets path
    """
    # setup
    dataset = 'my_dataset'
    bucket = 'my_bucket'
    bytesio = io.BytesIO()

    with ZipFile(bytesio, mode='w') as zf:
        zf.writestr(dataset, 'test_content')

    s3_mock = Mock()
    body_mock = Mock()
    body_mock.read.return_value = bytesio.getvalue()
    obj = {
        'Body': body_mock
    }
    s3_mock.get_object.return_value = obj
    boto3_mock.client.return_value = s3_mock
    boto3_mock.Session().get_credentials.return_value = None

    # run
    download_dataset(
        dataset,
        datasets_path=str(tmpdir),
        bucket=bucket
    )

    # asserts
    boto3_mock.client.assert_called_once_with(
        's3',
        config=AnyConfigWith(botocore.UNSIGNED)
    )
    s3_mock.get_object.assert_called_once_with(Bucket=bucket, Key=f'{dataset}.zip')
    with open(f'{tmpdir}/{dataset}') as dataset_file:
        assert dataset_file.read() == 'test_content'


@patch('sdgym.s3.boto3')
def test_download_dataset_private_bucket(boto3_mock, tmpdir):
    """Test the ``sdv.datasets.download_dataset`` method. It calls `download_dataset`
    with a dataset in a private bucket and uses aws credentials.

    Setup:
    The boto3 library for s3 access is patched, and mocks are created for the
    s3 bucket and dataset zipfile. The tmpdir fixture is used for the expected
    file creation.

    Input:
    - dataset name
    - datasets path
    - bucket
    - aws key
    - aws secret

    Output:
    - n/a

    Side effects:
    - s3 client creation with aws credentials
    - s3 method call to the bucket
    - file creation for dataset in datasets path
    """
    # setup
    dataset = 'my_dataset'
    bucket = 'my_bucket'
    aws_key = 'my_key'
    aws_secret = 'my_secret'
    bytesio = io.BytesIO()

    with ZipFile(bytesio, mode='w') as zf:
        zf.writestr(dataset, 'test_content')

    s3_mock = Mock()
    body_mock = Mock()
    body_mock.read.return_value = bytesio.getvalue()
    obj = {
        'Body': body_mock
    }
    s3_mock.get_object.return_value = obj
    boto3_mock.client.return_value = s3_mock

    # run
    download_dataset(
        dataset,
        datasets_path=str(tmpdir),
        bucket=bucket,
        aws_key=aws_key,
        aws_secret=aws_secret
    )

    # asserts
    boto3_mock.client.assert_called_once_with(
        's3',
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret
    )
    s3_mock.get_object.assert_called_once_with(Bucket=bucket, Key=f'{dataset}.zip')
    with open(f'{tmpdir}/{dataset}') as dataset_file:
        assert dataset_file.read() == 'test_content'
