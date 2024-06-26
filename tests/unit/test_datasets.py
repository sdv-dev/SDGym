import io
from pathlib import Path
from unittest.mock import Mock, call, patch
from zipfile import ZipFile

import botocore
import pandas as pd

from sdgym import get_available_datasets
from sdgym.datasets import (
    _download_dataset,
    _get_bucket_name,
    _get_dataset_path,
    get_dataset_paths,
    load_dataset,
)


class AnyConfigWith:
    """AnyConfigWith matches any s3 config with the specified signature version."""

    def __init__(self, signature_version):
        self.signature_version = signature_version

    def __eq__(self, other):
        return self.signature_version == other.signature_version


@patch('sdgym.s3.boto3')
def test__download_dataset_public_bucket(boto3_mock, tmpdir):
    """Test the ``sdv.datasets._download_dataset`` method.

    It calls `_download_dataset` with a dataset in a public bucket,
    and does not pass in any aws credentials.

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
    modality = 'single_table'
    dataset = 'my_dataset'
    bucket = 's3://my_bucket'
    bytesio = io.BytesIO()

    with ZipFile(bytesio, mode='w') as zf:
        zf.writestr(dataset, 'test_content')

    s3_mock = Mock()
    body_mock = Mock()
    body_mock.read.return_value = bytesio.getvalue()
    obj = {'Body': body_mock}
    s3_mock.get_object.return_value = obj
    boto3_mock.client.return_value = s3_mock
    boto3_mock.Session().get_credentials.return_value = None

    # run
    _download_dataset(modality, dataset, datasets_path=str(tmpdir), bucket=bucket)

    # asserts
    boto3_mock.client.assert_called_once_with('s3', config=AnyConfigWith(botocore.UNSIGNED))
    s3_mock.get_object.assert_called_once_with(
        Bucket='my_bucket', Key=f'{modality.upper()}/{dataset}.zip'
    )
    with open(f'{tmpdir}/{dataset}') as dataset_file:
        assert dataset_file.read() == 'test_content'


@patch('sdgym.s3.boto3')
def test__download_dataset_private_bucket(boto3_mock, tmpdir):
    """Test the ``sdv.datasets._download_dataset`` method.

    It calls `_download_dataset` with a dataset in a private bucket and uses aws credentials.

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
    modality = 'single_table'
    dataset = 'my_dataset'
    bucket = 's3://my_bucket'
    aws_key = 'my_key'
    aws_secret = 'my_secret'
    bytesio = io.BytesIO()

    with ZipFile(bytesio, mode='w') as zf:
        zf.writestr(dataset, 'test_content')

    s3_mock = Mock()
    body_mock = Mock()
    body_mock.read.return_value = bytesio.getvalue()
    obj = {'Body': body_mock}
    s3_mock.get_object.return_value = obj
    boto3_mock.client.return_value = s3_mock

    # run
    _download_dataset(
        modality,
        dataset,
        datasets_path=str(tmpdir),
        bucket=bucket,
        aws_key=aws_key,
        aws_secret=aws_secret,
    )

    # asserts
    boto3_mock.client.assert_called_once_with(
        's3', aws_access_key_id=aws_key, aws_secret_access_key=aws_secret
    )
    s3_mock.get_object.assert_called_once_with(
        Bucket='my_bucket', Key=f'{modality.upper()}/{dataset}.zip'
    )
    with open(f'{tmpdir}/{dataset}') as dataset_file:
        assert dataset_file.read() == 'test_content'


@patch('sdgym.datasets.Path')
def test__get_dataset_path(mock_path):
    """Test that the path to the dataset is returned if it already exists."""
    # Setup
    modality = 'single_table'
    dataset = 'test_dataset'
    datasets_path = 'local_path'
    mock_path.return_value.__rtruediv__.side_effect = [False, False, True]

    # Run
    path = _get_dataset_path(modality, dataset, datasets_path)

    # Assert
    assert path == mock_path.return_value


def test_get_bucket_name():
    """Test that the bucket name is returned for s3 path."""
    # Setup
    bucket = 's3://bucket-name'

    # Run
    bucket_name = _get_bucket_name(bucket)

    # Assert
    assert bucket_name == 'bucket-name'


def test_get_bucket_name_local_folder():
    """Test that the bucket name is returned for a local path."""
    # Setup
    bucket = 'bucket-name'

    # Run
    bucket_name = _get_bucket_name(bucket)

    # Assert
    assert bucket_name == 'bucket-name'


@patch('sdgym.datasets._get_available_datasets')
def test_get_available_datasets(helper_mock):
    """Test that the modality is set to single-table."""
    # Run
    get_available_datasets()

    # Assert
    helper_mock.assert_called_once_with('single_table')


def test_get_available_datasets_results():
    # Run
    tables_info = get_available_datasets()

    # Assert
    expected_table = pd.DataFrame({
        'dataset_name': [
            'adult',
            'alarm',
            'census',
            'child',
            'covtype',
            'expedia_hotel_logs',
            'insurance',
            'intrusion',
            'news',
        ],
        'size_MB': [
            '3.907448',
            '4.520128',
            '98.165608',
            '3.200128',
            '255.645408',
            '0.200128',
            '3.340128',
            '162.039016',
            '18.712096',
        ],
        'num_tables': [1] * 9,
    })
    expected_table['size_MB'] = expected_table['size_MB'].astype(float).round(2)
    assert len(expected_table.merge(tables_info.round(2))) == len(expected_table)


@patch('sdgym.datasets._get_dataset_path')
@patch('sdgym.datasets.ZipFile')
@patch('sdgym.datasets.Path')
def test_get_dataset_paths(path_mock, zipfile_mock, helper_mock):
    """Test that the dataset paths are generated correctly."""
    # Setup
    local_path = 'test_local_path'
    bucket_path_mock = Mock()
    bucket_path_mock.exists.return_value = True
    path_mock.side_effect = [Path('datasets_folder'), bucket_path_mock, bucket_path_mock]
    bucket_path_mock.iterdir.return_value = [
        Path('test_local_path/dataset_1.zip'),
        Path('test_local_path/dataset_2'),
    ]

    # Run
    get_dataset_paths(None, None, local_path, None, None)

    # Assert
    zipfile_mock.return_value.extractall.assert_called_once_with(Path('datasets_folder/dataset_1'))
    helper_mock.assert_has_calls([
        call(
            'single_table',
            Path('datasets_folder/dataset_1'),
            Path('datasets_folder'),
            'test_local_path',
            None,
            None,
        ),
        call(
            'single_table',
            Path('test_local_path/dataset_2'),
            Path('datasets_folder'),
            'test_local_path',
            None,
            None,
        ),
    ])


def test_load_dataset_limit_dataset_size():
    """Test ``limit_dataset_size`` selects a slice of the metadata and data."""
    # Run
    data, metadata_dict = load_dataset(
        modality='single_table', dataset='adult', limit_dataset_size=True
    )

    # Assert
    assert list(data.columns) == [
        'age',
        'workclass',
        'fnlwgt',
        'education',
        'education-num',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
    ]
    assert data.shape == (1000, 10)
    assert metadata_dict == {
        'columns': {
            'age': {'sdtype': 'numerical', 'computer_representation': 'Int64'},
            'workclass': {'sdtype': 'categorical'},
            'fnlwgt': {'sdtype': 'numerical', 'computer_representation': 'Int64'},
            'education': {'sdtype': 'categorical'},
            'education-num': {'sdtype': 'numerical', 'computer_representation': 'Int64'},
            'marital-status': {'sdtype': 'categorical'},
            'occupation': {'sdtype': 'categorical'},
            'relationship': {'sdtype': 'categorical'},
            'race': {'sdtype': 'categorical'},
            'sex': {'sdtype': 'categorical'},
        },
        'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1',
    }
