from pathlib import Path
from unittest.mock import ANY, Mock, call, patch

import botocore
import numpy as np
import pandas as pd
import pytest

from sdgym.datasets import (
    SDV_DATASETS_PRIVATE_BUCKET,
    SDV_DATASETS_PUBLIC_BUCKET,
    _download_dataset,
    _genereate_dataset_info,
    _get_bucket_name,
    _get_dataset_path_and_download,
    _load_dataset_with_client,
    _load_private_sdv_demo_dataset,
    _load_sdv_demo_dataset,
    _path_contains_data_and_metadata,
    _validate_modality,
    dataset_to_bucket,
    get_data_and_metadata_from_path,
    get_dataset_paths,
    load_dataset,
)


@patch('sdgym.datasets.get_s3_client')
@patch('sdgym.datasets._get_bucket_name')
@patch('sdgym.datasets._list_s3_bucket_contents')
@patch('sdgym.datasets.os.makedirs')
def test__download_dataset(makedirs_mock, list_mock, bucket_name_mock, s3_client_mock):
    """Test that the dataset is downloaded successfully when found in S3."""
    # Setup
    modality = 'single_table'
    dataset_name = 'test_dataset'
    datasets_path = Path('/tmp/datasets')
    bucket = 's3://fake-bucket'

    # Mocks
    s3_mock = Mock()
    s3_client_mock.return_value = s3_mock
    bucket_name_mock.return_value = 'fake-bucket'
    list_mock.return_value = [
        {'Key': f'{modality}/{dataset_name}/file1.csv'},
        {'Key': f'{modality}/{dataset_name}/file2.json'},
    ]

    # Run
    result = _download_dataset(modality, dataset_name, datasets_path, bucket)

    # Assert
    assert result == datasets_path
    s3_client_mock.assert_called_once_with()
    bucket_name_mock.assert_called_once_with(bucket)
    list_mock.assert_called_once_with(s3_mock, 'fake-bucket', f'{modality}/{dataset_name}/')

    expected_calls = [
        call('fake-bucket', f'{modality}/{dataset_name}/file1.csv', datasets_path / 'file1.csv'),
        call('fake-bucket', f'{modality}/{dataset_name}/file2.json', datasets_path / 'file2.json'),
    ]
    s3_mock.download_file.assert_has_calls(expected_calls)
    makedirs_mock.assert_called()


@patch('sdgym.datasets.get_s3_client')
@patch('sdgym.datasets._get_bucket_name')
@patch('sdgym.datasets._list_s3_bucket_contents', return_value=[])
def test__download_dataset_not_found(list_mock, bucket_name_mock, s3_client_mock):
    """Test that ValueError is raised if the dataset is not found in S3."""
    # Setup
    modality = 'single_table'
    dataset_name = 'missing_dataset'
    bucket = 's3://fake-bucket'
    datasets_path = Path('/tmp/datasets')

    bucket_name_mock.return_value = 'fake-bucket'
    s3_client_mock.return_value = Mock()

    # Run and Assert
    expected_message = (
        "Dataset 'missing_dataset' not found in bucket 's3://fake-bucket' for "
        "modality 'single_table'."
    )
    with pytest.raises(ValueError, match=expected_message):
        _download_dataset(modality, dataset_name, datasets_path, bucket)

    # Assert
    s3_client_mock.assert_called_once()
    bucket_name_mock.assert_called_with(bucket)
    expected_calls = [
        call(s3_client_mock.return_value, 'fake-bucket', 'single_table/missing_dataset/'),
        call(s3_client_mock.return_value, 'fake-bucket', 'sequential/missing_dataset/'),
        call(s3_client_mock.return_value, 'fake-bucket', 'multi_table/missing_dataset/'),
    ]
    list_mock.assert_has_calls(expected_calls, any_order=True)


@patch('sdgym.datasets.get_s3_client')
@patch('sdgym.datasets._get_bucket_name')
@patch('sdgym.datasets._list_s3_bucket_contents')
def test__download_dataset_with_credentials(list_mock, bucket_name_mock, s3_client_mock):
    """Test that AWS credentials and custom bucket are used correctly."""
    # Setup
    modality = 'single_table'
    dataset_name = 'secure_dataset'
    datasets_path = Path('/tmp/datasets')
    bucket = 's3://secure-bucket'

    s3_mock = Mock()
    s3_client_mock.return_value = s3_mock
    bucket_name_mock.return_value = 'secure-bucket'
    list_mock.return_value = [{'Key': f'{modality}/{dataset_name}/file.csv'}]

    # Run
    _download_dataset(modality, dataset_name, datasets_path, bucket, s3_mock)

    # Assert
    bucket_name_mock.assert_called_once_with(bucket)
    list_mock.assert_called_once_with(s3_mock, 'secure-bucket', f'{modality}/{dataset_name}/')
    s3_mock.download_file.assert_called_once()


def test__path_contains_data_and_metadata_true(monkeypatch):
    """Test that this method returns ``True`` when both metadata.json and data.zip are found."""
    # Setup
    dataset_path = Mock()
    dataset_path.iterdir.return_value = [
        Path('metadata.json'),
        Path('data.zip'),
        Path('notes.txt'),
    ]

    # Run
    result = _path_contains_data_and_metadata(dataset_path)

    # Assert
    assert result is True


def test__path_contains_data_and_metadata_missing_data(monkeypatch):
    """Test that this method returns ``False`` when only metadata file is present."""
    # Setup
    dataset_path = Mock()
    dataset_path.iterdir.return_value = [Path('metadata.json')]

    # Run
    result = _path_contains_data_and_metadata(dataset_path)

    # Assert
    assert result is False


def test__path_contains_data_and_metadata_missing_metadata(monkeypatch):
    """Test returns ``False`` when only data.zip is present."""
    # Setup
    dataset_path = Mock()
    dataset_path.iterdir.return_value = [Path('data.zip')]

    # Run
    result = _path_contains_data_and_metadata(dataset_path)

    # Assert
    assert result is False


def test__path_contains_data_and_metadata_no_relevant_files(monkeypatch):
    """Test returns ``False`` when neither ``metadata.json`` nor ``data.zip`` are found."""
    # Setup
    dataset_path = Mock()
    dataset_path.iterdir.return_value = [
        Path('metainfo.yaml'),
        Path('meta.json'),
        Path('sdv_meta.json'),
        Path('metadata.txt'),
        Path('data.csv'),
    ]

    # Run
    result = _path_contains_data_and_metadata(dataset_path)

    # Assert
    assert result is False


@patch('sdgym.datasets._download_dataset')
@patch('sdgym.datasets._path_contains_data_and_metadata', return_value=True)
@patch('sdgym.datasets.Path')
def test__get_dataset_path_and_download_local_exists(path_mock, contains_mock, download_mock):
    """Test that this function returns the dataset path directly if it already exists."""
    # Setup
    modality = 'single_table'
    dataset = 'local_dataset'
    datasets_path = Path('/tmp/datasets')

    dataset_path_mock = Mock()
    dataset_path_mock.exists.return_value = True
    path_mock.return_value = dataset_path_mock

    # Run
    result = _get_dataset_path_and_download(modality, dataset, datasets_path)

    # Assert
    assert result == dataset_path_mock
    download_mock.assert_not_called()
    contains_mock.assert_called_once_with(dataset_path_mock)


@patch('sdgym.datasets._download_dataset')
@patch('sdgym.datasets.get_s3_client')
@patch('sdgym.datasets._path_contains_data_and_metadata', return_value=False)
def test__get_dataset_path_and_download_triggers_download(
    contains_mock, s3_client_mock, download_mock
):
    """Test that `_get_dataset_path_and_download` triggers dataset download if not found locally."""
    # Setup
    modality = 'single_table'
    dataset = 'remote_dataset'
    datasets_path = Path('/tmp/datasets')
    bucket = 's3://remote-bucket'
    s3_client_mock.return_value = 's3_client'
    download_mock.return_value = Path('/tmp/datasets/single_table/remote_dataset')

    # Run
    result = _get_dataset_path_and_download(modality, dataset, datasets_path, bucket=bucket)

    # Assert
    download_mock.assert_called_once_with(
        modality, Path(dataset), datasets_path / Path(dataset), bucket, s3_client='s3_client'
    )
    assert result == download_mock.return_value


@pytest.mark.parametrize('modality', ['single_table', 'multi_table', 'sequential'])
def test__validate_modality_valid(modality):
    """Test that valid modalities do not raise an exception."""
    # Run and Assert
    _validate_modality(modality)


@pytest.mark.parametrize('invalid_modality', ['single-table', 'multi-table', 'timeseries'])
def test__validate_modality_invalid(invalid_modality):
    """Test that invalid modalities trigger a ``ValueError``."""
    # Run and Assert
    modalities_list = ', '.join(['single_table', 'multi_table', 'sequential'])
    expected_message = (
        f'Modality `{invalid_modality}` not recognized. Must be one of {modalities_list}'
    )
    with pytest.raises(ValueError, match=expected_message):
        _validate_modality(invalid_modality)


@patch('pathlib.Path.iterdir')
@patch('sdgym.datasets._read_metadata_json')
@patch('sdgym.datasets._read_zipped_data')
def test_get_data_and_metadata_both_found(read_data_mock, read_meta_mock, iterdir_mock, tmp_path):
    """Test returns both data and metadata when both files exist."""
    dataset_path = tmp_path
    metadata_file = dataset_path / 'metadata.json'
    data_file = dataset_path / 'data.zip'

    iterdir_mock.return_value = [metadata_file, data_file]
    read_meta_mock.return_value = {'info': 'meta'}
    read_data_mock.return_value = 'dataframe'

    # Run
    data, metadata = get_data_and_metadata_from_path(dataset_path, 'single_table')

    # Assert
    read_meta_mock.assert_called_once_with(metadata_file)
    read_data_mock.assert_called_once_with(zip_file_path=data_file, modality='single_table')
    assert data == 'dataframe'
    assert metadata == {'info': 'meta'}


@patch('sdgym.datasets._read_metadata_json')
@patch('sdgym.datasets._read_zipped_data')
def test_get_data_and_metadata_no_files(read_data_mock, read_meta_mock):
    """Test that this function returns None when neither data.zip nor metadata.json is found."""
    # Setup
    dataset_path = Mock()
    dataset_path.iterdir.return_value = [Path('readme.txt'), Path('notes.yaml')]

    # Run
    data, metadata = get_data_and_metadata_from_path(dataset_path, 'single_table')

    # Assert
    read_data_mock.assert_not_called()
    read_meta_mock.assert_not_called()
    assert data is None
    assert metadata is None


@patch('sdgym.datasets._parse_numeric_value')
@patch('sdgym.datasets._load_yaml_metainfo_from_s3')
def test__generate_dataset_info(load_yaml_mock, parse_value_mock):
    """Test `_genereate_dataset_info` when metainfo.yaml entry produces correct output."""
    # Setup
    s3_client = Mock()
    bucket_name = 'test-bucket'
    contents = [{'Key': 'single_table/test_dataset/metainfo.yaml'}]

    load_yaml_mock.return_value = {'dataset-size-mb': 25.5, 'num-tables': 3}
    parse_value_mock.side_effect = [25.5, 3]

    # Run
    result = _genereate_dataset_info(s3_client, bucket_name, contents)

    # Assert
    load_yaml_mock.assert_called_once_with(
        s3_client, bucket_name, 'single_table/test_dataset/metainfo.yaml'
    )
    parse_value_mock.assert_has_calls([
        call(25.5, 'test_dataset', field_name='dataset-size-mb', target_type=float),
        call(3, 'test_dataset', 'num-tables', target_type=int),
    ])
    assert result == {
        'dataset_name': ['test_dataset'],
        'size_MB': [25.5],
        'num_tables': [3],
    }


@patch('sdgym.datasets._parse_numeric_value')
@patch('sdgym.datasets._load_yaml_metainfo_from_s3')
def test__generate_dataset_info_missing_fields(load_yaml_mock, parse_value_mock):
    """Test when YAML is missing fields, np.nan is passed to parser."""
    # Setup
    s3_client = Mock()
    bucket_name = 'bucket'
    contents = [{'Key': 'single_table/datasetX/metainfo.yaml'}]

    load_yaml_mock.return_value = {}  # missing dataset-size-mb and num-tables
    parse_value_mock.side_effect = [np.nan, np.nan]

    # Run
    result = _genereate_dataset_info(s3_client, bucket_name, contents)

    # Assert
    parse_value_mock.assert_has_calls([
        call(np.nan, 'datasetX', field_name='dataset-size-mb', target_type=float),
        call(np.nan, 'datasetX', 'num-tables', target_type=int),
    ])
    assert result['dataset_name'] == ['datasetX']


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
@patch('sdgym.datasets.LOGGER')
def test_dataset_to_bucket_prefers_private(logger_mock, get_available_mock):
    """Test that datasets are mapped to private when duplicated across buckets."""
    # Setup
    get_available_mock.side_effect = [
        pd.DataFrame({'dataset_name': ['public_only', 'duplicate']}),
        pd.DataFrame({'dataset_name': ['private_only', 'duplicate']}),
    ]

    # Run
    result = dataset_to_bucket(
        'single_table',
        [SDV_DATASETS_PUBLIC_BUCKET, SDV_DATASETS_PRIVATE_BUCKET],
        s3_client='s3_client',
    )

    # Assert
    assert result == {
        'public_only': SDV_DATASETS_PUBLIC_BUCKET,
        'private_only': SDV_DATASETS_PRIVATE_BUCKET,
        'duplicate': SDV_DATASETS_PRIVATE_BUCKET,
    }
    get_available_mock.assert_has_calls([
        call('single_table', bucket=SDV_DATASETS_PUBLIC_BUCKET, s3_client='s3_client'),
        call('single_table', bucket=SDV_DATASETS_PRIVATE_BUCKET, s3_client='s3_client'),
    ])
    logger_mock.info.assert_called_once_with(
        "Dataset 'duplicate' appeared in multiple buckets. Using bucket 's3://sdv-datasets-private'."
    )


@patch('sdgym.datasets._get_available_datasets')
def test_dataset_to_bucket_skips_inaccessible_bucket(get_available_mock):
    """Test inaccessible buckets can be skipped while building the mapping."""
    # Setup
    error = botocore.exceptions.ClientError(
        {'Error': {'Code': 'AccessDenied', 'Message': 'denied'}},
        'ListObjectsV2',
    )
    get_available_mock.side_effect = [
        pd.DataFrame({'dataset_name': ['public_only']}),
        error,
    ]

    # Run
    result = dataset_to_bucket(
        'single_table',
        [SDV_DATASETS_PUBLIC_BUCKET, SDV_DATASETS_PRIVATE_BUCKET],
        s3_client='s3_client',
        skip_inaccessible=True,
    )

    # Assert
    assert result == {'public_only': SDV_DATASETS_PUBLIC_BUCKET}


@patch('sdgym.datasets._get_available_datasets')
def test_dataset_to_bucket_raises_inaccessible_bucket(get_available_mock):
    """Test inaccessible buckets raise by default."""
    # Setup
    get_available_mock.side_effect = botocore.exceptions.ClientError(
        {'Error': {'Code': 'AccessDenied', 'Message': 'denied'}},
        'ListObjectsV2',
    )

    # Run and Assert
    with pytest.raises(ValueError, match="Bucket 's3://sdv-datasets-private' is not accessible"):
        dataset_to_bucket(
            'single_table',
            [SDV_DATASETS_PRIVATE_BUCKET],
            s3_client='s3_client',
        )


@patch('sdgym.datasets.get_s3_client')
@patch('sdgym.datasets._get_metadata')
@patch('sdgym.datasets._load_data_from_zip')
@patch('sdgym.datasets._get_first_v1_metadata_bytes')
@patch('sdgym.datasets._get_data_from_bucket')
@patch('sdgym.datasets._find_data_zip_key')
@patch('sdgym.datasets._list_objects')
def test__load_private_sdv_demo_dataset(
    list_objects_mock,
    find_data_zip_key_mock,
    get_data_from_bucket_mock,
    get_first_v1_metadata_bytes_mock,
    load_data_from_zip_mock,
    get_metadata_mock,
    get_s3_client_mock,
):
    """Test the `_load_private_sdv_demo_dataset` method."""
    # Setup
    modality = 'single_table'
    dataset_name = 'demo'
    bucket = SDV_DATASETS_PRIVATE_BUCKET
    bucket_name = 'sdv-datasets-private'
    dataset_prefix = f'{modality}/{dataset_name}/'
    data_key = f'{dataset_prefix}data.zip'
    contents = [
        {'Key': f'{dataset_prefix}metadata.json'},
        {'Key': data_key},
    ]
    raw_data = b'fake zipped data'
    metadata_bytes = b'{"meta": "data"}'
    table_data = pd.DataFrame({'column': [1, 2, 3]})
    metadata_mock = Mock()

    s3_client_mock = Mock()
    list_objects_mock.return_value = contents
    find_data_zip_key_mock.return_value = data_key
    get_data_from_bucket_mock.return_value = raw_data
    get_first_v1_metadata_bytes_mock.return_value = metadata_bytes
    load_data_from_zip_mock.return_value = {'table_name': table_data}
    metadata_mock.to_dict.return_value = {'meta': 'data'}
    get_metadata_mock.return_value = metadata_mock

    # Run
    data, metadata = _load_private_sdv_demo_dataset(modality, dataset_name, bucket, s3_client_mock)

    # Assert
    get_s3_client_mock.assert_not_called()
    list_objects_mock.assert_called_once_with(
        dataset_prefix, bucket=bucket_name, client=s3_client_mock
    )
    find_data_zip_key_mock.assert_called_once_with(contents, dataset_prefix, bucket_name)
    get_data_from_bucket_mock.assert_called_once_with(
        data_key, bucket=bucket_name, client=s3_client_mock
    )
    get_first_v1_metadata_bytes_mock.assert_called_once_with(
        contents, dataset_prefix, bucket=bucket_name, client=s3_client_mock
    )
    load_data_from_zip_mock.assert_called_once_with(ANY, bucket_name, dataset_name)
    data_bytes = load_data_from_zip_mock.call_args.args[0]
    assert data_bytes.getvalue() == raw_data
    get_metadata_mock.assert_called_once_with(metadata_bytes, dataset_name)
    metadata_mock.to_dict.assert_called_once_with()
    pd.testing.assert_frame_equal(data, table_data)
    assert metadata == {'meta': 'data'}


@patch('sdgym.datasets.download_demo')
def test__load_sdv_demo_dataset_uses_download_demo(download_demo_mock):
    """Test SDV demo datasets are loaded through SDV's download_demo."""
    # Setup
    data = pd.DataFrame({'column': [1, 2]})
    metadata = Mock()
    metadata.to_dict.return_value = {'tables': {'demo': {'columns': {'column': {}}}}}
    download_demo_mock.return_value = data, metadata

    # Run
    result = _load_sdv_demo_dataset(
        modality='single_table',
        dataset_name='demo',
        bucket=SDV_DATASETS_PUBLIC_BUCKET,
    )

    # Assert
    result_data, result_metadata = result
    pd.testing.assert_frame_equal(result_data, data)
    assert result_metadata == metadata.to_dict.return_value
    download_demo_mock.assert_called_once_with(
        modality='single_table',
        dataset_name='demo',
        s3_bucket_name='sdv-datasets-public',
    )


@patch('sdgym.datasets._load_private_sdv_demo_dataset')
def test__load_sdv_demo_dataset_for_private_bucket(load_private_mock):
    """Test `_load_sdv_demo_dataset` with the SDV private bucket."""
    # Setup
    data = pd.DataFrame({'column': [1, 2]})
    metadata = {'tables': {'demo': {'columns': {'column': {}}}}}
    load_private_mock.return_value = data, metadata

    # Run
    result = _load_sdv_demo_dataset(
        modality='single_table',
        dataset_name='demo',
        bucket=SDV_DATASETS_PRIVATE_BUCKET,
        s3_client='s3_client',
    )

    # Assert
    result_data, result_metadata = result
    pd.testing.assert_frame_equal(result_data, data)
    assert result_metadata == metadata
    load_private_mock.assert_called_once_with(
        'single_table',
        'demo',
        SDV_DATASETS_PRIVATE_BUCKET,
        s3_client='s3_client',
    )


@patch('sdgym.datasets._get_dataset_subset')
@patch('sdgym.datasets.download_demo')
def test__load_sdv_demo_dataset_limits_dataset_size(download_demo_mock, subset_mock):
    """Test SDV demo dataset loading applies the dataset size limit."""
    # Setup
    data = pd.DataFrame({'column': [1, 2]})
    metadata = Mock()
    metadata.to_dict.return_value = {'tables': {'demo': {'columns': {'column': {}}}}}
    limited_data = pd.DataFrame({'column': [1]})
    limited_metadata = {'tables': {'demo': {'columns': {'column': {}}}}}
    download_demo_mock.return_value = data, metadata
    subset_mock.return_value = limited_data, limited_metadata

    # Run
    result_data, result_metadata = _load_sdv_demo_dataset(
        modality='single_table',
        dataset_name='demo',
        bucket=SDV_DATASETS_PUBLIC_BUCKET,
        limit_dataset_size=True,
    )

    # Assert
    pd.testing.assert_frame_equal(result_data, limited_data)
    assert result_metadata == limited_metadata
    subset_mock.assert_called_once_with(
        data,
        metadata.to_dict.return_value,
        modality='single_table',
    )


def test_get_dataset_paths_local_bucket(tmp_path):
    """Test datasets are discovered locally when bucket path exists."""
    # Setup
    modality = 'single_table'
    bucket = tmp_path / 'local_bucket'
    dataset1 = bucket / 'dataset_1'
    dataset2 = bucket / 'dataset_2'
    for dataset in (dataset1, dataset2):
        dataset.mkdir(parents=True)
        (dataset / 'metadata.json').touch()
        (dataset / 'data.zip').touch()

    # Run
    result = get_dataset_paths(modality, None, None, str(bucket))

    # Assert
    assert result == [dataset1, dataset2]


@patch('sdgym.datasets.get_s3_client')
@patch('sdgym.datasets._load_dataset_with_client')
def test_load_dataset_mock(mock_load_dataset_with_client, mock_get_s3_client):
    """Test `load_dataset` uses `_load_dataset_with_client` correctly."""
    # Setup
    modality = 'single_table'
    dataset_name = 'test_dataset'
    fake_data = 'dataframe'
    fake_metadata = {'meta': 'data'}
    mock_get_s3_client.return_value = 's3_client'
    mock_load_dataset_with_client.return_value = (fake_data, fake_metadata)

    # Run
    data, metadata = load_dataset(
        modality, dataset_name, aws_access_key_id='access_key', aws_secret_access_key='secret_key'
    )

    # Assert
    mock_get_s3_client.assert_called_once_with(
        aws_access_key_id='access_key', aws_secret_access_key='secret_key'
    )
    mock_load_dataset_with_client.assert_called_once_with(
        modality=modality,
        dataset_name=dataset_name,
        datasets_path=None,
        bucket=None,
        s3_client='s3_client',
        limit_dataset_size=False,
    )
    assert data == fake_data
    assert metadata == fake_metadata


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
        'METADATA_SPEC_VERSION': 'V1',
        'relationships': [],
        'tables': {
            'adult': {
                'columns': {
                    'age': {'computer_representation': 'Int64', 'sdtype': 'numerical'},
                    'education': {'sdtype': 'categorical'},
                    'education-num': {'computer_representation': 'Int64', 'sdtype': 'numerical'},
                    'fnlwgt': {'computer_representation': 'Int64', 'sdtype': 'numerical'},
                    'marital-status': {'sdtype': 'categorical'},
                    'occupation': {'sdtype': 'categorical'},
                    'race': {'sdtype': 'categorical'},
                    'relationship': {'sdtype': 'categorical'},
                    'sex': {'sdtype': 'categorical'},
                    'workclass': {'sdtype': 'categorical'},
                }
            }
        },
    }


@patch('sdgym.datasets._get_dataset_subset')
@patch('sdgym.datasets.get_data_and_metadata_from_path')
@patch('sdgym.datasets._get_dataset_path_and_download')
@patch('sdgym.datasets._validate_modality')
def test__load_dataset_with_client(
    validate_mock, path_or_download_mock, get_data_mock, subset_mock
):
    """Test that `_load_dataset_with_client` returns data and metadata without limiting size."""
    # Setup
    modality = 'single_table'
    dataset = 'test_dataset'
    fake_path = Mock()
    fake_data = 'dataframe'
    fake_metadata = {'meta': 1}

    path_or_download_mock.return_value = fake_path
    get_data_mock.return_value = (fake_data, fake_metadata)

    # Run
    data, metadata = _load_dataset_with_client(modality, dataset)

    # Assert
    validate_mock.assert_called_once_with(modality)
    path_or_download_mock.assert_called_once_with(modality, dataset, None, None, s3_client=None)
    get_data_mock.assert_called_once_with(fake_path, modality)
    subset_mock.assert_not_called()
    assert data == fake_data
    assert metadata == fake_metadata


@patch('sdgym.datasets._get_dataset_subset')
@patch('sdgym.datasets.get_data_and_metadata_from_path')
@patch('sdgym.datasets._get_dataset_path_and_download')
@patch('sdgym.datasets._validate_modality')
def test__load_dataset_with_limit(validate_mock, path_or_download_mock, get_data_mock, subset_mock):
    """Test `_load_dataset_with_client` applies dataset size limit when flag is True."""
    # Setup
    modality = 'sequential'
    dataset = 'tiny_dataset'
    fake_path = Mock()
    fake_data = 'original_data'
    fake_metadata = {'meta': 2}
    limited_data = 'limited_data'
    limited_metadata = {'meta': 'small'}

    path_or_download_mock.return_value = fake_path
    get_data_mock.return_value = (fake_data, fake_metadata)
    subset_mock.return_value = (limited_data, limited_metadata)

    # Run
    data, metadata = _load_dataset_with_client(modality, dataset, limit_dataset_size=True)

    # Assert
    validate_mock.assert_called_once_with(modality)
    get_data_mock.assert_called_once_with(fake_path, modality)
    subset_mock.assert_called_once_with(fake_data, fake_metadata, modality=modality)
    assert data == limited_data
    assert metadata == limited_metadata
