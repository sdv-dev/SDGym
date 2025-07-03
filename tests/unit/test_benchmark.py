import io
import json
import pickle
import re
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from unittest.mock import ANY, MagicMock, Mock, call, patch

import pandas as pd
import pytest
import yaml

from sdgym import benchmark_single_table
from sdgym.benchmark import (
    _check_write_permissions,
    _create_sdgym_script,
    _directory_exists,
    _format_output,
    _handle_deprecated_parameters,
    _setup_output_destination,
    _setup_output_destination_aws,
    _update_run_id_file,
    _upload_dataframe_to_s3,
    _upload_pickle_to_s3,
    _validate_aws_inputs,
    _validate_output_destination,
    _write_run_id_file,
)
from sdgym.synthesizers import GaussianCopulaSynthesizer


@patch('sdgym.benchmark.os.path')
def test_output_file_exists(path_mock):
    """Test the benchmark function when the output path already exists."""
    # Setup
    path_mock.exists.return_value = True
    output_filepath = 's3://test_output.csv'

    # Run and assert
    with pytest.raises(
        ValueError,
        match='test_output.csv already exists. Please provide a file that does not already exist.',
    ):
        benchmark_single_table(
            synthesizers=['DataIdentity', 'ColumnSynthesizer', 'UniformSynthesizer'],
            sdv_datasets=['student_placements'],
            output_filepath=output_filepath,
        )


@patch('sdgym.benchmark.tqdm.tqdm')
@patch('sdgym.benchmark._handle_deprecated_parameters')
def test_benchmark_single_table_deprecated_params(mock_handle_deprecated, tqdm_mock):
    """Test that the benchmarking function updates the progress bar on one line."""
    # Setup
    scores_mock = MagicMock()
    scores_mock.__iter__.return_value = [pd.DataFrame([1, 2, 3])]
    tqdm_mock.return_value = scores_mock

    # Run
    benchmark_single_table(
        synthesizers=['DataIdentity'],
        sdv_datasets=['student_placements'],
        show_progress=True,
    )

    # Assert
    mock_handle_deprecated.assert_called_once_with(None, None, None)
    tqdm_mock.assert_called_once_with(ANY, total=1, position=0, leave=True)


@patch('sdgym.benchmark._score')
@patch('sdgym.benchmark.multiprocessing')
def test_benchmark_single_table_with_timeout(mock_multiprocessing, mock__score):
    """Test that benchmark runs with timeout."""
    # Setup
    mocked_process = mock_multiprocessing.Process.return_value
    manager = mock_multiprocessing.Manager.return_value
    manager_dict = {'timeout': True, 'error': 'Synthesizer Timeout'}
    manager.__enter__.return_value.dict.return_value = manager_dict

    # Run
    scores = benchmark_single_table(
        synthesizers=['GaussianCopulaSynthesizer'],
        sdv_datasets=['student_placements'],
        timeout=1,
    )

    # Assert
    mocked_process.start.assert_called_once_with()
    mocked_process.join.assert_called_once_with(1)
    mocked_process.terminate.assert_called_once_with()
    expected_scores = pd.DataFrame({
        'Synthesizer': {0: 'GaussianCopulaSynthesizer'},
        'Dataset': {0: 'student_placements'},
        'Dataset_Size_MB': {0: None},
        'Train_Time': {0: None},
        'Peak_Memory_MB': {0: None},
        'Synthesizer_Size_MB': {0: None},
        'Sample_Time': {0: None},
        'Evaluate_Time': {0: None},
        'Diagnostic_Score': {0: None},
        'Quality_Score': {0: None},
        'Privacy_Score': {0: None},
        'error': {0: 'Synthesizer Timeout'},
    })
    pd.testing.assert_frame_equal(scores, expected_scores)


@patch('sdgym.benchmark.boto3.client')
def test__directory_exists(mock_client):
    # Setup
    mock_client.return_value.list_objects_v2.return_value = {
        'Contents': [
            {
                'Key': 'example.txt',
                'ETag': '"1234567890abcdef1234567890abcdef"',
                'Size': 1024,
                'StorageClass': 'STANDARD',
            },
            {
                'Key': 'example_folder/',
                'ETag': '"0987654321fedcba0987654321fedcba"',
                'Size': 0,
                'StorageClass': 'STANDARD',
            },
        ],
        'CommonPrefixes': [
            {'Prefix': 'example_folder/subfolder1/'},
            {'Prefix': 'example_folder/subfolder2/'},
        ],
    }

    # Run and Assert
    assert _directory_exists('bucket', 'file_path/mock.csv')

    # Setup Failure
    mock_client.return_value.list_objects_v2.return_value = {}

    # Run and Assert
    assert not _directory_exists('bucket', 'file_path/mock.csv')


def test__check_write_permissions():
    """Test the `_check_write_permissions` function."""
    # Setup
    mock_client = Mock()

    # Run and Assert
    assert _check_write_permissions(mock_client, 'bucket')
    mock_client.put_object.side_effect = Exception('Simulated error')
    assert not _check_write_permissions(mock_client, 'bucket')


@patch('sdgym.benchmark._directory_exists')
@patch('sdgym.benchmark._check_write_permissions')
@patch('sdgym.benchmark.boto3.session.Session')
@patch('sdgym.benchmark._create_instance_on_ec2')
def test_run_ec2_flag(create_ec2_mock, session_mock, mock_write_permissions, mock_directory_exists):
    """Test that the benchmarking function updates the progress bar on one line."""
    # Setup
    create_ec2_mock.return_value = MagicMock()
    session_mock.get_credentials.return_value = MagicMock()
    mock_write_permissions.return_value = True
    mock_directory_exists.return_value = True

    # Run
    benchmark_single_table(run_on_ec2=True, output_filepath='s3://BucketName/path')

    # Assert
    create_ec2_mock.assert_called_once()

    # Run
    with pytest.raises(
        ValueError, match=r'In order to run on EC2, please provide an S3 folder output.'
    ):
        benchmark_single_table(run_on_ec2=True)

    # Assert
    create_ec2_mock.assert_called_once()

    # Run
    with pytest.raises(
        ValueError,
        match=r"""Invalid S3 path format.
                         Expected 's3://<bucket_name>/<path_to_file>'.""",
    ):
        benchmark_single_table(run_on_ec2=True, output_filepath='Wrong_Format')

    # Assert
    create_ec2_mock.assert_called_once()

    # Setup for failure in permissions
    mock_write_permissions.return_value = False

    # Run
    with pytest.raises(ValueError, match=r'No write permissions allowed for the bucket.'):
        benchmark_single_table(run_on_ec2=True, output_filepath='s3://BucketName/path')

    # Setup for failure in directory exists
    mock_write_permissions.return_value = True
    mock_directory_exists.return_value = False

    # Run
    with pytest.raises(ValueError, match=r'Directories in mock/path do not exist'):
        benchmark_single_table(run_on_ec2=True, output_filepath='s3://BucketName/mock/path')


@patch('sdgym.benchmark._directory_exists')
@patch('sdgym.benchmark._check_write_permissions')
@patch('sdgym.benchmark.boto3.session.Session')
def test__create_sdgym_script(session_mock, mock_write_permissions, mock_directory_exists):
    """Test that the created SDGym script contains the expected values."""
    # Setup
    session_mock.get_credentials.return_value = MagicMock()
    test_params = {
        'synthesizers': [GaussianCopulaSynthesizer, 'CTGANSynthesizer'],
        'custom_synthesizers': None,
        'sdv_datasets': [
            'adult',
            'alarm',
            'census',
            'child',
            'expedia_hotel_logs',
            'insurance',
            'intrusion',
            'news',
            'covtype',
        ],
        'limit_dataset_size': True,
        'compute_quality_score': False,
        'compute_privacy_score': False,
        'compute_diagnostic_score': False,
        'sdmetrics': None,
        'timeout': 600,
        'output_filepath': 's3://sdgym-results/address_comments.csv',
        'detailed_results_folder': None,
        'additional_datasets_folder': 'Details/',
        'show_progress': False,
        'multi_processing_config': None,
        'dummy': True,
    }
    mock_write_permissions.return_value = True
    mock_directory_exists.return_value = True

    # Run
    result = _create_sdgym_script(test_params, 's3://Bucket/Filepath')

    # Assert
    assert 'synthesizers=[GaussianCopulaSynthesizer, CTGANSynthesizer]' in result
    assert 'detailed_results_folder=None' in result
    assert "additional_datasets_folder='Details/'" in result
    assert 'multi_processing_config=None' in result
    assert 'sdmetrics=None' in result
    assert 'timeout=600' in result
    assert 'compute_quality_score=False' in result
    assert 'compute_diagnostic_score=False' in result
    assert 'compute_privacy_score=False' in result
    assert 'import boto3' in result


def test__format_output():
    """Test the method ``_format_output`` and confirm that metrics are properly computed."""
    # Setup
    mock_dataframe = pd.DataFrame([])
    mock_output = {
        'timeout': False,
        'dataset_size': 3.907452,
        'synthetic_data': mock_dataframe,
        'train_time': 267.028721,
        'sample_time': 1.039627,
        'synthesizer_size': 0.936981,
        'peak_memory': 127.729832,
        'diagnostic_score': 1.0,
        'quality_score': 0.881,
        'privacy_score': 0.588,
        'quality_score_time': 1.0,
        'diagnostic_score_time': 3.0,
        'privacy_score_time': 4.0,
        'scores': [
            {
                'metric': 'NewRowSynthesis',
                'error': None,
                'score': 0.998,
                'normalized_score': 0.998,
                'metric_time': 6.0,
            },
            {
                'metric': 'NewMetric',
                'error': None,
                'score': 0.998,
                'normalized_score': 0.998,
                'metric_time': 5.0,
            },
        ],
    }

    # Run
    scores = _format_output(mock_output, 'mock_name', 'mock_dataset', True, True, True, False)

    # Assert
    expected_scores = pd.DataFrame({
        'Synthesizer': ['mock_name'],
        'Dataset': ['mock_dataset'],
        'Dataset_Size_MB': [mock_output.get('dataset_size')],
        'Train_Time': [mock_output.get('train_time')],
        'Peak_Memory_MB': [mock_output.get('peak_memory')],
        'Synthesizer_Size_MB': [mock_output.get('synthesizer_size')],
        'Sample_Time': [mock_output.get('sample_time')],
        'Evaluate_Time': [19.0],
        'Diagnostic_Score': [1.0],
        'Quality_Score': [0.881],
        'Privacy_Score': [0.588],
        'NewRowSynthesis': [0.998],
        'NewMetric': [0.998],
    })
    pd.testing.assert_frame_equal(scores, expected_scores)


def test__handle_deprecated_parameters():
    """Test the ``_handle_deprecated_parameters`` function."""
    # Setup
    output_filepath = 's3://BucketName/path'
    detailed_results_folder = 'mock/path'
    multi_processing_config = {'num_processes': 4}
    expected_message_1 = (
        "Parameters 'detailed_results_folder', 'output_filepath' are deprecated in the "
        "'benchmark_single_table' function and will be removed in October 2025. For saving"
        " results, please use the 'output_destination' parameter. For running SDGym remotely"
        " on AWS please use the 'benchmark_single_table_aws' method."
    )
    expected_message_2 = (
        "Parameters 'detailed_results_folder', 'multi_processing_config', 'output_filepath'"
        " are deprecated in the 'benchmark_single_table' function and will be removed in October"
        " 2025. For saving results, please use the 'output_destination' parameter. For running"
        " SDGym remotely on AWS please use the 'benchmark_single_table_aws' method."
    )

    # Run and Assert
    _handle_deprecated_parameters(None, None, None)
    with pytest.warns(FutureWarning, match=expected_message_1):
        _handle_deprecated_parameters(output_filepath, detailed_results_folder, None)

    with pytest.warns(FutureWarning, match=expected_message_2):
        _handle_deprecated_parameters(
            output_filepath, detailed_results_folder, multi_processing_config
        )


def test__validate_output_destination(tmp_path):
    """Test the `_validate_output_destination` function."""
    # Setup
    wrong_type = 12345
    aws_destination = 's3://valid-bucket/path/to/file'
    valid_destination = tmp_path / 'valid-destination'
    err_1 = re.escape(
        'The `output_destination` parameter must be a string representing the output path.'
    )
    err_2 = re.escape(
        'The `output_destination` parameter cannot be an S3 path. '
        'Please use `benchmark_single_table_aws` instead.'
    )

    # Run and Assert
    _validate_output_destination(str(valid_destination))
    with pytest.raises(ValueError, match=err_1):
        _validate_output_destination(wrong_type)

    with pytest.raises(ValueError, match=err_2):
        _validate_output_destination(aws_destination)


@patch('sdgym.benchmark._validate_aws_inputs')
def test__validate_output_destination_with_aws_keys(mock_validate):
    """Test the `_validate_output_destination` function with AWS keys."""
    # Setup
    output_destination = 's3://my-bucket/path/to/file'
    aws_keys = {
        'aws_access_key_id': 'mock_access_key',
        'aws_secret_access_key': 'mock_secret_key',
    }

    # Run
    _validate_output_destination(output_destination, aws_keys)

    # Assert
    mock_validate.assert_called_once_with(
        output_destination, aws_keys['aws_access_key_id'], aws_keys['aws_secret_access_key']
    )


def test__setup_output_destination(tmp_path):
    """Test the `_setup_output_destination` function."""
    # Setup
    output_destination = tmp_path / 'output_destination'
    synthesizers = ['GaussianCopulaSynthesizer', 'CTGANSynthesizer']
    datasets = ['adult', 'census']
    today = datetime.today().strftime('%m_%d_%Y')
    base_path = output_destination / f'SDGym_results_{today}'

    # Run
    result_1 = _setup_output_destination(None, synthesizers, datasets)
    result_2 = _setup_output_destination(output_destination, synthesizers, datasets)

    # Assert
    expected = {
        dataset: {
            **{
                synth: {
                    'synthesizer': str(
                        base_path / f'{dataset}_{today}' / synth / f'{synth}_synthesizer.pkl'
                    ),
                    'synthetic_data': str(
                        base_path / f'{dataset}_{today}' / synth / f'{synth}_synthetic_data.csv'
                    ),
                    'benchmark_result': str(
                        base_path / f'{dataset}_{today}' / synth / f'{synth}_benchmark_result.csv'
                    ),
                    'run_id': str(base_path / f'run_{today}_1.yaml'),
                    'results': str(base_path / f'results_{today}_1.csv'),
                }
                for synth in synthesizers
            },
        }
        for dataset in datasets
    }
    assert result_1 == {}
    assert json.loads(json.dumps(result_2)) == expected


@patch('sdgym.benchmark.datetime')
def test__write_run_id_file(mock_datetime, tmp_path):
    """Test the `_write_run_id_file` method."""
    # Setup
    output_destination = tmp_path / 'output_destination'
    output_destination.mkdir()
    mock_datetime.today.return_value.strftime.return_value = '06_26_2025'
    file_name = {'run_id': f'{output_destination}/run_06_26_2025_1.yaml'}
    jobs = [
        ({'name': 'GaussianCopulaSynthesizer'}, 'adult', None, file_name),
        ({'name': 'CTGANSynthesizer'}, 'census', None, None),
    ]
    expected_jobs = [['adult', 'GaussianCopulaSynthesizer'], ['census', 'CTGANSynthesizer']]
    synthesizers = ['GaussianCopulaSynthesizer', 'CTGANSynthesizer', 'RealTabFormerSynthesizer']

    # Run
    _write_run_id_file(synthesizers, jobs)

    # Assert
    assert Path(file_name['run_id']).exists()
    with open(file_name['run_id'], 'r') as file:
        run_id_data = yaml.safe_load(file)
        assert run_id_data['run_id'] == 'run_06_26_2025_1'
        assert run_id_data['starting_date'] == '06_26_2025'
        assert run_id_data['jobs'] == expected_jobs
        assert run_id_data['sdgym_version'] == version('sdgym')
        assert run_id_data['sdv_version'] == version('sdv')
        assert run_id_data['realtabformer_version'] == version('realtabformer')
        assert run_id_data['completed_date'] is None


@patch('sdgym.benchmark.datetime')
def test__update_run_id_file(mock_datetime, tmp_path):
    """Test the `_update_run_id_file` method."""
    # Setup
    output_destination = tmp_path / 'output_destination'
    output_destination.mkdir()
    metadata = {'run_id': 'run_06_25_2025_1', 'starting_date': '06_25_2025', 'completed_date': None}
    run_id_file = output_destination / 'run_06_25_2025_1.yaml'
    run_id = 'run_06_25_2025_1'
    mock_datetime.today.return_value.strftime.return_value = '06_26_2025'
    with open(run_id_file, 'w') as file:
        yaml.dump(metadata, file)

    # Run
    _update_run_id_file(run_id_file)

    # Assert
    with open(run_id_file, 'r') as file:
        run_id_data = yaml.safe_load(file)
        assert run_id_data['completed_date'] == '06_26_2025'
        assert run_id_data['starting_date'] == '06_25_2025'
        assert run_id_data['run_id'] == run_id


def test_setup_output_destination_aws():
    """Test the `_setup_output_destination_aws` function."""
    # Setup
    output_destination = 's3://my-bucket/results'
    synthesizers = ['GaussianCopulaSynthesizer', 'CTGANSynthesizer']
    datasets = ['Dataset1', 'Dataset2']
    s3_client_mock = Mock()

    # Run
    paths = _setup_output_destination_aws(
        output_destination, synthesizers, datasets, s3_client_mock
    )

    # Assert
    today = datetime.today().strftime('%m_%d_%Y')
    bucket_name = 'my-bucket'
    top_folder = f'results/SDGym_results_{today}'
    expected_calls = [call(Bucket=bucket_name, Key=top_folder + '/')]
    for dataset in datasets:
        dataset_folder = f'{top_folder}/{dataset}_{today}'
        expected_calls.append(call(Bucket=bucket_name, Key=dataset_folder + '/'))
        for synth in synthesizers:
            synth_folder = f'{dataset_folder}/{synth}'
            expected_calls.append(call(Bucket=bucket_name, Key=synth_folder + '/'))

    s3_client_mock.put_object.assert_has_calls(expected_calls, any_order=True)
    for dataset in datasets:
        for synth in synthesizers:
            assert 'synthesizer' in paths[dataset][synth]
            assert paths[dataset][synth]['synthesizer'] == (
                f's3://{bucket_name}/{top_folder}/{dataset}_{today}/{synth}/{synth}_synthesizer.pkl'
            )
            assert 'synthetic_data' in paths[dataset][synth]
            assert paths[dataset][synth]['synthetic_data'] == (
                f's3://{bucket_name}/{top_folder}/{dataset}_{today}/{synth}/{synth}_synthetic_data.csv'
            )


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


@patch('sdgym.benchmark.boto3.client')
@patch('sdgym.benchmark._check_write_permissions')
def test_validate_aws_inputs_valid(mock_check_write_permissions, mock_boto3_client):
    """Test `_validate_aws_inputs` with valid inputs and credentials."""
    # Setup
    valid_url = 's3://my-bucket/some/path'
    s3_client_mock = Mock()
    mock_boto3_client.return_value = s3_client_mock
    mock_check_write_permissions.return_value = True

    # Run
    result = _validate_aws_inputs(
        output_destination=valid_url, aws_access_key_id='AKIA...', aws_secret_access_key='SECRET'
    )

    # Assert
    mock_boto3_client.assert_called_once_with(
        's3', aws_access_key_id='AKIA...', aws_secret_access_key='SECRET'
    )
    s3_client_mock.head_bucket.assert_called_once_with(Bucket='my-bucket')
    mock_check_write_permissions.assert_called_once_with(s3_client_mock, 'my-bucket')
    assert result == s3_client_mock


def test_validate_aws_inputs_invalid():
    """Test `_validate_aws_inputs` raises ValueError for invalid inputs."""
    # Setup
    invalid_url_type = 123
    invalid_url_no_s3 = 'https://my-bucket/path'
    invalid_url_empty_bucket = 's3://'

    # Run and Assert
    with pytest.raises(
        ValueError,
        match=re.escape(
            'The `output_destination` parameter must be a string representing the S3 URL.'
        ),
    ):
        _validate_aws_inputs(invalid_url_type, None, None)

    with pytest.raises(
        ValueError,
        match=re.escape("'output_destination' must be an S3 URL starting with 's3://'. "),
    ):
        _validate_aws_inputs(invalid_url_no_s3, None, None)

    with pytest.raises(ValueError, match=re.escape(f'Invalid S3 URL: {invalid_url_empty_bucket}')):
        _validate_aws_inputs(invalid_url_empty_bucket, None, None)


@patch('sdgym.benchmark.boto3.client')
@patch('sdgym.benchmark._check_write_permissions')
def test_validate_aws_inputs_permission_error(mock_check_write_permissions, mock_boto3_client):
    """Test `_validate_aws_inputs` raises PermissionError when write permission is missing."""
    valid_url = 's3://my-bucket/some/path'
    s3_client_mock = Mock()
    mock_boto3_client.return_value = s3_client_mock
    mock_check_write_permissions.return_value = False

    with pytest.raises(
        PermissionError,
        match=re.escape(
            'No write permissions for the S3 bucket: my-bucket. '
            'Please check your AWS credentials or bucket policies.'
        ),
    ):
        _validate_aws_inputs(valid_url, None, None)
