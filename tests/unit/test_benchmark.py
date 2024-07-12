from unittest.mock import ANY, MagicMock, patch

import pandas as pd
import pytest

from sdgym import benchmark_single_table
from sdgym.benchmark import (
    _check_write_permissions,
    _create_sdgym_script,
    _directory_exists,
    _format_output,
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
def test_progress_bar_updates(tqdm_mock):
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


@patch('sdgym.benchmark.boto3.client')
def test__check_write_permissions(mock_client):
    # Setup
    mock_client.return_value.put_object.side_effect = Exception('Simulated error')

    # Run and Assert
    assert not _check_write_permissions('bucket')

    # Setup for success
    mock_client.return_value.put_object.side_effect = None

    # Run and Assert
    assert _check_write_permissions('bucket')


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
        'compute_diagnostic_score': False,
        'sdmetrics': [('NewRowSynthesis', {'synthetic_sample_size': 1000})],
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
    assert "sdmetrics=[('NewRowSynthesis', {'synthetic_sample_size': 1000})]" in result
    assert 'timeout=600' in result
    assert 'compute_quality_score=False' in result
    assert 'compute_diagnostic_score=False' in result
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
        'quality_score_time': 1.0,
        'diagnostic_score_time': 3.0,
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
    scores = _format_output(mock_output, 'mock_name', 'mock_dataset', True, True, False)

    # Assert
    expected_scores = pd.DataFrame({
        'Synthesizer': ['mock_name'],
        'Dataset': ['mock_dataset'],
        'Dataset_Size_MB': [mock_output.get('dataset_size')],
        'Train_Time': [mock_output.get('train_time')],
        'Peak_Memory_MB': [mock_output.get('peak_memory')],
        'Synthesizer_Size_MB': [mock_output.get('synthesizer_size')],
        'Sample_Time': [mock_output.get('sample_time')],
        'Evaluate_Time': [15.0],
        'Diagnostic_Score': [1.0],
        'Quality_Score': [0.881],
        'NewRowSynthesis': [0.998],
        'NewMetric': [0.998],
    })
    pd.testing.assert_frame_equal(scores, expected_scores)
