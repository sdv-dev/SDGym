from unittest.mock import ANY, MagicMock, patch

import pandas as pd
import pytest

from sdgym import benchmark_single_table
from sdgym.benchmark import _create_sdgym_script
from sdgym.synthesizers import CTGANSynthesizer, GaussianCopulaSynthesizer


@patch('sdgym.benchmark.os.path')
def test_output_file_exists(path_mock):
    """Test the benchmark function when the output path already exists."""
    # Setup
    path_mock.exists.return_value = True
    output_filepath = 'test_output.csv'

    # Run and assert
    with pytest.raises(
        ValueError,
        match='test_output.csv already exists. Please provide a file that does not already exist.',
    ):
        benchmark_single_table(
            synthesizers=['DataIdentity', 'IndependentSynthesizer', 'UniformSynthesizer'],
            sdv_datasets=['student_placements'],
            output_filepath=output_filepath
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
    manager_dict = {
        'timeout': True,
        'error': 'Synthesizer Timeout'
    }
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
        'Quality_Score': {0: None},
        'error': {0: 'Synthesizer Timeout'}
    })
    pd.testing.assert_frame_equal(scores, expected_scores)


@patch('sdgym.benchmark.boto3.session.Session')
@patch('sdgym.benchmark._create_instance_on_ec2')
def test_run_ec2_flag(create_ec2_mock, session_mock):
    """Test that the benchmarking function updates the progress bar on one line."""
    # Setup
    create_ec2_mock.return_value = MagicMock()
    session_mock.get_credentials.return_value = MagicMock()

    # Run
    benchmark_single_table(run_on_ec2=True, output_filepath='BucketName/path')

    # Assert
    create_ec2_mock.assert_called_once()

    # Run
    with pytest.raises(ValueError,
                       match=r'In order to run on EC2, please provide an S3 folder output.'):
        benchmark_single_table(run_on_ec2=True)

    # Assert
    create_ec2_mock.assert_called_once()

    # Run
    with pytest.raises(ValueError, match=r"""Invalid output_filepath.
                   The path should be structured as: <s3_bucket_name>/<path_to_file>
                   Please make sure the path exists and permissions are given."""):
        benchmark_single_table(run_on_ec2=True, output_filepath='Wrong_Format')

    # Assert
    create_ec2_mock.assert_called_once()


@patch('sdgym.benchmark.boto3.session.Session')
def test__create_sdgym_script(session_mock):
    session_mock.get_credentials.return_value = MagicMock()
    # Setup
    test_params = {
        'synthesizers': [GaussianCopulaSynthesizer, CTGANSynthesizer],
        'custom_synthesizers': None,
        'sdv_datasets': [
            'adult', 'alarm', 'census',
            'child', 'expedia_hotel_logs',
            'insurance', 'intrusion', 'news', 'covtype'
        ],
        'additional_datasets_folder': None,
        'limit_dataset_size': True,
        'compute_quality_score': False,
        'sdmetrics': [('NewRowSynthesis', {'synthetic_sample_size': 1000})],
        'timeout': 600,
        'output_filepath': 'sdgym-results/address_comments.csv',
        'detailed_results_folder': None,
        'show_progress': False,
        'multi_processing_config': None,
        'dummy': True
    }

    result = _create_sdgym_script(test_params, 'Bucket/Filepath')

    assert 'synthesizers=[GaussianCopulaSynthesizer, CTGANSynthesizer, ]' in result
    assert 'detailed_results_folder=None' in result
    assert 'multi_processing_config=None' in result
    assert "sdmetrics=[('NewRowSynthesis', {'synthetic_sample_size': 1000})]" in result
    assert 'timeout=600' in result
    assert 'compute_quality_score=False' in result
    assert 'import boto3' in result
