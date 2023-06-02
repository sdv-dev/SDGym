from unittest.mock import ANY, MagicMock, patch

import pandas as pd
import pytest

from sdgym.benchmark import benchmark_single_table
from sdgym.synthesizers.generate import create_sdv_synthesizer_variant


def test_benchmark_single_table_only_datasets():
    """Test it works when only the ``sdv_datasets`` argument is passed.

    This is the simplest possible test, since removing the ``sdv_datasets``
    argument could take days to run.
    """
    # Run
    scores = benchmark_single_table(sdv_datasets=['fake_companies'])

    # Assert
    assert len(scores.columns) == 10
    assert list(scores['Synthesizer']) == [
        'GaussianCopulaSynthesizer',
        'FastMLPreset',
        'CTGANSynthesizer'
    ]
    assert list(scores['Dataset']) == ['fake_companies'] * 3
    assert list(scores['Dataset_Size_MB']) == [.00128] * 3
    assert scores['Train_Time'].between(0, 1000).all()
    assert scores['Peak_Memory_MB'].between(0, 1000).all()
    assert scores['Synthesizer_Size_MB'].between(0, 1000).all()
    assert scores['Sample_Time'].between(0, 1000).all()
    assert scores['Evaluate_Time'].between(0, 1000).all()
    assert scores['Quality_Score'].between(.5, 1).all()
    assert list(scores['NewRowSynthesis']) == [1.0] * 3


def test_benchmark_single_table_synthesizers_none():
    """Test it works when ``synthesizers`` is None."""
    # Setup
    synthesizer_variant = create_sdv_synthesizer_variant(
        'FastMLVariant',
        'FastMLPreset',
        synthesizer_parameters={'name': 'FAST_ML'}
    )

    # Run
    scores = benchmark_single_table(
        synthesizers=None,
        custom_synthesizers=[synthesizer_variant],
        sdv_datasets=['fake_companies']
    )

    # Assert
    assert scores.shape == (1, 10)
    scores = scores.iloc[0]
    assert scores['Synthesizer'] == 'Variant:FastMLVariant'
    assert scores['Dataset'] == 'fake_companies'
    assert scores['Dataset_Size_MB'] == 0.00128
    assert .5 < scores['Quality_Score'] < 1
    assert scores[[
        'Train_Time',
        'Peak_Memory_MB',
        'Synthesizer_Size_MB',
        'Sample_Time',
        'Evaluate_Time'
    ]].between(0, 1000).all()


def test_benchmark_single_table_no_synthesizers():
    """Test it works when no synthesizers are passed.

    It should return an empty dataframe.
    """
    # Run
    result = benchmark_single_table(synthesizers=None)

    # Assert
    expected = pd.DataFrame({
        'Synthesizer': [],
        'Dataset': [],
        'Dataset_Size_MB': [],
        'Train_Time': [],
        'Peak_Memory_MB': [],
        'Synthesizer_Size_MB': [],
        'Sample_Time': [],
        'Evaluate_Time': [],
        'Quality_Score': [],
        'NewRowSynthesis': [],
    })
    pd.testing.assert_frame_equal(result, expected)


def test_benchmark_single_table_no_datasets():
    """Test it works when no datasets are passed.

    It should return an empty dataframe.
    """
    # Run
    result = benchmark_single_table(sdv_datasets=None)

    # Assert
    expected = pd.DataFrame({
        'Synthesizer': [],
        'Dataset': [],
        'Dataset_Size_MB': [],
        'Train_Time': [],
        'Peak_Memory_MB': [],
        'Synthesizer_Size_MB': [],
        'Sample_Time': [],
        'Evaluate_Time': [],
        'Quality_Score': [],
        'NewRowSynthesis': [],
    })
    pd.testing.assert_frame_equal(result, expected)


def test_benchmark_single_table_no_synthesizers_with_parameters():
    """Test it works when no synthesizers are passed but other parameters are."""
    # Run
    result = benchmark_single_table(
        synthesizers=None,
        sdv_datasets=['fake_companies'],
        sdmetrics=[('a', {'params'}), ('b', {'more_params'})],
        compute_quality_score=False
    )

    # Assert
    expected = pd.DataFrame({
        'Synthesizer': [],
        'Dataset': [],
        'Dataset_Size_MB': [],
        'Train_Time': [],
        'Peak_Memory_MB': [],
        'Synthesizer_Size_MB': [],
        'Sample_Time': [],
        'Evaluate_Time': [],
        'a': [],
        'b': []
    })
    pd.testing.assert_frame_equal(result, expected)


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
