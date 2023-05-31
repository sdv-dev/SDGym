from unittest.mock import ANY, MagicMock, patch

import pandas as pd
import pytest

from sdgym.benchmark import benchmark_single_table, _score_with_timeout


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
def test__score_with_timeout(mock_multiprocessing, mock__score):
    """Test ``_score_with_timeout``.

    Test that when ``_score_with_timeout`` is being called with timeout, this creates a process
    that has as ``target``  the ``_score`` function, then this process is started by getting called
    its ``start`` method and then it's ``join`` method is called with the given ``timeout`` and
    finally is being terminated by ``terminate``.
    """
    # Setup
    timeout = 10
    synthesizer = 'GaussianCopulaSynthesizer'
    data = 'data'
    metadata = 'metadata'
    metrics = 'metrics'
    max_rows = 10
    compute_quality_score = True
    modality = 'single_table'
    dataset_name = 'students'
    manager = mock_multiprocessing.Manager.return_value
    manager_dict = manager.__enter__.return_value.dict.return_value
    mocked_process = mock_multiprocessing.Process.return_value

    # Run
    result = _score_with_timeout(
        timeout=timeout,
        synthesizer=synthesizer,
        data=data,
        metadata=metadata,
        metrics=metrics,
        max_rows=max_rows,
        compute_quality_score=compute_quality_score,
        modality=modality,
        dataset_name=dataset_name,
    )

    # Assert
    assert result == {}
    mock_multiprocessing.Process.assert_called_once_with(
        target=mock__score,
        args=(
            'GaussianCopulaSynthesizer',
            'data',
            'metadata',
            'metrics',
            manager_dict,
            10,
            True,
            'single_table',
            'students'
        )
    )
    mocked_process.start.assert_called_once_with()
    mocked_process.join.assert_called_once_with(10)
    mocked_process.terminate.assert_called_once_with()
