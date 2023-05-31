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
    assert scores['Quality_Score'].between(.6, 1).all()
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
    assert .6 < scores['Quality_Score'] < 1
    assert scores[[
        'Train_Time',
        'Peak_Memory_MB',
        'Synthesizer_Size_MB',
        'Sample_Time',
        'Evaluate_Time'
    ]].between(0, 1000).all()


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
