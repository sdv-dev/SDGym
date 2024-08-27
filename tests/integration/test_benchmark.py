"""Tests for the benchmarking module."""

import contextlib
import io
import re
import time

import numpy as np
import pandas as pd
import pytest
from sdv.metadata.single_table import SingleTableMetadata
from sdv.single_table.copulas import GaussianCopulaSynthesizer

import sdgym
from sdgym import (
    benchmark_single_table,
    create_sdv_synthesizer_variant,
    create_single_table_synthesizer,
)


def test_benchmark_single_table_basic_synthsizers():
    """Test it with DataIdentity, ColumnSynthesizer and UniformSynthesizer."""
    # Run
    output = sdgym.benchmark_single_table(
        synthesizers=['DataIdentity', 'ColumnSynthesizer', 'UniformSynthesizer'],
        sdv_datasets=['student_placements'],
    )

    # Assert
    assert not output.empty
    assert 'Train_Time' in output
    assert 'Sample_Time' in output

    scores = output.groupby('Synthesizer').NewRowSynthesis.mean().sort_values()

    assert [
        'DataIdentity',
        'ColumnSynthesizer',
        'UniformSynthesizer',
    ] == scores.index.tolist()

    quality_scores = output.groupby('Synthesizer').Quality_Score.mean().sort_values()

    assert [
        'UniformSynthesizer',
        'ColumnSynthesizer',
        'DataIdentity',
    ] == quality_scores.index.tolist()


def test_benchmark_single_table_no_metrics():
    """Test it without metrics."""
    # Run
    output = sdgym.benchmark_single_table(
        synthesizers=['DataIdentity', 'ColumnSynthesizer', 'UniformSynthesizer'],
        sdv_datasets=['student_placements'],
        sdmetrics=[],
    )

    # Assert
    assert not output.empty
    assert 'Train_Time' in output
    assert 'Sample_Time' in output

    # Expect no metric columns.
    assert len(output.columns) == 10


def test_benchmarking_no_report_output():
    """Test that the benchmarking printing does not include report progress."""
    # Setup
    prints = io.StringIO()

    # Run
    with contextlib.redirect_stderr(prints):
        sdgym.benchmark_single_table(
            synthesizers=['DataIdentity', 'ColumnSynthesizer', 'UniformSynthesizer'],
            sdv_datasets=['student_placements'],
        )

    # Assert
    assert 'Creating report:' not in prints


def get_trained_synthesizer_err(data, metadata):
    """Get empty dict."""
    return {}


def sample_from_synthesizer_err(synthesizer, num_rows):
    """Get ValueError."""
    raise ValueError('random error')


def test_benchmark_single_table_error_handling():
    """Test it produces the correct errors."""
    # Setup
    erroring_synthesizer = create_single_table_synthesizer(
        'my_synth', get_trained_synthesizer_err, sample_from_synthesizer_err
    )

    # Run
    output = sdgym.benchmark_single_table(
        synthesizers=['DataIdentity', 'ColumnSynthesizer', 'UniformSynthesizer'],
        custom_synthesizers=[erroring_synthesizer],
        sdv_datasets=['student_placements'],
    )

    # Assert
    assert not output.empty
    assert 'Train_Time' in output
    assert 'Sample_Time' in output

    output = output[output['Synthesizer'] == 'Custom:my_synth'][['Train_Time', 'Sample_Time']]
    assert output.isna().all(1).all()


def test_benchmark_single_table_compute_quality_score():
    """Test ``compute_quality_score=False`` works."""
    # Run
    output = sdgym.benchmark_single_table(
        synthesizers=['DataIdentity', 'ColumnSynthesizer', 'UniformSynthesizer'],
        sdv_datasets=['student_placements'],
        compute_quality_score=False,
    )

    # Assert
    assert not output.empty
    assert 'Train_Time' in output
    assert 'Sample_Time' in output
    assert 'Quality_Score' not in output


def test_benchmark_single_table_compute_diagnostic_score():
    """Test ``compute_diagnostic_score=False`` works."""
    # Run
    output = sdgym.benchmark_single_table(
        synthesizers=['DataIdentity', 'ColumnSynthesizer', 'UniformSynthesizer'],
        sdv_datasets=['student_placements'],
        compute_diagnostic_score=False,
    )

    # Assert
    assert not output.empty
    assert 'Train_Time' in output
    assert 'Sample_Time' in output
    assert 'Diagnostic_Score' not in output


def test_benchmark_single_table_duplicate_synthesizers():
    """Test it raises an error when passed a duplicate synthesizer."""
    # Setup
    custom_synthesizer = create_single_table_synthesizer(
        'my_synth', get_trained_synthesizer_err, sample_from_synthesizer_err
    )

    # Run and Assert
    error_msg = re.escape(
        'Synthesizers must be unique. Please remove repeated values in the `synthesizers` '
        'and `custom_synthesizers` parameters.'
    )
    with pytest.raises(ValueError, match=error_msg):
        sdgym.benchmark_single_table(
            synthesizers=['GaussianCopulaSynthesizer', 'GaussianCopulaSynthesizer'],
            custom_synthesizers=[custom_synthesizer, custom_synthesizer],
        )


def test_benchmark_single_table():
    """Test all synthesizers, as well as some generated ones, against a dataset.

    The custom synthesizers should be generated from both ``create_single_table_synthesizer``
    and ``create_sdv_synthesizer_variant``, to test they work.
    """

    # Setup
    def get_trained_synthesizer(data, metadata):
        metadata_obj = SingleTableMetadata.load_from_dict(metadata)
        model = GaussianCopulaSynthesizer(metadata_obj)
        model.fit(data)
        return model

    def sample_from_synthesizer(synthesizer, n_samples):
        return synthesizer.sample(n_samples)

    test_synthesizer = create_single_table_synthesizer(
        display_name='TestSynthesizer',
        get_trained_synthesizer_fn=get_trained_synthesizer,
        sample_from_synthesizer_fn=sample_from_synthesizer,
    )

    ctgan_variant = create_sdv_synthesizer_variant(
        'CTGANVariant', 'CTGANSynthesizer', synthesizer_parameters={'epochs': 100}
    )

    # Run
    results = sdgym.benchmark_single_table(
        synthesizers=[
            'TVAESynthesizer',
            'CopulaGANSynthesizer',
            'GaussianCopulaSynthesizer',
            'DataIdentity',
            'ColumnSynthesizer',
            'UniformSynthesizer',
            'CTGANSynthesizer',
        ],
        custom_synthesizers=[test_synthesizer, ctgan_variant],
        sdv_datasets=['fake_companies'],
    )

    # Assert
    expected_synthesizers = pd.Series(
        [
            'TVAESynthesizer',
            'CopulaGANSynthesizer',
            'GaussianCopulaSynthesizer',
            'DataIdentity',
            'ColumnSynthesizer',
            'UniformSynthesizer',
            'CTGANSynthesizer',
            'Custom:TestSynthesizer',
            'Variant:CTGANVariant',
        ],
        name='Synthesizer',
    )
    pd.testing.assert_series_equal(results['Synthesizer'], expected_synthesizers)

    assert set(results['Dataset']) == {'fake_companies'}
    assert np.isclose(results['Dataset_Size_MB'][0], 0.00128, atol=4)
    assert results['Train_Time'].between(0, 1000).all()
    assert results['Peak_Memory_MB'].between(0, 100).all()
    assert results['Synthesizer_Size_MB'].between(0, 100).all()
    assert results['Sample_Time'].between(0, 100).all()
    assert results['Evaluate_Time'].between(0, 100).all()
    assert results['Quality_Score'].between(0.5, 1).all()

    # The IdentitySynthesizer never returns new rows, so its score is 0
    # Every other synthesizer should only return new rows, so their score is 1
    assert results['NewRowSynthesis'][3] == 0
    results['NewRowSynthesis'][3] = 1
    assert (results['NewRowSynthesis'] == 1).all()


def test_benchmark_single_table_timeout():
    """Test that benchmark times out if the ``timeout`` argument is given."""
    # Setup
    start_time = time.time()

    # Run
    scores = sdgym.benchmark_single_table(
        synthesizers=['GaussianCopulaSynthesizer'], sdv_datasets=['insurance'], timeout=2
    )
    total_time = time.time() - start_time

    # Assert
    assert total_time < 50.0  # Buffer time for code not in timeout
    expected_scores = pd.DataFrame({
        'Synthesizer': {0: 'GaussianCopulaSynthesizer'},
        'Dataset': {0: 'insurance'},
        'Dataset_Size_MB': {0: 3.340128},
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


def test_benchmark_single_table_only_datasets():
    """Test it works when only the ``sdv_datasets`` argument is passed.

    This is the simplest possible test, since removing the ``sdv_datasets``
    argument could take days to run.
    """
    # Run
    scores = benchmark_single_table(sdv_datasets=['fake_companies'])

    # Assert
    assert len(scores.columns) == 11
    assert list(scores['Synthesizer']) == ['GaussianCopulaSynthesizer', 'CTGANSynthesizer']
    assert list(scores['Dataset']) == ['fake_companies'] * 2
    assert [round(score, 5) for score in scores['Dataset_Size_MB']] == [0.00128] * 2
    assert scores['Train_Time'].between(0, 1000).all()
    assert scores['Peak_Memory_MB'].between(0, 1000).all()
    assert scores['Synthesizer_Size_MB'].between(0, 1000).all()
    assert scores['Sample_Time'].between(0, 1000).all()
    assert scores['Evaluate_Time'].between(0, 1000).all()
    assert scores['Quality_Score'].between(0.5, 1).all()
    assert (scores['Diagnostic_Score'] == 1.0).all()
    assert list(scores['NewRowSynthesis']) == [1.0] * 2


def test_benchmark_single_table_synthesizers_none():
    """Test it works when ``synthesizers`` is None."""
    # Setup
    synthesizer_variant = create_sdv_synthesizer_variant(
        'test_synth', 'GaussianCopulaSynthesizer', synthesizer_parameters={}
    )

    # Run
    scores = benchmark_single_table(
        synthesizers=None,
        custom_synthesizers=[synthesizer_variant],
        sdv_datasets=['fake_companies'],
    )

    # Assert
    assert scores.shape == (1, 11)
    scores = scores.iloc[0]
    assert scores['Synthesizer'] == 'Variant:test_synth'
    assert scores['Dataset'] == 'fake_companies'
    assert round(scores['Dataset_Size_MB'], 5) == 0.00128
    assert 0.5 < scores['Quality_Score'] < 1
    assert scores['Diagnostic_Score'] == 1.0
    assert (
        scores[
            ['Train_Time', 'Peak_Memory_MB', 'Synthesizer_Size_MB', 'Sample_Time', 'Evaluate_Time']
        ]
        .between(0, 1000)
        .all()
    )


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
        'Diagnostic_Score': [],
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
        'Diagnostic_Score': [],
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
        compute_quality_score=False,
        compute_diagnostic_score=False,
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
        'b': [],
    })
    pd.testing.assert_frame_equal(result, expected)


def test_benchmark_single_table_custom_synthesizer():
    """Test it works with the ``create_single_table_synthesizer`` method."""

    # Setup
    def get_trained_synthesizer(data, metadata):
        metadata_obj = SingleTableMetadata.load_from_dict(metadata)
        model = GaussianCopulaSynthesizer(metadata_obj)
        model.fit(data)
        return model

    def sample_from_synthesizer(synthesizer, n_samples):
        return synthesizer.sample(n_samples)

    test_synthesizer = create_single_table_synthesizer(
        display_name='TestSynthesizer',
        get_trained_synthesizer_fn=get_trained_synthesizer,
        sample_from_synthesizer_fn=sample_from_synthesizer,
    )

    # Run
    results = benchmark_single_table(
        synthesizers=None, custom_synthesizers=[test_synthesizer], sdv_datasets=['fake_companies']
    )

    # Assert
    results = results.iloc[0]
    assert results['Synthesizer'] == 'Custom:TestSynthesizer'
    assert results['Dataset'] == 'fake_companies'
    assert round(results['Dataset_Size_MB'], 5) == 0.00128
    assert 0.5 < results['Quality_Score'] < 1

    assert (
        results[
            ['Train_Time', 'Peak_Memory_MB', 'Synthesizer_Size_MB', 'Sample_Time', 'Evaluate_Time']
        ]
        .between(0, 1000)
        .all()
    )


def test_benchmark_single_table_limit_dataset_size():
    """Test it works with ``limit_dataset_size``."""

    # Run
    results = benchmark_single_table(
        synthesizers=['GaussianCopulaSynthesizer'], sdv_datasets=['adult'], limit_dataset_size=True
    )

    # Assert
    results = results.iloc[0]
    assert results['Synthesizer'] == 'GaussianCopulaSynthesizer'
    assert results['Dataset'] == 'adult'
    assert round(results['Dataset_Size_MB'], 4) == 0.0801
    assert 0.5 < results['Quality_Score'] < 1
    assert (
        results[
            ['Train_Time', 'Peak_Memory_MB', 'Synthesizer_Size_MB', 'Sample_Time', 'Evaluate_Time']
        ]
        .between(0, 1000)
        .all()
    )

    assert (
        results[
            ['Train_Time', 'Peak_Memory_MB', 'Synthesizer_Size_MB', 'Sample_Time', 'Evaluate_Time']
        ]
        .between(0, 1000)
        .all()
    )


def test_benchmark_single_table_instantiated_synthesizer():
    """Test it with instances of synthesizers instead of the class."""

    # Setup
    def get_trained_synthesizer(data, metadata):
        metadata_obj = SingleTableMetadata.load_from_dict(metadata)
        model = GaussianCopulaSynthesizer(metadata_obj)
        model.fit(data)
        return model

    def sample_from_synthesizer(synthesizer, n_samples):
        return synthesizer.sample(n_samples)

    test_synthesizer = create_single_table_synthesizer(
        display_name='TestSynthesizer',
        get_trained_synthesizer_fn=get_trained_synthesizer,
        sample_from_synthesizer_fn=sample_from_synthesizer,
    )
    test_synthesizer_instance = test_synthesizer()

    # Run
    results = benchmark_single_table(
        synthesizers=None,
        custom_synthesizers=[test_synthesizer_instance],
        sdv_datasets=['fake_companies'],
    )

    # Assert
    results = results.iloc[0]
    assert results['Synthesizer'] == 'Custom:TestSynthesizer'
    assert results['Dataset'] == 'fake_companies'
    assert round(results['Dataset_Size_MB'], 5) == 0.00128
    assert 0.5 < results['Quality_Score'] < 1

    assert (
        results[
            ['Train_Time', 'Peak_Memory_MB', 'Synthesizer_Size_MB', 'Sample_Time', 'Evaluate_Time']
        ]
        .between(0, 1000)
        .all()
    )
