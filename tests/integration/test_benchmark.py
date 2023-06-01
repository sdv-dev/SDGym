import contextlib
import io
import time

import pandas as pd
import pytest
from sdv.metadata.single_table import SingleTableMetadata
from sdv.single_table.copulas import GaussianCopulaSynthesizer

import sdgym
from sdgym import create_sdv_synthesizer_variant, create_single_table_synthesizer


def test_identity():
    output = sdgym.benchmark_single_table(
        synthesizers=['DataIdentity', 'IndependentSynthesizer', 'UniformSynthesizer'],
        sdv_datasets=['student_placements'],
    )

    assert not output.empty
    assert 'Train_Time' in output
    assert 'Sample_Time' in output

    scores = output.groupby('Synthesizer').NewRowSynthesis.mean().sort_values()

    assert [
        'DataIdentity',
        'IndependentSynthesizer',
        'UniformSynthesizer',
    ] == scores.index.tolist()

    quality_scores = output.groupby('Synthesizer').Quality_Score.mean().sort_values()

    assert [
        'UniformSynthesizer',
        'IndependentSynthesizer',
        'DataIdentity',
    ] == quality_scores.index.tolist()


def test_benchmarking_no_metrics():
    output = sdgym.benchmark_single_table(
        synthesizers=['DataIdentity', 'IndependentSynthesizer', 'UniformSynthesizer'],
        sdv_datasets=['student_placements'],
        sdmetrics=[],
    )

    assert not output.empty
    assert 'Train_Time' in output
    assert 'Sample_Time' in output
    # Expect no metric columns.
    assert len(output.columns) == 9


def test_benchmarking_no_report_output():
    """Test that the benchmarking printing does not include report progress."""
    prints = io.StringIO()
    with contextlib.redirect_stderr(prints):
        sdgym.benchmark_single_table(
            synthesizers=['DataIdentity', 'IndependentSynthesizer', 'UniformSynthesizer'],
            sdv_datasets=['student_placements'],
        )

    assert 'Creating report:' not in prints


def get_trained_synthesizer_err(data, metadata):
    return {}


def sample_from_synthesizer_err(synthesizer, num_rows):
    raise ValueError('random error')


def test_error_handling():
    erroring_synthesizer = create_single_table_synthesizer(
        'my_synth', get_trained_synthesizer_err, sample_from_synthesizer_err)
    output = sdgym.benchmark_single_table(
        synthesizers=['DataIdentity', 'IndependentSynthesizer', 'UniformSynthesizer'],
        custom_synthesizers=[erroring_synthesizer],
        sdv_datasets=['student_placements'],
    )

    assert not output.empty
    assert 'Train_Time' in output
    assert 'Sample_Time' in output
    assert (
        output[output['Synthesizer'] == 'Custom:my_synth'][['Train_Time', 'Sample_Time']]
    ).isna().all(1).all()


def test_compute_quality_score():
    output = sdgym.benchmark_single_table(
        synthesizers=['DataIdentity', 'IndependentSynthesizer', 'UniformSynthesizer'],
        sdv_datasets=['student_placements'],
        compute_quality_score=False,
    )

    assert not output.empty
    assert 'Train_Time' in output
    assert 'Sample_Time' in output
    assert 'Quality_Score' not in output


def test_duplicate_synthesizers():
    custom_synthesizer = create_single_table_synthesizer(
        'my_synth', get_trained_synthesizer_err, sample_from_synthesizer_err)
    with pytest.raises(
        ValueError,
        match=(
            'Synthesizers must be unique. Please remove repeated values in the `synthesizers` '
            'and `custom_synthesizers` parameters.'
        )
    ):
        sdgym.benchmark_single_table(
            synthesizers=['GaussianCopulaSynthesizer', 'GaussianCopulaSynthesizer'],
            custom_synthesizers=[custom_synthesizer, custom_synthesizer]
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

    TestSynthesizer = create_single_table_synthesizer(
        display_name='TestSynthesizer',
        get_trained_synthesizer_fn=get_trained_synthesizer,
        sample_from_synthesizer_fn=sample_from_synthesizer
    )

    CTGANVariant = create_sdv_synthesizer_variant(
        'CTGANVariant',
        'CTGANSynthesizer',
        synthesizer_parameters={'epochs': 100}
    )

    FastMLVariant = create_sdv_synthesizer_variant(
        'FastMLVariant',
        'FastMLPreset',
        synthesizer_parameters={'name': 'FAST_ML'}
    )

    # Run
    results = sdgym.benchmark_single_table(
        synthesizers=[
            'TVAESynthesizer',
            'CopulaGANSynthesizer',
            'GaussianCopulaSynthesizer',
            'FastMLPreset',
            'DataIdentity',
            'IndependentSynthesizer',
            'UniformSynthesizer',
            'CTGANSynthesizer'
        ],
        custom_synthesizers=[FastMLVariant, TestSynthesizer, CTGANVariant],
        sdv_datasets=['fake_companies']
    )

    # Assert
    expected_synthesizers = pd.Series([
        'TVAESynthesizer',
        'CopulaGANSynthesizer',
        'GaussianCopulaSynthesizer',
        'FastMLPreset',
        'DataIdentity',
        'IndependentSynthesizer',
        'UniformSynthesizer',
        'CTGANSynthesizer',
        'Variant:FastMLVariant',
        'Custom:TestSynthesizer',
        'Variant:CTGANVariant'
    ], name='Synthesizer')
    pd.testing.assert_series_equal(results['Synthesizer'], expected_synthesizers)

    assert set(results['Dataset']) == {'fake_companies'}
    assert set(results['Dataset_Size_MB']) == {0.00128}
    assert results['Train_Time'].between(0, 1000).all()
    assert results['Peak_Memory_MB'].between(0, 100).all()
    assert results['Synthesizer_Size_MB'].between(0, 100).all()
    assert results['Sample_Time'].between(0, 100).all()
    assert results['Evaluate_Time'].between(0, 100).all()
    assert results['Quality_Score'].between(.5, 1).all()

    # The IdentitySynthesizer never returns new rows, so its score is 0
    # Every other synthesizer should only return new rows, so their score is 1
    assert results['NewRowSynthesis'][4] == 0
    results['NewRowSynthesis'][4] = 1
    assert (results['NewRowSynthesis'] == 1).all()


def test_benchmark_single_table_timeout():
    """Test that benchmark times out if the ``timeout`` argument is given."""
    # Setup
    start_time = time.time()

    # Run
    scores = sdgym.benchmark_single_table(
        synthesizers=['GaussianCopulaSynthesizer'],
        sdv_datasets=['insurance'],
        timeout=2
    )
    total_time = time.time() - start_time

    # Assert
    assert total_time < 10.0
    expected_scores = pd.DataFrame({
        'Synthesizer': {0: 'GaussianCopulaSynthesizer'},
        'Dataset': {0: 'insurance'},
        'Dataset_Size_MB': {0: 3.340128},
        'Train_Time': {0: None},
        'Peak_Memory_MB': {0: None},
        'Synthesizer_Size_MB': {0: None},
        'Sample_Time': {0: None},
        'Evaluate_Time': {0: None},
        'Quality_Score': {0: None},
        'error': {0: 'Synthesizer Timeout'}
    })
    pd.testing.assert_frame_equal(scores, expected_scores)
