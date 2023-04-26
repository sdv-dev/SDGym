import contextlib
import io

import pandas as pd
import pytest

import sdgym
from sdgym.synthesizers import create_single_table_synthesizer
from sdgym.synthesizers.generate import create_sdv_synthesizer_variant


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
    """Test all synthesizers, as well as some generated ones, against a dataset."""
    # Setup
    CTGANVariant = create_sdv_synthesizer_variant(
        'CTGANVariant',
        'CTGANSynthesizer',
        synthesizer_parameters={
            'epochs': 100})
    FastMLVariant = create_sdv_synthesizer_variant(
        'FastMLVariant',
        'FastMLPreset',
        synthesizer_parameters={
            'name': 'FAST_ML'})

    # Run
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(sdgym.benchmark_single_table(
        synthesizers=[
            'GaussianCopulaSynthesizer', 'FastMLPreset', 'CTGANSynthesizer', 'DataIdentity',
            'IndependentSynthesizer', 'UniformSynthesizer'],
        custom_synthesizers=[CTGANVariant, FastMLVariant],
        sdv_datasets=['student_placements']
    ))

    # Assert
    assert 0
