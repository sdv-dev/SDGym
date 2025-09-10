"""Tests for the benchmarking module."""

import contextlib
import io
import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
import pytest
import yaml
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
        sdmetrics=[('NewRowSynthesis', {'synthetic_sample_size': 1000})],
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


@pytest.mark.skipif(sys.platform.startswith('darwin'), reason='Test not supported on github MacOS')
def test_benchmark_single_table_realtabformer_no_metrics():
    """Test it without metrics."""
    # Run
    custom_synthesizer = create_sdv_synthesizer_variant(
        display_name='RealTabFormerSynthesizer',
        synthesizer_class='RealTabFormerSynthesizer',
        synthesizer_parameters={'epochs': 2},
    )
    output = sdgym.benchmark_single_table(
        synthesizers=[],
        custom_synthesizers=[custom_synthesizer],
        sdv_datasets=['fake_companies'],
        sdmetrics=[],
    )

    # Assert
    train_time = output['Train_Time'][0]
    sample_time = output['Sample_Time'][0]
    assert isinstance(train_time, (int, float, complex)), 'Train_Time is not numerical'
    assert isinstance(sample_time, (int, float, complex)), 'Sample_Time is not numerical'
    assert train_time >= 0
    assert sample_time >= 0


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
    assert len(output.columns) == 11


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


def test_benchmark_single_table_compute_privacy_score():
    """Test ``compute_privacy_score=False`` works."""
    # Run
    output = sdgym.benchmark_single_table(
        synthesizers=['DataIdentity', 'ColumnSynthesizer', 'UniformSynthesizer'],
        sdv_datasets=['student_placements'],
        compute_privacy_score=False,
    )

    # Assert
    assert not output.empty
    assert 'Train_Time' in output
    assert 'Sample_Time' in output
    assert 'Privacy_Score' not in output


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
        sdmetrics=[('NewRowSynthesis', {'synthetic_sample_size': 1000})],
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
    timeout_scores = pd.Series(
        {
            'Synthesizer': 'GaussianCopulaSynthesizer',
            'Dataset': 'insurance',
            'Dataset_Size_MB': 3.340128,
            'Train_Time': None,
            'Peak_Memory_MB': None,
            'Synthesizer_Size_MB': None,
            'Sample_Time': None,
            'Evaluate_Time': None,
            'Diagnostic_Score': None,
            'Quality_Score': None,
            'Privacy_Score': None,
            'error': 'Synthesizer Timeout',
        },
        name=0,
    )
    pd.testing.assert_series_equal(scores.T[0], timeout_scores)


def test_benchmark_single_table_only_datasets():
    """Test it works when only the ``sdv_datasets`` argument is passed.

    This is the simplest possible test, since removing the ``sdv_datasets``
    argument could take days to run.
    """
    # Run
    scores = benchmark_single_table(
        sdv_datasets=['fake_companies'],
        sdmetrics=[('NewRowSynthesis', {'synthetic_sample_size': 1000})],
    )

    # Assert
    assert len(scores.columns) == 12
    assert list(scores['Synthesizer']) == [
        'GaussianCopulaSynthesizer',
        'CTGANSynthesizer',
        'UniformSynthesizer',
    ]
    assert list(scores['Dataset']) == ['fake_companies'] * 3
    assert [round(score, 5) for score in scores['Dataset_Size_MB']] == [0.00128] * 3
    assert scores['Train_Time'].between(0, 1000).all()
    assert scores['Peak_Memory_MB'].between(0, 1000).all()
    assert scores['Synthesizer_Size_MB'].between(0, 1000).all()
    assert scores['Sample_Time'].between(0, 1000).all()
    assert scores['Evaluate_Time'].between(0, 1000).all()
    assert scores['Quality_Score'].between(0.5, 1).all()
    assert scores['Privacy_Score'].between(0.5, 1).all()
    assert (scores['Diagnostic_Score'][0:2] == 1.0).all()
    assert scores['Diagnostic_Score'][2:].between(0.5, 1.0).all()
    assert list(scores['NewRowSynthesis']) == [1.0] * 3


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
    assert scores.shape == (2, 11)
    for name, iloc in (('UniformSynthesizer', 0), ('Variant:test_synth', 1)):
        _scores = scores.iloc[iloc]
        assert _scores['Synthesizer'] == name
        assert _scores['Dataset'] == 'fake_companies'
        assert round(_scores['Dataset_Size_MB'], 5) == 0.00128
        assert 0.5 < _scores['Quality_Score'] < 1
        assert 0.5 < _scores['Privacy_Score'] <= 1.0
        if name == 'Variant:test_synth':
            assert _scores['Diagnostic_Score'] == 1.0
        else:
            assert 0.5 < _scores['Diagnostic_Score'] <= 1.0
        assert (
            _scores[
                [
                    'Train_Time',
                    'Peak_Memory_MB',
                    'Synthesizer_Size_MB',
                    'Sample_Time',
                    'Evaluate_Time',
                ]
            ]
            .between(0, 1000)
            .all()
        )


def test_benchmark_single_table_no_synthesizers():
    """Test it works when no synthesizers are passed.

    It should still run UniformSynthesizer.
    """
    # Run
    result = benchmark_single_table(
        synthesizers=None,
        sdv_datasets=['fake_companies'],
        sdmetrics=[('NewRowSynthesis', {'synthetic_sample_size': 1000})],
    )

    # Assert
    assert result.shape == (1, 12)
    result = result.iloc[0]
    assert result['Synthesizer'] == 'UniformSynthesizer'
    assert result['Dataset'] == 'fake_companies'
    assert round(result['Dataset_Size_MB'], 5) == 0.00128
    assert 0.5 < result['Quality_Score'] < 1
    assert 0.5 < result['Privacy_Score'] <= 1.0
    assert 0.5 < result['Diagnostic_Score'] <= 1.0
    assert 0 < result['NewRowSynthesis'] <= 1.0
    assert (
        result[
            ['Train_Time', 'Peak_Memory_MB', 'Synthesizer_Size_MB', 'Sample_Time', 'Evaluate_Time']
        ]
        .between(0, 1000)
        .all()
    )


def test_benchmark_single_table_no_datasets():
    """Test it works when no datasets are passed.

    It should return an empty dataframe.
    """
    # Run
    result = benchmark_single_table(
        sdv_datasets=None,
        sdmetrics=[('NewRowSynthesis', {'synthetic_sample_size': 1000})],
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
        'Diagnostic_Score': [],
        'Quality_Score': [],
        'Privacy_Score': [],
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
        compute_privacy_score=False,
    )

    # Assert
    assert result.shape == (1, 9)
    result = result.iloc[0]
    assert result['Synthesizer'] == 'UniformSynthesizer'
    assert result['Dataset'] == 'fake_companies'
    assert round(result['Dataset_Size_MB'], 5) == 0.00128
    assert (
        result[['Train_Time', 'Peak_Memory_MB', 'Synthesizer_Size_MB', 'Sample_Time']]
        .between(0, 1000)
        .all()
    )
    assert result['Evaluate_Time'] is None
    assert result['error'] == 'ValueError: Unknown single-table metric: a'


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
    results = results.iloc[1]
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
    results = results.iloc[1]
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


def test_benchmark_single_table_no_warnings():
    """Test that the benchmark does not raise any FutureWarnings."""
    # Run
    with warnings.catch_warnings(record=True) as w:
        benchmark_single_table(
            synthesizers=['GaussianCopulaSynthesizer'], sdv_datasets=['fake_companies']
        )
        future_warnings = [warning for warning in w if issubclass(warning.category, FutureWarning)]
        assert len(future_warnings) == 0


def test_benchmark_single_table_with_output_destination(tmp_path):
    """Test it works with the ``output_destination`` argument."""
    # Setup
    output_destination = str(tmp_path / 'benchmark_output')
    today_date = pd.Timestamp.now().strftime('%m_%d_%Y')

    # Run
    results = benchmark_single_table(
        synthesizers=['GaussianCopulaSynthesizer', 'TVAESynthesizer'],
        sdv_datasets=['fake_companies'],
        output_destination=output_destination,
    )

    # Assert
    directions = os.listdir(output_destination)
    score_saved_separately = pd.DataFrame()
    assert directions == [f'SDGym_results_{today_date}']
    subdirections = os.listdir(os.path.join(output_destination, directions[0]))
    assert set(subdirections) == {
        f'results_{today_date}_1.csv',
        f'fake_companies_{today_date}',
        f'run_{today_date}_1.yaml',
    }
    with open(
        os.path.join(output_destination, directions[0], f'run_{today_date}_1.yaml'), 'r'
    ) as f:
        metadata = yaml.safe_load(f)
        assert metadata['completed_date'] is not None
        assert metadata['sdgym_version'] == sdgym.__version__

    synthesizer_directions = os.listdir(
        os.path.join(output_destination, directions[0], f'fake_companies_{today_date}')
    )
    assert set(synthesizer_directions) == {
        'TVAESynthesizer',
        'GaussianCopulaSynthesizer',
        'UniformSynthesizer',
    }
    for synthesizer in sorted(synthesizer_directions):
        synthesizer_files = os.listdir(
            os.path.join(
                output_destination, directions[0], f'fake_companies_{today_date}', synthesizer
            )
        )
        assert set(synthesizer_files) == {
            f'{synthesizer}.pkl',
            f'{synthesizer}_synthetic_data.csv',
            f'{synthesizer}_benchmark_result.csv',
        }
        score = pd.read_csv(
            os.path.join(
                output_destination,
                directions[0],
                f'fake_companies_{today_date}',
                synthesizer,
                f'{synthesizer}_benchmark_result.csv',
            )
        )
        score_saved_separately = pd.concat([score_saved_separately, score], ignore_index=True)

    saved_result = pd.read_csv(
        f'{output_destination}/SDGym_results_{today_date}/results_{today_date}_1.csv'
    )
    pd.testing.assert_frame_equal(results, saved_result, check_dtype=False)
    pd.testing.assert_frame_equal(results, score_saved_separately, check_dtype=False)


def test_benchmark_single_table_with_output_destination_multiple_runs(tmp_path):
    """Test saving in ``output_destination`` with multiple runs.

    Here two benchmark runs are performed with different synthesizers
    on the same dataset, and the results are saved in the same output directory.
    The directory contains a `results.csv` file with the combined results
    and a subdirectory for each synthesizer with its own results.
    """
    # Setup
    output_destination = str(tmp_path / 'benchmark_output')
    today_date = pd.Timestamp.now().strftime('%m_%d_%Y')

    # Run
    result_1 = benchmark_single_table(
        synthesizers=['GaussianCopulaSynthesizer'],
        sdv_datasets=['fake_companies'],
        output_destination=output_destination,
    )
    result_2 = benchmark_single_table(
        synthesizers=['TVAESynthesizer'],
        sdv_datasets=['fake_companies'],
        output_destination=output_destination,
    )

    # Assert
    score_saved_separately = pd.DataFrame()
    directions = os.listdir(output_destination)
    assert directions == [f'SDGym_results_{today_date}']
    subdirections = os.listdir(os.path.join(output_destination, directions[0]))
    assert set(subdirections) == {
        f'results_{today_date}_1.csv',
        f'results_{today_date}_2.csv',
        f'fake_companies_{today_date}',
        f'run_{today_date}_1.yaml',
        f'run_{today_date}_2.yaml',
    }
    with open(
        os.path.join(output_destination, directions[0], f'run_{today_date}_1.yaml'), 'r'
    ) as f:
        metadata = yaml.safe_load(f)
        assert metadata['completed_date'] is not None
        assert metadata['sdgym_version'] == sdgym.__version__

    synthesizer_directions = os.listdir(
        os.path.join(output_destination, directions[0], f'fake_companies_{today_date}')
    )
    assert set(synthesizer_directions) == {
        'TVAESynthesizer',
        'GaussianCopulaSynthesizer',
        'UniformSynthesizer',
    }
    for synthesizer in sorted(synthesizer_directions):
        synthesizer_files = os.listdir(
            os.path.join(
                output_destination, directions[0], f'fake_companies_{today_date}', synthesizer
            )
        )
        assert set(synthesizer_files) == {
            f'{synthesizer}.pkl',
            f'{synthesizer}_synthetic_data.csv',
            f'{synthesizer}_benchmark_result.csv',
        }
        score = pd.read_csv(
            os.path.join(
                output_destination,
                directions[0],
                f'fake_companies_{today_date}',
                synthesizer,
                f'{synthesizer}_benchmark_result.csv',
            )
        )
        score_saved_separately = pd.concat([score_saved_separately, score], ignore_index=True)

    saved_result_1 = pd.read_csv(
        f'{output_destination}/SDGym_results_{today_date}/results_{today_date}_1.csv'
    )
    saved_result_2 = pd.read_csv(
        f'{output_destination}/SDGym_results_{today_date}/results_{today_date}_2.csv'
    )
    pd.testing.assert_frame_equal(result_1, saved_result_1, check_dtype=False)
    pd.testing.assert_frame_equal(result_2, saved_result_2, check_dtype=False)
