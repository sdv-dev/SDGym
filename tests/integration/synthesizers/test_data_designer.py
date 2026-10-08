import pytest

import pandas as pd
from sdgym import load_dataset
from sdgym.benchmark import benchmark_single_table
from sdgym.synthesizers import DataDesignerSynthesizer

pytest.importorskip('data_designer.interface')


def test_datadesigner_end_to_end():
    """Test DataDesignerSynthesizer end to end."""
    # Setup
    data, metadata_dict = load_dataset(
        'single_table', 'student_placements', limit_dataset_size=False
    )
    datadesigner_instance = DataDesignerSynthesizer()
    datadesigner_instance._MODEL_KWARGS = {'temperature': 0.5}

    # Run
    trained_synthesizer = datadesigner_instance.get_trained_synthesizer(data, metadata_dict)
    sampled_data = datadesigner_instance.sample_from_synthesizer(trained_synthesizer, n_samples=10)

    # Assert
    assert sampled_data.shape[1] == data.shape[1], (
        f'Sampled data shape {sampled_data.shape} does not match original data shape {data.shape}'
    )

    assert set(sampled_data.columns) == set(data.columns)

def test_benchmark_single_table_with_data_designer():
    """Test single table benchmark running data designer"""
    # Run
    result = benchmark_single_table(
        synthesizers=['DataDesignerSynthesizer'],
        sdv_datasets=['adult'],
        compute_quality_score=True,
        compute_diagnostic_score=True,
        compute_privacy_score=False,
    )
    print(result)

    # Assert
    assert 'Synthesizer' in result.columns
    assert 'Dataset' in result.columns
    assert 'Quality_Score' in result.columns
    assert 'DataDesignerSynthesizer' in result['Synthesizer'].to_numpy()
    assert 'adult' in result['Dataset'].to_numpy()
    assert not pd.isna(result[['Synthesizer', 'Dataset', 'Quality_Score']]).any().any()
