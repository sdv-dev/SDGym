import pytest

from sdgym import load_dataset
from sdgym.synthesizers import DataDesignerSynthesizer

pytest.importorskip('data_designer.interface')


def test_datadesigner_end_to_end():
    """Test it without metrics."""
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
