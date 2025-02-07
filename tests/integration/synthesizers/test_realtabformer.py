import sys

import pytest

from sdgym import load_dataset
from sdgym.synthesizers import RealTabFormerSynthesizer


@pytest.mark.skipif(sys.platform.startswith('darwin'), reason='Test not supported on github MacOS')
def test_realtabformer_end_to_end():
    """Test it without metrics."""
    # Setup
    data, metadata_dict = load_dataset(
        'single_table', 'student_placements', limit_dataset_size=False
    )
    realtabformer_instance = RealTabFormerSynthesizer()
    realtabformer_instance._MODEL_KWARGS = {'epochs': 10}

    # Run
    trained_synthesizer = realtabformer_instance.get_trained_synthesizer(data, metadata_dict)
    sampled_data = realtabformer_instance.sample_from_synthesizer(trained_synthesizer, n_samples=10)

    # Assert
    assert sampled_data.shape[1] == data.shape[1], (
        f'Sampled data shape {sampled_data.shape} does not match original data shape {data.shape}'
    )

    assert set(sampled_data.columns) == set(data.columns)
