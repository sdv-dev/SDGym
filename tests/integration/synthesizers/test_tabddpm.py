import sys

import pytest

from sdgym import load_dataset
from sdgym.synthesizers import TabDDPMSynthesizer


@pytest.mark.skipif(sys.platform.startswith('darwin'), reason='Test not supported on github MacOS')
def test_tabddpm_end_to_end():
    """Test it without metrics."""
    # Setup
    data, metadata_dict = load_dataset(
        'single_table', 'student_placements', limit_dataset_size=False
    )
    tabddpm_instance = TabDDPMSynthesizer()
    tabddpm_instance._MODEL_KWARGS = {'steps': 5000, 'num_timesteps': 1000}

    # Run
    trained_synthesizer = tabddpm_instance.get_trained_synthesizer(data, metadata_dict)
    sampled_data = tabddpm_instance.sample_from_synthesizer(trained_synthesizer, n_samples=10)

    # Assert
    assert sampled_data.shape[1] == data.shape[1], (
        f'Sampled data shape {sampled_data.shape} does not match original data shape {data.shape}'
    )

    assert set(sampled_data.columns) == set(data.columns)
