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
    real_table = data['student_placements']
    sampled_table = sampled_data['student_placements']
    assert sampled_table.shape[1] == real_table.shape[1], (
        f'Sampled data shape {sampled_table.shape} does not match '
        f'original data shape {real_table.shape}'
    )
    assert set(sampled_table.columns) == set(real_table.columns)
