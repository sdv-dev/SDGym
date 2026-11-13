import os
import sys

import pytest

from sdgym import load_dataset
from sdgym.synthesizers import RealTabFormerSynthesizer


@pytest.mark.skipif(sys.platform.startswith('darwin'), reason='Test not supported on github MacOS')
@pytest.mark.skipif(
    sys.platform.startswith('win') and os.environ.get('SDGYM_MINIMUM_TESTS') == 'true',
    reason='Test fails on github Windows with the minimum dependency versions',
)
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
    real_table = data['student_placements']
    sampled_table = sampled_data['student_placements']
    assert sampled_table.shape[1] == real_table.shape[1], (
        f'Sampled data shape {sampled_table.shape} does not match '
        f'original data shape {real_table.shape}'
    )
    assert set(sampled_table.columns) == set(real_table.columns)
