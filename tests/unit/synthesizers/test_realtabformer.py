import numpy as np
import pytest
import pandas as pd

from sdgym.synthesizers import RealTabFormerSynthesizer


class TestRealTabFormerSynthesizer:
    """Unit tests for RealTabFormerSynthesizer integration with SDGym."""

    @pytest.fixture
    def sample_data(self):
        """Provide sample data for testing."""
        n_samples = 10
        num_values = np.random.normal(size=n_samples)

        return pd.DataFrame({
            'num': num_values,
        })


    def test_initialization(self):
        """Test that RealTabFormerSynthesizer initializes correctly."""
        synthesizer = RealTabFormerSynthesizer()
        assert synthesizer is not None, "Failed to initialize RealTabFormerSynthesizer"

    def test_training(self, sample_data):
        """Test that RealTabFormerSynthesizer trains successfully on sample data."""
        synthesizer = RealTabFormerSynthesizer()

        trained_model = synthesizer.get_trained_synthesizer(sample_data, {})

        assert trained_model is not None, "RealTabFormerSynthesizer failed to train on sample data"

    def test_sampling(self, sample_data):
        """Test that RealTabFormerSynthesizer can generate synthetic data."""
        synthesizer = RealTabFormerSynthesizer()
        trained_model = synthesizer._get_trained_synthesizer(sample_data, metadata=None)

        n_sample = 10
        synthetic_data = synthesizer._sample_from_synthesizer(trained_model, n_sample)

        assert synthetic_data.shape[0] == n_sample, f"Expected {n_sample} rows, but got {synthetic_data.shape[0]}"

