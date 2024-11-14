"""Tests for the realtabformer module."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import pandas as pd
from sdgym.synthesizers import RealTabFormerSynthesizer


@pytest.fixture
def sample_data():
    """Provide sample data for testing."""
    n_samples = 10
    num_values = np.random.normal(size=n_samples)

    return pd.DataFrame({
        'num': num_values,
    })


class TestRealTabFormerSynthesizer:
    """Unit tests for RealTabFormerSynthesizer integration with SDGym."""

    def test_get_trained_synthesizer(self):
        """Test _get_trained_synthesizer initializes and fits REaLTabFormer with correct parameters."""
        data = MagicMock()
        metadata = MagicMock()

        with patch('realtabformer.REaLTabFormer') as MockREaLTabFormer:
            mock_model = MagicMock()
            MockREaLTabFormer.return_value = mock_model
            synthesizer = RealTabFormerSynthesizer()
            result = synthesizer._get_trained_synthesizer(data, metadata)
            MockREaLTabFormer.assert_called_once_with(model_type='tabular')
            mock_model.fit.assert_called_once_with(data, device='cpu')
            assert result == mock_model, "Expected the trained model to be returned."

    def test_sample_from_synthesizer(self):
        """Test _sample_from_synthesizer generates data with the specified sample size."""
        trained_model = MagicMock()
        trained_model.sample.return_value = MagicMock(shape=(10, 5))  # Mock sample data shape
        n_sample = 10
        synthesizer = RealTabFormerSynthesizer()
        synthetic_data = synthesizer._sample_from_synthesizer(trained_model, n_sample)
        trained_model.sample.assert_called_once_with(n_sample, device='cpu')
        assert synthetic_data.shape[0] == n_sample, f"Expected {n_sample} rows, but got {synthetic_data.shape[0]}"