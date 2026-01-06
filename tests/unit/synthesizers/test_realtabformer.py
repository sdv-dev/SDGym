"""Tests for the realtabformer module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

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

    @patch('realtabformer.REaLTabFormer')
    def test__get_trained_synthesizer(self, mock_real_tab_former):
        """Test _get_trained_synthesizer

        Initializes REaLTabFormer and fits REaLTabFormer with
        correct parameters.
        """
        # Setup
        mock_model = MagicMock()
        mock_real_tab_former.return_value = mock_model
        data = MagicMock()
        metadata = MagicMock()
        synthesizer = RealTabFormerSynthesizer()

        # Run
        result = synthesizer._get_trained_synthesizer(data, metadata)

        # Assert
        mock_real_tab_former.assert_called_once_with(model_type='tabular')
        mock_model.fit.assert_called_once_with(data, save_full_every_epoch=0, gen_kwargs={})
        assert result == mock_model, 'Expected the trained model to be returned.'

    def test__sample_from_synthesizer(self):
        """Test _sample_from_synthesizer generates data with the specified sample size."""
        # Setup
        trained_model = MagicMock()
        trained_model.sample.return_value = MagicMock(shape=(10, 5))  # Mock sample data shape
        n_sample = 10
        synthesizer = RealTabFormerSynthesizer()

        # Run
        synthetic_data = synthesizer._sample_from_synthesizer(trained_model, n_sample)

        # Assert
        trained_model.sample.assert_called_once_with(n_sample)
        assert synthetic_data.shape[0] == n_sample, (
            f'Expected {n_sample} rows, but got {synthetic_data.shape[0]}'
        )
