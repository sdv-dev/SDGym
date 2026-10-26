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
        table_data = MagicMock()
        data = {'table': table_data}
        metadata = MagicMock()
        synthesizer = RealTabFormerSynthesizer()

        # Run
        result = synthesizer._get_trained_synthesizer(data, metadata)

        # Assert
        mock_real_tab_former.assert_called_once_with(model_type='tabular')
        mock_model.fit.assert_called_once_with(table_data)
        assert result._internal_synthesizer == mock_model
        assert result._table_name == 'table'
        assert isinstance(result, RealTabFormerSynthesizer)

    def test__sample_from_synthesizer(self):
        """Test _sample_from_synthesizer generates data with the specified sample size."""
        # Setup
        trained_model = MagicMock()
        trained_model._table_name = 'table'
        trained_model._internal_synthesizer = MagicMock()
        trained_model._internal_synthesizer.sample.return_value = MagicMock(
            shape=(10, 5)
        )  # Mock sample data shape
        n_sample = 10
        synthesizer = RealTabFormerSynthesizer()

        # Run
        synthetic_data = synthesizer._sample_from_synthesizer(trained_model, n_sample)

        # Assert
        trained_model._internal_synthesizer.sample.assert_called_once_with(n_sample)
        assert isinstance(synthetic_data, dict)
        assert synthetic_data['table'].shape[0] == n_sample, (
            f'Expected {n_sample} rows, but got {synthetic_data["table"].shape[0]}'
        )
