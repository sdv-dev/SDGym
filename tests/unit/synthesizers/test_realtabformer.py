"""Tests for the realtabformer module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from sdgym.synthesizers import RealTabFormerSynthesizer
from sdgym.synthesizers.realtabformer import _remove_unsupported_training_args


@pytest.fixture
def sample_data():
    """Provide sample data for testing."""
    n_samples = 10
    num_values = np.random.normal(size=n_samples)

    return pd.DataFrame({
        'num': num_values,
    })


@pytest.mark.parametrize(
    ('field_names', 'expected'),
    [
        (['output_dir'], {'output_dir': 'checkpoints', 'num_train_epochs': 10}),
        (
            ['output_dir', 'overwrite_output_dir'],
            {'output_dir': 'checkpoints', 'overwrite_output_dir': True, 'num_train_epochs': 10},
        ),
    ],
)
@patch('sdgym.synthesizers.realtabformer.dataclasses.fields')
def test__remove_unsupported_training_args(fields_mock, field_names, expected):
    """Test ``overwrite_output_dir`` is only dropped when transformers does not accept it."""
    # Setup
    fields = []
    for field_name in field_names:
        field = MagicMock()
        field.name = field_name
        fields.append(field)

    fields_mock.return_value = fields
    model = MagicMock()
    model.training_args_kwargs = {
        'output_dir': 'checkpoints',
        'overwrite_output_dir': True,
        'num_train_epochs': 10,
    }

    # Run
    _remove_unsupported_training_args(model)

    # Assert
    assert model.training_args_kwargs == expected


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
        mock_model.fit.assert_called_once_with(data)
        assert result._internal_synthesizer == mock_model
        assert isinstance(result, RealTabFormerSynthesizer)

    def test__sample_from_synthesizer(self):
        """Test _sample_from_synthesizer generates data with the specified sample size."""
        # Setup
        trained_model = MagicMock()
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
        assert synthetic_data.shape[0] == n_sample, (
            f'Expected {n_sample} rows, but got {synthetic_data.shape[0]}'
        )
