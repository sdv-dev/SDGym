import re
import warnings
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from sdv.metadata import Metadata

from sdgym.synthesizers.base import (
    BaselineSynthesizer,
    MultiTableBaselineSynthesizer,
    _is_valid_modality,
    _validate_modality,
)


@pytest.mark.parametrize(
    'modality, result',
    [
        ('single_table', True),
        ('multi_table', True),
        ('invalid_modality', False),
    ],
)
def test__is_valid_modality(modality, result):
    """Test the `_is_valid_modality` method."""
    assert _is_valid_modality(modality) == result


def test__validate_modality():
    """Test the `_validate_modality` method."""
    # Setup
    valid_modality = 'single_table'
    invalid_modality = 'invalid_modality'
    expected_error = re.escape(
        f"Modality '{invalid_modality}' is not valid. Must be either "
        "'single_table' or 'multi_table'."
    )

    # Run and Assert
    _validate_modality(valid_modality)
    with pytest.raises(ValueError, match=expected_error):
        _validate_modality(invalid_modality)


class TestBaselineSynthesizer:
    @patch('sdgym.synthesizers.base.BaselineSynthesizer.get_subclasses')
    @patch('sdgym.synthesizers.base._validate_modality')
    def test__get_supported_synthesizers_mock(self, mock_validate_modality, mock_get_subclasses):
        """Test the `_get_supported_synthesizers` method with mocks."""
        # Setup
        mock_get_subclasses.return_value = {
            'Variant:Synthesizer': Mock(_NATIVELY_SUPPORTED=False, _MODALITY_FLAG='single_table'),
            'Custom:MySynthesizer': Mock(_NATIVELY_SUPPORTED=False, _MODALITY_FLAG='single_table'),
            'ColumnSynthesizer': Mock(_NATIVELY_SUPPORTED=True, _MODALITY_FLAG='single_table'),
            'UniformSynthesizer': Mock(_NATIVELY_SUPPORTED=True, _MODALITY_FLAG='single_table'),
            'MultiTableSynthesizer': Mock(_NATIVELY_SUPPORTED=True, _MODALITY_FLAG='multi_table'),
            'DataIdentity': Mock(_NATIVELY_SUPPORTED=True, _MODALITY_FLAG='single_table'),
        }
        expected_synthesizers = [
            'ColumnSynthesizer',
            'DataIdentity',
            'UniformSynthesizer',
        ]

        # Run
        synthesizers = BaselineSynthesizer._get_supported_synthesizers('single_table')

        # Assert
        mock_validate_modality.assert_called_once_with('single_table')
        mock_get_subclasses.assert_called_once_with(include_parents=True)
        assert synthesizers == expected_synthesizers

    def test_get_trained_synthesizer(self):
        """Test it calls the correct methods and returns the correct values."""
        # Setup
        data = pd.DataFrame({
            'pk': [1, 2, 3, 4],
            'col': [1, 2, 3, 4],
        })
        metadata = Metadata()
        metadata.add_table('table')
        metadata.add_column('pk', 'table', sdtype='id')
        metadata.add_column('col', 'table', sdtype='numerical')
        instance = BaselineSynthesizer()
        mock_synthesizer = Mock()
        instance._get_trained_synthesizer = Mock(return_value=mock_synthesizer)

        # Run
        with warnings.catch_warnings(record=True) as recorded_warnings:
            instance.get_trained_synthesizer(data, metadata.to_dict())
            assert len(recorded_warnings) == 0

        # Assert
        instance._get_trained_synthesizer.assert_called_once()
        args = instance._get_trained_synthesizer.call_args[0]
        assert (args[0] == data).all().all()
        assert args[1].to_dict() == metadata.to_dict()
        assert isinstance(args[1], Metadata)
        assert instance._get_trained_synthesizer.return_value == mock_synthesizer


class TestMultiTableBaselineSynthesizer:
    @pytest.mark.parametrize(
        'scale, expected_scale',
        [
            (None, 1.0),
            (2.0, 2.0),
        ],
    )
    def test_sample_from_synthesizer_valid(self, scale, expected_scale):
        """Test that valid calls return correct values and call underlying method."""
        synthesizer = MultiTableBaselineSynthesizer()
        mock_synthesizer = Mock()
        synthesizer._sample_from_synthesizer = Mock(return_value='sampled_data')

        # Run
        if scale is None:
            result = synthesizer.sample_from_synthesizer(mock_synthesizer)
        else:
            result = synthesizer.sample_from_synthesizer(mock_synthesizer, scale)

        # Assert call
        synthesizer._sample_from_synthesizer.assert_called_with(mock_synthesizer, expected_scale)

        assert result == 'sampled_data'
        assert synthesizer._MODALITY_FLAG == 'multi_table'

    def test_sample_from_synthesizer_raises_on_unexpected_kwarg(self):
        """Test that passing n_samples raises a TypeError."""
        synthesizer = MultiTableBaselineSynthesizer()
        mock_synthesizer = Mock()

        expected_error = re.escape(
            "sample_from_synthesizer() got an unexpected keyword argument 'n_samples'"
        )

        with pytest.raises(TypeError, match=expected_error):
            synthesizer.sample_from_synthesizer(
                mock_synthesizer,
                n_samples=10,
            )
