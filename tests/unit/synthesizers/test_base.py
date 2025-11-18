import re
import warnings
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from sdv.metadata import Metadata

from sdgym.synthesizers.base import BaselineSynthesizer, _is_valid_modality, _validate_modality


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
    def test__validate_modality_flag(self):
        """Test the `_validate_modality_flag` method."""
        # Setup
        instance = BaselineSynthesizer()
        instance._MODALITY_FLAG = 'single_table'
        expected_error = re.escape(
            "The `_MODALITY_FLAG` 'None' of the synthesizer is not valid. Must be"
            " either 'single_table' or 'multi_table'."
        )

        # Run and Assert
        instance._validate_modality_flag()
        instance._MODALITY_FLAG = None
        with pytest.raises(ValueError, match=expected_error):
            instance._validate_modality_flag()

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

    @patch('sdgym.synthesizers.base.BaselineSynthesizer._validate_modality_flag')
    def test_get_trained_synthesizer(self, mock_validate_modality_flag):
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
        mock_validate_modality_flag.assert_called_once_with()
        instance._get_trained_synthesizer.assert_called_once()
        args = instance._get_trained_synthesizer.call_args[0]
        assert (args[0] == data).all().all()
        assert args[1].to_dict() == metadata.to_dict()
        assert isinstance(args[1], Metadata)
        assert instance._get_trained_synthesizer.return_value == mock_synthesizer
