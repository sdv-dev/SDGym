import warnings
from unittest.mock import Mock, patch

import pandas as pd
from sdv.metadata import Metadata

from sdgym.synthesizers.base import BaselineSynthesizer


class TestBaselineSynthesizer:
    @patch('sdgym.synthesizers.utils.BaselineSynthesizer.get_subclasses')
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
