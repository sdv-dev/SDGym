import warnings
from unittest.mock import Mock

import pandas as pd
from sdv.metadata import Metadata

from sdgym.synthesizers.base import BaselineSynthesizer


class TestBaselineSynthesizer:
    def test_get_trained_synthesizer(self):
        """Test it ."""
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
