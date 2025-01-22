from unittest.mock import patch

import pandas as pd
from sdv.metadata import Metadata, SingleTableMetadata

from sdgym.synthesizers import ColumnSynthesizer


class TestColumnSynthesizer:
    @patch('sdgym.synthesizers.column.GaussianMixture')
    def test__get_trained_synthesizer(self, gm_mock):
        """Expect that GaussianMixture is instantiated with 4 components."""
        # Setup
        column_synthesizer = ColumnSynthesizer()
        column_synthesizer.length = 10
        data = pd.DataFrame({'col': [1, 2, 3, 4]})
        metadata = Metadata()
        metadata.add_table('table')
        metadata.add_column('col', 'table', sdtype='numerical')

        # Run
        column_synthesizer._get_trained_synthesizer(data, metadata)

        # Assert
        gm_mock.assert_called_once_with(4)

    @patch('sdgym.synthesizers.column.GaussianMixture')
    def test__get_trained_synthesizer_single_table_metadata(self, gm_mock):
        """Expect that GaussianMixture is instantiated with 4 components."""
        # Setup
        column_synthesizer = ColumnSynthesizer()
        column_synthesizer.length = 10
        data = pd.DataFrame({'col': [1, 2, 3, 4]})
        metadata = SingleTableMetadata()
        metadata.add_column('col', sdtype='numerical')

        # Run
        column_synthesizer._get_trained_synthesizer(data, metadata)

        # Assert
        gm_mock.assert_called_once_with(4)
