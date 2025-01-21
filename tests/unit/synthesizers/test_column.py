from unittest.mock import patch

import pandas as pd
from sdv.metadata import SingleTableMetadata

from sdgym.synthesizers import ColumnSynthesizer


class TestColumnSynthesizer:
    @patch('sdgym.synthesizers.column.GaussianMixture')
    def test__get_trained_synthesizer(self, gm_mock):
        """Expect that GaussianMixture is instantiated with 4 components."""
        # Setup
        column_synthesizer = ColumnSynthesizer()
        column_synthesizer.length = 10
        data = pd.DataFrame({'col1': [1, 2, 3, 4]})
        metadata = SingleTableMetadata()
        metadata.add_column('col1', sdtype='numerical')

        # Run
        column_synthesizer._get_trained_synthesizer(data, metadata)

        # Assert
        gm_mock.assert_called_once_with(4)
