from unittest.mock import Mock, patch

import pandas as pd

from sdgym.synthesizers import ColumnSynthesizer


class TestColumnSynthesizer:
    @patch('sdgym.synthesizers.column.GaussianMixture')
    def test__get_trained_synthesizer(self, gm_mock):
        """Expect that GaussianMixture is instantiated with 4 components."""
        # Setup
        column_synthesizer = ColumnSynthesizer()
        column_synthesizer.length = 10
        data = pd.DataFrame({'col1': [1, 2, 3, 4]})

        # Run
        column_synthesizer._get_trained_synthesizer(data, Mock())

        # Assert
        gm_mock.assert_called_once_with(4)
