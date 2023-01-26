from unittest.mock import Mock, patch

import pandas as pd

from sdgym.synthesizers import IndependentSynthesizer


class TestIndependentSynthesizer:

    @patch('sdgym.synthesizers.independent.GaussianMixture')
    def test__get_trained_synthesizer(self, gm_mock):
        """Expect that GaussianMixture is instantiated with 4 components."""
        # Setup
        independent = IndependentSynthesizer()
        independent.length = 10
        data = pd.DataFrame({'col1': [1, 2, 3, 4]})

        # Run
        independent._get_trained_synthesizer(data, Mock())

        # Assert
        gm_mock.assert_called_once_with(4)
