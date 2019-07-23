from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from sdgym.synthesizers.utils import DiscretizeTransformer, Transformer


class TestTransformer(TestCase):

    def test_get_metadata(self):
        """get_metadata returns information about the dataframe."""
        # Setup
        data = pd.DataFrame({
            'numerical': [0, 1, 2, 3, 4, 5],
            'categorical': list('AAABBC')
        }).values

        expected_result = [
            {
                'name': 0,
                'type': 'continuous',
                'min': 0,
                'max': 5
            },
            {
                "name": 1,
                "type": 'categorical',
                "size": 3,
                "i2s": ['A', 'B', 'C']
            }
        ]

        categorical_columns = [1]
        ordinal_columns = []

        # Run
        result = Transformer.get_metadata(data, categorical_columns, ordinal_columns)

        # Check
        assert result == expected_result


class TestDiscretizeTransformer(TestCase):

    def test___init__(self):
        """On init attributes are set as None, and n_bins assigned."""
        # Setup
        n_bins = 5

        # Run
        instance = DiscretizeTransformer(n_bins=n_bins)

        # Check
        assert instance.n_bins == 5
        assert instance.meta is None
        assert instance.column_index is None
        assert instance.discretizer is None

    @patch('sdgym.synthesizers.utils.KBinsDiscretizer', autospec=True)
    def test_fit(self, kbins_mock):
        # Setup
        n_bins = 2
        instance = DiscretizeTransformer(n_bins=n_bins)
        data = pd.DataFrame({
            'A': [1 / (x + 1) for x in range(10)],
            'B': [x for x in range(10)]
        }).values
        kbins_instance = kbins_mock.return_value

        # Run
        instance.fit(data, [], [])

        # Check
        assert instance.column_index == [0, 1]
        assert instance.discretizer == kbins_instance
        assert instance.meta == [
            {
                'name': 0,
                'type': 'continuous',
                'min': 0.1,
                'max': 1.0
            },
            {
                'name': 1,
                'type': 'continuous',
                'min': 0.0,
                'max': 9.0
            }
        ]

        kbins_mock.assert_called_once_with(n_bins=2, encode='ordinal', strategy='uniform')
        call_list = kbins_instance.fit.call_args_list
        assert len(call_list) == 1
        call_args, call_kwargs = call_list[0]
        assert call_kwargs == {}
        assert len(call_args) == 1
        np.testing.assert_equal(call_args[0], data)

    def test_transform(self):
        """transform continous columns into discrete bins."""
        # Setup
        instance = DiscretizeTransformer(n_bins=2)
        data = pd.DataFrame({
            'A': [x for x in range(10)],
            'B': [2 * x for x in range(10)]
        }).values
        expected_result = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
        ])
        instance.fit(data)

        # Run
        result = instance.transform(data)

        # Check
        np.testing.assert_equal(result, expected_result)

    def test_inverse_transform(self):
        """Transform discrete values back into its original space."""
        # Setup
        n_bins = 2
        instance = DiscretizeTransformer(n_bins=n_bins)
        data = pd.DataFrame({
            'A': [1 / (x + 1) for x in range(10)],
            'B': [x for x in range(10)]
        }).values
        instance.fit(data)
        transformed_data = instance.transform(data)
        expected_result = pd.DataFrame({
            'A': [0.775, 0.325, 0.325, 0.325, 0.325, 0.325, 0.325, 0.325, 0.325, 0.325],
            'B': [2.25, 2.25, 2.25, 2.25, 2.25, 6.75, 6.75, 6.75, 6.75, 6.75]
        })

        # Run
        result = instance.inverse_transform(transformed_data)

        # Check
        np.testing.assert_allclose(result, expected_result)
