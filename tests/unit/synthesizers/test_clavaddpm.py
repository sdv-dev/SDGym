"""Tests for the clavaddmp module."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sdv.metadata import Metadata

from sdgym.synthesizers.clavaddpm import (
    ClavaDDPM,
    ClavaDDPMSynthesizer,
    guess_array_datetime_format,
)


def test_guess_array_datetime_format_empty_input_returns_none():
    """Test guess array with an empty input returns None."""
    # Run
    result = guess_array_datetime_format([])

    # Assert
    assert result is None


def test_guess_array_datetime_format_all_null_returns_none():
    """Test guess array with all null inputs returns None."""
    # Run
    result = guess_array_datetime_format([None, None, float('nan')])

    # Assert
    assert result is None


def test_guess_array_datetime_format_with_datetime_column():
    """Test guess array works as expected with datetime values."""
    # Setup
    values = pd.to_datetime(['2020-01-01', '2020-01-02'])

    # Run
    result = guess_array_datetime_format(values)

    # Assert
    assert result == '%Y-%m-%d'


def test_guess_array_datetime_format_non_date_numeric_values_return_none():
    """Test guess array doesn't parse small integers as datetime format."""
    # Run
    result = guess_array_datetime_format([1, 2, 3, 4, 5])

    # Assert
    assert result is None


def test_guess_array_datetime_format_consistent_format_detected():
    """Test that guess array datetime works as expected."""
    # Setup
    values = ['2020-01-05', '2020-02-14', '2020-03-30']

    # Run
    result = guess_array_datetime_format(values)

    # Assert
    assert result == '%Y-%m-%d'


def test_majority_format_wins_over_minority():
    """Test that guess array datetime works finds the majority format."""
    # Setup
    values = ['2020-01-05'] * 7 + ['01/05/2020'] * 3

    # Run
    result = guess_array_datetime_format(values)

    # Assert
    assert result == '%Y-%m-%d'


def test_guess_array_datetime_format_with_mixed_values():
    """Test guess array returns the majority of valid dates."""
    # Setup
    values = ['2020-01-05', '2020-02-14', '2020-03-30'] + [''] * 5 + ['n/a'] * 5

    # Run
    result = guess_array_datetime_format(values)

    # Assert
    assert result == '%Y-%m-%d'


def test_guess_array_datetime_format_with_whitespace_only_strings_are_ignored():
    """Test guess array returns ignores whitespace."""
    # Setup
    values = ['2020-01-05', '2020-02-14', '', '   ', '2020-03-30', '\t']

    # Run
    result = guess_array_datetime_format(values)

    # Assert
    assert result == '%Y-%m-%d'


def test_guess_array_datetime_format_with_all_unparseable_returns_none():
    """Test guess array returns None when all values are not dates."""
    # Setup
    values = ['n/a', 'not a date', 'xx', '']

    # Run
    result = guess_array_datetime_format(values)

    # Assert
    assert result is None


@pytest.mark.parametrize(
    'values, expected',
    [
        (['2020-01-05', '2020-02-14', '2020-03-30'] + [None] * 10, '%Y-%m-%d'),
        (['2020/01/05', '2020/02/14', '2020/03/30', '2020-02-14', '2020-03-30'], '%Y/%m/%d'),
        (['2020-01-05 00:10:05', '2020-02-14 12:12:05', '2020-03-30'], '%Y-%m-%d %H:%M:%S'),
    ],
)
def test_guess_array_datetime_format_with_expected_format(values, expected):
    """Test guess array returns expected formats."""
    # Run
    result = guess_array_datetime_format(values)

    # Assert
    assert result == expected


@pytest.mark.parametrize(
    'dayfirst, expected',
    [
        (False, '%m-%d-%Y'),
        (True, '%d-%m-%Y'),
    ],
)
def test_guess_array_datetime_format_dayfirst_changes_ambiguous_guess(dayfirst, expected):
    """Test guess array with dayfirst parameter clears up ambiguity."""
    # Setup
    values = ['01-02-2020'] * 5

    # Run
    result = guess_array_datetime_format(values, dayfirst=dayfirst)

    # Assert
    assert result == expected


def test_sample_size_larger_than_population_does_not_error():
    """Test guess array with larger sample size works."""
    # Setup
    values = ['2020-01-01', '2020-01-02']

    # Run
    result = guess_array_datetime_format(values, sample_size=1000)

    # Assert
    assert result == '%Y-%m-%d'


def test_sample_size_zero_returns_none():
    """Test guess array with zero sample size returns None."""
    # Setup
    values = ['2020-01-01', '2020-01-02']

    # Run
    result = guess_array_datetime_format(values, sample_size=0)

    # Assert
    assert result is None


def test_reproducible_with_fixed_random_state():
    """Test guess array retuns the same result."""
    # Setup
    values = ['2020-01-05'] * 150 + ['01/05/2020'] * 150

    # Run
    result_a = guess_array_datetime_format(values, sample_size=10)
    result_b = guess_array_datetime_format(values, sample_size=10)

    # Assert
    assert result_a == result_b


class TestClavaDDPMSynthesizer:
    @patch('sdgym.synthesizers.clavaddpm.ClavaDDPM')
    def test__fit_mock(self, mock_clavaddpm):
        """Test `_fit` builds and fits mock ClavaDDPM."""
        # Setup
        synthesizer = ClavaDDPMSynthesizer()
        data = {'table1': pd.DataFrame({'col1': [1, 2, 3]})}
        metadata = MagicMock()

        # Run
        synthesizer._fit(data, metadata)

        # Assert
        mock_clavaddpm.assert_called_once_with(metadata)
        mock_clavaddpm.return_value.fit.assert_called_once_with(data)
        assert synthesizer._internal_synthesizer is mock_clavaddpm.return_value

    @patch('sdgym.synthesizers.clavaddpm.ClavaDDPM')
    def test__fit_passes_model_kwargs(self, mock_clavaddpm):
        """Test `_fit` forwards `_MODEL_KWARGS` to ClavaDDPM."""
        # Setup
        synthesizer = ClavaDDPMSynthesizer()
        synthesizer._MODEL_KWARGS = {'num_clusters': 5, 'num_timesteps': 100}
        data = {'table1': pd.DataFrame({'col1': [1, 2, 3]})}
        metadata = MagicMock()

        # Run
        synthesizer._fit(data, metadata)

        # Assert
        mock_clavaddpm.assert_called_once_with(metadata, num_clusters=5, num_timesteps=100)

    def test__fit(self):
        """Test the `_fit` method."""
        # Setup
        synthesizer = ClavaDDPMSynthesizer()
        synthesizer._MODEL_KWARGS = {'diffusion_iterations': 300, 'num_timesteps': 100}
        data = {
            'table1': pd.DataFrame({
                'col1': [1, 2, 3],
                'col2': ['A', 'B', 'C'],
            }),
            'table2': pd.DataFrame({
                'col3': [10.0, 20.0, 30.0],
                'col4': [True, False, True],
            }),
        }

        metadata = Metadata.load_from_dict({
            'tables': {
                'table1': {
                    'columns': {
                        'col1': {'sdtype': 'numerical'},
                        'col2': {'sdtype': 'categorical'},
                    },
                    'primary_key': 'col1',
                },
                'table2': {
                    'columns': {
                        'col3': {'sdtype': 'numerical'},
                        'col4': {'sdtype': 'boolean'},
                    },
                    'primary_key': 'col3',
                },
            },
            'relationships': [],
        })

        # Run
        synthesizer._fit(data, metadata)

        # Assert
        assert isinstance(synthesizer._internal_synthesizer, ClavaDDPM)
        assert isinstance(synthesizer, ClavaDDPMSynthesizer)

    @patch('sdgym.synthesizers.clavaddpm.ClavaDDPM')
    def test__get_trained_synthesizer(self, mock_clavaddpm):
        """Test `_get_trained_synthesizer` with mock."""
        # Setup
        synthesizer = ClavaDDPMSynthesizer()
        data = {'table1': pd.DataFrame({'col1': [1, 2, 3]})}
        metadata = MagicMock()

        # Run
        trained_synthesizer = synthesizer._get_trained_synthesizer(data, metadata)

        # Assert
        assert isinstance(trained_synthesizer, ClavaDDPMSynthesizer)
        assert trained_synthesizer._internal_synthesizer is mock_clavaddpm.return_value
        mock_clavaddpm.return_value.fit.assert_called_once_with(data)
