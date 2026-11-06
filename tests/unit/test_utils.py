import sys
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from sdgym.utils import (
    _get_single_table_dataframe,
    _is_list_of_type,
    _set_column_width,
    calculate_score_time,
    get_duplicates,
    get_size_of,
    get_utc_now,
)


def test_get_size_of():
    """Test that the correct size is returned."""
    # Setup
    test_obj = {'key': 'value'}

    # Run
    size = get_size_of(test_obj)

    # Assert
    assert size == sys.getsizeof('value')


def test_get_size_of_nested_obj():
    """Test that the correct size is returned when given a nested object."""
    # Setup
    test_inner_obj = {'inner_key': 'inner_value'}
    test_obj = {'key1': 'value', 'key2': test_inner_obj}

    # Run
    size = get_size_of(test_obj)

    # Assert
    assert size == sys.getsizeof('value') + sys.getsizeof('inner_value')


def test_get_duplicates():
    """Test that the correct duplicates are returned."""
    # Setup
    items = ['a', 'a', 'b', 'c', 'd', 'd', 'd']

    # Run
    duplicates = get_duplicates(items)

    # Assert
    assert duplicates == {'a', 'd'}


def test_get_utc_now():
    # Run
    now = get_utc_now()

    # Assert
    assert isinstance(now, datetime)
    assert now.tzinfo == timezone.utc


def test_calculate_score_time():
    # Setup
    start = get_utc_now()

    # Run
    total_secs = calculate_score_time(start)

    # Assert
    assert isinstance(total_secs, float)


@patch('sdgym.utils.get_column_letter')
def test_set_column_width_sets_expected_width(mock_get_column_letter):
    """Test `_set_column_width` sets correct column widths."""
    # Setup
    df = pd.DataFrame({'A': ['aa', 'bbbb'], 'LongCol': ['x', 'yy']})
    mock_get_column_letter.side_effect = ['A', 'B']
    worksheet = Mock()
    worksheet.column_dimensions = {
        'A': Mock(),
        'B': Mock(),
    }
    writer = Mock()
    writer.sheets = {'Sheet1': worksheet}

    # Run
    _set_column_width(writer, df, 'Sheet1')

    # Assert
    assert worksheet.column_dimensions['A'].width == 6
    assert worksheet.column_dimensions['B'].width == 9


def test__is_list_of_type():
    """Test `_is_list_of_type` method"""
    assert _is_list_of_type(['a', 'b'])
    assert not _is_list_of_type(['a', 1])
    assert not _is_list_of_type([1, 2])
    assert not _is_list_of_type(1)
    assert not _is_list_of_type('a')


def test__get_single_table_dataframe():
    """Test `_get_single_table_dataframe` returns the only table in the dictionary."""
    # Setup
    table = pd.DataFrame({'a': [1, 2, 3]})
    data = {'table': table}

    # Run
    result = _get_single_table_dataframe(data)

    # Assert
    pd.testing.assert_frame_equal(result, table)


@pytest.mark.parametrize(
    'data',
    [
        pd.DataFrame({'a': [1, 2, 3]}),
        'invalid',
        {'table': [1, 2, 3]},
        {'table': pd.DataFrame({'a': [1]}), 'other': 'not a dataframe'},
    ],
)
def test__get_single_table_dataframe_not_dict_of_dataframes(data):
    """Test `_get_single_table_dataframe` errors if the input is not a dict of dataframe."""
    # Setup
    expected_message = 'The data must be a dictionary mapping a single table name to a dataframe.'

    # Run and Assert
    with pytest.raises(ValueError, match=expected_message):
        _get_single_table_dataframe(data)


@pytest.mark.parametrize(
    'data, num_tables',
    [
        ({}, 0),
        ({'table_1': pd.DataFrame({'a': [1]}), 'table_2': pd.DataFrame({'b': [2]})}, 2),
    ],
)
def test__get_single_table_dataframe_wrong_number_of_tables(data, num_tables):
    """Test `_get_single_table_dataframe` raises an error if there is not exactly one table."""
    # Setup
    expected_message = (
        f'The data must contain exactly one table, but {num_tables} tables were found.'
    )

    # Run and Assert
    with pytest.raises(ValueError, match=expected_message):
        _get_single_table_dataframe(data)
