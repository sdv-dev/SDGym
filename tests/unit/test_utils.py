import sys
from datetime import datetime, timezone

from sdgym.utils import calculate_score_time, get_duplicates, get_size_of, get_utc_now


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
