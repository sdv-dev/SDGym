import pytest

from sdgym._run_benchmark._utils import get_run_name


def test_get_run_name():
    """Test the `get_run_name` method."""
    # Setup
    expected_error_message = 'Invalid date format: invalid-date. Expected YYYY-MM-DD.'

    # Run and Assert
    assert get_run_name('2023-10-01') == 'SDGym_results_10_01_2023'
    with pytest.raises(ValueError, match=expected_error_message):
        get_run_name('invalid-date')
