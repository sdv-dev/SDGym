import pytest

from sdgym.run_benchmark.utils import get_result_folder_name


def test_get_result_folder_name():
    """Test the `get_result_folder_name` method."""
    # Setup
    expected_error_message = 'Invalid date format: invalid-date. Expected YYYY-MM-DD.'

    # Run and Assert
    assert get_result_folder_name('2023-10-01') == 'SDGym_results_10_01_2023'
    with pytest.raises(ValueError, match=expected_error_message):
        get_result_folder_name('invalid-date')
