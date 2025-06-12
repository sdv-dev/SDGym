from sdgym import get_available_datasets
import pytest
import re

def test_get_available_datasets():
    # Run
    df = get_available_datasets('single_table')

    # Assert
    assert df.columns.tolist() == ['dataset_name', 'size_MB', 'num_tables']
    assert df['num_tables'].unique().tolist() == [1]


def test_get_available_datasets_raises():
    # Setup
    msg = re.escape("'modality' must be in ['single_table'].")

    # Run and Assert
    with pytest.raises(ValueError, match=msg):
        get_available_datasets('multi-table')

