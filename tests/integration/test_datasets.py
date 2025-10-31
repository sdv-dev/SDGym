from sdgym import get_available_datasets


def test_get_available_datasets_single_table():
    """Test that `get_available_datasets` returns single table datasets with expected properties."""
    # Run
    df = get_available_datasets('single_table')

    # Assert
    assert df.columns.tolist() == ['dataset_name', 'size_MB', 'num_tables']
    assert all(df['num_tables'] == 1)


def test_get_available_datasets_multi_table():
    """Test that `get_available_datasets` returns multi table datasets with expected properties."""
    # Run
    df = get_available_datasets('multi_table')

    # Assert
    assert df.columns.tolist() == ['dataset_name', 'size_MB', 'num_tables']
    assert all(df['num_tables'] > 1)


def test_get_available_datasets_sequential():
    """Test that `get_available_datasets` returns sequential datasets with expected properties."""
    # Run
    df = get_available_datasets('sequential')

    # Assert
    assert df.columns.tolist() == ['dataset_name', 'size_MB', 'num_tables']
    assert all(df['num_tables'] == 1)
