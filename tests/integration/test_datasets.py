from sdgym import DatasetExplorer


def test_list_datasets_single_table():
    """Test that it lists single table datasets with expected properties."""
    # Run
    dataframe = DatasetExplorer().list_datasets('single_table')

    # Assert
    assert dataframe.columns.tolist() == ['dataset_name', 'size_MB', 'num_tables']
    assert all(dataframe['num_tables'] == 1)


def test_list_datasets_multi_table():
    """Test that it lists multi table datasets with expected properties."""
    # Run
    dataframe = DatasetExplorer().list_datasets('multi_table')

    # Assert
    assert dataframe.columns.tolist() == ['dataset_name', 'size_MB', 'num_tables']
    assert all(dataframe['num_tables'] > 1)


def test_list_datasets_sequential():
    """Test that it lists sequential datasets with expected properties."""
    # Run
    dataframe = DatasetExplorer().list_datasets('sequential')

    # Assert
    assert dataframe.columns.tolist() == ['dataset_name', 'size_MB', 'num_tables']
    assert all(dataframe['num_tables'] == 1)
