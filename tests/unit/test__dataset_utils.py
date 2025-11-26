import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from sdgym._dataset_utils import (
    _get_dataset_subset,
    _get_multi_table_dataset_subset,
    _parse_numeric_value,
    _read_csv_from_zip,
    _read_metadata_json,
    _read_zipped_data,
)


@pytest.mark.parametrize(
    'value,expected',
    [
        ('3.14', 3.14),
        ('not-a-number', np.nan),
        (None, np.nan),
    ],
)
def test__parse_numeric_value(value, expected):
    """Test numeric parsing with fallback to NaN."""
    # Setup / Run
    result = _parse_numeric_value(value, 'dataset', 'field')

    # Assert
    if np.isnan(expected):
        assert np.isnan(result)
    else:
        assert result == expected


@patch('sdgym._dataset_utils.poc.get_random_subset')
@patch('sdgym._dataset_utils.Metadata')
def test__get_multi_table_dataset_subset(mock_metadata, mock_subset):
    """Test multi-table subset selection calls SDV and trims columns."""
    # Setup
    df_main = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    df_other = pd.DataFrame({'x': [5, 6], 'y': [7, 8]})

    data = {'main': df_main, 'other': df_other}

    metadata_dict = {
        'tables': {
            'main': {'columns': {'a': {}, 'b': {}}},
            'other': {'columns': {'x': {}, 'y': {}}},
        }
    }

    mock_meta_obj = MagicMock()
    mock_meta_obj.tables = {
        'main': MagicMock(columns={'a': {}, 'b': {}}),
        'other': MagicMock(columns={'x': {}, 'y': {}}),
    }
    mock_meta_obj._get_all_keys.return_value = []
    mock_metadata.load_from_dict.return_value = mock_meta_obj

    mock_subset.return_value = {'main': df_main[:1], 'other': df_other[:1]}

    # Run
    result_data, result_meta = _get_multi_table_dataset_subset(data, metadata_dict)

    # Assert
    assert 'main' in result_data
    assert 'other' in result_data
    mock_subset.assert_called_once()


def test__get_dataset_subset_single_table():
    """Test tabular dataset subset reduces rows and columns."""
    # Setup
    df = pd.DataFrame({f'c{i}': range(2000) for i in range(15)})
    metadata = {'tables': {'table': {'columns': {f'c{i}': {} for i in range(15)}}}}

    # Run
    result_df, result_meta = _get_dataset_subset(df, metadata, modality='regular')

    # Assert
    assert len(result_df) <= 1000
    assert len(result_df.columns) == 10
    assert 'tables' in result_meta


def test__get_dataset_subset_sequential():
    """Test sequential dataset preserves mandatory columns."""
    # Setup
    df = pd.DataFrame({
        'seq_id': range(20),
        'seq_key': range(20),
        **{f'c{i}': range(20) for i in range(20)},
    })

    metadata = {
        'tables': {
            'table': {
                'columns': {col: {'sdtype': 'numerical'} for col in df.columns.to_list()},
                'sequence_index': 'seq_id',
                'sequence_key': 'seq_key',
            }
        }
    }

    # Run
    subset_df, _ = _get_dataset_subset(df, metadata, modality='sequential')

    # Assert
    assert 'seq_id' in subset_df.columns
    assert 'seq_key' in subset_df.columns
    assert len(subset_df.columns) <= 10


@patch('sdgym._dataset_utils._get_multi_table_dataset_subset')
def test__get_dataset_subset_multi_table(mock_multi):
    """Test multi-table dispatch calls the correct function."""
    # Setup
    data = {'table': pd.DataFrame({'a': [1, 2]})}
    metadata = {'tables': {}}
    mock_multi.return_value = ('DATA', 'META')

    # Run
    out_data, out_meta = _get_dataset_subset(data, metadata, modality='multi_table')

    # Assert
    assert out_data == 'DATA'
    assert out_meta == 'META'
    mock_multi.assert_called_once()


@patch('sdgym._dataset_utils._read_csv_from_zip')
def test__read_zipped_data_multitable(mock_read):
    """Test zipped CSV reading returns a dict for multi-table."""
    # Setup
    mock_read.return_value = pd.DataFrame({'a': [1]})

    mock_zip = MagicMock()
    mock_zip.__enter__.return_value = mock_zip
    mock_zip.namelist.return_value = ['table1.csv', 'table2.csv']

    # Run
    with patch('sdgym._dataset_utils.ZipFile', return_value=mock_zip):
        data_multi = _read_zipped_data('fake.zip', modality='multi_table')

    # Assert
    assert isinstance(data_multi, dict)
    assert mock_read.call_count == 2


@patch('sdgym._dataset_utils._read_csv_from_zip')
def test__read_zipped_data_single(mock_read):
    """Test zipped CSV reading returns a DataFrame for single-table."""
    # Setup
    mock_read.return_value = pd.DataFrame({'a': [1]})

    mock_zip = MagicMock()
    mock_zip.__enter__.return_value = mock_zip
    mock_zip.namelist.return_value = ['table1.csv']

    # Run
    with patch('sdgym._dataset_utils.ZipFile', return_value=mock_zip):
        data_single = _read_zipped_data('fake.zip', modality='single')

    # Assert
    assert isinstance(data_single, pd.DataFrame)
    assert mock_read.call_count == 1


@patch('sdgym._dataset_utils.pd')
def test__read_csv_from_zip(mock_pd):
    """Test CSV is read from zip and returned as DataFrame."""
    # Setup
    csv_bytes = b'a,b\n1,2\n3,4\n'
    returned_bytes = csv_bytes.decode().splitlines()
    mock_zip = MagicMock()
    mock_zip.open.return_value.__enter__.return_value = returned_bytes

    # Run
    result = _read_csv_from_zip(mock_zip, 'fake.csv')

    # Assert
    mock_pd.read_csv.assert_called_once_with(returned_bytes, low_memory=False)
    assert result == mock_pd.read_csv.return_value


def test__read_metadata_json(tmp_path):
    """Test reading metadata JSON file."""
    # Setup
    meta = {'tables': {'a': {}}}
    path = tmp_path / 'meta.json'
    path.write_text(json.dumps(meta))

    # Run
    result = _read_metadata_json(path)

    # Assert
    assert result == meta
