"""Utility functions for handling datasets."""

import json
import logging
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sdv.metadata import Metadata
from sdv.utils import poc

LOGGER = logging.getLogger(__name__)

MAX_NUM_COLUMNS = 10
MAX_NUM_ROWS = 1000


def _parse_numeric_value(value, dataset_name, field_name, target_type=float):
    """Generic parser for numeric values with logging and NaN fallback."""
    try:
        return target_type(value)
    except (ValueError, TypeError):
        LOGGER.info(
            f"Could not cast {field_name} '{value}' to {target_type.__name__} for dataset "
            f"'{dataset_name}' defaulting to NaN."
        )
        return np.nan


def _filter_columns(columns, mandatory_columns):
    """Given a dictionary of columns and a list of mandatory ones, return a filtered subset."""
    mandatory_columns = [m_col for m_col in mandatory_columns if m_col in columns]
    optional_columns = [col for col in columns if col not in mandatory_columns]
    keep_columns = mandatory_columns + optional_columns[:MAX_NUM_COLUMNS]
    return {col: columns[col] for col in keep_columns if col in columns}


def _get_multi_table_dataset_subset(data, metadata_dict):
    """Create a smaller, referentially consistent subset of multi-table data.

    This function limits each table to at most 10 columns by keeping all
    mandatory columns and, if needed, a subset of the remaining columns, then
    trims the underlying DataFrames to match the updated metadata. Finally, it
    uses SDV's multi-table utility to sample up to 1,000 rows from
    the main table and a consistent subset of rows from all related tables
    while preserving referential integrity.

    Args:
        data (dict):
            A dictionary where keys are table names and values are DataFrames
            representing tables.
        metadata_dict (dict):
            Metadata dictionary containing schema information for each table.

    Returns:
        tuple:
            A tuple containing:
                - dict: The subset of the input data with reduced columns and rows.
                - dict: The updated metadata dictionary reflecting the reduced column sets.
    """
    metadata = Metadata.load_from_dict(metadata_dict)
    for table_name, table in metadata.tables.items():
        table_columns = table.columns
        mandatory_columns = list(metadata._get_all_keys(table_name))
        subset_column_schema = _filter_columns(
            columns=table_columns, mandatory_columns=mandatory_columns
        )
        metadata_dict['tables'][table_name]['columns'] = subset_column_schema

    # Re-load the metadata object that will be used with the `SDV` utility function
    metadata = Metadata.load_from_dict(metadata_dict)
    largest_table_name = max(data, key=lambda table_name: len(data[table_name]))

    # Trim the data to contain only the subset of columns
    for table_name, table in metadata.tables.items():
        data[table_name] = data[table_name][list(table.columns)]

    # Subsample the data mantaining the referential integrity
    data = poc.get_random_subset(
        data=data,
        metadata=metadata,
        main_table_name=largest_table_name,
        num_rows=MAX_NUM_ROWS,
        verbose=False,
    )
    return data, metadata_dict


def _get_dataset_subset(data, metadata_dict, modality):
    """Limit the size of a dataset for faster evaluation or testing.

    This function reduces a dataset to a smaller subset by restricting the number
    of rows and columns to 1000 rows and 10 columns. It ensures that essential
    columns—such as sequence indices and keys in sequential datasets—are always retained.

    Args:
        data (pd.DataFrame or dict):
            The dataset to be reduced.
        metadata_dict (dict):
            A dictionary representing the dataset's metadata.
        modality (str):
            The dataset modality.

    Returns:
        tuple[pd.DataFrame, dict]:
            A tuple containing:
            - The reduced dataset as a DataFrame or Dictionary.
            - The updated metadata dictionary reflecting any removed columns.
    """
    if modality == 'multi_table':
        return _get_multi_table_dataset_subset(data, metadata_dict)

    tables = metadata_dict.get('tables', {})
    mandatory_columns = []
    table_name, table_info = next(iter(tables.items()))
    columns = table_info.get('columns', {})

    seq_index = table_info.get('sequence_index')
    seq_key = table_info.get('sequence_key')
    mandatory_columns = [column for column in (seq_index, seq_key) if column]
    filtered = _filter_columns(columns=columns, mandatory_columns=mandatory_columns)

    table_info['columns'] = filtered
    data = data[list(filtered)]
    max_rows = min(MAX_NUM_ROWS, len(data))
    data = data.sample(max_rows)

    return data, metadata_dict


def _read_zipped_data(zip_file_path, modality):
    data = {}
    with ZipFile(zip_file_path, 'r') as zf:
        for file_name in zf.namelist():
            if file_name.endswith('.csv'):
                key = Path(file_name).stem
                data[key] = _read_csv_from_zip(zf, csv_file_name=file_name)

    if modality != 'multi_table':
        data = next(iter(data.values()))

    return data


def _read_csv_from_zip(zip_file, csv_file_name):
    """Read a single CSV file from an open ZipFile and return a DataFrame."""
    with zip_file.open(csv_file_name) as csv_file:
        return pd.read_csv(csv_file, low_memory=False)


def _read_metadata_json(metadata_path):
    with open(metadata_path) as metadata_file:
        metadata_dict = json.load(metadata_file)

    return metadata_dict
