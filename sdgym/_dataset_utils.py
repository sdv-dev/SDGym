"""Utility functions for handling datasets."""

import json
import logging
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


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


def _get_dataset_subset(data, metadata_dict, modality):
    """Limit the size of a dataset for faster evaluation or testing.

    This function reduces a dataset to a smaller subset by restricting the number
    of rows and columns to 1000 rows and 10 columns. It ensures that essential
    columns—such as sequence indices and keys in sequential datasets—are always retained.

    Args:
        data (pd.DataFrame):
            The dataset to be reduced.
        metadata_dict (dict):
            A dictionary containing the dataset's metadata.
        modality (str):
            The dataset modality. Must be one of: ``'single_table'``, ``'sequential'``.

    Returns:
        tuple[pd.DataFrame, dict]:
            A tuple containing:
            - The reduced dataset as a DataFrame.
            - The updated metadata dictionary reflecting any removed columns.

    Raises:
        ValueError:
            If the provided modality is ``'multi_table'``.
    """
    if modality == 'multi_table':
        raise ValueError('limit_dataset_size is not supported for multi-table datasets.')

    max_rows, max_columns = (1000, 10)
    tables = metadata_dict.get('tables', {})
    mandatory_columns = []
    table_name, table_info = next(iter(tables.items()))

    columns = table_info.get('columns', {})
    keep_columns = list(columns)
    if modality == 'sequential':
        seq_index = table_info.get('sequence_index')
        seq_key = table_info.get('sequence_key')
        mandatory_columns = [col for col in (seq_index, seq_key) if col]

    optional_columns = [col for col in columns if col not in mandatory_columns]

    # If we have too many columns, drop extras but never mandatory ones
    if len(columns) > max_columns:
        keep_count = max_columns - len(mandatory_columns)
        keep_columns = mandatory_columns + optional_columns[:keep_count]
        table_info['columns'] = {
            column_name: column_definition
            for column_name, column_definition in columns.items()
            if column_name in keep_columns
        }

    data = data[list(keep_columns)]
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
