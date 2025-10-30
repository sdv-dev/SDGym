"""SDGym module to handle datasets."""

import io
import itertools
import json
import logging
import os
from pathlib import Path
from zipfile import ZipFile

import appdirs
import pandas as pd

from sdgym.s3 import get_s3_client

LOGGER = logging.getLogger(__name__)

DATASETS_PATH = Path(appdirs.user_data_dir()) / 'SDGym' / 'datasets'
BUCKET = 's3://sdv-datasets-public' # 's3://sdv-demo-datasets'
BUCKET_URL = 'https://{}.s3.amazonaws.com/'
TIMESERIES_FIELDS = ['sequence_index', 'entity_columns', 'context_columns', 'deepecho_version']
MODALITIES = ['single_table', 'multi_table', 'sequential']
S3_PREFIX = 's3://'


def _get_bucket_name(bucket):
    return bucket[len(S3_PREFIX) :] if bucket.startswith(S3_PREFIX) else bucket


def _download_dataset(
    modality,
    dataset_name,
    datasets_path=None,
    bucket=None,
    aws_access_key_id=None,
    aws_secret_access_key=None,
):
    """Download a dataset into the given ``datasets_path`` / ``modality``."""
    datasets_path = datasets_path or DATASETS_PATH / modality / dataset_name
    bucket = bucket or BUCKET
    bucket_name = _get_bucket_name(bucket)

    LOGGER.info('Downloading dataset %s from %s', dataset_name, bucket)
    s3_client = get_s3_client(aws_access_key_id, aws_secret_access_key)
    prefix = f'{modality.lower()}/{dataset_name}/'

    contents = []
    paginator = s3_client.get_paginator('list_objects_v2')
    for resp in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        contents.extend(resp.get('Contents', []))

    if not contents:
        raise ValueError(f"No objects found under '{prefix}' in bucket '{BUCKET}'.")

    for obj in contents:
        s3_path = obj['Key']
        local_path = datasets_path / os.path.relpath(s3_path, prefix)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3_client.download_file(bucket_name, s3_path, local_path)

    return datasets_path


def _path_contains_data_and_metadata(dataset_path):
    metadata_found = False
    data_zip_found = False
    for file_name in dataset_path.iterdir():
        if 'metadata' in file_name.stem and file_name.suffix == '.json':
            metadata_found = True

        if 'data' in file_name.stem and file_name.suffix == '.zip':
            data_zip_found = True

    return metadata_found and data_zip_found



def _get_dataset_path_or_download(
    modality,
    dataset,
    datasets_path,
    bucket=None,
    aws_access_key_id=None,
    aws_secret_access_key=None,
):
    dataset = Path(dataset)
    if dataset.exists():
        return dataset

    datasets_path = datasets_path or DATASETS_PATH / modality
    dataset_path = datasets_path / dataset
    if dataset_path.exists() and _path_contains_data_and_metadata(dataset_path):
        return dataset_path

    bucket = bucket or BUCKET
    if not bucket.startswith(S3_PREFIX):
        local_path = Path(bucket) / modality / dataset if bucket else Path(dataset)
        if local_path.exists() and _path_contains_data_and_metadata(local_path):
            return local_path

    print(dataset, dataset_path)
    dataset_path = _download_dataset(
        modality, dataset, dataset_path, bucket, aws_access_key_id, aws_secret_access_key
    )
    return dataset_path


def _get_dataset_subset(data, metadata_dict, modality):
    if modality == 'multi_table':
        raise ValueError('limit_dataset_size is not supported for multi-table datasets.')

    max_rows, max_columns = (1000, 10)
    tables = metadata_dict.get('tables', {})
    mandatory_columns = []
    for table_name, table_info in tables.items():
        columns = table_info.get('columns', {})

        if modality == 'sequential':
            seq_index = table_info.get('sequence_index')
            seq_key = table_info.get('sequence_key')
            mandatory_columns = [seq_index, seq_key]

        optional_columns = [col for col in columns if col not in mandatory_columns]

        # If we have too many columns, drop extras but never mandatory ones
        if len(columns) > max_columns:
            keep_count = max_columns - len(mandatory_columns)
            keep_columns = mandatory_columns.union(set(optional_columns[:keep_count]))
            table_info['columns'] = {
                column_name: column_definition
                for column_name, column_definition in columns.items()
                if column_name in keep_columns
            }

    data = data[keep_columns]
    data = data.head(max_rows)
    return data, metadata_dict


def _validate_modality(modality):
    if modality not in MODALITIES:
        modalities_list = ', '.join(MODALITIES)
        raise ValueError(f'Modality `{modality}` not recognized. Must be one of {modalities_list}')


def load_dataset(
    modality,
    dataset,
    datasets_path=None,
    bucket=None,
    aws_access_key_id=None,
    aws_secret_access_key=None,
    limit_dataset_size=False
):
    """Get the data and metadata of a dataset.

    Args:
        modality (str):
            It must be ``'single_table'``, ``'multi_table'`` or ``'sequential'``.
        dataset (str):
            The path of the dataset as a string.
        dataset_path (PurePath):
            The path of the dataset as an object. This will only be used if the given ``dataset``
            doesn't exist.
        bucket (str):
            The AWS bucket where to get the dataset. This will only be used if both ``dataset``
            and ``dataset_path`` don't exist.
        aws_access_key_id (str):
            The access key id that will be used to communicate with s3, if provided.
        aws_secret_access_key (str):
            The secret access key that will be used to communicate with s3, if provided.
        limit_dataset_size (bool):
            Use this flag to limit the size of the datasets for faster evaluation. If ``True``,
            limit the size of every table to 1,000 rows (randomly sampled) and the first 10
            columns.

    Returns:
        pd.DataFrame, dict:
            The data and medatata of a dataset.
    """
    _validate_modality(modality)
    dataset_path = _get_dataset_path_or_download(
        modality,
        dataset,
        datasets_path,
        bucket,
        aws_access_key_id,
        aws_secret_access_key
    )

    data, metadata_dict = get_data_and_metadata_from_path(dataset_path, modality)
    if limit_dataset_size:
        data, metadata_dict = _get_dataset_subset(data, metadata_dict, modality=modality)

    return data, metadata_dict


def get_data_and_metadata_from_path(dataset_path, modality):
    metadata_dict = None
    data = None
    for file_name in dataset_path.iterdir():
        if 'metadata' in file_name.stem and file_name.suffix == '.json':
            metadata_dict = _read_metadata_json(dataset_path / file_name)

        elif 'data' in file_name.stem and file_name.suffix == '.zip':
            data = _read_zipped_data(zip_file_path=(dataset_path / file_name), modality=modality)

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
        return pd.read_csv(csv_file)


def _read_metadata_json(metadata_path):
    with open(metadata_path) as metadata_file:
        metadata_dict = json.load(metadata_file)

    return metadata_dict


def get_available_datasets(modality='single_table'):
    """Get available single_table datasets.

    Args:
        modality (str):
            The modality of the datasets: ``'single_table'`` (Default).

    Return:
        pd.DataFrame:
            Table of available datasets and their sizes.
    """
    possible_modalities = ['single_table']
    if modality not in possible_modalities:
        raise ValueError(f"'modality' must be in {possible_modalities}.")

    return _get_available_datasets(modality)


def _get_available_datasets(
    modality, bucket=None, aws_access_key_id=None, aws_secret_access_key=None
):
    _validate_modality(modality)
    s3 = get_s3_client(aws_access_key_id, aws_secret_access_key)
    bucket = bucket or BUCKET
    bucket_name = _get_bucket_name(bucket)

    response = s3.list_objects(Bucket=bucket_name, Prefix=modality.upper())
    datasets = []
    for content in response['Contents']:
        key = content['Key']
        metadata = s3.head_object(Bucket=bucket_name, Key=key)['ResponseMetadata']['HTTPHeaders']
        size = metadata.get('x-amz-meta-size-mb')
        size = float(size) if size is not None else size
        num_tables = metadata.get('x-amz-meta-num-tables')
        num_tables = int(num_tables) if num_tables is not None else num_tables
        if key.endswith('.zip'):
            datasets.append({
                'dataset_name': key[: -len('.zip')].lstrip(f'{modality.upper()}/'),
                'size_MB': size,
                'num_tables': num_tables,
            })

    return pd.DataFrame(datasets)


def get_dataset_paths(
    modality,
    datasets=None,
    datasets_path=None,
    bucket=None,
    aws_access_key_id=None,
    aws_secret_access_key=None,
):
    """Build the full path to datasets and ensure they exist.

    Args:
        datasets (list):
            List of datasets.
        dataset_path (str):
            The path of the datasets.
        bucket (str):
            The AWS bucket where to get the dataset or folder.
        aws_access_key_id (str):
            The access key id that will be used to communicate with s3, if provided.
        aws_secret_access_key (str):
            The secret access key that will be used to communicate with s3, if provided.

    Returns:
        list:
            List of the full path of the datasets.
    """
    _validate_modality(modality)
    bucket = bucket or BUCKET
    is_remote = bucket.startswith(S3_PREFIX)

    if datasets_path is None:
        datasets_path = DATASETS_PATH / modality
    else:
        datasets_path = Path(datasets_path)

    if datasets is None:
        # local path
        if not is_remote and Path(bucket).exists():
            datasets = []
            folder_items = list(Path(bucket).iterdir())
            for dataset in folder_items:
                if _path_contains_data_and_metadata(dataset) and dataset not in datasets:
                    datasets.append(dataset)
        else:
            datasets = _get_available_datasets(modality, bucket=bucket)
            datasets = datasets['dataset_name'].tolist()

    dataset_paths = []
    for dataset in datasets:
        available_dataset = _get_dataset_path_or_download(
            modality,
            dataset,
            datasets_path,
            bucket,
            aws_access_key_id,
            aws_secret_access_key
        )
        dataset_paths.append(available_dataset)

    return dataset_paths
