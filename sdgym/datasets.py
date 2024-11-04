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
BUCKET = 's3://sdv-demo-datasets'
BUCKET_URL = 'https://{}.s3.amazonaws.com/'
TIMESERIES_FIELDS = ['sequence_index', 'entity_columns', 'context_columns', 'deepecho_version']
MODALITIES = ['single_table', 'multi_table', 'sequential']
S3_PREFIX = 's3://'


def _get_bucket_name(bucket):
    return bucket[len(S3_PREFIX) :] if bucket.startswith(S3_PREFIX) else bucket


def _download_dataset(
    modality, dataset_name, datasets_path=None, bucket=None, aws_key=None, aws_secret=None
):
    """Download a dataset and extract it into the given ``datasets_path``."""
    datasets_path = datasets_path or DATASETS_PATH / dataset_name
    bucket = bucket or BUCKET
    bucket_name = _get_bucket_name(bucket)

    LOGGER.info('Downloading dataset %s from %s', dataset_name, bucket)
    s3 = get_s3_client(aws_key, aws_secret)
    obj = s3.get_object(Bucket=bucket_name, Key=f'{modality.upper()}/{dataset_name}.zip')
    bytes_io = io.BytesIO(obj['Body'].read())

    LOGGER.info('Extracting dataset into %s', datasets_path)
    with ZipFile(bytes_io) as zf:
        zf.extractall(datasets_path)


def _get_dataset_path(modality, dataset, datasets_path, bucket=None, aws_key=None, aws_secret=None):
    dataset = Path(dataset)
    if dataset.exists():
        return dataset

    datasets_path = datasets_path or DATASETS_PATH
    dataset_path = datasets_path / dataset
    if dataset_path.exists():
        return dataset_path

    bucket = bucket or BUCKET
    if not bucket.startswith(S3_PREFIX):
        local_path = Path(bucket) / dataset if bucket else Path(dataset)
        if local_path.exists():
            return local_path

    _download_dataset(
        modality, dataset, dataset_path, bucket=bucket, aws_key=aws_key, aws_secret=aws_secret
    )
    return dataset_path


def _get_dataset_subset(data, metadata_dict):
    if 'tables' in metadata_dict.keys():
        raise ValueError('limit_dataset_size is not supported for multi-table datasets.')

    max_rows, max_columns = (1000, 10)
    columns = metadata_dict['columns']
    if len(columns) > max_columns:
        columns = dict(itertools.islice(columns.items(), max_columns))
        metadata_dict['columns'] = columns
        data = data[columns.keys()]

    data = data.head(max_rows)

    return data, metadata_dict


def load_dataset(
    modality,
    dataset,
    datasets_path=None,
    bucket=None,
    aws_key=None,
    aws_secret=None,
    limit_dataset_size=None,
):
    """Get the data and metadata of a dataset.

    Args:
        modality (str):
            It must be ``'single-table'``, ``'multi-table'`` or ``'time-series'``.
        dataset (str):
            The path of the dataset as a string.
        dataset_path (PurePath):
            The path of the dataset as an object. This will only be used if the given ``dataset``
            doesn't exist.
        bucket (str):
            The AWS bucket where to get the dataset. This will only be used if both ``dataset``
            and ``dataset_path`` don't exist.
        aws_key (str):
            The access key id that will be used to communicate with s3, if provided.
        aws_secret (str):
            The secret access key that will be used to communicate with s3, if provided.
        limit_dataset_size (bool):
            Use this flag to limit the size of the datasets for faster evaluation. If ``True``,
            limit the size of every table to 1,000 rows (randomly sampled) and the first 10
            columns.

    Returns:
        pd.DataFrame, dict:
            The data and medatata of a dataset.
    """
    dataset_path = _get_dataset_path(modality, dataset, datasets_path, bucket, aws_key, aws_secret)
    with open(dataset_path / f'{dataset_path.name}.csv') as data_csv:
        data = pd.read_csv(data_csv)

    metadata_filename = 'metadata.json'
    if not os.path.exists(f'{dataset_path}/{metadata_filename}'):
        metadata_filename = 'metadata_v1.json'

    with open(dataset_path / metadata_filename) as metadata_file:
        metadata_dict = json.load(metadata_file)

    if limit_dataset_size:
        data, metadata_dict = _get_dataset_subset(data, metadata_dict)

    return data, metadata_dict


def get_available_datasets():
    """Get available single_table datasets.

    Return:
        pd.DataFrame:
            Table of available datasets and their sizes.
    """
    return _get_available_datasets('single_table')


def _get_available_datasets(modality, bucket=None, aws_key=None, aws_secret=None):
    if modality not in MODALITIES:
        modalities_list = ', '.join(MODALITIES)
        raise ValueError(f'Modality `{modality}` not recognized. Must be one of {modalities_list}')

    s3 = get_s3_client(aws_key, aws_secret)
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
    datasets=None, datasets_path=None, bucket=None, aws_key=None, aws_secret=None
):
    """Build the full path to datasets and ensure they exist.

    Args:
        datasets (list):
            List of datasets.
        dataset_path (str):
            The path of the datasets.
        bucket (str):
            The AWS bucket where to get the dataset.
        aws_key (str):
            The access key id that will be used to communicate with s3, if provided.
        aws_secret (str):
            The secret access key that will be used to communicate with s3, if provided.

    Returns:
        list:
            List of the full path of the datasets.
    """
    bucket = bucket or BUCKET
    is_remote = bucket.startswith(S3_PREFIX)

    if datasets_path is None:
        datasets_path = DATASETS_PATH

    datasets_path = Path(datasets_path)
    if datasets is None:
        # local path
        if not is_remote and Path(bucket).exists():
            datasets = []
            folder_items = list(Path(bucket).iterdir())
            for dataset in folder_items:
                if not dataset.name.startswith('.'):
                    if dataset.name.endswith('zip'):
                        dataset_name = os.path.splitext(dataset.name)[0]
                        dataset_path = datasets_path / dataset_name
                        ZipFile(dataset).extractall(dataset_path)

                        datasets.append(dataset_path)
                    elif dataset not in datasets:
                        datasets.append(dataset)
        else:
            datasets = _get_available_datasets('single_table', bucket=bucket)[
                'dataset_name'
            ].tolist()

    return [
        _get_dataset_path('single_table', dataset, datasets_path, bucket, aws_key, aws_secret)
        for dataset in datasets
    ]
