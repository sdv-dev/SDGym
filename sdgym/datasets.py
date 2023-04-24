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
    return bucket[len(S3_PREFIX):] if bucket.startswith(S3_PREFIX) else bucket


def _download_dataset(modality, dataset_name, datasets_path=None, bucket=None, aws_key=None,
                      aws_secret=None):
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


def _get_dataset_path(modality, dataset, datasets_path, bucket=None, aws_key=None,
                      aws_secret=None):
    dataset = Path(dataset)
    if dataset.exists():
        return dataset

    datasets_path = datasets_path or DATASETS_PATH
    dataset_path = datasets_path / dataset
    if dataset_path.exists():
        return dataset_path

    if not bucket.startswith(S3_PREFIX):
        local_path = Path(bucket) / dataset if bucket else Path(dataset)
        if local_path.exists():
            return local_path

    _download_dataset(
        modality, dataset, dataset_path, bucket=bucket, aws_key=aws_key, aws_secret=aws_secret)
    return dataset_path


def _apply_max_columns_to_metadata(metadata, max_columns):
    tables = metadata['tables']
    for table in tables.values():
        fields = table['fields']
        if len(fields) > max_columns:
            fields = dict(itertools.islice(fields.items(), max_columns))
            table['fields'] = fields

        structure = table.get('structure')
        if structure:
            structure['structure'] = structure['structure'][:max_columns]
            structure['states'] = structure['states'][:max_columns]


def load_dataset(modality, dataset, datasets_path=None, bucket=None, aws_key=None,
                 aws_secret=None, max_columns=None):
    """Get the data and metadata of a dataset."""
    dataset_path = _get_dataset_path(modality, dataset, datasets_path, bucket, aws_key, aws_secret)
    with open(dataset_path / f'{dataset_path.name}.csv') as data_csv:
        data = pd.read_csv(data_csv)

    metadata_filename = 'metadata.json'
    if not os.path.exists(f'{dataset_path}/{metadata_filename}'):
        metadata_filename = 'metadata_v1.json'

    with open(dataset_path / metadata_filename) as metadata_file:
        metadata_content = json.load(metadata_file)

    if max_columns:
        if 'tables' in metadata_content.keys():
            raise ValueError('max_columns is not supported for multi-table datasets')

        _apply_max_columns_to_metadata(metadata_content, max_columns)

    return data, metadata_content


def _get_available_datasets(modality, bucket=None, aws_key=None, aws_secret=None):
    if modality not in MODALITIES:
        modalities_list = ', '.join(MODALITIES)
        raise ValueError(
            f'Modality `{modality}` not recognized. Must be one of {modalities_list}')

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
                'dataset_name': key[:-len('.zip')].lstrip(f'{modality.upper()}/'),
                'size_MB': size,
                'num_tables': num_tables,
            })

    return pd.DataFrame(datasets)


def get_available_datasets():
    return _get_available_datasets('single_table')


def get_downloaded_datasets(datasets_path=None):
    datasets_path = Path(datasets_path or DATASETS_PATH)
    if not datasets_path.is_dir():
        return pd.DataFrame(columns=['name', 'modality', 'tables', 'size'])

    datasets = []
    for dataset_path in datasets_path.iterdir():
        dataset = load_dataset(dataset_path)
        datasets.append({
            'name': dataset_path.name,
            'modality': dataset._metadata['modality'],
            'tables': len(dataset.get_tables()),
            'size': sum(csv.stat().st_size for csv in dataset_path.glob('*.csv')),
        })

    return pd.DataFrame(datasets)


def get_dataset_paths(datasets, datasets_path, bucket, aws_key, aws_secret):
    """Build the full path to datasets and ensure they exist."""
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
            datasets = _get_available_datasets(
                'single_table', bucket=bucket)['dataset_name'].tolist()

    return [
        _get_dataset_path('single_table', dataset, datasets_path, bucket, aws_key, aws_secret)
        for dataset in datasets
    ]
