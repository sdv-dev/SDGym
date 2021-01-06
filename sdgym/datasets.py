import io
import logging
import urllib.request
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import appdirs
import pandas as pd
from sdv import Metadata

LOGGER = logging.getLogger(__name__)

DATASETS_PATH = Path(appdirs.user_data_dir()) / 'SDGym' / 'datasets'
BUCKET = 'sdv-datasets'
BUCKET_URL = 'https://{}.s3.amazonaws.com/'
TIMESERIES_FIELDS = ['sequence_index', 'entity_columns', 'context_columns', 'deepecho_version']


def download_dataset(dataset_name, datasets_path=None, bucket=None):
    datasets_path = datasets_path or DATASETS_PATH
    bucket = bucket or BUCKET
    url = BUCKET_URL.format(bucket) + f'{dataset_name}.zip'

    LOGGER.info('Downloading dataset %s from %s', dataset_name, url)
    response = urllib.request.urlopen(url)
    bytes_io = io.BytesIO(response.read())

    LOGGER.info('Extracting dataset into %s', datasets_path)
    with ZipFile(bytes_io) as zf:
        zf.extractall(datasets_path)


def _get_dataset_path(dataset, datasets_path, bucket=None):
    dataset = Path(dataset)
    if dataset.exists():
        return dataset

    datasets_path = datasets_path or DATASETS_PATH
    dataset_path = datasets_path / dataset
    if dataset_path.exists():
        return dataset_path

    download_dataset(dataset, datasets_path, bucket=bucket)
    return dataset_path


def load_dataset(dataset, datasets_path=None, bucket=None):
    dataset_path = _get_dataset_path(dataset, datasets_path, bucket)
    metadata = Metadata(str(dataset_path / 'metadata.json'))
    tables = metadata.get_tables()
    if not hasattr(metadata, 'modality'):
        if len(tables) > 1:
            modality = 'multi-table'
        else:
            table = metadata.get_table_meta(tables[0])
            if any(table.get(field) for field in TIMESERIES_FIELDS):
                modality = 'timeseries'
            else:
                modality = 'single-table'

        metadata._metadata['modality'] = modality
        metadata.modality = modality

    if not hasattr(metadata, 'name'):
        metadata._metadata['name'] = dataset_path.name
        metadata.name = dataset_path.name

    return metadata


def load_tables(metadata):
    real_data = metadata.load_tables()
    for table_name, table in real_data.items():
        fields = metadata.get_fields(table_name)
        columns = [
            column
            for column in table.columns
            if column in fields
        ]
        real_data[table_name] = table[columns]

    return real_data


def get_available_datasets(bucket=None):
    bucket_url = BUCKET_URL.format(bucket or BUCKET)
    response = urllib.request.urlopen(bucket_url)
    tree = ElementTree.fromstring(response.read())
    datasets = []
    for content in tree.findall('{*}Contents'):
        key = content.find('{*}Key').text
        size = int(content.find('{*}Size').text)
        if key.endswith('.zip'):
            datasets.append({
                'name': key[:-len('.zip')],
                'size': size
            })

    return pd.DataFrame(datasets)


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


def get_dataset_paths(datasets, datasets_path, bucket):
    """Build the full path to datasets and ensure they exist."""
    if datasets_path is None:
        datasets_path = DATASETS_PATH

    datasets_path = Path(datasets_path)
    if datasets is None:
        if datasets_path.exists():
            datasets = list(datasets_path.iterdir())

        if not datasets:
            datasets = get_available_datasets()['name'].tolist()

    return [
        _get_dataset_path(dataset, datasets_path, bucket)
        for dataset in datasets
    ]
