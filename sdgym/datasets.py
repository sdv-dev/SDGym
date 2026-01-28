"""SDGym module to handle datasets."""

import logging
import os
from pathlib import Path

import appdirs
import numpy as np
import pandas as pd

from sdgym._dataset_utils import (
    _get_dataset_subset,
    _parse_numeric_value,
    _read_metadata_json,
    _read_zipped_data,
)
from sdgym.s3 import (
    _list_s3_bucket_contents,
    _load_yaml_metainfo_from_s3,
    get_s3_client,
)

LOGGER = logging.getLogger(__name__)

DATASETS_PATH = Path(appdirs.user_data_dir()) / 'SDGym' / 'datasets'
SDV_DATASETS_PUBLIC_BUCKET = 's3://sdv-datasets-public'
SDV_DATASETS_PRIVATE_BUCKET = 's3://sdv-datasets-private'
BUCKET_URL = 'https://{}.s3.amazonaws.com/'
TIMESERIES_FIELDS = ['sequence_index', 'entity_columns', 'context_columns', 'deepecho_version']
MODALITIES = ['single_table', 'multi_table', 'sequential']
S3_PREFIX = 's3://'


def _get_bucket_name(bucket):
    return bucket[len(S3_PREFIX) :] if bucket.startswith(S3_PREFIX) else bucket


def _raise_dataset_not_found_error(
    s3_client,
    bucket_name,
    dataset_name,
    current_modality,
    bucket,
    modality,
):
    display_name = dataset_name
    if isinstance(dataset_name, Path):
        display_name = dataset_name.name

    available_modalities = []
    for other_modality in MODALITIES:
        if other_modality == current_modality:
            continue

        other_prefix = f'{other_modality.lower()}/{dataset_name}/'
        other_contents = _list_s3_bucket_contents(s3_client, bucket_name, other_prefix)
        if other_contents:
            available_modalities.append(other_modality)

    if available_modalities:
        available_list = ', '.join(sorted(available_modalities))
        raise ValueError(
            f"Dataset '{display_name}' not found in bucket '{bucket}' "
            f"for modality '{modality}'. It is available under modality: '{available_list}'."
        )
    else:
        raise ValueError(
            f"Dataset '{display_name}' not found in bucket '{bucket}' for modality '{modality}'."
        )


def _download_dataset(
    modality,
    dataset_name,
    datasets_path=None,
    bucket=None,
    s3_client=None,
):
    """Download a dataset into the given ``datasets_path`` / ``modality``."""
    datasets_path = datasets_path or DATASETS_PATH / modality / dataset_name
    bucket = bucket or SDV_DATASETS_PUBLIC_BUCKET
    bucket_name = _get_bucket_name(bucket)

    LOGGER.info('Downloading dataset %s from %s', dataset_name, bucket)
    s3_client = s3_client or get_s3_client()
    prefix = f'{modality.lower()}/{dataset_name}/'
    contents = _list_s3_bucket_contents(s3_client, bucket_name, prefix)
    if not contents:
        _raise_dataset_not_found_error(
            s3_client,
            bucket_name,
            dataset_name,
            modality,
            bucket,
            modality,
        )

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


def _get_dataset_path_and_download(
    modality,
    dataset,
    datasets_path,
    bucket=None,
    s3_client=None,
):
    dataset = Path(dataset)
    if dataset.exists() and _path_contains_data_and_metadata(dataset):
        return dataset

    datasets_path = datasets_path or DATASETS_PATH / modality
    dataset_path = datasets_path / dataset
    if dataset_path.exists() and _path_contains_data_and_metadata(dataset_path):
        return dataset_path

    bucket = bucket or SDV_DATASETS_PUBLIC_BUCKET
    if not bucket.startswith(S3_PREFIX):
        local_path = Path(bucket) / modality / dataset
        if local_path.exists() and _path_contains_data_and_metadata(local_path):
            return local_path

    s3_client = s3_client or get_s3_client()

    return _download_dataset(
        modality,
        dataset,
        dataset_path,
        bucket,
        s3_client=s3_client,
    )


def _validate_modality(modality):
    if modality not in MODALITIES:
        modalities_list = ', '.join(MODALITIES)
        raise ValueError(f'Modality `{modality}` not recognized. Must be one of {modalities_list}')


def get_data_and_metadata_from_path(dataset_path, modality):
    """Load dataset data and metadata from a given local path.

    Args:
        dataset_path (Path):
            The local path to the dataset directory containing data and metadata files.
        modality (str):
            The dataset modality. Used by the data-reading function to correctly
            interpret and load the dataset (e.g., ``'single_table'``, ``'multi_table'``).

    Returns:
        tuple[pd.DataFrame | dict, dict | None]:
            A tuple containing:
                - The loaded dataset as a ``pandas.DataFrame`` or a dictionary with
                  ``pandas.DataFrame`` if ``multi_table``.
            - The dataset metadata as a dictionary.
    """
    metadata_dict = None
    data = None
    for file_name in dataset_path.iterdir():
        if 'metadata' in file_name.stem and file_name.suffix == '.json':
            metadata_dict = _read_metadata_json(file_name)
        elif 'data' in file_name.stem and file_name.suffix == '.zip':
            data = _read_zipped_data(zip_file_path=(file_name), modality=modality)

        if data is not None and metadata_dict is not None:
            break

    return data, metadata_dict


def _genereate_dataset_info(s3_client, bucket_name, contents):
    """Generate summarized dataset information from S3 bucket.

    Args:
        s3_client (boto3.client):
            An initialized boto3 S3 client used to access S3 objects.
        bucket_name (str):
            The name of the S3 bucket containing the datasets.
        contents (list[dict]):
            A list of S3 object metadata dictionaries (e.g., from
            ``s3_client.list_objects_v2()['Contents']``). Each dictionary should
            include a ``'Key'`` field representing the S3 object path.

    Returns:
        dict:
            A dictionary containing dataset summary information with the
            following keys:
                - ``'dataset_name'`` (list[str]): Names of the datasets discovered.
                - ``'size_MB'`` (list[float]): Corresponding dataset sizes in megabytes.
                - ``'num_tables'`` (list[int]): Number of tables in each dataset.
    """
    tables_info = {'dataset_name': [], 'size_MB': [], 'num_tables': []}
    for obj in contents:
        key = obj.get('Key', '')
        if key.endswith('metainfo.yaml'):
            parts = key.split('/')
            if len(parts) >= 3:
                dataset_name = parts[-2]
                yaml_key = key
                info = _load_yaml_metainfo_from_s3(s3_client, bucket_name, yaml_key)
                size_mb = _parse_numeric_value(
                    info.get('dataset-size-mb', np.nan),
                    dataset_name,
                    field_name='dataset-size-mb',
                    target_type=float,
                )
                num_tables = _parse_numeric_value(
                    info.get('num-tables', np.nan), dataset_name, 'num-tables', target_type=int
                )
                tables_info['dataset_name'].append(dataset_name)
                tables_info['size_MB'].append(size_mb)
                tables_info['num_tables'].append(num_tables)

    return tables_info


def _get_available_datasets(
    modality,
    bucket=None,
    s3_client=None,
):
    _validate_modality(modality)
    s3_client = s3_client or get_s3_client()
    bucket = bucket or SDV_DATASETS_PUBLIC_BUCKET
    bucket_name = _get_bucket_name(bucket)
    contents = _list_s3_bucket_contents(
        s3_client,
        bucket_name,
        f'{modality}/',
    )
    datasets_info = _genereate_dataset_info(s3_client, bucket_name, contents)
    return pd.DataFrame(datasets_info)


def load_dataset(
    modality,
    dataset,
    datasets_path=None,
    bucket=None,
    aws_access_key_id=None,
    aws_secret_access_key=None,
    limit_dataset_size=False,
):
    """Get the data and metadata of a dataset.

    Args:
        modality (str):
            It must be ``'single_table'``, ``'multi_table'`` or ``'sequential'``.
        dataset (str):
            The path of the dataset as a string or the name of the dataset.
        dataset_path (PurePath):
            The path of the dataset as an object. This will only be used if the given ``dataset``
            doesn't exist.
        bucket (str):
            The AWS bucket where to get the dataset. This will only be used if both ``dataset``
            and ``dataset_path`` don't exist.
        aws_access_key_id (str or None):
            The access key id that will be used to communicate with s3, if provided.
            Defaults to ``None``.
        aws_secret_access_key (str or None):
            The secret access key that will be used to communicate with s3, if provided.
            Defaults to ``None``.
        limit_dataset_size (bool):
            Use this flag to limit the size of the datasets for faster evaluation. If ``True``,
            limit the size of every table to 1,000 rows (randomly sampled) and the first 10
            columns. Defauts to ``False``.

    Returns:
        pd.DataFrame | dict, dict:
            The data and medatata for a dataset.
    """
    s3_client = get_s3_client(
        aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key
    )
    return _load_dataset_with_client(
        modality=modality,
        dataset=dataset,
        datasets_path=datasets_path,
        bucket=bucket,
        s3_client=s3_client,
        limit_dataset_size=limit_dataset_size,
    )


def _load_dataset_with_client(
    modality,
    dataset,
    datasets_path=None,
    bucket=None,
    s3_client=None,
    limit_dataset_size=False,
):
    """Get the data and metadata of a dataset using a given s3 client."""
    _validate_modality(modality)
    dataset_path = _get_dataset_path_and_download(
        modality, dataset, datasets_path, bucket, s3_client=s3_client
    )

    data, metadata_dict = get_data_and_metadata_from_path(dataset_path, modality)
    if limit_dataset_size:
        data, metadata_dict = _get_dataset_subset(data, metadata_dict, modality=modality)

    return data, metadata_dict


def get_dataset_paths(
    modality,
    datasets=None,
    datasets_path=None,
    bucket=None,
    s3_client=None,
):
    """Build the full path to datasets and ensure they exist.

    Args:
        datasets (list):
            List of datasets.
        dataset_path (str):
            The path of the datasets.
        bucket (str):
            The AWS bucket where to get the dataset or folder.
        s3_client (boto3.client):
            The s3 client that will be used to communicate with s3, if provided.
            Defaults to ``None``.

    Returns:
        list:
            List of the full path of the datasets.
    """
    _validate_modality(modality)
    bucket = bucket or SDV_DATASETS_PUBLIC_BUCKET
    is_remote = bucket.startswith(S3_PREFIX)

    if datasets_path is None:
        datasets_path = DATASETS_PATH / modality
    else:
        datasets_path = Path(datasets_path)

    if datasets is None:
        if not is_remote and Path(bucket).exists():
            datasets = []
            folder_items = list(Path(bucket).iterdir())
            for dataset in folder_items:
                if _path_contains_data_and_metadata(dataset) and dataset not in datasets:
                    datasets.append(dataset)
        else:
            datasets = _get_available_datasets(
                modality,
                bucket=bucket,
                s3_client=s3_client,
            )
            datasets = datasets['dataset_name'].tolist()

    dataset_paths = []
    for dataset in datasets:
        dataset_paths.append(
            _get_dataset_path_and_download(
                modality,
                dataset,
                datasets_path,
                bucket=bucket,
                s3_client=s3_client,
            )
        )

    return dataset_paths
