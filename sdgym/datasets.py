"""SDGym module to handle datasets."""

import io
import json
import logging
import os
from pathlib import Path

import appdirs
import botocore
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
    parse_s3_path,
)

LOGGER = logging.getLogger(__name__)

DATASETS_PATH = Path(appdirs.user_data_dir()) / 'SDGym' / 'datasets'
SDV_DATASETS_PUBLIC_BUCKET = 's3://sdv-datasets-public'
SDV_DATASETS_PRIVATE_BUCKET = 's3://sdv-datasets-private'
BUCKET_URL = 'https://{}.s3.amazonaws.com/'
TIMESERIES_FIELDS = ['sequence_index', 'entity_columns', 'context_columns', 'deepecho_version']
MODALITIES = ['single_table', 'multi_table', 'sequential']
S3_PREFIX = 's3://'


class _S3DatasetReference:
    """Reference to a dataset that should be loaded from a specific S3 bucket."""

    def __init__(self, name, bucket):
        self.name = str(name)
        self.bucket = bucket


def _is_bucket_list(bucket):
    return isinstance(bucket, list)


def _validate_bucket(bucket):
    if bucket is None or isinstance(bucket, str):
        return

    if _is_bucket_list(bucket) and all(isinstance(item, str) for item in bucket):
        return

    raise TypeError("The 'bucket' parameter must be a string, a list of strings, or None.")


def _get_bucket_name(bucket):
    return bucket[len(S3_PREFIX) :] if bucket.startswith(S3_PREFIX) else bucket


def _get_bucket_name_and_prefix(bucket):
    if bucket.startswith(S3_PREFIX):
        path_without_prefix = bucket[len(S3_PREFIX) :].rstrip('/')
        if '/' not in path_without_prefix:
            return _get_bucket_name(bucket), ''

        bucket_name, prefix = parse_s3_path(bucket)
        if prefix and not prefix.endswith('/'):
            prefix = f'{prefix}/'

        return bucket_name, prefix

    return bucket, ''


def _get_buckets(bucket):
    _validate_bucket(bucket)
    if bucket is None:
        return [SDV_DATASETS_PUBLIC_BUCKET, SDV_DATASETS_PRIVATE_BUCKET]

    if _is_bucket_list(bucket):
        return bucket

    return [bucket]


def _is_multi_bucket(bucket):
    return bucket is None or _is_bucket_list(bucket)


def _get_dataset_display_name(dataset):
    if isinstance(dataset, _S3DatasetReference):
        return dataset.name

    return Path(dataset).name


def _format_error_dataset_not_found(dataset, modality, bucket, available_modalities=None):
    """Format a consistent dataset-not-found error message."""
    if isinstance(dataset, list):
        dataset_to_print = "', '".join(_get_dataset_display_name(item) for item in dataset)
        dataset_label = 'Dataset(s)'
    else:
        dataset_to_print = _get_dataset_display_name(dataset)
        dataset_label = 'Dataset'

    if isinstance(bucket, list):
        bucket_to_print = "', '".join(bucket)
        bucket_label = 'buckets'
    else:
        bucket_to_print = bucket
        bucket_label = 'bucket'

    message = (
        f"{dataset_label} '{dataset_to_print}' not found in {bucket_label} "
        f"'{bucket_to_print}' for modality '{modality}'."
    )
    if available_modalities:
        available_list = ', '.join(sorted(available_modalities))
        message = f"{message} It is available under modality: '{available_list}'."

    return message


def _raise_dataset_not_found_error(
    s3_client,
    bucket_name,
    dataset_name,
    current_modality,
    bucket,
    modality,
    bucket_prefix='',
):
    display_name = dataset_name
    if isinstance(dataset_name, Path):
        display_name = dataset_name.name

    available_modalities = []
    for other_modality in MODALITIES:
        if other_modality == current_modality:
            continue

        other_prefix = f'{bucket_prefix}{other_modality.lower()}/{dataset_name}/'
        other_contents = _list_s3_bucket_contents(s3_client, bucket_name, other_prefix)
        if other_contents:
            available_modalities.append(other_modality)

    raise ValueError(
        _format_error_dataset_not_found(display_name, modality, bucket, available_modalities)
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
    bucket_name, bucket_prefix = _get_bucket_name_and_prefix(bucket)

    LOGGER.info('Downloading dataset %s from %s', dataset_name, bucket)
    s3_client = s3_client or get_s3_client()
    prefix = f'{bucket_prefix}{modality.lower()}/{dataset_name}/'
    contents = _list_s3_bucket_contents(s3_client, bucket_name, prefix)
    if not contents:
        _raise_dataset_not_found_error(
            s3_client,
            bucket_name,
            dataset_name,
            modality,
            bucket,
            modality,
            bucket_prefix,
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


def _get_existing_dataset_path(modality, dataset, datasets_path=None, bucket=None):
    if isinstance(dataset, _S3DatasetReference):
        return None

    dataset = Path(dataset)
    if dataset.exists() and _path_contains_data_and_metadata(dataset):
        return dataset

    datasets_path = datasets_path or DATASETS_PATH / modality
    dataset_path = datasets_path / dataset
    if dataset_path.exists() and _path_contains_data_and_metadata(dataset_path):
        return dataset_path

    if bucket and isinstance(bucket, str) and not bucket.startswith(S3_PREFIX):
        local_path = Path(bucket) / modality / dataset
        if local_path.exists() and _path_contains_data_and_metadata(local_path):
            return local_path


def _get_dataset_bucket_mapping(modality, buckets, s3_client, skip_inaccessible=False):
    dataset_buckets = {}
    for bucket in buckets:
        try:
            available_datasets = _get_available_datasets(
                modality,
                bucket=bucket,
                s3_client=s3_client,
            )
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as error:
            if skip_inaccessible:
                LOGGER.info("Skipping inaccessible bucket '%s': %s", bucket, error)
                continue

            raise ValueError(
                f"Bucket '{bucket}' is not accessible with the provided credentials."
            ) from error

        for dataset_name in available_datasets['dataset_name'].tolist():
            if bucket == SDV_DATASETS_PRIVATE_BUCKET:
                dataset_buckets[dataset_name] = bucket
            else:
                dataset_buckets.setdefault(dataset_name, bucket)

    return dataset_buckets


def _get_bucket_for_dataset(modality, dataset, bucket, s3_client, skip_inaccessible=False):
    dataset_name = _get_dataset_display_name(dataset)
    buckets = _get_buckets(bucket)
    dataset_buckets = _get_dataset_bucket_mapping(
        modality,
        buckets,
        s3_client,
        skip_inaccessible=skip_inaccessible,
    )
    if dataset_name in dataset_buckets:
        return dataset_buckets[dataset_name]

    raise ValueError(_format_error_dataset_not_found(dataset_name, modality, buckets))


def _get_dataset_path_and_download(
    modality,
    dataset,
    datasets_path,
    bucket=None,
    s3_client=None,
):
    _validate_bucket(bucket)
    if isinstance(dataset, _S3DatasetReference):
        bucket = dataset.bucket
        dataset = dataset.name

    existing_path = _get_existing_dataset_path(modality, dataset, datasets_path, bucket)
    if existing_path:
        return existing_path

    dataset = Path(dataset)
    datasets_path = datasets_path or DATASETS_PATH / modality
    dataset_path = datasets_path / dataset
    s3_client = s3_client or get_s3_client()
    if _is_multi_bucket(bucket):
        bucket = _get_bucket_for_dataset(
            modality,
            dataset,
            bucket,
            s3_client,
            skip_inaccessible=bucket is None,
        )
        return _S3DatasetReference(dataset.name, bucket)
    else:
        bucket = bucket or SDV_DATASETS_PUBLIC_BUCKET

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


def _load_data_and_metadata_from_s3(modality, dataset_name, bucket, s3_client):
    """Load dataset data and metadata from S3 without writing files locally."""
    bucket_name, bucket_prefix = _get_bucket_name_and_prefix(bucket)
    prefix = f'{bucket_prefix}{modality.lower()}/{dataset_name}/'
    contents = _list_s3_bucket_contents(s3_client, bucket_name, prefix)
    if not contents:
        _raise_dataset_not_found_error(
            s3_client,
            bucket_name,
            dataset_name,
            modality,
            bucket,
            modality,
            bucket_prefix,
        )

    metadata_key = None
    data_key = None
    for obj in contents:
        s3_path = obj['Key']
        path = Path(s3_path)
        if 'metadata' in path.stem and path.suffix == '.json':
            metadata_key = s3_path
        elif 'data' in path.stem and path.suffix == '.zip':
            data_key = s3_path

        if metadata_key and data_key:
            break

    if metadata_key is None or data_key is None:
        raise ValueError(
            f"Dataset '{dataset_name}' in bucket '{bucket}' does not contain both "
            'metadata JSON and data ZIP files.'
        )

    metadata_response = s3_client.get_object(Bucket=bucket_name, Key=metadata_key)
    metadata_dict = json.loads(metadata_response['Body'].read().decode('utf-8'))

    data_response = s3_client.get_object(Bucket=bucket_name, Key=data_key)
    data = _read_zipped_data(
        zip_file_path=io.BytesIO(data_response['Body'].read()),
        modality=modality,
    )

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
    if _is_bucket_list(bucket):
        raise TypeError("The 'bucket' parameter must be a string or None.")

    _validate_bucket(bucket)
    s3_client = s3_client or get_s3_client()
    bucket = bucket or SDV_DATASETS_PUBLIC_BUCKET
    bucket_name, bucket_prefix = _get_bucket_name_and_prefix(bucket)
    contents = _list_s3_bucket_contents(
        s3_client,
        bucket_name,
        f'{bucket_prefix}{modality}/',
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
    _validate_bucket(bucket)
    if _is_multi_bucket(bucket) or isinstance(dataset, _S3DatasetReference):
        existing_path = _get_existing_dataset_path(modality, dataset, datasets_path)
        if existing_path:
            data, metadata_dict = get_data_and_metadata_from_path(existing_path, modality)
        else:
            s3_client = s3_client or get_s3_client()
            dataset_name = _get_dataset_display_name(dataset)
            bucket = (
                dataset.bucket
                if isinstance(dataset, _S3DatasetReference)
                else _get_bucket_for_dataset(
                    modality,
                    dataset,
                    bucket,
                    s3_client,
                    skip_inaccessible=bucket is None,
                )
            )
            data, metadata_dict = _load_data_and_metadata_from_s3(
                modality,
                dataset_name,
                bucket,
                s3_client,
            )

        if limit_dataset_size:
            data, metadata_dict = _get_dataset_subset(data, metadata_dict, modality=modality)

        return data, metadata_dict

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
    original_bucket = bucket
    buckets = _get_buckets(bucket)
    is_multi_bucket = _is_multi_bucket(bucket)
    bucket = buckets[0]
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
        elif is_multi_bucket:
            s3_client = s3_client or get_s3_client()
            dataset_buckets = _get_dataset_bucket_mapping(
                modality,
                buckets,
                s3_client,
                skip_inaccessible=original_bucket is None,
            )
            datasets = [
                _S3DatasetReference(dataset, dataset_bucket)
                for dataset, dataset_bucket in dataset_buckets.items()
            ]
        else:
            datasets = _get_available_datasets(
                modality,
                bucket=bucket,
                s3_client=s3_client,
            )
            datasets = datasets['dataset_name'].tolist()
    elif is_multi_bucket:
        existing_datasets = {
            dataset: _get_existing_dataset_path(modality, dataset, datasets_path)
            for dataset in datasets
        }
        remote_datasets = [
            dataset for dataset, dataset_path in existing_datasets.items() if dataset_path is None
        ]
        dataset_buckets = {}
        if remote_datasets:
            s3_client = s3_client or get_s3_client()
            dataset_buckets = _get_dataset_bucket_mapping(
                modality,
                buckets,
                s3_client,
                skip_inaccessible=original_bucket is None,
            )

        missing_datasets = [
            _get_dataset_display_name(dataset)
            for dataset in remote_datasets
            if _get_dataset_display_name(dataset) not in dataset_buckets
        ]
        if missing_datasets:
            raise ValueError(_format_error_dataset_not_found(missing_datasets, modality, buckets))

        datasets = [
            existing_datasets[dataset]
            if existing_datasets[dataset] is not None
            else _S3DatasetReference(dataset, dataset_buckets[_get_dataset_display_name(dataset)])
            for dataset in datasets
        ]

    dataset_paths = []
    for dataset in datasets:
        if isinstance(dataset, _S3DatasetReference):
            dataset_paths.append(dataset)
        else:
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
