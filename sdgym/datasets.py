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


def _get_default_buckets():
    """Get the default dataset buckets"""
    return [SDV_DATASETS_PUBLIC_BUCKET, SDV_DATASETS_PRIVATE_BUCKET]


def _validate_bucket(bucket):
    if bucket is None or isinstance(bucket, str):
        return

    if isinstance(bucket, list) and all(isinstance(item, str) for item in bucket):
        return

    raise TypeError("The 'bucket' parameter must be a string, a list of strings, or None.")


def _validate_buckets(buckets):
    if buckets is None:
        return

    if isinstance(buckets, list) and all(isinstance(bucket, str) for bucket in buckets):
        return

    raise TypeError("The 'buckets' parameter must be a list of strings or None.")


def _parse_bucket(bucket):
    """Return the bucket name and root key prefix for a dataset bucket URL."""
    if bucket.startswith(S3_PREFIX):
        bucket_name, prefix = parse_s3_path(bucket)
        if prefix and not prefix.endswith('/'):
            prefix = f'{prefix}/'

        return bucket_name, prefix

    return bucket, ''


def _get_buckets(bucket):
    _validate_bucket(bucket)
    if bucket is None:
        return _get_default_buckets()

    if isinstance(bucket, list):
        return bucket

    return [bucket]


def _should_resolve_bucket(bucket):
    """Whether dataset names need to be mapped to a bucket before loading."""
    return bucket is None or isinstance(bucket, list)


def _is_s3_dataset_path(dataset):
    """Whether the dataset is an S3 dataset path."""
    return isinstance(dataset, str) and dataset.startswith(S3_PREFIX)


def _get_dataset_display_name(dataset):
    """Get the display name of a dataset."""
    if _is_s3_dataset_path(dataset):
        bucket_name, key = parse_s3_path(dataset)
        return Path(key or bucket_name).name

    return Path(dataset).name


def _make_s3_dataset_path(bucket, modality, dataset_name):
    """Make an S3 dataset path."""
    return f'{bucket.rstrip("/")}/{modality}/{dataset_name}'


def _get_bucket_and_dataset_from_s3_path(dataset, modality):
    """Get the bucket and dataset name from an S3 dataset path."""
    bucket_name, key = parse_s3_path(dataset)
    parts = key.rstrip('/').split('/')
    if len(parts) < 2 or parts[-2] != modality:
        raise ValueError(f"Invalid S3 dataset path for modality '{modality}': '{dataset}'.")

    dataset_name = parts[-1]
    bucket_prefix = '/'.join(parts[:-2])
    bucket = f'{S3_PREFIX}{bucket_name}'
    if bucket_prefix:
        bucket = f'{bucket}/{bucket_prefix}'

    return bucket, dataset_name


def _format_error_dataset_not_found(dataset, modality, bucket):
    """Format a dataset-not-found error message."""
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

    return message


def _raise_dataset_not_found_error(
    dataset_name,
    bucket,
    modality,
):
    display_name = dataset_name
    if isinstance(dataset_name, Path):
        display_name = dataset_name.name

    raise ValueError(_format_error_dataset_not_found(display_name, modality, bucket))


def _download_dataset(
    modality,
    dataset_name,
    datasets_path=None,
    bucket=None,
    s3_client=None,
):
    """Download a dataset into the given ``datasets_path`` / ``modality``."""
    datasets_path = datasets_path or DATASETS_PATH / modality / dataset_name
    s3_client = s3_client or get_s3_client()
    if bucket is None:
        bucket = _get_bucket_for_dataset(
            modality,
            dataset_name,
            _get_default_buckets(),
            s3_client,
            skip_inaccessible=True,
        )

    bucket_name, bucket_prefix = _parse_bucket(bucket)

    LOGGER.info('Downloading dataset %s from %s', dataset_name, bucket)
    prefix = f'{bucket_prefix}{modality.lower()}/{dataset_name}/'
    contents = _list_s3_bucket_contents(s3_client, bucket_name, prefix)
    if not contents:
        _raise_dataset_not_found_error(
            dataset_name,
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


def _get_existing_dataset_path(modality, dataset, datasets_path=None, bucket=None):
    if _is_s3_dataset_path(dataset):
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
    """Map datasets to buckets.

    Args:
        modality (str):
            The dataset modality.
        buckets (list):
            The list of buckets to map datasets to.
        s3_client (boto3.client):
            The S3 client to use to access the buckets.
        skip_inaccessible (bool):
            Whether to skip inaccessible buckets.
    """
    dataset_buckets = {}
    for bucket in buckets:
        try:
            available_datasets = _get_available_datasets(
                modality,
                buckets=[bucket],
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


def _get_dataset_paths_from_buckets(
    modality,
    buckets,
    s3_client,
    skip_inaccessible=False,
):
    dataset_buckets = _get_dataset_bucket_mapping(
        modality,
        buckets,
        s3_client,
        skip_inaccessible=skip_inaccessible,
    )
    return [
        _make_s3_dataset_path(dataset_bucket, modality, dataset)
        for dataset, dataset_bucket in dataset_buckets.items()
    ]


def _get_bucket_for_dataset(
    modality,
    dataset,
    buckets,
    s3_client,
    skip_inaccessible=False,
):
    """Get the bucket containing a dataset."""
    dataset_buckets = _get_dataset_bucket_mapping(
        modality,
        buckets,
        s3_client,
        skip_inaccessible=skip_inaccessible,
    )
    dataset_name = _get_dataset_display_name(dataset)
    bucket = dataset_buckets.get(dataset_name)
    if bucket is None:
        raise ValueError(_format_error_dataset_not_found(dataset_name, modality, buckets))

    return bucket


def _resolve_dataset_name_to_path(
    modality,
    dataset,
    datasets_path,
    buckets,
    s3_client,
    skip_inaccessible=False,
):
    """Resolve one dataset name to a local path or the matching S3 dataset path."""
    existing_path = _get_existing_dataset_path(modality, dataset, datasets_path)
    if existing_path is not None:
        return existing_path

    dataset_name = _get_dataset_display_name(dataset)
    bucket = _get_bucket_for_dataset(
        modality,
        dataset_name,
        buckets,
        s3_client,
        skip_inaccessible=skip_inaccessible,
    )
    return _make_s3_dataset_path(bucket, modality, dataset_name)


def _resolve_dataset_names_to_paths(
    modality,
    datasets,
    datasets_path,
    buckets,
    s3_client,
    skip_inaccessible=False,
):
    existing_datasets = {
        dataset: _get_existing_dataset_path(modality, dataset, datasets_path)
        for dataset in datasets
    }
    remote_datasets = [
        dataset for dataset, dataset_path in existing_datasets.items() if dataset_path is None
    ]
    dataset_buckets = {}
    if remote_datasets:
        dataset_buckets = _get_dataset_bucket_mapping(
            modality,
            buckets,
            s3_client,
            skip_inaccessible=skip_inaccessible,
        )

    missing_datasets = [
        _get_dataset_display_name(dataset)
        for dataset in remote_datasets
        if _get_dataset_display_name(dataset) not in dataset_buckets
    ]
    if missing_datasets:
        raise ValueError(_format_error_dataset_not_found(missing_datasets, modality, buckets))

    return [
        existing_datasets[dataset]
        if existing_datasets[dataset] is not None
        else _make_s3_dataset_path(
            dataset_buckets[_get_dataset_display_name(dataset)],
            modality,
            _get_dataset_display_name(dataset),
        )
        for dataset in datasets
    ]


def _get_dataset_path_and_download(
    modality,
    dataset,
    datasets_path,
    bucket=None,
    s3_client=None,
):
    _validate_bucket(bucket)
    if _is_s3_dataset_path(dataset):
        return dataset

    existing_path = _get_existing_dataset_path(modality, dataset, datasets_path, bucket)
    if existing_path:
        return existing_path

    dataset = Path(dataset)
    datasets_path = datasets_path or DATASETS_PATH / modality
    dataset_path = datasets_path / dataset
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


def _load_data_and_metadata_from_s3(modality, dataset_name, bucket, s3_client):
    """Load dataset data and metadata from S3 without writing files locally."""
    bucket_name, bucket_prefix = _parse_bucket(bucket)
    prefix = f'{bucket_prefix}{modality.lower()}/{dataset_name}/'
    contents = _list_s3_bucket_contents(s3_client, bucket_name, prefix)
    if not contents:
        _raise_dataset_not_found_error(
            dataset_name,
            bucket,
            modality,
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
    buckets=None,
    s3_client=None,
):
    _validate_modality(modality)
    _validate_buckets(buckets)
    s3_client = s3_client or get_s3_client()
    buckets = buckets or _get_default_buckets()
    available_datasets = []
    for bucket in buckets:
        bucket_name, bucket_prefix = _parse_bucket(bucket)
        contents = _list_s3_bucket_contents(
            s3_client,
            bucket_name,
            f'{bucket_prefix}{modality}/',
        )
        datasets_info = _genereate_dataset_info(s3_client, bucket_name, contents)
        available_datasets.append(pd.DataFrame(datasets_info))

    return pd.concat(available_datasets, ignore_index=True)


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
    if _should_resolve_bucket(bucket) or _is_s3_dataset_path(dataset):
        s3_client = s3_client or get_s3_client()
        if _is_s3_dataset_path(dataset):
            bucket, dataset_name = _get_bucket_and_dataset_from_s3_path(dataset, modality)
            data, metadata_dict = _load_data_and_metadata_from_s3(
                modality,
                dataset_name,
                bucket,
                s3_client,
            )
        else:
            dataset_path = _resolve_dataset_name_to_path(
                modality,
                dataset,
                datasets_path or DATASETS_PATH / modality,
                _get_buckets(bucket),
                s3_client,
                skip_inaccessible=bucket is None,
            )
            if _is_s3_dataset_path(dataset_path):
                bucket, dataset_name = _get_bucket_and_dataset_from_s3_path(dataset_path, modality)
                data, metadata_dict = _load_data_and_metadata_from_s3(
                    modality,
                    dataset_name,
                    bucket,
                    s3_client,
                )
            else:
                data, metadata_dict = get_data_and_metadata_from_path(dataset_path, modality)

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
    _validate_bucket(bucket)
    should_resolve_bucket = _should_resolve_bucket(bucket)
    buckets = _get_buckets(bucket) if should_resolve_bucket else None
    single_bucket = bucket if isinstance(bucket, str) else None

    if datasets_path is None:
        datasets_path = DATASETS_PATH / modality
    else:
        datasets_path = Path(datasets_path)

    if datasets is None:
        if (
            single_bucket
            and not single_bucket.startswith(S3_PREFIX)
            and Path(single_bucket).exists()
        ):
            datasets = []
            folder_items = list(Path(single_bucket).iterdir())
            for dataset in folder_items:
                if _path_contains_data_and_metadata(dataset) and dataset not in datasets:
                    datasets.append(dataset)
        elif should_resolve_bucket:
            s3_client = s3_client or get_s3_client()
            datasets = _get_dataset_paths_from_buckets(
                modality, buckets, s3_client, skip_inaccessible=bucket is None
            )
        else:
            datasets = _get_available_datasets(
                modality,
                buckets=[single_bucket],
                s3_client=s3_client,
            )
            datasets = datasets['dataset_name'].tolist()
    elif should_resolve_bucket:
        s3_client = s3_client or get_s3_client()
        datasets = _resolve_dataset_names_to_paths(
            modality,
            datasets,
            datasets_path,
            buckets,
            s3_client,
            skip_inaccessible=bucket is None,
        )

    dataset_paths = []
    for dataset in datasets:
        if _is_s3_dataset_path(dataset):
            dataset_paths.append(dataset)
        else:
            dataset_paths.append(
                _get_dataset_path_and_download(
                    modality,
                    dataset,
                    datasets_path,
                    bucket=single_bucket,
                    s3_client=s3_client,
                )
            )

    return dataset_paths
