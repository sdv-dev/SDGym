"""Utils for the CLI module."""

import io
import pathlib
from pathlib import Path

import pandas as pd
import tqdm

from sdgym.datasets import DATASETS_PATH, load_dataset
from sdgym.s3 import get_s3_client, is_s3_path, parse_s3_path


def read_file(path, aws_key, aws_secret):
    """Read file from path.

    The path can either be a local path or an s3 directory.

    Args:
        path (str):
            The path to the file.
        aws_key (str):
            The access key id that will be used to communicate with s3, if provided.
        aws_secret (str):
            The secret access key that will be used to communicate with s3, if provided.

    Returns:
        bytes:
            The content of the file in bytes.
    """
    if is_s3_path(path):
        s3 = get_s3_client(aws_key, aws_secret)
        bucket_name, key = parse_s3_path(path)
        obj = s3.get_object(Bucket=bucket_name, Key=key)
        contents = obj['Body'].read()
    else:
        with open(path, 'r') as f:
            contents = f.read().encode('utf-8')

    return contents


def read_csv(path, aws_key, aws_secret):
    """Read csv file from path.

    The path can either be a local path or an s3 directory.

    Args:
        path (str):
            The path to the csv file.
        aws_key (str):
            The access key id that will be used to communicate with s3, if provided.
        aws_secret (str):
            The secret access key that will be used to communicate with s3, if provided.

    Returns:
        pandas.DataFrame:
            A DataFrame containing the contents of the csv file.
    """
    contents = read_file(path, aws_key, aws_secret)
    return pd.read_csv(io.BytesIO(contents))


def read_csv_from_path(path, aws_key, aws_secret):
    """Read all csv content within a path.

    All csv content within a path will be read and returned in a
    DataFrame. The path can be either local or an s3 directory.

    Args:
        path (str):
            The path to read from, which can be either local or an s3 path.
        aws_key (str):
            The access key id that will be used to communicate with s3, if provided.
        aws_secret (str):
            The secret access key that will be used to communicate with s3, if provided.

    Returns:
        pandas.DataFrame:
            A DataFrame of all the csv contents in the path.
    """
    csv_contents = []
    if is_s3_path(path):
        s3 = get_s3_client(aws_key, aws_secret)
        bucket_name, key_prefix = parse_s3_path(path)
        resp = s3.list_objects(Bucket=bucket_name, Prefix=key_prefix)
        csv_files = [f for f in resp['Contents'] if f['Key'].endswith('.csv')]
        for csv_file in csv_files:
            csv_file_key = csv_file['Key']
            csv_contents.append(read_csv(f's3://{bucket_name}/{csv_file_key}', aws_key, aws_secret))

    else:
        run_path = pathlib.Path(path)
        for csv_path in tqdm.tqdm(list(run_path.glob('**/*.csv'))):
            csv_contents.append(pd.read_csv(csv_path))

    return pd.concat(csv_contents)


def get_downloaded_datasets(datasets_path=None):
    """Get downloaded datatsets."""
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
