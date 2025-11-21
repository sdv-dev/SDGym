"""S3 module."""

import io
import logging
import pickle
from urllib.parse import urlparse

import boto3
import botocore
import pandas as pd
import yaml

S3_PREFIX = 's3://'
S3_REGION = 'us-east-1'
LOGGER = logging.getLogger(__name__)


def is_s3_path(path):
    """Determine if the given path is an s3 path.

    Args:
        path (str):
            The path, which might be an s3 path.

    Returns:
        bool:
            A boolean representing if the path is an s3 path or not.
    """
    return isinstance(path, str) and S3_PREFIX in path


def parse_s3_path(path):
    """Parse a s3 path into the bucket and key prefix.

    Args:
        path (str):
            The s3 path to parse. The expected format for the s3 path is
            `s3://<bucket-name>/path/to/dir`.

    Returns:
        tuple: (bucket_name, key_prefix)
    """
    if not path.startswith(S3_PREFIX):
        raise ValueError(f'Invalid S3 URI: {path}')

    path_without_prefix = path[len(S3_PREFIX) :]
    parts = path_without_prefix.split('/', 1)
    bucket_name = parts[0]
    key_prefix = parts[1] if len(parts) > 1 else ''

    return bucket_name, key_prefix


def get_s3_client(aws_access_key_id=None, aws_secret_access_key=None):
    """Get the boto client for interfacing with AWS s3.

    Args:
        aws_access_key_id (str):
            The access key id that will be used to communicate with
            s3, if provided.
        aws_secret_access_key (str):
            The secret access key that will be used to communicate
            with s3, if provided.

    Returns:
        boto3.session.Session.client:
            The s3 client that can be used to read / write to s3.
    """
    if aws_access_key_id is not None and aws_secret_access_key is not None:
        # credentials available
        return boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=S3_REGION,
        )
    else:
        if boto3.Session().get_credentials():
            # credentials available and will be detected automatically
            config = None
        else:
            # no credentials available, make unsigned requests
            config = botocore.config.Config(signature_version=botocore.UNSIGNED)

        return boto3.client('s3', config=config)


def write_file(data_contents, path, aws_access_key_id, aws_secret_access_key):
    """Write a file to the given path with the given contents.

    If the path is an s3 directory, we will use the given aws credentials
    to write to s3.

    Args:
        data_contents (bytes):
            The contents that will be written to the file.
        path (str):
            The path to write the file to, which can be either local
            or an s3 path.
        aws_access_key_id (str):
            The access key id that will be used to communicate with s3,
            if provided.
        aws_secret_access_key (str):
            The secret access key that will be used to communicate
            with s3, if provided.

    Returns:
        none
    """
    content_encoding = ''
    write_mode = 'w'
    if path.endswith('gz') or path.endswith('gzip'):
        content_encoding = 'gzip'
        write_mode = 'wb'
    elif isinstance(data_contents, bytes):
        write_mode = 'wb'

    if is_s3_path(path):
        s3 = get_s3_client(aws_access_key_id, aws_secret_access_key)
        bucket_name, key = parse_s3_path(path)
        s3.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=data_contents,
            ContentEncoding=content_encoding,
        )
    else:
        with open(path, write_mode) as f:
            if write_mode == 'w':
                f.write(data_contents.decode('utf-8'))
            else:
                f.write(data_contents)


def write_csv(data, path, aws_access_key_id, aws_secret_access_key):
    """Write a csv file to the given path with the given contents.

    If the path is an s3 directory, we will use the given aws credentials
    to write to s3.

    Args:
        data (pandas.DataFrame):
            The data that will be written to the csv file.
        path (str):
            The path to write the file to, which can be either local
            or an s3 path.
        aws_access_key_id (str):
            The access key id that will be used to communicate with s3,
            if provided.
        aws_secret_access_key (str):
            The secret access key that will be used to communicate
            with s3, if provided.

    Returns:
        none
    """
    data_contents = data.to_csv(index=False).encode('utf-8')
    write_file(data_contents, path, aws_access_key_id, aws_secret_access_key)


def _parse_s3_paths(s3_paths_dict):
    bucket = None
    keys = {}
    for k, s3_uri in s3_paths_dict.items():
        b, key = parse_s3_path(s3_uri)
        if bucket is None:
            bucket = b
        elif bucket != b:
            raise ValueError('Different buckets found in the paths dict')

        keys[k] = key

    return bucket, keys


def _upload_pickle_to_s3(obj, s3_client, bucket_name, key):
    bytes_buffer = io.BytesIO()
    pickle.dump(obj, bytes_buffer)
    bytes_buffer.seek(0)
    s3_client.put_object(Bucket=bucket_name, Key=key, Body=bytes_buffer)


def _upload_dataframe_to_s3(data, s3_client, bucket_name, key, append=False):
    """Upload a dataframe to S3, optionally appending if the file already exists."""
    if append:
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=key)
            existing_csv = response['Body'].read().decode('utf-8')
            existing_df = pd.read_csv(io.StringIO(existing_csv))
            data = pd.concat([existing_df, data], ignore_index=True)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] != 'NoSuchKey':
                raise e
            else:
                LOGGER.info(f'File {key} does not exist, creating a new one.')

    csv_buffer = io.StringIO()
    data.to_csv(csv_buffer, index=False)
    s3_client.put_object(Bucket=bucket_name, Key=key, Body=csv_buffer.getvalue())


def _get_s3_client(output_destination, aws_access_key_id=None, aws_secret_access_key=None):
    parsed_url = urlparse(output_destination)
    bucket_name = parsed_url.netloc
    if not bucket_name:
        raise ValueError(f'Invalid S3 URL: {output_destination}')

    if aws_access_key_id and aws_secret_access_key:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=S3_REGION,
        )
    else:
        s3_client = boto3.client('s3')

    s3_client.head_bucket(Bucket=bucket_name)

    return s3_client


def _read_data_from_bucket_key(s3_client, bucket_name, object_key):
    response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
    return response['Body'].read()


def _list_s3_bucket_contents(s3_client, bucket_name, prefix):
    contents = []
    paginator = s3_client.get_paginator('list_objects_v2')
    for resp in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        contents.extend(resp.get('Contents', []))

    return contents


def _load_yaml_metainfo_from_s3(s3_client, bucket_name, yaml_key):
    """Load and parse YAML metainfo from an S3 key."""
    raw_data = _read_data_from_bucket_key(s3_client, bucket_name, yaml_key)
    return yaml.safe_load(raw_data) or {}


def _validate_s3_url(s3_url):
    """Validate the provided S3 URL and return the bucket name.

    Args:
        s3_url (str):
            The S3 URL that should point to a bucket, e.g. ``'s3://my-bucket'``.

    Returns:
        str:
            The validated bucket name.

    Raises:
        ValueError:
            If the S3 URL is malformed or does not point to a bucket.
    """
    error_msg = (
        f"The provided s3_url parameter ('{s3_url}') is not a valid S3 URL.\n"
        "Please provide a string that starts with 's3://' and refers to a AWS bucket."
    )
    if not isinstance(s3_url, str):
        raise ValueError(error_msg)

    parsed = urlparse(s3_url)
    if parsed.scheme != 's3':
        raise ValueError(error_msg)

    bucket_name = parsed.netloc
    # Only bucket-level URLs are allowed (no keys/paths), allow optional trailing slash
    if not bucket_name or parsed.path not in ('', '/'):
        raise ValueError(error_msg)
    if parsed.params or parsed.query or parsed.fragment:
        raise ValueError(error_msg)

    return bucket_name
