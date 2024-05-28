"""S3 module."""

import boto3
import botocore

S3_PREFIX = 's3://'


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
        tuple:
            A tuple containing (`bucket_name`, `key_prefix`) where `bucket_name`
            is the name of the s3 bucket, and `key_prefix` is the remainder
            of the s3 path.
    """
    bucket_parts = path.replace(S3_PREFIX, '').split('/')
    bucket_name = bucket_parts[0]

    key_prefix = ''
    if len(bucket_parts) > 1:
        key_prefix = '/'.join(bucket_parts[1:])

    return bucket_name, key_prefix


def get_s3_client(aws_key=None, aws_secret=None):
    """Get the boto client for interfacing with AWS s3.

    Args:
        aws_key (str):
            The access key id that will be used to communicate with
            s3, if provided.
        aws_secret (str):
            The secret access key that will be used to communicate
            with s3, if provided.

    Returns:
        boto3.session.Session.client:
            The s3 client that can be used to read / write to s3.
    """
    if aws_key is not None and aws_secret is not None:
        # credentials available
        return boto3.client('s3', aws_access_key_id=aws_key, aws_secret_access_key=aws_secret)
    else:
        if boto3.Session().get_credentials():
            # credentials available and will be detected automatically
            config = None
        else:
            # no credentials available, make unsigned requests
            config = botocore.config.Config(signature_version=botocore.UNSIGNED)

        return boto3.client('s3', config=config)


def write_file(data_contents, path, aws_key, aws_secret):
    """Write a file to the given path with the given contents.

    If the path is an s3 directory, we will use the given aws credentials
    to write to s3.

    Args:
        data_contents (bytes):
            The contents that will be written to the file.
        path (str):
            The path to write the file to, which can be either local
            or an s3 path.
        aws_key (str):
            The access key id that will be used to communicate with s3,
            if provided.
        aws_secret (str):
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
        s3 = get_s3_client(aws_key, aws_secret)
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


def write_csv(data, path, aws_key, aws_secret):
    """Write a csv file to the given path with the given contents.

    If the path is an s3 directory, we will use the given aws credentials
    to write to s3.

    Args:
        data (pandas.DataFrame):
            The data that will be written to the csv file.
        path (str):
            The path to write the file to, which can be either local
            or an s3 path.
        aws_key (str):
            The access key id that will be used to communicate with s3,
            if provided.
        aws_secret (str):
            The secret access key that will be used to communicate
            with s3, if provided.

    Returns:
        none
    """
    data_contents = data.to_csv(index=False).encode('utf-8')
    write_file(data_contents, path, aws_key, aws_secret)
