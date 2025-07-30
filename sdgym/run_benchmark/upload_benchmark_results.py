"""Script to upload benchmark results to S3."""

import json
import logging
import os
import sys

import boto3
from botocore.exceptions import ClientError

from sdgym.result_writer import S3ResultsWriter
from sdgym.run_benchmark.utils import OUTPUT_DESTINATION_AWS
from sdgym.s3 import S3_REGION, parse_s3_path
from sdgym.sdgym_result_explorer.result_explorer import SDGymResultsExplorer

LOGGER = logging.getLogger(__name__)


def get_latest_run_from_file(s3_client, bucket, key):
    """Get the latest run folder name from the benchmark dates file in S3."""
    try:
        object = s3_client.get_object(Bucket=bucket, Key=key)
        body = object['Body'].read().decode('utf-8')
        data = json.loads(body)
        latest = sorted(data['runs'], key=lambda x: x['date'])[-1]
        return latest['folder_name']
    except s3_client.exceptions.ClientError as e:
        raise RuntimeError(f'Failed to read {key} from S3: {e}')


def write_uploaded_marker(s3_client, bucket, prefix, folder_name):
    """Write a marker file to indicate that the upload is complete."""
    s3_client.put_object(
        Bucket=bucket, Key=f'{prefix}{folder_name}/upload_complete.marker', Body=b'Upload complete'
    )


def upload_already_done(s3_client, bucket, prefix, folder_name):
    """Check if the upload has already been done by looking for the marker file."""
    try:
        s3_client.head_object(Bucket=bucket, Key=f'{prefix}{folder_name}/upload_complete.marker')
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False

        raise


def get_result_folder_name_and_s3_vars(aws_access_key_id, aws_secret_access_key):
    """Get the result folder name and S3 client variables."""
    bucket, prefix = parse_s3_path(OUTPUT_DESTINATION_AWS)
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=S3_REGION,
    )
    folder_name = get_latest_run_from_file(s3_client, bucket, f'{prefix}_BENCHMARK_DATES.json')

    return folder_name, s3_client, bucket, prefix


def upload_results(
    aws_access_key_id, aws_secret_access_key, folder_name, s3_client, bucket, prefix
):
    """Upload benchmark results to S3."""
    result_explorer = SDGymResultsExplorer(
        OUTPUT_DESTINATION_AWS,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    result_writer = S3ResultsWriter(s3_client)

    if not result_explorer.all_runs_complete(folder_name):
        LOGGER.warning(f'Run {folder_name} is not complete yet. Exiting.')
        sys.exit(0)

    LOGGER.info(f'Run {folder_name} is complete! Proceeding with summarization...')
    summary, _ = result_explorer.summarize(folder_name)
    result_writer.write_dataframe(
        summary, f'{OUTPUT_DESTINATION_AWS}{folder_name}/{folder_name}_summary.csv', index=True
    )
    write_uploaded_marker(s3_client, bucket, prefix, folder_name)


def main():
    """Main function to upload benchmark results."""
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    folder_name, s3_client, bucket, prefix = get_result_folder_name_and_s3_vars(
        aws_access_key_id, aws_secret_access_key
    )
    if upload_already_done(s3_client, bucket, prefix, folder_name):
        LOGGER.warning('Benchmark results have already been uploaded. Exiting.')
        sys.exit(0)

    upload_results(aws_access_key_id, aws_secret_access_key, folder_name, s3_client, bucket, prefix)


if __name__ == '__main__':
    main()
