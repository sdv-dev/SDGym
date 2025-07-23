import argparse
import logging
import os
import sys
from datetime import datetime

import boto3
from botocore.exceptions import ClientError

from sdgym._run_benchmark import OUTPUT_DESTINATION_AWS
from sdgym.result_writer import S3ResultsWriter
from sdgym.s3 import S3_REGION, parse_s3_path
from sdgym.sdgym_result_explorer.result_explorer import SDGymResultsExplorer

LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, help='Benchmark date (YYYY-MM-DD)')
    return parser.parse_args()


def get_run_name(date_str):
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f'Invalid date format: {date_str}. Expected YYYY-MM-DD.')

    return f'SDGym_results_{date.month:02d}_{date.day:02d}_{date.year}'


def write_uploaded_marker(s3_client, bucket, prefix, run_name):
    s3_client.put_object(
        Bucket=bucket, Key=f'{prefix}{run_name}/upload_complete.marker', Body=b'Upload complete'
    )


def upload_already_done(s3_client, bucket, prefix, run_name):
    try:
        s3_client.head_object(Bucket=bucket, Key=f'{prefix}{run_name}/upload_complete.marker')
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False

        raise


def get_run_name_and_s3_vars(aws_access_key_id, aws_secret_access_key):
    args = parse_args()
    if args.date:
        date_str = args.date
    else:
        date_str = datetime.utcnow().replace(day=1).strftime('%Y-%m-%d')

    run_name = get_run_name(date_str)
    bucket, prefix = parse_s3_path(OUTPUT_DESTINATION_AWS)
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=S3_REGION,
    )

    return run_name, s3_client, bucket, prefix


def upload_results(aws_access_key_id, aws_secret_access_key, run_name, s3_client, bucket, prefix):
    result_explorer = SDGymResultsExplorer(
        OUTPUT_DESTINATION_AWS,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    result_writer = S3ResultsWriter(s3_client)

    if not result_explorer.all_runs_complete(run_name):
        LOGGER.info(f'Run {run_name} is not complete yet. Exiting.')
        sys.exit(0)

    LOGGER.info(f'Run {run_name} is complete! Proceeding with summarization...')
    summary, _ = result_explorer.summarize(run_name)
    result_writer.write_dataframe(
        summary, f'{OUTPUT_DESTINATION_AWS}{run_name}/{run_name}_summary.csv', index=True
    )
    write_uploaded_marker(s3_client, bucket, prefix, run_name)


def main():
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    run_name, s3_client, bucket, prefix = get_run_name_and_s3_vars(
        aws_access_key_id, aws_secret_access_key
    )
    if upload_already_done(s3_client, bucket, prefix, run_name):
        LOGGER.info('Benchmark results have already been uploaded. Exiting.')
        sys.exit(0)

    upload_results(aws_access_key_id, aws_secret_access_key, run_name, s3_client, bucket, prefix)


if __name__ == '__main__':
    main()
