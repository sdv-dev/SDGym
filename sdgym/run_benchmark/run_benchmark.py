"""Script to run a benchmark and upload results to S3."""

import argparse
import json
import os
from datetime import datetime, timezone

from botocore.exceptions import ClientError

from sdgym.benchmark import _benchmark_multi_table_compute_gcp, benchmark_single_table_aws
from sdgym.run_benchmark.utils import (
    KEY_DATE_FILE,
    OUTPUT_DESTINATION_AWS,
    SYNTHESIZERS_SPLIT_MULTI_TABLE,
    SYNTHESIZERS_SPLIT_SINGLE_TABLE,
    get_result_folder_name,
    post_benchmark_launch_message,
)
from sdgym.s3 import get_s3_client, parse_s3_path


def append_benchmark_run(
    aws_access_key_id, aws_secret_access_key, date_str, modality='single_table'
):
    """Append a new benchmark run to the benchmark dates file in S3."""
    s3_client = get_s3_client(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    bucket, prefix = parse_s3_path(OUTPUT_DESTINATION_AWS)
    try:
        object = s3_client.get_object(Bucket=bucket, Key=f'{prefix}{modality}{KEY_DATE_FILE}')
        body = object['Body'].read().decode('utf-8')
        data = json.loads(body)
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            data = {'runs': []}
        else:
            raise RuntimeError(f'Failed to read {KEY_DATE_FILE} from S3: {e}')

    data['runs'].append({'date': date_str, 'folder_name': get_result_folder_name(date_str)})
    data['runs'] = sorted(data['runs'], key=lambda x: x['date'])
    s3_client.put_object(
        Bucket=bucket, Key=f'{prefix}{KEY_DATE_FILE}', Body=json.dumps(data).encode('utf-8')
    )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--modality',
        choices=['single_table', 'multi_table'],
        default='single_table',
        help='Benchmark modality to run.',
    )
    return parser.parse_args()


def main():
    """Main function to run the benchmark and upload results."""
    args = _parse_args()
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    if args.modality == 'single_table':
        for synthesizer_group in SYNTHESIZERS_SPLIT_SINGLE_TABLE:
            benchmark_single_table_aws(
                output_destination=OUTPUT_DESTINATION_AWS,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                synthesizers=synthesizer_group,
                compute_privacy_score=False,
                timeout=345600,  # 4 days
            )

        append_benchmark_run(
            aws_access_key_id, aws_secret_access_key, date_str, modality='single_table'
        )

    else:
        for synthesizer_group in SYNTHESIZERS_SPLIT_MULTI_TABLE:
            _benchmark_multi_table_compute_gcp(
                output_destination='s3://sdgym-benchmark/Debug/GCP/',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                synthesizers=synthesizer_group,
                compute_privacy_score=False,
                timeout=345600,  # 4 days
            )
        append_benchmark_run(
            aws_access_key_id, aws_secret_access_key, date_str, modality='multi_table'
        )

    post_benchmark_launch_message(date_str, compute_service='GCP')


if __name__ == '__main__':
    main()
