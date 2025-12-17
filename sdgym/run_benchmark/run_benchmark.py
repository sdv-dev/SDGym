"""Script to run a benchmark and upload results to S3."""

import argparse
import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from botocore.exceptions import ClientError

from sdgym._benchmark.benchmark import _benchmark_multi_table_compute_gcp
from sdgym.benchmark import benchmark_single_table_aws
from sdgym.run_benchmark.utils import (
    GCP_PROJECT,
    GCP_ZONE,
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


def _load_gcp_service_account_from_env():
    """Load GCP service account JSON from env.

    Supports:
      - raw JSON string
      - base64-encoded JSON string
    """
    raw = os.getenv('GCP_SERVICE_ACCOUNT_JSON', '') or ''
    if not raw.strip():
        return {}

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        decoded = base64.b64decode(raw).decode('utf-8')
        return json.loads(decoded)


def create_credentials_file(filepath):
    """Create credentials file used by the benchmark launcher."""
    gcp_sa = _load_gcp_service_account_from_env()

    credentials = {
        'aws': {
            'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
            'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
        },
        'gcp': {
            **gcp_sa,
            'gcp_project': GCP_PROJECT,
            'gcp_zone': GCP_ZONE,
        },
        'sdv': {
            'username': os.getenv('SDV_ENTERPRISE_USERNAME'),
            'license_key': os.getenv('SDV_ENTERPRISE_LICENSE_KEY'),
        },
    }

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(credentials, f, indent=2, sort_keys=True)
        f.write('\n')


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--modality',
        choices=['single_table', 'multi_table'],
        default='single_table',
        help='Benchmark modality to run.',
    )
    parser.add_argument(
        '--gcp-output-destination',
        default='s3://sdgym-benchmark/Debug/GCP/',
        help='Where to store GCP benchmark results (S3).',
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
            aws_access_key_id,
            aws_secret_access_key,
            date_str,
            modality='single_table',
        )
        compute_service = 'AWS'

    else:
        runner_temp = os.environ.get('RUNNER_TEMP', '/tmp')
        cred_path = os.path.join(runner_temp, 'credentials.json')
        create_credentials_file(cred_path)

        for synthesizer_group in SYNTHESIZERS_SPLIT_MULTI_TABLE:
            _benchmark_multi_table_compute_gcp(
                output_destination=args.gcp_output_destination,
                credential_filepath=cred_path,
                synthesizers=synthesizer_group,
                compute_privacy_score=False,
                timeout=345600,  # 4 days
            )

        append_benchmark_run(
            aws_access_key_id,
            aws_secret_access_key,
            date_str,
            modality='multi_table',
        )
        compute_service = 'GCP'

    post_benchmark_launch_message(date_str, compute_service=compute_service)


if __name__ == '__main__':
    main()
