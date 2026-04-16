"""Script to run a benchmark and upload results to S3."""

import json
import os
from datetime import datetime, timezone

from botocore.exceptions import ClientError

from sdgym._benchmark_launcher.benchmark_config import BenchmarkConfig
from sdgym._benchmark_launcher.benchmark_launcher import BenchmarkLauncher
from sdgym._benchmark_launcher.utils import _resolve_modality_config
from sdgym.run_benchmark.utils import (
    KEY_BENCHMARK_LAUNCHER,
    KEY_DATE_FILE,
    OUTPUT_DESTINATION_AWS,
    _parse_args,
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
        object = s3_client.get_object(Bucket=bucket, Key=f'{prefix}{modality}/{KEY_DATE_FILE}')
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
        Bucket=bucket,
        Key=f'{prefix}{modality}/{KEY_DATE_FILE}',
        Body=json.dumps(data).encode('utf-8'),
    )


def _get_config(modality):
    config_dict = _resolve_modality_config(modality)
    config = BenchmarkConfig.load_from_dict(config_dict)
    config.validate()

    return config


def main():
    """Main function to run the benchmark."""
    args = _parse_args()
    modality = args.modality
    date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

    config = _get_config(modality)
    launcher = BenchmarkLauncher(config)
    launcher.launch()
    launcher.save_to_cloud(
        f'{OUTPUT_DESTINATION_AWS}{modality}/{get_result_folder_name(date_str)}/{KEY_BENCHMARK_LAUNCHER}'
    )

    append_benchmark_run(aws_access_key_id, aws_secret_access_key, date_str, modality=modality)
    post_benchmark_launch_message(date_str, compute_service='GCP', modality=modality)


if __name__ == '__main__':
    main()
