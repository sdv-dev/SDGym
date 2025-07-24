import json
import os
from datetime import datetime, timezone

from botocore.exceptions import ClientError

import sdgym._run_benchmark as run_benchmark
from sdgym._run_benchmark._utils import get_run_name
from sdgym.benchmark import benchmark_single_table_aws
from sdgym.s3 import get_s3_client, parse_s3_path


def append_benchmark_run(aws_access_key_id, aws_secret_access_key, date_str):
    s3_client = get_s3_client(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    bucket, prefix = parse_s3_path(run_benchmark.OUTPUT_DESTINATION_AWS)
    key = '_BENCHMARK_DATES.json'
    try:
        object = s3_client.get_object(Bucket=bucket, Key=f'{prefix}{key}')
        body = object['Body'].read().decode('utf-8')
        data = json.loads(body)
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            data = {'runs': []}
        else:
            raise RuntimeError(f'Failed to read {key} from S3: {e}')

    data['runs'].append({'date': date_str, 'run_name': get_run_name(date_str)})
    data['runs'] = sorted(data['runs'], key=lambda x: x['date'])
    s3_client.put_object(Bucket=bucket, Key=f'{prefix}{key}', Body=json.dumps(data).encode('utf-8'))


def main():
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    for synthesizer in run_benchmark.SYNTHESIZERS:
        benchmark_single_table_aws(
            output_destination=run_benchmark.OUTPUT_DESTINATION_AWS,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            synthesizers=[synthesizer],
            compute_privacy_score=False,
        )

    append_benchmark_run(aws_access_key_id, aws_secret_access_key, date_str)


if __name__ == '__main__':
    main()
