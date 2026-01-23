"""Script to run a benchmark and upload results to S3."""

import json
import os

from botocore.exceptions import ClientError

from sdgym._benchmark.benchmark import (
    _benchmark_multi_table_compute_gcp,
    _benchmark_single_table_compute_gcp,
)
from sdgym.run_benchmark.utils import (
    KEY_DATE_FILE,
    OUTPUT_DESTINATION_AWS,
    SYNTHESIZERS_SPLIT_MULTI_TABLE,
    SYNTHESIZERS_SPLIT_SINGLE_TABLE,
    _parse_args,
    get_result_folder_name,
)
from sdgym.s3 import get_s3_client, parse_s3_path

MODALITY_TO_SETUP = {
    'single_table': {
        'method': _benchmark_single_table_compute_gcp,
        'synthesizers_split': SYNTHESIZERS_SPLIT_SINGLE_TABLE,
    },
    'multi_table': {
        'method': _benchmark_multi_table_compute_gcp,
        'synthesizers_split': SYNTHESIZERS_SPLIT_MULTI_TABLE,
    },
}


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


def main():
    """Main function to run the benchmark and upload results."""
    args = _parse_args()
    modality = args.modality
    dataset = 'expedia_hotel_logs' if modality == 'single_table' else 'fake_hotels'
    for synthesizer_group in MODALITY_TO_SETUP[modality]['synthesizers_split']:
        MODALITY_TO_SETUP[modality]['method'](
            output_destination=OUTPUT_DESTINATION_AWS,
            sdv_datasets=[dataset],
            credential_filepath=os.getenv('CREDENTIALS_FILEPATH'),
            synthesizers=synthesizer_group,
            timeout=345600,  # 4 days
        )


if __name__ == '__main__':
    main()
