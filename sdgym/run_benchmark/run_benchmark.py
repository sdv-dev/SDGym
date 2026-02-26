"""Script to run a benchmark and upload results to S3."""

import json
import os
from datetime import datetime, timezone

from botocore.exceptions import ClientError

from sdgym._benchmark.benchmark import (
    _benchmark_multi_table_compute_gcp,
    _benchmark_single_table_compute_gcp,
)
from sdgym.run_benchmark.utils import (
    KEY_DATE_FILE,
    OUTPUT_DESTINATION_AWS,
    _exclude_datasets,
    _parse_args,
    get_result_folder_name,
    post_benchmark_launch_message,
)
from sdgym.s3 import get_s3_client, parse_s3_path

SINGLE_TABLE_DATASETS = [
    'adult',
    'alarm',
    'census',
    'child',
    'covtype',
    'expedia_hotel_logs',
    'insurance',
    'intrusion',
    'news',
]
MULTI_TABLE_DATASETS = [
    'WebKP',
    'DCG',
    'UW_std',
    'Same_gen',
    'CORA',
    'got_families',
    'SalesDB',
    'UTube',
    'Student_loan',
    'Hepatitis_std',
    'Elti',
    'Bupa',
    'Toxicology',
    'imdb_ijs',
    'ftp',
    'imdb_small',
    'imdb_MovieLens',
    'Pima',
    'university',
    'legalActs',
    'Dunur',
    'Mesh',
    'world',
    'airbnb-simplified',
    'trains',
    'FNHK',
    'fake_hotels',
    'SAT',
    'genes',
    'Biodegradability',
    'Pyrimidine',
    'mutagenesis',
    'restbase',
    'Triazine',
    'Carcinogenesis',
    'fake_hotels_extended',
    'Mooney_Family',
    'PTE',
    'Facebook',
    'multi_table_ID_demo_dataset',
    'SAP',
    'Chess',
    'Countries',
    'NCAA',
    'Atherosclerosis',
    'nations',
    'TubePricing',
    'financial',
    'Accidents',
    'MuskSmall',
    'NBA',
    'AustralianFootball',
    'PremierLeague',
    'OMOP_CDM_dayz',
]


def _get_benchmark_setup(modality):
    """Get the setup dict for a given modality ('single_table' or 'multi_table')."""
    if modality == 'single_table':
        real_tab_former_to_exclude = ['covtype', 'intrusion', 'expedia_hotel_logs', 'census']
        gan_to_exclude = ['covtype', 'intrusion']
        job_split = [
            (['ColumnSynthesizer', 'GaussianCopulaSynthesizer'], SINGLE_TABLE_DATASETS),
            (['TVAESynthesizer'], SINGLE_TABLE_DATASETS),
            (['SegmentSynthesizer'], SINGLE_TABLE_DATASETS),
            (['XGCSynthesizer'], SINGLE_TABLE_DATASETS),
            (['BootstrapSynthesizer'], SINGLE_TABLE_DATASETS),
            (['CTGANSynthesizer'], _exclude_datasets(SINGLE_TABLE_DATASETS, gan_to_exclude)),
            (['CopulaGANSynthesizer'], _exclude_datasets(SINGLE_TABLE_DATASETS, gan_to_exclude)),
            (
                ['RealTabFormerSynthesizer'],
                _exclude_datasets(SINGLE_TABLE_DATASETS, real_tab_former_to_exclude),
            ),
        ]
        for dataset in real_tab_former_to_exclude:
            job_split.append((['RealTabFormerSynthesizer'], [dataset]))

        for dataset in gan_to_exclude:
            job_split.append((['CTGANSynthesizer'], [dataset]))
            job_split.append((['CopulaGANSynthesizer'], [dataset]))

        return {
            'method': _benchmark_single_table_compute_gcp,
            'job_split': job_split,
        }

    if modality == 'multi_table':
        hma_to_exclude = [
            'Accidents',
            'AustralianFootball',
            'Countries',
            'MuskSmall',
            'NBA',
            'OMOP_CDM_dayz',
            'PremierLeague',
            'SalesDB',
            'airbnb-simplified',
            'imdb_ijs',
            'legalActs',
            'SAP',
            'imdb_MovieLens',
        ]
        job_split = [
            (['HSASynthesizer', 'IndependentSynthesizer'], MULTI_TABLE_DATASETS),
            (['HMASynthesizer'], _exclude_datasets(MULTI_TABLE_DATASETS, hma_to_exclude)),
        ]
        for dataset in hma_to_exclude:
            job_split.append((['HMASynthesizer'], [dataset]))

        return {
            'method': _benchmark_multi_table_compute_gcp,
            'job_split': job_split,
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
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    modality = args.modality
    benchmark_setup = _get_benchmark_setup(modality)
    for job in benchmark_setup['job_split']:
        benchmark_setup['method'](
            output_destination=OUTPUT_DESTINATION_AWS,
            credential_filepath=os.getenv('CREDENTIALS_FILEPATH'),
            synthesizers=job[0],
            sdv_datasets=job[1],
            timeout=345600,  # 4 days
        )

    append_benchmark_run(aws_access_key_id, aws_secret_access_key, date_str, modality=modality)
    post_benchmark_launch_message(date_str, compute_service='GCP', modality=modality)


if __name__ == '__main__':
    main()
