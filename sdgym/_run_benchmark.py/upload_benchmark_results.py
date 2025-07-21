import argparse
import os
import sys
from datetime import datetime

import boto3

from sdgym.result_writer import S3ResultsWriter
from sdgym.run_benchmark import OUTPUT_DESTINATION_AWS, RESULT_UPLOADED
from sdgym.sdgym_result_explorer.result_explorer import SDGymResultsExplorer


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


def main():
    if RESULT_UPLOADED:
        print('Benchmark results have already been uploaded. Exiting.')
        sys.exit(0)

    args = parse_args()
    if args.date:
        date_str = args.date
    else:
        date_str = datetime.utcnow().replace(day=1).strftime('%Y-%m-%d')

    run_name = get_run_name(date_str)
    print(f'Checking benchmark results for run: {run_name}')  # noqa: T201

    aws_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
    bucket_path = OUTPUT_DESTINATION_AWS
    summary_filepath = f'{bucket_path}/{run_name}_summary.csv'
    result_explorer = SDGymResultsExplorer(
        bucket_path, aws_access_key_id=aws_key, aws_secret_access_key=aws_secret
    )
    summary = result_explorer.
    s3_client = boto3.client('s3', aws_access_key_id=aws_key, aws_secret_access_key=aws_secret)
    result_writer = S3ResultsWriter(s3_client)
    if not result_explorer.all_runs_complete(run_name):
        print(f'Run {run_name} is not complete yet. Exiting.')  # noqa: T201
        sys.exit(0)

    print(f'Run {run_name} is complete! Proceeding with summarization...')  # noqa: T201
    summary, _ = result_explorer.summarize(run_name)
    result_writer.write_dataframe(summary, f'{bucket_path}/{run_name}_summary.csv', index=True)
    RESULT_UPLOADED = True

if __name__ == '__main__':
    main()
