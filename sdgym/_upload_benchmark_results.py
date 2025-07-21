import argparse
import os
import sys
from datetime import datetime

from sdgym.sdgym_result_explorer.result_explorer import SDGymResultsExplorer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, help='Benchmark date (YYYY-MM-DD)')
    return parser.parse_args()


def get_run_name(date_str):
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD.")
    return f'SDGym_results_{date.month:02d}_{date.day:02d}_{date.year}'


def main():
    args = parse_args()

    if args.date:
        date_str = args.date
    else:
        date_str = datetime.utcnow().replace(day=1).strftime('%Y-%m-%d')

    run_name = get_run_name(date_str)
    print(f"Checking benchmark results for run: {run_name}")

    aws_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
    bucket_path = 's3://sdgym-benchmark/Debug/Issue_425'
    result_explorer = SDGymResultsExplorer(
        bucket_path, aws_access_key_id=aws_key, aws_secret_access_key=aws_secret
    )

    if not result_explorer.all_runs_complete(run_name):
        print(f"Run {run_name} is not complete yet. Exiting.")
        sys.exit(0)

    print(f"Run {run_name} is complete! Proceeding with summarization...")
    # Call summarization/upload here
    result_explorer.summarize_and_publish(run_name)


if __name__ == '__main__':
    main()
