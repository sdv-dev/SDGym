"""Utils file for the run_benchmark module."""

from datetime import datetime

from sdgym.benchmark import SDV_SINGLE_TABLE_SYNTHESIZERS

OUTPUT_DESTINATION_AWS = 's3://sdgym-benchmark/Debug/Issue_425/'
UPLOAD_DESTINATION_AWS = 's3://sdgym-benchmark/Debug/Issue_425/'
DEBUG_SLACK_CHANNEL = 'sdv-alerts-debug'
SLACK_CHANNEL = 'sdv-alerts'
KEY_DATE_FILE = '_BENCHMARK_DATES.json'
SYNTHESIZERS = SDV_SINGLE_TABLE_SYNTHESIZERS


def get_result_folder_name(date_str):
    """Get the result folder name based on the date string."""
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f'Invalid date format: {date_str}. Expected YYYY-MM-DD.')

    return f'SDGym_results_{date.month:02d}_{date.day:02d}_{date.year}'
