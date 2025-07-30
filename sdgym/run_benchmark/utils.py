"""Utils file for the run_benchmark module."""

import os
from datetime import datetime

from slack_sdk import WebClient

OUTPUT_DESTINATION_AWS = 's3://sdgym-benchmark/Debug/Issue_425/'
UPLOAD_DESTINATION_AWS = 's3://sdgym-benchmark/Debug/Issue_425/'
DEBUG_SLACK_CHANNEL = 'sdv-alerts-debug'
SLACK_CHANNEL = 'sdv-alerts'
KEY_DATE_FILE = '_BENCHMARK_DATES.json'

# The synthesizers inside the same list will be run by the same ec2 instance
SYNTHESIZERS_SPLIT = [
    ['UniformSynthesizer', 'ColumnSynthesizer', 'GaussianCopulaSynthesizer'],
    ['TVAESynthesizer'],
]

"""
['CopulaGANSynthesizer'],
['CTGANSynthesizer'],
['RealTabFormerSynthesizer'],
"""


def get_result_folder_name(date_str):
    """Get the result folder name based on the date string."""
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f'Invalid date format: {date_str}. Expected YYYY-MM-DD.')

    return f'SDGym_results_{date.month:02d}_{date.day:02d}_{date.year}'


def _get_slack_client():
    """Create an authenticated Slack client.

    Returns:
        WebClient:
            An authenticated Slack WebClient instance.
    """
    token = os.getenv('SLACK_TOKEN')
    client = WebClient(token=token)
    return client


def post_slack_message(channel, text):
    """Post a message to a Slack channel."""
    client = _get_slack_client()
    client.chat_postMessage(channel=channel, text=text)


def post_benchmark_launch_message(date_str):
    """Post a message to the SDV Alerts Slack channel when the benchmark is launched."""
    channel = DEBUG_SLACK_CHANNEL
    body = '🏃 SDGym benchmark has been launched! EC2 Instances are running. '
    body += 'Intermediate results can be found '
    body += f'<{OUTPUT_DESTINATION_AWS}{get_result_folder_name(date_str)} |here>.\n'
    post_slack_message(channel, body)


def post_run_summary(folder_name):
    """Post run summary to sdv-alerts slack channel."""
    channel = DEBUG_SLACK_CHANNEL
    body = ''
    body += f'🤸🏻‍♀️ SDGym benchmark results for {folder_name} are available!🏋️‍♀️ \n'
    body += (
        f'Check the results <{OUTPUT_DESTINATION_AWS}{folder_name}/{folder_name}_summary'
        '.csv|here>.\n'
    )

    post_slack_message(channel, body)
