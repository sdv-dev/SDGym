"""Utils file for the run_benchmark module."""

import os
from datetime import datetime

import numpy as np
from slack_sdk import WebClient

from sdgym.s3 import parse_s3_path

OUTPUT_DESTINATION_AWS = 's3://sdgym-benchmark/Benchmarks/'
UPLOAD_DESTINATION_AWS = 's3://sdgym-benchmark/Benchmarks/'
DEBUG_SLACK_CHANNEL = 'sdv-alerts-debug'
SLACK_CHANNEL = 'sdv-alerts'
KEY_DATE_FILE = '_BENCHMARK_DATES.json'
PLOTLY_MARKERS = [
    'circle',
    'square',
    'diamond',
    'cross',
    'x',
    'triangle-up',
    'triangle-down',
    'triangle-left',
    'triangle-right',
    'pentagon',
    'hexagon',
    'hexagon2',
    'octagon',
    'star',
    'hexagram',
    'star-triangle-up',
    'star-triangle-down',
    'star-square',
    'star-diamond',
    'diamond-tall',
    'diamond-wide',
    'hourglass',
    'bowtie',
    'circle-cross',
    'circle-x',
    'square-cross',
    'square-x',
    'diamond-cross',
    'diamond-x',
]

# The synthesizers inside the same list will be run by the same ec2 instance
SYNTHESIZERS_SPLIT = [
    ['UniformSynthesizer', 'ColumnSynthesizer', 'GaussianCopulaSynthesizer', 'TVAESynthesizer'],
    ['CopulaGANSynthesizer'],
    ['CTGANSynthesizer'],
    ['RealTabFormerSynthesizer'],
]


def get_result_folder_name(date_str):
    """Get the result folder name based on the date string."""
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f'Invalid date format: {date_str}. Expected YYYY-MM-DD.')

    return f'SDGym_results_{date.month:02d}_{date.day:02d}_{date.year}'


def get_s3_console_link(bucket, prefix):
    """Get the S3 console link for the specified bucket and prefix."""
    return (
        f'https://s3.console.aws.amazon.com/s3/buckets/{bucket}?prefix={prefix}&showversions=false'
    )


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
    folder_name = get_result_folder_name(date_str)
    bucket, prefix = parse_s3_path(OUTPUT_DESTINATION_AWS)
    url_link = get_s3_console_link(bucket, f'{prefix}{folder_name}/')
    body = '🏃 SDGym benchmark has been launched! EC2 Instances are running. '
    body += f'Intermediate results can be found <{url_link}|here>.\n'
    post_slack_message(channel, body)


def post_benchmark_uploaded_message(folder_name, commit_url=None):
    """Post benchmark uploaded message to sdv-alerts slack channel."""
    channel = DEBUG_SLACK_CHANNEL
    bucket, prefix = parse_s3_path(OUTPUT_DESTINATION_AWS)
    url_link = get_s3_console_link(bucket, f'{prefix}{folder_name}/{folder_name}_summary.csv')
    body = (
        f'🤸🏻‍♀️ SDGym benchmark results for *{folder_name}* are available! 🏋️‍♀️\n'
        f'Check the results <{url_link} |here>'
    )
    if commit_url:
        body += f' or on GitHub: <{commit_url}|Commit Link>\n'

    post_slack_message(channel, body)


def get_df_to_plot(benchmark_result):
    """Get the data to plot from the benchmark result.

    Args:
        benchmark_result (DataFrame): The benchmark result DataFrame.

    Returns:
        DataFrame: The data to plot.
    """
    df_to_plot = benchmark_result.copy()
    df_to_plot['total_time'] = df_to_plot['Train_Time'] + df_to_plot['Sample_Time']
    df_to_plot['Aggregated_Time'] = df_to_plot.groupby('Synthesizer')['total_time'].transform('sum')
    df_to_plot = (
        df_to_plot.groupby('Synthesizer')[['Aggregated_Time', 'Quality_Score']].mean().reset_index()
    )
    df_to_plot['Log10 Aggregated_Time'] = df_to_plot['Aggregated_Time'].apply(
        lambda x: np.log10(x) if x > 0 else 0
    )
    df_to_plot = df_to_plot.sort_values(
        ['Aggregated_Time', 'Quality_Score'], ascending=[True, False]
    )
    df_to_plot['Cumulative Quality Score'] = df_to_plot['Quality_Score'].cummax()
    pareto_points = df_to_plot.loc[
        df_to_plot['Quality_Score'] == df_to_plot['Cumulative Quality Score']
    ]
    df_to_plot['Pareto'] = df_to_plot.index.isin(pareto_points.index)
    df_to_plot['Color'] = df_to_plot['Pareto'].apply(lambda x: '#01E0C9' if x else '#03AFF1')
    df_to_plot['Synthesizer'] = df_to_plot['Synthesizer'].str.replace(
        'Synthesizer', '', regex=False
    )

    synthesizers = df_to_plot['Synthesizer'].unique()
    marker_map = {
        synth: PLOTLY_MARKERS[i % len(PLOTLY_MARKERS)] for i, synth in enumerate(synthesizers)
    }
    df_to_plot['Marker'] = df_to_plot['Synthesizer'].map(marker_map)

    return df_to_plot.drop(columns=['Cumulative Quality Score']).reset_index(drop=True)
