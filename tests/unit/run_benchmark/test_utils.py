from unittest.mock import patch

import pandas as pd
import pytest

from sdgym.run_benchmark.utils import (
    DEBUG_SLACK_CHANNEL,
    GDRIVE_LINK,
    OUTPUT_DESTINATION_AWS,
    _get_slack_client,
    get_df_to_plot,
    get_result_folder_name,
    get_s3_console_link,
    post_benchmark_launch_message,
    post_benchmark_uploaded_message,
    post_slack_message,
)


def test_get_result_folder_name():
    """Test the `get_result_folder_name` method."""
    # Setup
    expected_error_message = 'Invalid date format: invalid-date. Expected YYYY-MM-DD.'

    # Run and Assert
    assert get_result_folder_name('2023-10-01') == 'SDGym_results_10_01_2023'
    with pytest.raises(ValueError, match=expected_error_message):
        get_result_folder_name('invalid-date')


def test_get_s3_console_link():
    """Test the `get_s3_console_link` method."""
    # Setup
    bucket = 'my-bucket'
    prefix = 'my-prefix/'

    # Run
    link = get_s3_console_link(bucket, prefix)

    # Assert
    expected_link = (
        f'https://s3.console.aws.amazon.com/s3/buckets/{bucket}?prefix={prefix}&showversions=false'
    )
    assert link == expected_link


@patch('sdgym.run_benchmark.utils.WebClient')
@patch('sdgym.run_benchmark.utils.os.getenv')
def test_get_slack_client(mock_getenv, mock_web_client):
    """Test the `_get_slack_client` method."""
    # Setup
    mock_getenv.return_value = 'xoxb-test-token'

    # Run
    client = _get_slack_client()

    # Assert
    mock_getenv.assert_called_once_with('SLACK_TOKEN')
    mock_web_client.assert_called_once_with(token='xoxb-test-token')
    assert client is mock_web_client.return_value


@patch('sdgym.run_benchmark.utils._get_slack_client')
def test_post_slack_message(mock_get_slack_client):
    """Test the `post_slack_message` method."""
    # Setup
    mock_slack_client = mock_get_slack_client.return_value
    channel = 'test-channel'
    text = 'Test message'

    # Run
    post_slack_message(channel, text)

    # Assert
    mock_get_slack_client.assert_called_once()
    mock_slack_client.chat_postMessage.assert_called_once_with(channel=channel, text=text)


@patch('sdgym.run_benchmark.utils.post_slack_message')
@patch('sdgym.run_benchmark.utils.get_s3_console_link')
@patch('sdgym.run_benchmark.utils.parse_s3_path')
@patch('sdgym.run_benchmark.utils.get_result_folder_name')
def test_post_benchmark_launch_message(
    mock_get_result_folder_name,
    mock_parse_s3_path,
    mock_get_s3_console_link,
    mock_post_slack_message,
):
    """Test the `post_benchmark_launch_message` method."""
    # Setup
    date_str = '2023-10-01'
    folder_name = 'SDGym_results_10_01_2023'
    mock_get_result_folder_name.return_value = folder_name
    mock_parse_s3_path.return_value = ('my-bucket', 'my-prefix/')
    url = 'https://s3.console.aws.amazon.com/'
    mock_get_s3_console_link.return_value = url
    expected_body = (
        '🏃 SDGym benchmark has been launched! EC2 Instances are running. '
        f'Intermediate results can be found <{url}|here>.\n'
    )
    # Run
    post_benchmark_launch_message(date_str)

    # Assert
    mock_get_result_folder_name.assert_called_once_with(date_str)
    mock_parse_s3_path.assert_called_once_with(OUTPUT_DESTINATION_AWS)
    mock_get_s3_console_link.assert_called_once_with('my-bucket', f'my-prefix/{folder_name}/')
    mock_post_slack_message.assert_called_once_with(DEBUG_SLACK_CHANNEL, expected_body)


@patch('sdgym.run_benchmark.utils.post_slack_message')
@patch('sdgym.run_benchmark.utils.get_s3_console_link')
@patch('sdgym.run_benchmark.utils.parse_s3_path')
def test_post_benchmark_uploaded_message(
    mock_parse_s3_path,
    mock_get_s3_console_link,
    mock_post_slack_message,
):
    """Test the `post_benchmark_uploaded_message` method."""
    # Setup
    folder_name = 'SDGym_results_10_01_2023'
    mock_parse_s3_path.return_value = ('my-bucket', 'my-prefix/')
    url = 'https://s3.console.aws.amazon.com/'
    mock_get_s3_console_link.return_value = url
    expected_body = (
        f'🤸🏻‍♀️ SDGym benchmark results for *{folder_name}* are available! 🏋️‍♀️\n'
        f'Check the results:\n'
        f' - On GDrive: <{GDRIVE_LINK}|link>\n'
        f' - On S3: <{url}|link>\n'
    )

    # Run
    post_benchmark_uploaded_message(folder_name)

    # Assert
    mock_post_slack_message.assert_called_once_with(DEBUG_SLACK_CHANNEL, expected_body)
    mock_parse_s3_path.assert_called_once_with(OUTPUT_DESTINATION_AWS)
    mock_get_s3_console_link.assert_called_once_with(
        'my-bucket', f'my-prefix/{folder_name}/{folder_name}_summary.csv'
    )


@patch('sdgym.run_benchmark.utils.post_slack_message')
@patch('sdgym.run_benchmark.utils.get_s3_console_link')
@patch('sdgym.run_benchmark.utils.parse_s3_path')
def test_post_benchmark_uploaded_message_with_commit(
    mock_parse_s3_path,
    mock_get_s3_console_link,
    mock_post_slack_message,
):
    """Test the `post_benchmark_uploaded_message` with a commit URL."""
    # Setup
    folder_name = 'SDGym_results_10_01_2023'
    commit_url = 'https://github.com/user/repo/pull/123'
    mock_parse_s3_path.return_value = ('my-bucket', 'my-prefix/')
    url = 'https://s3.console.aws.amazon.com/'
    mock_get_s3_console_link.return_value = url
    expected_body = (
        f'🤸🏻‍♀️ SDGym benchmark results for *{folder_name}* are available! 🏋️‍♀️\n'
        f'Check the results:\n'
        f' - On GDrive: <{GDRIVE_LINK}|link>\n'
        f' - On S3: <{url}|link>\n'
        f' - On GitHub: <{commit_url}|link>\n'
    )

    # Run
    post_benchmark_uploaded_message(folder_name, commit_url)

    # Assert
    mock_post_slack_message.assert_called_once_with(DEBUG_SLACK_CHANNEL, expected_body)
    mock_parse_s3_path.assert_called_once_with(OUTPUT_DESTINATION_AWS)
    mock_get_s3_console_link.assert_called_once_with(
        'my-bucket', f'my-prefix/{folder_name}/{folder_name}_summary.csv'
    )


def test_get_df_to_plot():
    """Test the `get_df_to_plot` method."""
    # Setup
    data = pd.DataFrame({
        'Synthesizer': (
            ['GaussianCopulaSynthesizer'] * 2 + ['CTGANSynthesizer'] * 2 + ['TVAESynthesizer'] * 2
        ),
        'Dataset': ['Dataset1', 'Dataset2'] * 3,
        'Train_Time': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'Sample_Time': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        'Quality_Score': [0.8, 0.9, 0.7, 0.6, 0.5, 0.4],
    })

    # Run
    result = get_df_to_plot(data)

    # Assert
    expected_result = pd.DataFrame({
        'Synthesizer': ['GaussianCopula', 'CTGAN', 'TVAE'],
        'Aggregated_Time': [3.3, 7.7, 12.1],
        'Quality_Score': [0.85, 0.65, 0.45],
        'Log10 Aggregated_Time': [0.5185139398778875, 0.8864907251724818, 1.08278537031645],
        'Pareto': [True, False, False],
        'Color': ['#01E0C9', '#03AFF1', '#03AFF1'],
        'Marker': ['circle', 'square', 'diamond'],
    })
    pd.testing.assert_frame_equal(result, expected_result)
