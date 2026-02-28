from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sdgym.run_benchmark.utils import (
    OUTPUT_DESTINATION_AWS,
    _add_pareto_curve_extremity_points,
    _exclude_datasets,
    _extract_google_file_id,
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
        '🏃 SDGym single-table benchmark has been launched on AWS! '
        f'Intermediate results can be found <{url}|here>.\n'
    )
    # Run
    post_benchmark_launch_message(date_str)

    # Assert
    mock_get_result_folder_name.assert_called_once_with(date_str)
    mock_parse_s3_path.assert_called_once_with(OUTPUT_DESTINATION_AWS)
    mock_get_s3_console_link.assert_called_once_with(
        'my-bucket', f'my-prefix/single_table/{folder_name}/'
    )
    mock_post_slack_message.assert_called_once_with('sdgym', expected_body)


@patch('sdgym.run_benchmark.utils.post_slack_message')
@patch('sdgym.run_benchmark.utils.get_s3_console_link')
@patch('sdgym.run_benchmark.utils.parse_s3_path')
@patch('sdgym.run_benchmark.utils._get_filename_to_gdrive_link')
def test_post_benchmark_uploaded_message(
    mock__get_filename_to_gdrive_link,
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
    mock__get_filename_to_gdrive_link.return_value = {
        '[Single-table]_SDGym_Runs.xlsx': 'https://drive.google.com/file/d/example/view?usp=sharing'
    }
    expected_body = (
        f'🤸🏻‍♀️ SDGym Single-table benchmark results for *{folder_name}* are available! 🏋️‍♀️\n'
        f'Check the results:\n'
        f' - On GDrive: <https://drive.google.com/file/d/example/view?usp=sharing|link>\n'
        f' - On S3: <{url}|link>\n'
    )

    # Run
    post_benchmark_uploaded_message(folder_name)

    # Assert
    mock_post_slack_message.assert_called_once_with('sdgym', expected_body)
    mock__get_filename_to_gdrive_link.assert_called_once()
    mock_parse_s3_path.assert_called_once_with(OUTPUT_DESTINATION_AWS)
    mock_get_s3_console_link.assert_called_once_with(
        'my-bucket', 'my-prefix%2F%5BSingle-table%5D_SDGym_Runs.xlsx'
    )


@patch('sdgym.run_benchmark.utils.post_slack_message')
@patch('sdgym.run_benchmark.utils.get_s3_console_link')
@patch('sdgym.run_benchmark.utils.parse_s3_path')
@patch('sdgym.run_benchmark.utils._get_filename_to_gdrive_link')
def test_post_benchmark_uploaded_message_with_commit(
    mock__get_filename_to_gdrive_link,
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
    mock__get_filename_to_gdrive_link.return_value = {
        '[Single-table]_SDGym_Runs.xlsx': 'https://drive.google.com/file/d/example/view?usp=sharing'
    }
    expected_body = (
        f'🤸🏻‍♀️ SDGym Single-table benchmark results for *{folder_name}* are available! 🏋️‍♀️\n'
        f'Check the results:\n'
        f' - On GDrive: <https://drive.google.com/file/d/example/view?usp=sharing|link>\n'
        f' - On S3: <{url}|link>\n'
        f' - On GitHub: <{commit_url}|link>\n'
    )

    # Run
    post_benchmark_uploaded_message(folder_name, commit_url)

    # Assert
    mock__get_filename_to_gdrive_link.assert_called_once()
    mock_post_slack_message.assert_called_once_with('sdgym', expected_body)
    mock_parse_s3_path.assert_called_once_with(OUTPUT_DESTINATION_AWS)
    mock_get_s3_console_link.assert_called_once_with(
        'my-bucket', 'my-prefix%2F%5BSingle-table%5D_SDGym_Runs.xlsx'
    )


def test_add_pareto_curve_extremity_points():
    """Test `_add_pareto_curve_extremity_points` adds two extremity points."""
    # Setup
    df_to_plot = pd.DataFrame({
        'Synthesizer': ['GaussianCopula', 'CTGAN', 'TVAE'],
        'Aggregated_Time': [3.3, 7.7, 12.1],
        'Quality_Score': [0.82, 0.9, 0.475],
        'Log10 Aggregated_Time': np.log10([3.3, 7.7, 12.1]),
        'Pareto': [True, True, False],
        'Color': ['#01E0C9', '#01E0C9', '#03AFF1'],
        'Marker': ['circle', 'square', 'diamond'],
    })
    expected_results = pd.DataFrame({
        'Synthesizer': ['GaussianCopula', 'CTGAN', 'TVAE', np.nan, np.nan],
        'Aggregated_Time': [3.3, 7.7, 12.1, 1.8557263731281517, 21.517180861470965],
        'Quality_Score': [0.82, 0.9, 0.475, 0.7656487452490014, 0.9970266958089053],
        'Log10 Aggregated_Time': [
            0.5185139398778874,
            0.8864907251724818,
            1.08278537031645,
            0.2685139398778874,
            1.33278537031645,
        ],
        'Pareto': [True, True, False, True, True],
        'Color': ['#01E0C9', '#01E0C9', '#03AFF1', '#01E0C9', '#01E0C9'],
        'Marker': ['circle', 'square', 'diamond', np.nan, np.nan],
    })

    # Run
    result = _add_pareto_curve_extremity_points(df_to_plot)

    # Assert
    assert len(result) == len(df_to_plot) + 2
    pd.testing.assert_frame_equal(result, expected_results)


def test_add_pareto_curve_extremity_points_single_pareto():
    """Test `_add_pareto_curve_extremity_points` does nothing with <2 Pareto points."""
    # Setup
    df_to_plot = pd.DataFrame({
        'Synthesizer': ['GaussianCopula', 'CTGAN', 'TVAE'],
        'Aggregated_Time': [3.3, 7.7, 12.1],
        'Quality_Score': [0.9, 0.8, 0.7],
        'Log10 Aggregated_Time': np.log10([3.3, 7.7, 12.1]),
        'Pareto': [True, False, False],
        'Color': ['#01E0C9', '#03AFF1', '#03AFF1'],
        'Marker': ['circle', 'square', 'diamond'],
    })

    # Run
    result = _add_pareto_curve_extremity_points(df_to_plot)

    # Assert
    pd.testing.assert_frame_equal(result, df_to_plot)


def test_get_df_to_plot():
    """Test the `get_df_to_plot` method."""
    # Setup
    data = pd.DataFrame({
        'Synthesizer': (
            ['GaussianCopulaSynthesizer'] * 2 + ['CTGANSynthesizer'] * 2 + ['TVAESynthesizer'] * 2
        ),
        'Dataset': ['Dataset1', 'Dataset2'] * 3,
        'Train_Time': [0.9, 2.0, 3.0, 4.0, 5.0, 6.0],
        'Sample_Time': [0.1, 0.2, 0.3, 0.2, 0.5, 0.6],
        'Quality_Score': [0.80, 0.82, 0.92, 0.90, 0.50, 0.55],
        'Adjusted_Quality_Score': [0.81, 0.83, 0.91, 0.89, 0.45, 0.50],
        'Adjusted_Total_Time': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
    })

    # Run
    result = get_df_to_plot(data)

    # Assert
    expected_result = pd.DataFrame({
        'Synthesizer': ['GaussianCopula', 'CTGAN', 'TVAE', np.nan, np.nan],
        'Aggregated_Time': [3.3000000000000003, 7.7, 12.1, 1.8557263731281521, 21.517180861470965],
        'Quality_Score': [0.8200000000000001, 0.9, 0.475, 0.7656487452490015, 0.9970266958089052],
        'Log10 Aggregated_Time': [
            0.5185139398778875,
            0.8864907251724818,
            1.08278537031645,
            0.2685139398778875,
            1.33278537031645,
        ],
        'Pareto': [True, True, False, True, True],
        'Color': ['#01E0C9', '#01E0C9', '#03AFF1', '#01E0C9', '#01E0C9'],
        'Marker': ['circle', 'square', 'diamond', np.nan, np.nan],
    })
    pd.testing.assert_frame_equal(result, expected_result)


@pytest.mark.parametrize(
    'url',
    [
        'https://drive.google.com/file/d/1A2B3C4D5E6F7G8H9I0J/view?usp=sharing',
        'https://drive.google.com/open?id=1A2B3C4D5E6F7G8H9I0J',
        'https://docs.google.com/uc?id=1A2B3C4D5E6F7G8H9I0J&export=download',
    ],
)
def test_extract_google_file_id(url):
    """Test the `_extract_google_file_id` method."""
    # Run
    file_id = _extract_google_file_id(url)

    # Assert
    assert file_id == '1A2B3C4D5E6F7G8H9I0J'


def test_extract_google_file_id_invalid_url():
    """Test the `_extract_google_file_id` method with an invalid URL."""
    # Setup
    invalid_url = 'https://example.com/some/invalid/url'
    expected_message = 'Invalid Google Drive link format: https://example.com/some/invalid/url'

    # Run and Assert
    with pytest.raises(ValueError, match=expected_message):
        _extract_google_file_id(invalid_url)


def test__exclude_datasets():
    """Test the `_exclude_datasets` method."""
    # Setup
    datasets = ['dataset1', 'dataset2', 'dataset3', 'dataset4']
    dataset_to_exclude = ['dataset2', 'dataset4']

    # Run
    result = _exclude_datasets(datasets, dataset_to_exclude)

    # Assert
    expected_result = ['dataset1', 'dataset3']
    assert result == expected_result
