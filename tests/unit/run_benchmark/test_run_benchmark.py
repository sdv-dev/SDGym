import json
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest
from botocore.exceptions import ClientError

from sdgym.run_benchmark.run_benchmark import (
    _get_config,
    append_benchmark_run,
    main,
)
from sdgym.run_benchmark.utils import (
    OUTPUT_DESTINATION_AWS,
)


@patch('sdgym.run_benchmark.run_benchmark.get_s3_client')
@patch('sdgym.run_benchmark.run_benchmark.parse_s3_path')
@patch('sdgym.run_benchmark.run_benchmark.get_result_folder_name')
def test_append_benchmark_run(mock_get_result_folder_name, mock_parse_s3_path, mock_get_s3_client):
    """Test the `append_benchmark_run` method."""
    # Setup
    aws_access_key_id = 'my_access_key'
    aws_secret_access_key = 'my_secret_key'
    date = '2023-10-01'
    mock_get_result_folder_name.return_value = 'SDGym_results_10_01_2023'
    mock_parse_s3_path.return_value = ('my-bucket', 'my-prefix/')
    mock_s3_client = Mock()
    benchmark_date = {
        'runs': [
            {'date': '2023-09-30', 'folder_name': 'SDGym_results_09_30_2023'},
        ]
    }
    mock_get_s3_client.return_value = mock_s3_client
    mock_s3_client.get_object.return_value = {
        'Body': Mock(read=lambda: json.dumps(benchmark_date).encode('utf-8'))
    }
    expected_data = {
        'runs': [
            {'date': '2023-09-30', 'folder_name': 'SDGym_results_09_30_2023'},
            {'date': date, 'folder_name': 'SDGym_results_10_01_2023'},
        ]
    }

    # Run
    append_benchmark_run(aws_access_key_id, aws_secret_access_key, date)

    # Assert
    mock_get_s3_client.assert_called_once_with(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    mock_parse_s3_path.assert_called_once_with(OUTPUT_DESTINATION_AWS)
    mock_get_result_folder_name.assert_called_once_with(date)
    mock_s3_client.get_object.assert_called_once_with(
        Bucket='my-bucket', Key='my-prefix/single_table/_BENCHMARK_DATES.json'
    )
    mock_s3_client.put_object.assert_called_once_with(
        Bucket='my-bucket',
        Key='my-prefix/single_table/_BENCHMARK_DATES.json',
        Body=json.dumps(expected_data).encode('utf-8'),
    )


@patch('sdgym.run_benchmark.run_benchmark.get_s3_client')
@patch('sdgym.run_benchmark.run_benchmark.parse_s3_path')
@patch('sdgym.run_benchmark.run_benchmark.get_result_folder_name')
def test_append_benchmark_run_new_file(
    mock_get_result_folder_name, mock_parse_s3_path, mock_get_s3_client
):
    """Test the `append_benchmark_run` with a new file."""
    # Setup
    aws_access_key_id = 'my_access_key'
    aws_secret_access_key = 'my_secret_key'
    date = '2023-10-01'
    mock_get_result_folder_name.return_value = 'SDGym_results_10_01_2023'
    mock_parse_s3_path.return_value = ('my-bucket', 'my-prefix/')
    mock_s3_client = Mock()
    mock_get_s3_client.return_value = mock_s3_client
    mock_s3_client.get_object.side_effect = ClientError(
        {'Error': {'Code': 'NoSuchKey'}}, 'GetObject'
    )
    expected_data = {
        'runs': [
            {'date': date, 'folder_name': 'SDGym_results_10_01_2023'},
        ]
    }

    # Run
    append_benchmark_run(aws_access_key_id, aws_secret_access_key, date)

    # Assert
    mock_get_s3_client.assert_called_once_with(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    mock_parse_s3_path.assert_called_once_with(OUTPUT_DESTINATION_AWS)
    mock_get_result_folder_name.assert_called_once_with(date)
    mock_s3_client.get_object.assert_called_once_with(
        Bucket='my-bucket', Key='my-prefix/single_table/_BENCHMARK_DATES.json'
    )
    mock_s3_client.put_object.assert_called_once_with(
        Bucket='my-bucket',
        Key='my-prefix/single_table/_BENCHMARK_DATES.json',
        Body=json.dumps(expected_data).encode('utf-8'),
    )


@patch('sdgym.run_benchmark.run_benchmark._resolve_modality_config')
@patch('sdgym.run_benchmark.run_benchmark.BenchmarkConfig.load_from_dict')
def test__get_config(
    mock_load_from_dict,
    mock_resolve_modality_config,
):
    """Test the `_get_config` method."""
    # Setup
    modality = 'single_table'
    config_dict = {'modality': modality}
    mock_resolve_modality_config.return_value = config_dict
    config_obj = Mock()
    mock_load_from_dict.return_value = config_obj

    # Run
    config = _get_config(modality)

    # Assert
    mock_resolve_modality_config.assert_called_once_with(modality)
    mock_load_from_dict.assert_called_once_with(config_dict)
    config.validate.assert_called_once()
    assert config == config_obj


@pytest.mark.parametrize('modality', ['single_table', 'multi_table'])
@patch('sdgym.run_benchmark.run_benchmark.post_benchmark_launch_message')
@patch('sdgym.run_benchmark.run_benchmark.append_benchmark_run')
@patch('sdgym.run_benchmark.run_benchmark.os.getenv')
@patch('sdgym.run_benchmark.run_benchmark._parse_args')
@patch('sdgym.run_benchmark.run_benchmark._get_config')
@patch('sdgym.run_benchmark.run_benchmark.BenchmarkLauncher')
@patch('sdgym.run_benchmark.run_benchmark.get_result_folder_name')
def test_main(
    mock_get_result_folder_name,
    mock_benchmark_launcher,
    mock_get_config,
    mock_parse_args,
    mock_getenv,
    mock_append_benchmark_run,
    mock_post_benchmark_launch_message,
    modality,
):
    """Test the `main` function with both single_table and multi_table modalities."""
    # Setup
    mock_parse_args.return_value = Mock(modality=modality)
    mock_getenv.side_effect = lambda key: {
        'AWS_ACCESS_KEY_ID': 'my_access_key',
        'AWS_SECRET_ACCESS_KEY': 'my_secret_key',
        'CREDENTIALS_FILEPATH': '/path/to/creds.json',
    }.get(key)
    date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    config = Mock()
    mock_get_config.return_value = config
    launcher = Mock()
    mock_benchmark_launcher.return_value = launcher
    mock_launch = Mock()
    launcher.launch = mock_launch
    folder_name = f'SDGym_results_{date}'
    mock_get_result_folder_name.return_value = folder_name

    # Run
    main()

    # Assert
    launcher.save_to_cloud.assert_called_once_with(
        f'{OUTPUT_DESTINATION_AWS}{modality}/{folder_name}/_BENCHMARK_LAUNCHER.pkl'
    )
    mock_benchmark_launcher.assert_called_once_with(config)
    launcher.launch.assert_called_once()
    mock_append_benchmark_run.assert_called_once_with(
        'my_access_key',
        'my_secret_key',
        date,
        modality=modality,
    )
    mock_post_benchmark_launch_message.assert_called_once_with(
        date,
        compute_service='GCP',
        modality=modality,
    )
