from unittest.mock import Mock, patch

import pytest
from botocore.exceptions import ClientError

from sdgym._run_benchmark.upload_benchmark_results import (
    get_run_name_and_s3_vars,
    main,
    upload_already_done,
    upload_results,
    write_uploaded_marker,
)
from sdgym.s3 import S3_REGION


def test_write_uploaded_marker():
    """Test the `write_uploaded_marker` method."""
    # Setup
    s3_client = Mock()
    bucket = 'test-bucket'
    prefix = 'test-prefix/'
    run_name = 'test_run'

    # Run
    write_uploaded_marker(s3_client, bucket, prefix, run_name)

    # Assert
    s3_client.put_object.assert_called_once_with(
        Bucket=bucket, Key=f'{prefix}{run_name}/upload_complete.marker', Body=b'Upload complete'
    )


def test_upload_already_done():
    """Test the `upload_already_done` method."""
    # Setup
    s3_client = Mock()
    bucket = 'test-bucket'
    prefix = 'test-prefix/'
    run_name = 'test_run'
    s3_client.head_object.side_effect = [
        '',
        ClientError(
            error_response={'Error': {'Code': '404', 'Message': 'Not Found'}},
            operation_name='HeadObject',
        ),
        ClientError(
            error_response={'Error': {'Code': '405', 'Message': 'Other Error'}},
            operation_name='HeadObject',
        ),
    ]

    # Run
    result = upload_already_done(s3_client, bucket, prefix, run_name)
    result_false = upload_already_done(s3_client, bucket, prefix, run_name)
    with pytest.raises(ClientError):
        upload_already_done(s3_client, bucket, prefix, run_name)

    # Assert
    assert result is True
    assert result_false is False


@patch('sdgym._run_benchmark.upload_benchmark_results.boto3.client')
@patch('sdgym._run_benchmark.upload_benchmark_results.parse_s3_path')
@patch('sdgym._run_benchmark.upload_benchmark_results.OUTPUT_DESTINATION_AWS')
@patch('sdgym._run_benchmark.upload_benchmark_results.get_latest_run_from_file')
def test_get_run_name_and_s3_vars(
    mock_get_latest_run_from_file,
    mock_output_destination_aws,
    mock_parse_s3_path,
    mock_boto_client,
):
    """Test the `get_run_name_and_s3_vars` method."""
    # Setup
    aws_access_key_id = 'my_access_key'
    aws_secret_access_key = 'my_secret_key'
    expected_result = ('SDGym_results_10_01_2023', 's3_client', 'bucket', 'prefix')
    mock_boto_client.return_value = 's3_client'
    mock_parse_s3_path.return_value = ('bucket', 'prefix')
    mock_get_latest_run_from_file.return_value = 'SDGym_results_10_01_2023'

    # Run
    result = get_run_name_and_s3_vars(aws_access_key_id, aws_secret_access_key)

    # Assert
    assert result == expected_result
    mock_boto_client.assert_called_once_with(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=S3_REGION,
    )
    mock_parse_s3_path.assert_called_once_with(mock_output_destination_aws)
    mock_get_latest_run_from_file.assert_called_once_with(
        's3_client', 'bucket', 'prefix_BENCHMARK_DATES.json'
    )


@patch('sdgym._run_benchmark.upload_benchmark_results.SDGymResultsExplorer')
@patch('sdgym._run_benchmark.upload_benchmark_results.S3ResultsWriter')
@patch('sdgym._run_benchmark.upload_benchmark_results.write_uploaded_marker')
@patch('sdgym._run_benchmark.upload_benchmark_results.LOGGER')
@patch('sdgym._run_benchmark.upload_benchmark_results.OUTPUT_DESTINATION_AWS')
def test_upload_results(
    mock_output_destination_aws,
    mock_logger,
    mock_write_uploaded_marker,
    mock_s3_results_writer,
    mock_sdgym_results_explorer,
):
    """Test the `upload_results` method."""
    # Setup
    aws_access_key_id = 'my_access_key'
    aws_secret_access_key = 'my_secret_key'
    run_name = 'SDGym_results_10_01_2023'
    s3_client = 's3_client'
    bucket = 'bucket'
    prefix = 'prefix'
    result_explorer_instance = mock_sdgym_results_explorer.return_value
    result_explorer_instance.all_runs_complete.return_value = True
    result_explorer_instance.summarize.return_value = ('summary', 'results')

    # Run
    upload_results(aws_access_key_id, aws_secret_access_key, run_name, s3_client, bucket, prefix)

    # Assert
    mock_logger.info.assert_called_once_with(
        f'Run {run_name} is complete! Proceeding with summarization...'
    )
    mock_sdgym_results_explorer.assert_called_once_with(
        mock_output_destination_aws,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    result_explorer_instance.all_runs_complete.assert_called_once_with(run_name)
    result_explorer_instance.summarize.assert_called_once_with(run_name)
    mock_s3_results_writer.return_value.write_dataframe.assert_called_once()
    mock_write_uploaded_marker.assert_called_once_with(s3_client, bucket, prefix, run_name)


@patch('sdgym._run_benchmark.upload_benchmark_results.SDGymResultsExplorer')
@patch('sdgym._run_benchmark.upload_benchmark_results.S3ResultsWriter')
@patch('sdgym._run_benchmark.upload_benchmark_results.write_uploaded_marker')
@patch('sdgym._run_benchmark.upload_benchmark_results.LOGGER')
@patch('sdgym._run_benchmark.upload_benchmark_results.OUTPUT_DESTINATION_AWS')
def test_upload_results_not_all_runs_complete(
    mock_output_destination_aws,
    mock_logger,
    mock_write_uploaded_marker,
    mock_s3_results_writer,
    mock_sdgym_results_explorer,
):
    """Test the `upload_results` when not all runs are complete."""
    # Setup
    aws_access_key_id = 'my_access_key'
    aws_secret_access_key = 'my_secret_key'
    run_name = 'SDGym_results_10_01_2023'
    s3_client = 's3_client'
    bucket = 'bucket'
    prefix = 'prefix'
    result_explorer_instance = mock_sdgym_results_explorer.return_value
    result_explorer_instance.all_runs_complete.return_value = False
    result_explorer_instance.summarize.return_value = ('summary', 'results')

    # Run
    with pytest.raises(SystemExit, match='0'):
        upload_results(
            aws_access_key_id, aws_secret_access_key, run_name, s3_client, bucket, prefix
        )

    # Assert
    mock_logger.warning.assert_called_once_with(f'Run {run_name} is not complete yet. Exiting.')
    mock_sdgym_results_explorer.assert_called_once_with(
        mock_output_destination_aws,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    result_explorer_instance.all_runs_complete.assert_called_once_with(run_name)
    result_explorer_instance.summarize.assert_not_called()
    mock_s3_results_writer.return_value.write_dataframe.assert_not_called()
    mock_write_uploaded_marker.assert_not_called()


@patch('sdgym._run_benchmark.upload_benchmark_results.get_run_name_and_s3_vars')
@patch('sdgym._run_benchmark.upload_benchmark_results.upload_results')
@patch('sdgym._run_benchmark.upload_benchmark_results.upload_already_done')
@patch('sdgym._run_benchmark.upload_benchmark_results.LOGGER')
@patch('sdgym._run_benchmark.upload_benchmark_results.os.getenv')
def test_main_already_upload(
    mock_getenv,
    mock_logger,
    mock_upload_already_done,
    mock_upload_results,
    mock_get_run_name_and_s3_vars,
):
    """Test the `method` when results are already uploaded."""
    # Setup
    mock_getenv.side_effect = ['my_access_key', 'my_secret_key']
    mock_get_run_name_and_s3_vars.return_value = ('run_name', 's3_client', 'bucket', 'prefix')
    mock_upload_already_done.return_value = True
    expected_log_message = 'Benchmark results have already been uploaded. Exiting.'

    # Run
    with pytest.raises(SystemExit, match='0'):
        main()

    # Assert
    mock_get_run_name_and_s3_vars.assert_called_once_with('my_access_key', 'my_secret_key')
    mock_logger.warning.assert_called_once_with(expected_log_message)
    mock_upload_results.assert_not_called()


@patch('sdgym._run_benchmark.upload_benchmark_results.get_run_name_and_s3_vars')
@patch('sdgym._run_benchmark.upload_benchmark_results.upload_results')
@patch('sdgym._run_benchmark.upload_benchmark_results.upload_already_done')
@patch('sdgym._run_benchmark.upload_benchmark_results.os.getenv')
def test_main(
    mock_getenv, mock_upload_already_done, mock_upload_results, mock_get_run_name_and_s3_vars
):
    """Test the `main` method."""
    # Setup
    mock_getenv.side_effect = ['my_access_key', 'my_secret_key']
    mock_get_run_name_and_s3_vars.return_value = ('run_name', 's3_client', 'bucket', 'prefix')
    mock_upload_already_done.return_value = False

    # Run
    main()

    # Assert
    mock_get_run_name_and_s3_vars.assert_called_once_with('my_access_key', 'my_secret_key')
    mock_upload_already_done.assert_called_once_with('s3_client', 'bucket', 'prefix', 'run_name')
    mock_upload_results.assert_called_once_with(
        'my_access_key', 'my_secret_key', 'run_name', 's3_client', 'bucket', 'prefix'
    )
