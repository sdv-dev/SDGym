import json
import re
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from botocore.exceptions import ClientError

from sdgym.run_benchmark.upload_benchmark_results import (
    get_result_folder_name_and_s3_vars,
    main,
    upload_already_done,
    upload_results,
    upload_to_drive,
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
    modality = 'single_table'

    # Run
    write_uploaded_marker(s3_client, bucket, prefix, run_name)

    # Assert
    s3_client.put_object.assert_called_once_with(
        Bucket=bucket,
        Key=f'{prefix}{modality}/{run_name}/upload_complete.marker',
        Body=b'Upload complete',
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


@patch('sdgym.run_benchmark.upload_benchmark_results.boto3.client')
@patch('sdgym.run_benchmark.upload_benchmark_results.parse_s3_path')
@patch('sdgym.run_benchmark.upload_benchmark_results.OUTPUT_DESTINATION_AWS')
@patch('sdgym.run_benchmark.upload_benchmark_results.get_latest_run_from_file')
def test_get_result_folder_name_and_s3_vars(
    mock_get_latest_run_from_file,
    mock_output_destination_aws,
    mock_parse_s3_path,
    mock_boto_client,
):
    """Test the `get_result_folder_name_and_s3_vars` method."""
    # Setup
    aws_access_key_id = 'my_access_key'
    aws_secret_access_key = 'my_secret_key'
    expected_result = ('SDGym_results_10_01_2023', 's3_client', 'bucket', 'prefix/')
    mock_boto_client.return_value = 's3_client'
    mock_parse_s3_path.return_value = ('bucket', 'prefix/')
    mock_get_latest_run_from_file.return_value = 'SDGym_results_10_01_2023'

    # Run
    result = get_result_folder_name_and_s3_vars(aws_access_key_id, aws_secret_access_key)

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
        's3_client', 'bucket', 'prefix/single_table/_BENCHMARK_DATES.json'
    )


@patch('sdgym.run_benchmark.upload_benchmark_results.GoogleDrive')
@patch('sdgym.run_benchmark.upload_benchmark_results.GoogleAuth')
@patch('sdgym.run_benchmark.upload_benchmark_results.OAuth2Credentials')
@patch('sdgym.run_benchmark.upload_benchmark_results.os.environ', new_callable=dict)
def test_upload_to_drive_success(mock_environ, mock_oauth, mock_auth, mock_drive, tmp_path):
    """Test `upload_to_drive` uploads a file successfully."""
    # Setup
    file_path = tmp_path / 'test.xlsx'
    file_path.write_text('dummy content')

    creds_dict = {
        'access_token': 'token',
        'client_id': 'client',
        'client_secret': 'secret',
        'refresh_token': 'refresh',
    }
    mock_environ['PYDRIVE_TOKEN'] = json.dumps(creds_dict)

    mock_drive_instance = Mock()
    mock_drive.return_value = mock_drive_instance
    mock_file = Mock()
    mock_drive_instance.CreateFile.return_value = mock_file

    # Run
    upload_to_drive(file_path, 'fake_file_id')

    # Assert
    mock_oauth.assert_called_once_with(
        access_token='token',
        client_id='client',
        client_secret='secret',
        refresh_token='refresh',
        token_expiry=None,
        token_uri='https://oauth2.googleapis.com/token',
        user_agent=None,
    )
    mock_auth.assert_called_once()
    mock_drive.assert_called_once_with(mock_auth.return_value)
    mock_drive_instance.CreateFile.assert_called_once_with({'id': 'fake_file_id'})
    mock_file.SetContentFile.assert_called_once_with(file_path)
    mock_file.Upload.assert_called_once_with(param={'supportsAllDrives': True})


def test_upload_to_drive_file_not_found(tmp_path):
    """Test `upload_to_drive` raises FileNotFoundError for missing file."""
    # Setup
    missing_file = str(Path(tmp_path / 'missing.xlsx'))

    # Run and Assert
    with pytest.raises(FileNotFoundError, match=re.escape(f'File not found: {missing_file}')):
        upload_to_drive(missing_file, 'fake_file_id')


@patch('sdgym.run_benchmark.upload_benchmark_results.ResultsExplorer')
@patch('sdgym.run_benchmark.upload_benchmark_results.write_uploaded_marker')
@patch('sdgym.run_benchmark.upload_benchmark_results.LOGGER')
@patch('sdgym.run_benchmark.upload_benchmark_results.OUTPUT_DESTINATION_AWS')
@patch('sdgym.run_benchmark.upload_benchmark_results.LocalResultsWriter')
@patch('sdgym.run_benchmark.upload_benchmark_results.os.environ.get')
@patch('sdgym.run_benchmark.upload_benchmark_results.get_df_to_plot')
@patch('sdgym.run_benchmark.upload_benchmark_results.upload_to_drive')
@patch('sdgym.run_benchmark.upload_benchmark_results._extract_google_file_id')
def test_upload_results(
    mock_extract_google_file_id,
    mock_upload_to_drive,
    mock_get_df_to_plot,
    mock_os_environ_get,
    mock_local_results_writer,
    mock_output_destination_aws,
    mock_logger,
    mock_write_uploaded_marker,
    mock_sdgym_results_explorer,
):
    """Test the `upload_results` method."""
    # Setup
    aws_access_key_id = 'my_access_key'
    aws_secret_access_key = 'my_secret_key'
    folder_infos = {'folder_name': 'SDGym_results_10_01_2023', 'date': '10_01_2023'}
    run_name = folder_infos['folder_name']
    s3_client = Mock()
    bucket = 'bucket'
    prefix = 'prefix'
    result_explorer_instance = mock_sdgym_results_explorer.return_value
    result_explorer_instance.all_runs_complete.return_value = True
    result_explorer_instance.summarize.return_value = ('summary', 'results')
    mock_os_environ_get.return_value = '/tmp/sdgym_results'
    mock_get_df_to_plot.return_value = 'df_to_plot'
    datas = {
        'Wins': 'summary',
        '10_01_2023_Detailed_results': 'results',
        '10_01_2023_plot_data': 'df_to_plot',
    }
    local_path = str(Path('/tmp/sdgym_results/[single_table] SDGym Monthly Run.xlsx'))
    mock_extract_google_file_id.return_value = 'google_file_id'

    # Run
    upload_results(
        aws_access_key_id,
        aws_secret_access_key,
        folder_infos,
        s3_client,
        bucket,
        prefix,
        github_env=None,
    )

    # Assert
    mock_upload_to_drive.assert_called_once_with(local_path, 'google_file_id')
    mock_logger.info.assert_called_once_with(
        f'Run {run_name} is complete! Proceeding with summarization...'
    )
    mock_sdgym_results_explorer.assert_called_once_with(
        mock_output_destination_aws,
        modality='single_table',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    result_explorer_instance.all_runs_complete.assert_called_once_with(run_name)
    result_explorer_instance.summarize.assert_called_once_with(run_name)
    mock_write_uploaded_marker.assert_called_once_with(
        s3_client, bucket, prefix, run_name, modality='single_table'
    )
    mock_local_results_writer.return_value.write_xlsx.assert_called_once_with(datas, local_path)
    mock_get_df_to_plot.assert_called_once_with('results')


@patch('sdgym.run_benchmark.upload_benchmark_results.ResultsExplorer')
@patch('sdgym.run_benchmark.upload_benchmark_results.write_uploaded_marker')
@patch('sdgym.run_benchmark.upload_benchmark_results.LOGGER')
@patch('sdgym.run_benchmark.upload_benchmark_results.OUTPUT_DESTINATION_AWS')
def test_upload_results_not_all_runs_complete(
    mock_output_destination_aws,
    mock_logger,
    mock_write_uploaded_marker,
    mock_sdgym_results_explorer,
):
    """Test the `upload_results` when not all runs are complete."""
    # Setup
    aws_access_key_id = 'my_access_key'
    aws_secret_access_key = 'my_secret_key'
    folder_infos = {'folder_name': 'SDGym_results_10_01_2023', 'date': '10_01_2023'}
    run_name = folder_infos['folder_name']
    s3_client = Mock()
    bucket = 'bucket'
    prefix = 'prefix'
    result_explorer_instance = mock_sdgym_results_explorer.return_value
    result_explorer_instance.all_runs_complete.return_value = False
    result_explorer_instance.summarize.return_value = ('summary', 'results')

    # Run
    with pytest.raises(SystemExit, match='0'):
        upload_results(
            aws_access_key_id,
            aws_secret_access_key,
            folder_infos,
            s3_client,
            bucket,
            prefix,
            github_env=None,
        )

    # Assert
    mock_logger.warning.assert_called_once_with(f'Run {run_name} is not complete yet. Exiting.')
    mock_sdgym_results_explorer.assert_called_once_with(
        mock_output_destination_aws,
        modality='single_table',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    result_explorer_instance.all_runs_complete.assert_called_once_with(run_name)
    result_explorer_instance.summarize.assert_not_called()
    mock_write_uploaded_marker.assert_not_called()


@patch('sdgym.run_benchmark.upload_benchmark_results.get_result_folder_name_and_s3_vars')
@patch('sdgym.run_benchmark.upload_benchmark_results.upload_results')
@patch('sdgym.run_benchmark.upload_benchmark_results.upload_already_done')
@patch('sdgym.run_benchmark.upload_benchmark_results.LOGGER')
@patch('sdgym.run_benchmark.upload_benchmark_results.os.getenv')
@patch('sdgym.run_benchmark.upload_benchmark_results._parse_args')
def test_main_already_upload(
    mock_parse_args,
    mock_getenv,
    mock_logger,
    mock_upload_already_done,
    mock_upload_results,
    mock_get_result_folder_name_and_s3_vars,
):
    """Test the `method` when results are already uploaded."""
    # Setup
    mock_parse_args.return_value = Mock(modality='single_table')
    mock_getenv.side_effect = ['my_access_key', 'my_secret_key', None]
    folder_infos = {'folder_name': 'SDGym_results_10_01_2023', 'date': '10_01_2023'}
    mock_get_result_folder_name_and_s3_vars.return_value = (
        folder_infos,
        's3_client',
        'bucket',
        'prefix',
    )
    mock_upload_already_done.return_value = True
    expected_log_message = 'Benchmark results have already been uploaded. Exiting.'

    # Run
    with pytest.raises(SystemExit, match='0'):
        main()

    # Assert
    mock_parse_args.assert_called_once()
    mock_get_result_folder_name_and_s3_vars.assert_called_once_with(
        'my_access_key', 'my_secret_key', modality='single_table'
    )
    mock_logger.warning.assert_called_once_with(expected_log_message)
    mock_upload_results.assert_not_called()


@patch('sdgym.run_benchmark.upload_benchmark_results.get_result_folder_name_and_s3_vars')
@patch('sdgym.run_benchmark.upload_benchmark_results.upload_results')
@patch('sdgym.run_benchmark.upload_benchmark_results.upload_already_done')
@patch('sdgym.run_benchmark.upload_benchmark_results.os.getenv')
@patch('sdgym.run_benchmark.upload_benchmark_results._parse_args')
def test_main(
    mock_parse_args,
    mock_getenv,
    mock_upload_already_done,
    mock_upload_results,
    mock_get_result_folder_name_and_s3_vars,
):
    """Test the `main` method."""
    # Setup
    mock_parse_args.return_value = Mock(modality='single_table')
    mock_getenv.side_effect = ['my_access_key', 'my_secret_key', None]
    folder_infos = {'folder_name': 'SDGym_results_10_11_2024', 'date': '10_11_2024'}
    mock_get_result_folder_name_and_s3_vars.return_value = (
        folder_infos,
        's3_client',
        'bucket',
        'prefix',
    )
    mock_upload_already_done.return_value = False

    # Run
    main()

    # Assert
    mock_get_result_folder_name_and_s3_vars.assert_called_once_with(
        'my_access_key', 'my_secret_key', modality='single_table'
    )
    mock_upload_already_done.assert_called_once_with(
        's3_client', 'bucket', 'prefix', folder_infos['folder_name'], 'single_table'
    )
    mock_upload_results.assert_called_once_with(
        'my_access_key',
        'my_secret_key',
        folder_infos,
        's3_client',
        'bucket',
        'prefix',
        None,
        'single_table',
    )
