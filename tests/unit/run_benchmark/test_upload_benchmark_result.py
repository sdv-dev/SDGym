import io
import json
import re
from pathlib import Path
from unittest.mock import Mock, call, patch

import pandas as pd
import pytest
from botocore.exceptions import ClientError
from pandas.testing import assert_frame_equal

from sdgym.datasets import SDV_DATASETS_PRIVATE_BUCKET, SDV_DATASETS_PUBLIC_BUCKET
from sdgym.run_benchmark.upload_benchmark_results import (
    get_all_results,
    get_dataset_details,
    get_model_details,
    get_result_folder_name_and_s3_vars,
    get_upload_status,
    main,
    update_details_files,
    update_table_aws,
    upload_all_results,
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
    write_uploaded_marker(s3_client, bucket, prefix, run_name, modality)

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
    modality = 'single_table'
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
    result = upload_already_done(s3_client, bucket, prefix, run_name, modality)
    result_false = upload_already_done(s3_client, bucket, prefix, run_name, modality)
    with pytest.raises(ClientError):
        upload_already_done(s3_client, bucket, prefix, run_name, modality)

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


@patch('sdgym.run_benchmark.upload_benchmark_results.DatasetExplorer')
@patch(
    'sdgym.run_benchmark.upload_benchmark_results.DATASET_DETAILS_COLUMNS',
    ['Dataset', 'Rows', 'Availability', 'Best Model', 'Type'],
)
def test_get_dataset_details(mock_dataset_explorer):
    """Test the `get_dataset_details` method"""
    # Setup
    aws_access_key_id = 'access'
    aws_secret_access_key = 'secret'
    modality = 'single_table'
    dataset_bucket = [SDV_DATASETS_PUBLIC_BUCKET, SDV_DATASETS_PRIVATE_BUCKET]
    results = pd.DataFrame({
        'Dataset': ['A', 'A', 'B', 'C', 'D'],
        'Synthesizer': [
            'GaussianCopulaSynthesizer',
            'TVAESynthesizer',
            'GaussianCopulaSynthesizer',
            'CTGANSynthesizer',
            'CopulaGANSynthesizer',
        ],
        'Adjusted_Quality_Score': [0.7, 0.9, 0.5, 0.8, 0.1],
    })

    public_explorer = Mock()
    private_explorer = Mock()
    mock_dataset_explorer.side_effect = [public_explorer, private_explorer]

    public_explorer.summarize_datasets.return_value = pd.DataFrame({
        'Dataset': ['A', 'B'],
        'Rows': [10, 20],
    })
    private_explorer.summarize_datasets.return_value = pd.DataFrame({
        'Dataset': ['C'],
        'Rows': [30],
    })

    # Run
    dataset_details = get_dataset_details(
        results, modality, aws_access_key_id, aws_secret_access_key
    )

    # Assert
    assert mock_dataset_explorer.call_count == 2
    for _call, bucket in zip(mock_dataset_explorer.call_args_list, dataset_bucket):
        assert _call.kwargs == {
            's3_url': bucket,
            'aws_access_key_id': aws_access_key_id,
            'aws_secret_access_key': aws_secret_access_key,
        }

    public_explorer.summarize_datasets.assert_called_once_with(modality=modality)
    private_explorer.summarize_datasets.assert_called_once_with(modality=modality)
    expected = pd.DataFrame({
        'Dataset': ['A', 'B', 'C'],
        'Rows': [10, 20, 30],
        'Availability': ['Public', 'Public', 'Private'],
        'Best Model': ['TVAESynthesizer', 'GaussianCopulaSynthesizer', 'CTGANSynthesizer'],
        'Type': [modality, modality, modality],
    })
    assert_frame_equal(
        dataset_details.sort_values('Dataset').reset_index(drop=True),
        expected.sort_values('Dataset').reset_index(drop=True),
        check_dtype=False,
    )


@patch('sdgym.run_benchmark.upload_benchmark_results.DatasetExplorer')
@patch(
    'sdgym.run_benchmark.upload_benchmark_results.DATASET_DETAILS_COLUMNS',
    ['Dataset', 'Rows', 'Availability', 'Best Model', 'Type'],
)
def test_get_dataset_details_returns_empty_when_no_datasets_found(mock_dataset_explorer):
    """Test the `get_dataset_details` method returns empty DataFrame when no datasets are found."""
    # Setup
    results = pd.DataFrame({
        'Dataset': ['A', 'B'],
        'Synthesizer': ['S1', 'S2'],
        'Quality_Score': [0.1, 0.2],
    })
    modality = 'single_table'

    public_explorer = Mock()
    private_explorer = Mock()
    mock_dataset_explorer.side_effect = [public_explorer, private_explorer]
    public_explorer.summarize_datasets.return_value = pd.DataFrame({
        'Dataset': ['X'],
        'Rows': [999],
    })
    private_explorer.summarize_datasets.return_value = pd.DataFrame({
        'Dataset': ['Y'],
        'Rows': [999],
    })

    # Run
    out = get_dataset_details(results, modality, 'access', 'secret')

    # Assert
    assert list(out.columns) == ['Dataset', 'Rows', 'Availability', 'Best Model', 'Type']
    assert out.empty is True


@patch(
    'sdgym.run_benchmark.upload_benchmark_results.EXTERNAL_SYNTHESIZER_TO_LIBRARY',
    {'GaussianCopulaSynthesizer': 'external_lib'},
)
def test_get_model_details():
    """Test the `get_model_details` method."""
    # Setup
    modality = 'single_table'
    synthesizer_description = {
        'GaussianCopulaSynthesizer': {'type': 'GAN', 'description': 'A model A'},
    }

    summary = pd.DataFrame({
        'Synthesizer': ['GaussianCopulaSynthesizer', 'CTGANSynthesizer'],
        'Wins': [2, 0],
    })

    results = pd.DataFrame({
        'Dataset': ['D1', 'D2', 'D3', 'D1', 'D2'],
        'Synthesizer': [
            'GaussianCopulaSynthesizer',
            'GaussianCopulaSynthesizer',
            'GaussianCopulaSynthesizer',
            'CTGANSynthesizer',
            'CTGANSynthesizer',
        ],
        'Quality_Score': [0.1, 0.2, 0.3, 0.15, 0.25],
        'error': [
            'Synthesizer Timeout',  # timeout on D1 for GaussianCopulaSynthesizer
            'Other Error',  # error on D2 for GaussianCopulaSynthesizer
            None,  # no error on D3 for GaussianCopulaSynthesizer
            None,
            None,
        ],
    })
    df_to_plot = pd.DataFrame({
        'Synthesizer': ['GaussianCopula', 'CTGAN'],
        'Pareto': [True, False],
    })

    # Run
    model_details = get_model_details(
        summary, results, df_to_plot, modality, synthesizer_description
    )

    # Assert
    model_idx = model_details.set_index('Synthesizer')

    assert model_idx.loc['GaussianCopulaSynthesizer', 'Data Type'] == modality
    assert model_idx.loc['CTGANSynthesizer', 'Data Type'] == modality
    assert model_idx.loc['GaussianCopulaSynthesizer', 'Source'] == 'external_lib'
    assert model_idx.loc['CTGANSynthesizer', 'Source'] == 'sdv'
    assert model_idx.loc['GaussianCopulaSynthesizer', 'Type'] == 'GAN'
    assert model_idx.loc['GaussianCopulaSynthesizer', 'Description'] == 'A model A'
    assert model_idx.loc['CTGANSynthesizer', 'Type'] == 'Unknown'
    assert model_idx.loc['CTGANSynthesizer', 'Description'] == 'No description available.'
    assert model_idx.loc['GaussianCopulaSynthesizer', 'Number of dataset - Wins'] == 2
    assert model_idx.loc['CTGANSynthesizer', 'Number of dataset - Wins'] == 0
    assert model_idx.loc['GaussianCopulaSynthesizer', 'Number of dataset - Timeout'] == 1
    assert model_idx.loc['GaussianCopulaSynthesizer', 'Number of dataset - Errors'] == 1
    assert model_idx.loc['CTGANSynthesizer', 'Number of dataset - Timeout'] == 0
    assert model_idx.loc['CTGANSynthesizer', 'Number of dataset - Errors'] == 0
    assert model_idx.loc['GaussianCopulaSynthesizer', 'On the Pareto Curve']
    assert not model_idx.loc['CTGANSynthesizer', 'On the Pareto Curve']


def test_update_table_aws_merges_with_existing_table():
    """Test the `update_table_aws` method."""
    # Setup
    s3_client = Mock()
    bucket = 'bucket'
    filename = 'prefix/table.xlsx'
    reference_column = 'Dataset'

    def _excel_bytes_from_df(df):
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        buffer.seek(0)
        return buffer.getvalue()

    existing = pd.DataFrame({'Dataset': ['A', 'B'], 'Rows': [10, 20]})
    new = pd.DataFrame({'Dataset': ['B', 'C'], 'Rows': [999, 30]})

    body = Mock()
    body.read.return_value = _excel_bytes_from_df(existing)
    s3_client.get_object.return_value = {'Body': body}

    # Run
    updated = update_table_aws(s3_client, bucket, filename, new, reference_column)

    # Assert
    expected = pd.DataFrame({'Dataset': ['A', 'B', 'C'], 'Rows': [10, 999, 30]})
    assert_frame_equal(
        updated.sort_values('Dataset').reset_index(drop=True),
        expected.sort_values('Dataset').reset_index(drop=True),
    )
    s3_client.get_object.assert_called_once_with(Bucket=bucket, Key=filename)
    assert s3_client.upload_fileobj.call_count == 1
    args, _ = s3_client.upload_fileobj.call_args
    assert args[1] == bucket
    assert args[2] == filename


def test_update_table_aws_creates_new_table_when_missing_key():
    """Test the `update_table_aws` method uses `table` when S3 key is missing."""
    # Setup
    s3_client = Mock()
    bucket = 'bucket'
    filename = 'prefix/table.xlsx'
    reference_column = 'Dataset'

    table = pd.DataFrame({'Dataset': ['A'], 'Rows': [10]})
    s3_client.get_object.side_effect = ClientError(
        error_response={'Error': {'Code': 'NoSuchKey', 'Message': 'Not Found'}},
        operation_name='GetObject',
    )

    # Run
    updated = update_table_aws(s3_client, bucket, filename, table, reference_column)

    # Assert
    assert_frame_equal(updated.reset_index(drop=True), table.reset_index(drop=True))
    assert s3_client.upload_fileobj.call_count == 1


def test_update_table_aws_raises_for_unexpected_client_error():
    """Test the `update_table_aws` method raises for non-NoSuchKey errors."""
    # Setup
    s3_client = Mock()
    s3_client.get_object.side_effect = ClientError(
        error_response={'Error': {'Code': 'AccessDenied', 'Message': 'Denied'}},
        operation_name='GetObject',
    )

    # Run and Assert
    with pytest.raises(ClientError):
        update_table_aws(
            s3_client,
            bucket='bucket',
            filename='prefix/table.xlsx',
            table=pd.DataFrame({'Dataset': ['A'], 'Rows': [1]}),
            reference_column='Dataset',
        )


@patch('sdgym.run_benchmark.upload_benchmark_results.update_table_aws')
def test_update_details_file(mock_update_table_aws, tmp_path):
    """Test the `update_details_files` method."""
    # Setup
    s3_client = Mock()
    bucket = 'bucket'
    prefix = 'prefix/'

    df_updated_1 = Mock(spec=pd.DataFrame)
    df_updated_2 = Mock(spec=pd.DataFrame)
    mock_update_table_aws.side_effect = [df_updated_1, df_updated_2]

    details_list = [
        (pd.DataFrame({'Dataset': ['A']}), 'Dataset_Details.xlsx', 'Dataset'),
        (pd.DataFrame({'Synthesizer': ['S']}), 'Model_Details.xlsx', 'Synthesizer'),
    ]

    # Run
    update_details_files(s3_client, bucket, prefix, str(tmp_path), details_list)

    # Assert
    mock_update_table_aws.assert_has_calls([
        call(s3_client, bucket, f'{prefix}Dataset_Details.xlsx', details_list[0][0], 'Dataset'),
        call(
            s3_client,
            bucket,
            f'{prefix}Model_Details.xlsx',
            details_list[1][0],
            'Synthesizer',
        ),
    ])
    df_updated_1.to_excel.assert_called_once_with(
        Path(tmp_path) / 'Dataset_Details.xlsx', index=False
    )
    df_updated_2.to_excel.assert_called_once_with(
        Path(tmp_path) / 'Model_Details.xlsx', index=False
    )


@patch('sdgym.run_benchmark.upload_benchmark_results.update_table_aws')
def test_update_details_files_updates_s3_without_local_export(mock_update_table_aws):
    """Test the `update_details_files` method does not write locally when no dir is provided."""
    # Setup
    s3_client = Mock()
    bucket = 'bucket'
    prefix = 'prefix/'

    df_updated = Mock(spec=pd.DataFrame)
    mock_update_table_aws.return_value = df_updated

    details_list = [(pd.DataFrame({'Dataset': ['A']}), 'Dataset_Details.xlsx', 'Dataset')]

    # Run
    update_details_files(s3_client, bucket, prefix, None, details_list)

    # Assert
    mock_update_table_aws.assert_called_once_with(
        s3_client, bucket, f'{prefix}Dataset_Details.xlsx', details_list[0][0], 'Dataset'
    )
    df_updated.to_excel.assert_not_called()


@patch('sdgym.run_benchmark.upload_benchmark_results.get_model_details')
@patch('sdgym.run_benchmark.upload_benchmark_results.get_df_to_plot')
@patch('sdgym.run_benchmark.upload_benchmark_results.get_dataset_details')
@patch('sdgym.run_benchmark.upload_benchmark_results.SYNTHESIZER_DESCRIPTION', {'X': {}})
def test_get_all_results_mock(
    mock_get_dataset_details,
    mock_get_df_to_plot,
    mock_get_model_details,
):
    """Test the `get_all_results` method."""
    # Setup
    result_explorer = Mock()
    folder_name = 'SDGym_results_10_01_2023'
    modality = 'single_table'
    aws_access_key_id = 'access'
    aws_secret_access_key = 'secret'

    summary = Mock()
    results = Mock()
    df_to_plot = Mock()
    dataset_details = Mock()
    model_details = Mock()

    result_explorer.summarize.return_value = (summary, results)
    mock_get_dataset_details.return_value = dataset_details
    mock_get_df_to_plot.return_value = df_to_plot
    mock_get_model_details.return_value = model_details

    # Run
    output = get_all_results(
        result_explorer, folder_name, modality, aws_access_key_id, aws_secret_access_key
    )

    # Assert
    result_explorer.summarize.assert_called_once_with(folder_name)
    mock_get_dataset_details.assert_called_once_with(
        results, modality, aws_access_key_id, aws_secret_access_key
    )
    mock_get_df_to_plot.assert_called_once_with(results)
    mock_get_model_details.assert_called_once_with(
        summary, results, df_to_plot, modality, {'X': {}}
    )
    assert output == (summary, results, df_to_plot, dataset_details, model_details)


@patch('sdgym.run_benchmark.upload_benchmark_results.upload_to_drive')
@patch('sdgym.run_benchmark.upload_benchmark_results._extract_google_file_id')
@patch('sdgym.run_benchmark.upload_benchmark_results.update_details_files')
@patch('sdgym.run_benchmark.upload_benchmark_results.LocalResultsWriter')
@patch('sdgym.run_benchmark.upload_benchmark_results.os.environ.get')
@patch('sdgym.run_benchmark.upload_benchmark_results.SDGYM_RUNS_FILENAME', 'SDGym_Runs.xlsx')
@patch(
    'sdgym.run_benchmark.upload_benchmark_results.DATASET_DETAILS_FILENAME', 'Dataset_Details.xlsx'
)
@patch('sdgym.run_benchmark.upload_benchmark_results.MODEL_DETAILS_FILENAME', 'Model_Details.xlsx')
def test_upload_all_results_writes_and_uploads_and_uploads_to_drive(
    mock_environ_get,
    mock_local_results_writer,
    mock_update_details_files,
    mock_extract_google_file_id,
    mock_upload_to_drive,
    tmp_path,
):
    """Test the `upload_all_results` method."""
    # Setup
    modality = 'single_table'
    s3_client = Mock()
    bucket = 'bucket'
    prefix = 'prefix/'

    datas = {'Wins': Mock()}
    dataset_details = Mock()
    model_details = Mock()
    mock_environ_get.return_value = str(tmp_path)
    s3_client.download_file.side_effect = ClientError(
        error_response={'Error': {'Code': '404', 'Message': 'Not Found'}},
        operation_name='DownloadFile',
    )

    with patch(
        'sdgym.run_benchmark.upload_benchmark_results.FILE_TO_GDRIVE_LINK',
        {
            '[Multi-table]_SDGym_Runs.xlsx': 'skip_link',
            'Dataset_Details.xlsx': 'dataset_link',
            'Model_Details.xlsx': 'model_link',
        },
    ):
        mock_extract_google_file_id.side_effect = ['dataset_id', 'model_id']

        # Run
        out_dir = upload_all_results(
            datas, dataset_details, model_details, modality, s3_client, bucket, prefix
        )

    # Assert
    assert out_dir == str(tmp_path)
    local_writer_instance = mock_local_results_writer.return_value
    expected_runs_filename = '[Single-table]_SDGym_Runs.xlsx'
    expected_local_result = str(Path(tmp_path) / expected_runs_filename)
    expected_s3_key_result = f'{prefix}{expected_runs_filename}'

    s3_client.download_file.assert_called_once_with(
        bucket, expected_s3_key_result, expected_local_result
    )
    local_writer_instance.write_xlsx.assert_called_once_with(datas, expected_local_result)
    mock_update_details_files.assert_called_once_with(
        s3_client,
        bucket,
        prefix,
        str(tmp_path),
        [
            (dataset_details, 'Dataset_Details.xlsx', 'Dataset'),
            (model_details, 'Model_Details.xlsx', 'Synthesizer'),
        ],
    )
    s3_client.upload_file.assert_called_once_with(
        expected_local_result, bucket, expected_s3_key_result
    )
    mock_extract_google_file_id.assert_has_calls([call('dataset_link'), call('model_link')])
    mock_upload_to_drive.assert_has_calls(
        [
            call(str(Path(tmp_path) / 'Dataset_Details.xlsx'), 'dataset_id'),
            call(str(Path(tmp_path) / 'Model_Details.xlsx'), 'model_id'),
        ],
        any_order=True,
    )


@patch('sdgym.run_benchmark.upload_benchmark_results.ResultsExplorer')
@patch('sdgym.run_benchmark.upload_benchmark_results.LOGGER')
@patch('sdgym.run_benchmark.upload_benchmark_results.OUTPUT_DESTINATION_AWS')
def test_get_upload_status_exits_and_sets_skip_upload_true(
    mock_output_destination_aws,
    mock_logger,
    mock_results_explorer,
    tmp_path,
):
    """Test the `get_upload_status` method exits when runs are not complete and writes env file."""
    # Setup
    folder_name = 'SDGym_results_10_01_2023'
    modality = 'single_table'
    aws_access_key_id = 'access'
    aws_secret_access_key = 'secret'
    github_env = str(tmp_path / 'github.env')

    explorer_instance = mock_results_explorer.return_value
    explorer_instance.all_runs_complete.return_value = False

    # Run and Assert
    with pytest.raises(SystemExit, match='0'):
        get_upload_status(
            folder_name, modality, aws_access_key_id, aws_secret_access_key, github_env
        )

    mock_logger.warning.assert_called_once_with(f'Run {folder_name} is not complete yet. Exiting.')
    explorer_instance.all_runs_complete.assert_called_once_with(folder_name)

    env_content = Path(github_env).read_text()
    assert env_content == 'SKIP_UPLOAD=true\n'


@patch('sdgym.run_benchmark.upload_benchmark_results.ResultsExplorer')
@patch('sdgym.run_benchmark.upload_benchmark_results.LOGGER')
@patch('sdgym.run_benchmark.upload_benchmark_results.OUTPUT_DESTINATION_AWS')
def test_get_upload_status_returns_explorer_and_writes_env(
    mock_output_destination_aws,
    mock_logger,
    mock_results_explorer,
    tmp_path,
):
    """Test the `get_upload_status` method returns explorer and writes env vars when complete."""
    # Setup
    folder_name = 'SDGym_results_10_01_2023'
    modality = 'single_table'
    aws_access_key_id = 'access'
    aws_secret_access_key = 'secret'
    github_env = str(tmp_path / 'github.env')

    explorer_instance = mock_results_explorer.return_value
    explorer_instance.all_runs_complete.return_value = True

    # Run
    out = get_upload_status(
        folder_name, modality, aws_access_key_id, aws_secret_access_key, github_env
    )

    # Assert
    assert out == explorer_instance
    mock_logger.info.assert_called_once_with(
        f'Run {folder_name} is complete! Proceeding with summarization...'
    )
    explorer_instance.all_runs_complete.assert_called_once_with(folder_name)

    env_content = Path(github_env).read_text()
    assert env_content == f'SKIP_UPLOAD=false\nFOLDER_NAME={folder_name}\n'


@patch('sdgym.run_benchmark.upload_benchmark_results.ResultsExplorer')
@patch('sdgym.run_benchmark.upload_benchmark_results.write_uploaded_marker')
@patch('sdgym.run_benchmark.upload_benchmark_results.LOGGER')
@patch('sdgym.run_benchmark.upload_benchmark_results.OUTPUT_DESTINATION_AWS')
@patch('sdgym.run_benchmark.upload_benchmark_results.LocalResultsWriter')
@patch('sdgym.run_benchmark.upload_benchmark_results.os.environ.get')
@patch('sdgym.run_benchmark.upload_benchmark_results.get_df_to_plot')
@patch('sdgym.run_benchmark.upload_benchmark_results.upload_to_drive')
@patch('sdgym.run_benchmark.upload_benchmark_results._extract_google_file_id')
@patch('sdgym.run_benchmark.upload_benchmark_results.get_dataset_details')
@patch('sdgym.run_benchmark.upload_benchmark_results.get_model_details')
@patch('sdgym.run_benchmark.upload_benchmark_results.update_details_files')
@patch('sdgym.run_benchmark.upload_benchmark_results.SYNTHESIZER_DESCRIPTION')
def test_upload_results(
    mock_synthesizer_description,
    mock_update_details_files,
    mock_get_model_details,
    mock_get_dataset_details,
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
    summary = Mock()
    result_details = Mock()
    df_to_plot = Mock()
    dataset_details = Mock()
    result_explorer_instance = mock_sdgym_results_explorer.return_value
    result_explorer_instance.all_runs_complete.return_value = True
    result_explorer_instance.summarize.return_value = (summary, result_details)
    mock_os_environ_get.return_value = '/tmp/sdgym_results'
    mock_get_df_to_plot.return_value = df_to_plot
    datas = {
        'Wins': summary,
        '10_01_2023_Detailed_results': result_details,
        '10_01_2023_plot_data': df_to_plot,
    }
    local_path = str(Path('/tmp/sdgym_results/[Single-table]_SDGym_Runs.xlsx'))
    dataset_path = str(Path('/tmp/sdgym_results/Dataset_Details.xlsx'))
    model_path = str(Path('/tmp/sdgym_results/Model_Details.xlsx'))
    mock_extract_google_file_id.side_effect = ['Result_file_id', 'Dataset_file_id', 'Model_file_id']
    model_details = Mock()
    mock_get_dataset_details.return_value = dataset_details
    mock_get_model_details.return_value = model_details
    details = [
        (dataset_details, 'Dataset_Details.xlsx', 'Dataset'),
        (model_details, 'Model_Details.xlsx', 'Synthesizer'),
    ]

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
    mock_update_details_files.assert_called_once_with(
        s3_client, bucket, prefix, '/tmp/sdgym_results', details
    )
    mock_get_dataset_details.assert_called_once_with(
        result_details, 'single_table', aws_access_key_id, aws_secret_access_key
    )
    mock_get_model_details.assert_called_once_with(
        summary,
        result_details,
        df_to_plot,
        'single_table',
        mock_synthesizer_description,
    )
    mock_upload_to_drive.assert_has_calls([
        call(local_path, 'Result_file_id'),
        call(dataset_path, 'Dataset_file_id'),
        call(model_path, 'Model_file_id'),
    ])
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
    mock_get_df_to_plot.assert_called_once_with(result_details)


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


@patch('sdgym.run_benchmark.upload_benchmark_results.get_upload_status')
@patch('sdgym.run_benchmark.upload_benchmark_results.get_all_results')
@patch('sdgym.run_benchmark.upload_benchmark_results.upload_all_results')
@patch('sdgym.run_benchmark.upload_benchmark_results.write_uploaded_marker')
def test_upload_results_mock(
    mock_write_uploaded_marker,
    mock_upload_all_results,
    mock_get_all_results,
    mock_get_upload_status,
):
    """Test the `upload_results` method with mocks for internal functions."""
    # Setup
    aws_access_key_id = 'my_access_key'
    aws_secret_access_key = 'my_secret_key'
    folder_infos = {'folder_name': 'SDGym_results_10_01_2023', 'date': '10_01_2023'}
    s3_client = Mock()
    bucket = 'bucket'
    prefix = 'prefix'
    summary_data = Mock()
    detailed_data = Mock()
    plot_data = Mock()
    dataset_details = Mock()
    model_details = Mock()
    result_explorer_instance = Mock()
    mock_get_all_results.return_value = (
        summary_data,
        detailed_data,
        plot_data,
        dataset_details,
        model_details,
    )
    mock_get_upload_status.return_value = result_explorer_instance
    mock_upload_all_results.return_value = None
    expected_datas = {
        'Wins': summary_data,
        '10_01_2023_Detailed_results': detailed_data,
        '10_01_2023_plot_data': plot_data,
    }

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
    mock_get_upload_status.assert_called_once_with(
        'SDGym_results_10_01_2023', 'single_table', aws_access_key_id, aws_secret_access_key, None
    )
    mock_get_all_results.assert_called_once_with(
        result_explorer_instance,
        'SDGym_results_10_01_2023',
        'single_table',
        aws_access_key_id,
        aws_secret_access_key,
    )
    mock_upload_all_results.assert_called_once_with(
        expected_datas, dataset_details, model_details, 'single_table', s3_client, bucket, prefix
    )
    mock_write_uploaded_marker.assert_called_once_with(
        s3_client, bucket, prefix, folder_infos['folder_name'], modality='single_table'
    )


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
