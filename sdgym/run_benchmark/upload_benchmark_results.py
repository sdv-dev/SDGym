"""Script to upload benchmark results to S3."""

import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
import pandas as pd
from sdgym import DatasetExplorer
import boto3
from botocore.exceptions import ClientError
from oauth2client.client import OAuth2Credentials
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from sdgym.datasets import SDV_DATASETS_PRIVATE_BUCKET, SDV_DATASETS_PUBLIC_BUCKET
from sdgym.result_explorer.result_explorer import ResultsExplorer
from sdgym.result_writer import LocalResultsWriter
from sdgym.run_benchmark.utils import (
    FILE_TO_GDRIVE_LINK,
    OUTPUT_DESTINATION_AWS,
    _extract_google_file_id,
    _parse_args,
    get_df_to_plot,
)
from sdgym.s3 import S3_REGION, parse_s3_path

LOGGER = logging.getLogger(__name__)
SYNTHESIZER_TO_GLOBAL_POSITION = {
    'CTGAN': 'middle right',
    'TVAE': 'middle left',
    'GaussianCopula': 'bottom center',
    'Uniform': 'top center',
    'Column': 'top center',
    'CopulaGAN': 'top center',
    'RealTabFormer': 'bottom center',
    'HSA': 'bottom center',
    'Independent': 'top center',
}
RESULT_FILENAME = 'SDGym Run.xlsx'
DATASET_DETAILS_NAME = 'Dataset Details.xlsx'
DATASET_DETAILS_COLUMNS = [
    'Dataset', 'Type', 'Best Model', 'Total_Num_Columns', 'Total_Num_Rows',
    'Total_Num_Columns_Categorical', 'Total_Num_Columns_Numerical',
    'Num_Tables', 'Num_Relationships', 'Max_Schema_Depth',
    'Availability', 'Datasize_Size_MB',
]


def get_latest_run_from_file(s3_client, bucket, key):
    """Get the latest run folder name from the benchmark dates file in S3."""
    try:
        object = s3_client.get_object(Bucket=bucket, Key=key)
        body = object['Body'].read().decode('utf-8')
        data = json.loads(body)
        latest = sorted(data['runs'], key=lambda x: x['date'])[-1]
        return latest
    except s3_client.exceptions.ClientError as e:
        raise RuntimeError(f'Failed to read {key} from S3: {e}')


def write_uploaded_marker(s3_client, bucket, prefix, folder_name, modality='single_table'):
    """Write a marker file to indicate that the upload is complete."""
    s3_client.put_object(
        Bucket=bucket,
        Key=f'{prefix}{modality}/{folder_name}/upload_complete.marker',
        Body=b'Upload complete',
    )


def upload_already_done(s3_client, bucket, prefix, folder_name, modality='single_table'):
    """Check if the upload has already been done by looking for the marker file."""
    try:
        s3_client.head_object(
            Bucket=bucket, Key=f'{prefix}{modality}/{folder_name}/upload_complete.marker'
        )
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False

        raise


def get_result_folder_name_and_s3_vars(
    aws_access_key_id, aws_secret_access_key, modality='single_table'
):
    """Get the result folder name and S3 client variables."""
    bucket, prefix = parse_s3_path(OUTPUT_DESTINATION_AWS)
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=S3_REGION,
    )
    folder_infos = get_latest_run_from_file(
        s3_client, bucket, f'{prefix}{modality}/_BENCHMARK_DATES.json'
    )

    return folder_infos, s3_client, bucket, prefix


def upload_to_drive(file_path, file_id):
    """Upload a local file to a Google Drive folder.

    Args:
        file_path (str or Path): Path to the local file to upload.
        file_id (str): Google Drive file ID.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f'File not found: {file_path}')

    creds_dict = json.loads(os.environ['PYDRIVE_TOKEN'])
    creds = OAuth2Credentials(
        access_token=creds_dict['access_token'],
        client_id=creds_dict.get('client_id'),
        client_secret=creds_dict.get('client_secret'),
        refresh_token=creds_dict.get('refresh_token'),
        token_expiry=None,
        token_uri='https://oauth2.googleapis.com/token',
        user_agent=None,
    )
    gauth = GoogleAuth()
    gauth.credentials = creds
    drive = GoogleDrive(gauth)

    gfile = drive.CreateFile({'id': file_id})
    gfile.SetContentFile(file_path)
    gfile.Upload(param={'supportsAllDrives': True})


def get_dataset_details(results, modality, aws_access_key_id, aws_secret_access_key):
    dataset_list = results['Dataset'].unique().tolist()
    explorers = {
        'Public': DatasetExplorer(
            s3_url=SDV_DATASETS_PUBLIC_BUCKET,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        ),
        'Private': DatasetExplorer(
            s3_url=SDV_DATASETS_PRIVATE_BUCKET,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        ),
    }

    dataset_infos = []
    remaining_datasets = set(dataset_list)
    for availability, explorer in explorers.items():
        summary = explorer.summarize_datasets(modality=modality)
        summary = (
            summary
            .set_index('Dataset')
            .reindex(remaining_datasets)
            .dropna(how='all')
            .reset_index()
        )
        if summary.empty:
            continue

        summary['Availability'] = availability
        dataset_infos.append(summary)
        remaining_datasets -= set(summary['Dataset'])

    if not dataset_infos:
        return pd.DataFrame(columns=DATASET_DETAILS_COLUMNS)

    dataset_infos = pd.concat(dataset_infos, ignore_index=True)
    best_model_map = (
        results
        .sort_values('Quality_Score', ascending=False)
        .groupby('Dataset')['Synthesizer']
        .first()
    )
    dataset_infos['Best Model'] = dataset_infos['Dataset'].map(best_model_map)
    dataset_infos['Type'] = modality

    return dataset_infos[DATASET_DETAILS_COLUMNS]


def upload_results(
    aws_access_key_id,
    aws_secret_access_key,
    folder_infos,
    s3_client,
    bucket,
    prefix,
    github_env,
    modality='single_table',
):
    """Upload benchmark results to S3, GDrive, and save locally."""
    folder_name = folder_infos['folder_name']
    run_date = folder_infos['date']
    result_explorer = ResultsExplorer(
        OUTPUT_DESTINATION_AWS,
        modality=modality,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    local_results_writer = LocalResultsWriter()
    if not result_explorer.all_runs_complete(folder_name):
        LOGGER.warning(f'Run {folder_name} is not complete yet. Exiting.')
        if github_env:
            with open(github_env, 'a') as env_file:
                env_file.write('SKIP_UPLOAD=true\n')

        sys.exit(0)

    LOGGER.info(f'Run {folder_name} is complete! Proceeding with summarization...')
    if github_env:
        with open(github_env, 'a') as env_file:
            env_file.write('SKIP_UPLOAD=false\n')
            env_file.write(f'FOLDER_NAME={folder_name}\n')

    summary, results = result_explorer.summarize(folder_name)
    dataset_details = get_dataset_details(
        results, modality, aws_access_key_id, aws_secret_access_key
    )
    df_to_plot = get_df_to_plot(results)
    local_export_dir = os.environ.get('GITHUB_LOCAL_RESULTS_DIR')
    temp_dir = None
    if not local_export_dir:
        temp_dir = tempfile.mkdtemp()
        local_export_dir = temp_dir

    Path(local_export_dir).mkdir(parents=True, exist_ok=True)
    local_filepath_result = str(Path(local_export_dir) / f'[{modality}] {RESULT_FILENAME}')
    local_filepath_dataset_details = str(Path(local_export_dir) / DATASET_DETAILS_NAME)
    s3_key_result = f'{prefix}{modality}/{RESULT_FILENAME}'
    s3_key_dataset_details = f'{prefix}/{DATASET_DETAILS_NAME}'
    try:
        s3_client.download_file(bucket, s3_key_result, local_filepath_result)
    except ClientError as e:
        if not e.response['Error']['Code'] == '404':
            raise

    try:
        s3_client.download_file(bucket, s3_key_dataset_details, local_filepath_dataset_details)
    except ClientError as e:
        if not e.response['Error']['Code'] == '404':
            raise

    datas = {
        'Wins': summary,
        f'{run_date}_Detailed_results': results,
        f'{run_date}_plot_data': df_to_plot,
    }
    local_results_writer.write_xlsx(datas, local_filepath_result)
    upload_to_drive((local_filepath_result), _extract_google_file_id(FILE_TO_GDRIVE_LINK[modality]))
    s3_client.upload_file(local_filepath_result, bucket, s3_key_result)
    write_uploaded_marker(s3_client, bucket, prefix, folder_name, modality=modality)
    if temp_dir:
        shutil.rmtree(temp_dir)


def main():
    """Main function to upload benchmark results."""
    args = _parse_args()
    modality = args.modality
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    folder_infos, s3_client, bucket, prefix = get_result_folder_name_and_s3_vars(
        aws_access_key_id, aws_secret_access_key, modality=modality
    )
    github_env = os.getenv('GITHUB_ENV')
    if upload_already_done(s3_client, bucket, prefix, folder_infos['folder_name'], modality):
        LOGGER.warning('Benchmark results have already been uploaded. Exiting.')
        if github_env:
            with open(github_env, 'a') as env_file:
                env_file.write('SKIP_UPLOAD=true\n')

        sys.exit(0)

    upload_results(
        aws_access_key_id,
        aws_secret_access_key,
        folder_infos,
        s3_client,
        bucket,
        prefix,
        github_env,
        modality,
    )


if __name__ == '__main__':
    main()
