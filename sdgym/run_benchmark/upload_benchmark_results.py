"""Script to upload benchmark results to S3."""

import io
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import boto3
import pandas as pd
import yaml
from botocore.exceptions import ClientError
from oauth2client.client import OAuth2Credentials
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

from sdgym import DatasetExplorer
from sdgym.datasets import SDV_DATASETS_PRIVATE_BUCKET, SDV_DATASETS_PUBLIC_BUCKET
from sdgym.result_explorer.result_explorer import ResultsExplorer
from sdgym.result_writer import LocalResultsWriter
from sdgym.run_benchmark.utils import (
    OUTPUT_DESTINATION_AWS,
    _extract_google_file_id,
    _get_filename_to_gdrive_link,
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
SDGYM_RUNS_FILENAME = 'SDGym_Runs.xlsx'
MODEL_DETAILS_FILENAME = 'Model_Details.xlsx'
DATASET_DETAILS_FILENAME = 'Dataset_Details.xlsx'
SYNTHESIZER_DESCRIPTION_PATH = 'sdgym/synthesizer_descriptions.yaml'
DATASET_DETAILS_COLUMNS = [
    'Dataset',
    'Data Modality',
    'Best Model',
    'Total_Num_Columns',
    'Total_Num_Rows',
    'Total_Num_Columns_Categorical',
    'Total_Num_Columns_Numerical',
    'Num_Tables',
    'Num_Relationships',
    'Max_Schema_Depth',
    'Availability',
    'Datasize_Size_MB',
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


def write_uploaded_marker(s3_client, bucket, prefix, folder_name, modality):
    """Write a marker file to indicate that the upload is complete."""
    s3_client.put_object(
        Bucket=bucket,
        Key=f'{prefix}{modality}/{folder_name}/upload_complete.marker',
        Body=b'Upload complete',
    )


def upload_already_done(s3_client, bucket, prefix, folder_name, modality):
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
    """Get dataset details DataFrame.

    Based on the generated `results`, create a DataFrame containing details about each dataset
    used in the benchmark.

    Args:
        results (`pd.DataFrame`):
            Detailed results DataFrame.
        modality (str):
            Benchmark modality.
        aws_access_key_id (str):
            AWS access key ID.
        aws_secret_access_key (str):
            AWS secret access key.

    Returns:
        `pd.DataFrame`:
            Dataset details DataFrame.
    """
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
            summary.set_index('Dataset').reindex(remaining_datasets).dropna(how='all').reset_index()
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
        .sort_values('Adjusted_Quality_Score', ascending=False)
        .groupby('Dataset')['Synthesizer']
        .first()
    )
    dataset_infos['Best Model'] = dataset_infos['Dataset'].map(best_model_map)
    dataset_infos['Data Modality'] = modality

    return dataset_infos[DATASET_DETAILS_COLUMNS]


def get_model_details(summary, results, df_to_plot, modality):
    """Get model details DataFrame.

    Based on the generated `results`, `summary` and `df_to_plot`, create a DataFrame
    containing details about each synthesizer used in the benchmark.

    Args:
        summary (`pd.DataFrame`):
            Summary Wins DataFrame.
        results (`pd.DataFrame`):
            Detailed results DataFrame.
        df_to_plot (`pd.DataFrame`):
            DataFrame used for plotting.
        modality (str):
            Benchmark modality.

    Returns:
        DataFrame: Model details DataFrame.
    """
    with open(SYNTHESIZER_DESCRIPTION_PATH, 'r', encoding='utf-8') as f:
        synthesizer_info = yaml.safe_load(f) or {}

    err_column = 'error' if 'error' in results.columns else 'Error'
    paretos_synthesizers = (
        df_to_plot.loc[df_to_plot['Pareto'].eq(True), 'Synthesizer'].astype(str).add('Synthesizer')
    )
    wins_col = next(c for c in summary.columns if c != 'Synthesizer')
    model_details = results[['Synthesizer']].drop_duplicates().copy()
    model_details['Model Name'] = model_details['Synthesizer'].str.replace(
        'Synthesizer', '', regex=False
    )
    model_details['Data Modality'] = modality
    synthesizers = model_details['Synthesizer'].unique().tolist()
    filtered_desc = {k: v for k, v in (synthesizer_info or {}).items() if k in synthesizers}
    metadata_spec = {
        'Source': ('library', 'sdv'),
        'Organization': ('organization', 'Unknown'),
        'Data Modality': ('modality', 'Unknown'),
        'Type': ('type', 'Unknown'),
        'Description': ('description', 'No description available.'),
    }
    for column, (yaml_key, default) in metadata_spec.items():
        mapping = {k: (v or {}).get(yaml_key) for k, v in filtered_desc.items()}
        model_details[column] = model_details['Synthesizer'].map(mapping).fillna(default)

    wins = summary.set_index('Synthesizer')[wins_col]
    model_details['Number of datasets - Wins'] = (
        model_details['Synthesizer'].map(wins).fillna(0).astype(int)
    )
    timeout_counts = (
        results
        .loc[results[err_column].eq('Synthesizer Timeout')]
        .groupby('Synthesizer')['Dataset']
        .nunique()
    )
    error_counts = (
        results
        .loc[results[err_column].notna() & ~results[err_column].eq('Synthesizer Timeout')]
        .groupby('Synthesizer')['Dataset']
        .nunique()
    )
    model_details['Number of datasets - Timeout'] = (
        model_details['Synthesizer'].map(timeout_counts).fillna(0).astype(int)
    )
    model_details['Number of datasets - Errors'] = (
        model_details['Synthesizer'].map(error_counts).fillna(0).astype(int)
    )
    model_details['On the Pareto Curve'] = model_details['Synthesizer'].isin(paretos_synthesizers)

    return model_details


def update_table_aws(s3_client, bucket, filename, table, reference_column):
    """Update a table stored on S3 by merging with a new table.

    Args:
        s3_client:
            Boto3 S3 client.
        bucket:
            S3 bucket name.
        filename:
            S3 key of the table file.
        table (`pd.DataFrame`):
            DataFrame with new data to merge.
        reference_column:
            Column name to use as reference for merging.

    Returns:
        `pd.DataFrame`: The updated table.
    """
    try:
        existing_obj = s3_client.get_object(Bucket=bucket, Key=filename)
        existing_content = existing_obj['Body'].read()
        existing_table = pd.read_excel(io.BytesIO(existing_content), engine='openpyxl')
    except ClientError as e:
        if e.response['Error']['Code'] != 'NoSuchKey':
            raise

        existing_table = table.copy()

    existing_table = existing_table[~existing_table[reference_column].isin(table[reference_column])]
    updated_table = pd.concat([existing_table, table], ignore_index=True)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        updated_table.to_excel(writer, index=False)

    output.seek(0)
    s3_client.upload_fileobj(output, bucket, filename)

    return updated_table


def update_details_files(s3_client, bucket, prefix, details_list, local_export_dir=None):
    """Update details files on S3 and optionally save them locally.

    Args:
        s3_client:
            Boto3 S3 client.
        bucket:
            S3 bucket name.
        prefix:
            S3 prefix path.
        details_list:
            List of tuples (data, filename, reference_column)
        local_export_dir (`str`, optional):
            Local directory to save updated files. Default to None.
    """
    for data, filename, reference_column in details_list:
        key = f'{prefix}{filename}'
        updated_data = update_table_aws(s3_client, bucket, key, data, reference_column)
        if local_export_dir:
            local_path = Path(local_export_dir) / filename
            updated_data.to_excel(local_path, index=False)


def get_all_results(
    result_explorer, folder_name, modality, aws_access_key_id, aws_secret_access_key
):
    """Get all the benchmark results that will be saved.

    Compute and return all the benchmark results including:
    - Win Summary,
    - Detailed results,
    - Data for plotting the Pareto plot,
    - Dataset details,
    - Model details.

    The first three tables will be saved in the same Excel file, while the last two
    will be saved in their own Excel files.

    Args:
        result_explorer (ResultsExplorer):
            The ResultsExplorer instance to use.
        folder_name (str):
            The folder name of the benchmark run.
        modality (str):
            The benchmark modality.
        aws_access_key_id (str):
            AWS access key ID.
        aws_secret_access_key (str):
            AWS secret access key.
    """
    summary, results = result_explorer.summarize(folder_name)
    dataset_details = get_dataset_details(
        results, modality, aws_access_key_id, aws_secret_access_key
    )
    df_to_plot = get_df_to_plot(results)
    model_details = get_model_details(
        summary,
        results,
        df_to_plot,
        modality,
    )

    return summary, results, df_to_plot, dataset_details, model_details


def upload_all_results(datas, dataset_details, model_details, modality, s3_client, bucket, prefix):
    """Upload all benchmark results to S3 and GDrive.

    Args:
        datas (dict[str, DataFrame]):
            Dictionary of DataFrames to save in the main results Excel file.
        dataset_details (`pd.DataFrame`):
            Dataset details DataFrame.
        model_details (`pd.DataFrame`):
            Model details DataFrame.
        modality (str):
            Benchmark modality.
        s3_client:
            Boto3 S3 client.
        bucket:
            S3 bucket name.
        prefix:
            S3 prefix path.
        run_date (str):
            Date string of the benchmark run.

    Returns:
        str or None:
            Path to the temporary directory used for local storage, if any.
    """
    local_results_writer = LocalResultsWriter()
    local_export_dir = os.environ.get('GITHUB_LOCAL_RESULTS_DIR', tempfile.mkdtemp())
    Path(local_export_dir).mkdir(parents=True, exist_ok=True)
    sdgym_runs_filename = f'[{modality.replace("_", "-").capitalize()}]_{SDGYM_RUNS_FILENAME}'
    local_filepath_result = str(Path(local_export_dir) / sdgym_runs_filename)
    s3_key_result = f'{prefix}{sdgym_runs_filename}'
    try:
        s3_client.download_file(bucket, s3_key_result, local_filepath_result)
    except ClientError as e:
        if not e.response['Error']['Code'] == '404':
            raise

    local_results_writer.write_xlsx(datas, local_filepath_result)
    update_details_files(
        s3_client,
        bucket,
        prefix,
        [
            (dataset_details, DATASET_DETAILS_FILENAME, 'Data Modality'),
            (model_details, MODEL_DETAILS_FILENAME, 'Data Modality'),
        ],
        local_export_dir,
    )
    s3_client.upload_file(local_filepath_result, bucket, s3_key_result)
    file_to_gdrive_link = _get_filename_to_gdrive_link()
    for filename, link in file_to_gdrive_link.items():
        if link is None:
            continue

        other_modality = '[Multi-table]' if modality == 'single_table' else '[Single-table]'
        if filename == f'{other_modality}_{SDGYM_RUNS_FILENAME}':
            continue

        local_gdrive_path = str(Path(local_export_dir) / Path(filename))
        upload_to_drive(local_gdrive_path, _extract_google_file_id(link))

    return local_export_dir


def get_result_explorer(
    folder_name, modality, aws_access_key_id, aws_secret_access_key, github_env
):
    """Get the ResultsExplorer instance after checking if all runs are complete.

    The function checks if all runs for the given benchmark `folder_name` and `modality`
    are complete. If they are not complete, it logs a warning and sets the `SKIP_UPLOAD` flag
    in the GitHub environment file (if provided) to `true`, then exits the program.

    If all runs are complete, it logs an info message and sets the `SKIP_UPLOAD` flag to `false`
    along with the `FOLDER_NAME` in the GitHub environment file (if provided).

    This method returns a `ResultsExplorer` instance for further processing (summarization etc.).

    Args:
        folder_name (str):
            The folder name of the benchmark run.
        modality (str):
            The benchmark modality.
        aws_access_key_id (str):
            AWS access key ID.
        aws_secret_access_key (str):
            AWS secret access key.
        github_env (str or None):
            Path to the GitHub environment file, or None if not running in GitHub Actions.
    """
    result_explorer = ResultsExplorer(
        OUTPUT_DESTINATION_AWS,
        modality=modality,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
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

    return result_explorer


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
    """Upload benchmark results to S3, GDrive, and save locally.

    Args:
        aws_access_key_id (str):
            AWS access key ID.
        aws_secret_access_key (str):
            AWS secret access key.
        folder_infos (dict):
            Dictionary containing folder information such as 'folder_name' and 'date'.
        s3_client:
            Boto3 S3 client.
        bucket (str):
            S3 bucket name.
        prefix (str):
            S3 prefix path.
        github_env (str or None):
            Path to the GitHub environment file, or None if not running in GitHub Actions.
        modality (str):
            Benchmark modality.
            Default to 'single_table'.
    """
    folder_name = folder_infos['folder_name']
    run_date = folder_infos['date']
    result_explorer = get_result_explorer(
        folder_name,
        modality,
        aws_access_key_id,
        aws_secret_access_key,
        github_env,
    )

    summary, detailed_results, df_to_plot, dataset_details, model_details = get_all_results(
        result_explorer,
        folder_name,
        modality,
        aws_access_key_id,
        aws_secret_access_key,
    )
    datas = {
        'Wins': summary,
        f'{run_date}_Detailed_results': detailed_results,
        f'{run_date}_plot_data': df_to_plot,
    }
    temp_dir = upload_all_results(
        datas,
        dataset_details,
        model_details,
        modality,
        s3_client,
        bucket,
        prefix,
    )

    write_uploaded_marker(s3_client, bucket, prefix, folder_name, modality=modality)
    if temp_dir and os.getenv('GITHUB_LOCAL_RESULTS_DIR') is None:
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
