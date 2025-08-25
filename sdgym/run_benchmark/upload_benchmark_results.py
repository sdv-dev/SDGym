"""Script to upload benchmark results to S3."""

import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import boto3
import numpy as np
import plotly.express as px
from botocore.exceptions import ClientError
from oauth2client.client import OAuth2Credentials
from plotly import graph_objects as go
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from scipy.interpolate import interp1d

from sdgym.result_writer import LocalResultsWriter
from sdgym.run_benchmark.utils import OUTPUT_DESTINATION_AWS, get_df_to_plot
from sdgym.s3 import S3_REGION, parse_s3_path
from sdgym.sdgym_result_explorer.result_explorer import SDGymResultsExplorer

LOGGER = logging.getLogger(__name__)
SYNTHESIZER_TO_GLOBAL_POSITION = {
    'CTGAN': 'middle right',
    'TVAE': 'middle left',
    'GaussianCopula': 'bottom center',
    'Uniform': 'top center',
    'Column': 'top center',
    'CopulaGAN': 'top center',
    'RealTabFormer': 'bottom center',
}
SDGYM_FILE_ID = '1W3tsGOOtbtTw3g0EVE0irLgY_TN_cy2W4ONiZQ57OPo'
RESULT_FILENAME = 'SDGym Monthly Run.xlsx'


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


def write_uploaded_marker(s3_client, bucket, prefix, folder_name):
    """Write a marker file to indicate that the upload is complete."""
    s3_client.put_object(
        Bucket=bucket, Key=f'{prefix}{folder_name}/upload_complete.marker', Body=b'Upload complete'
    )


def upload_already_done(s3_client, bucket, prefix, folder_name):
    """Check if the upload has already been done by looking for the marker file."""
    try:
        s3_client.head_object(Bucket=bucket, Key=f'{prefix}{folder_name}/upload_complete.marker')
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False

        raise


def get_result_folder_name_and_s3_vars(aws_access_key_id, aws_secret_access_key):
    """Get the result folder name and S3 client variables."""
    bucket, prefix = parse_s3_path(OUTPUT_DESTINATION_AWS)
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=S3_REGION,
    )
    folder_infos = get_latest_run_from_file(s3_client, bucket, f'{prefix}_BENCHMARK_DATES.json')

    return folder_infos, s3_client, bucket, prefix


def generate_graph(plot_table):
    """Generate a scatter plot for the benchmark results."""
    fig = px.scatter(
        plot_table,
        x='Aggregated_Time',
        y='Quality_Score',
        color='Synthesizer',
        text='Synthesizer',
        title='Mean Quality Score vs Aggregated Time (Over All Datasets)',
        labels={'Aggregated_Time': 'Aggregated Time [s]', 'Quality_Score': 'Mean Quality Score'},
        log_x=True,
    )

    for trace in fig.data:
        synthesizer_name = trace.name
        shape = plot_table.loc[plot_table['Synthesizer'] == synthesizer_name, 'Marker'].values[0]
        color = plot_table.loc[plot_table['Synthesizer'] == synthesizer_name, 'Color'].values[0]
        trace_positions = SYNTHESIZER_TO_GLOBAL_POSITION.get(synthesizer_name, 'top center')
        trace.update(
            marker=dict(size=14, color=color), textposition=trace_positions, marker_symbol=shape
        )

    fig.update_layout(
        xaxis=dict(
            tickformat='.0e',
            tickmode='array',
            tickvals=[1e1, 1e2, 1e3, 1e4, 1e5, 1e6],
            ticktext=[
                '10<sup>1</sup>',
                '10<sup>2</sup>',
                '10<sup>3</sup>',
                '10<sup>4</sup>',
                '10<sup>5</sup>',
                '10<sup>6</sup>',
            ],
            showgrid=False,
            zeroline=False,
            title='Aggregated Time [s]',
            range=[0.6, 6],
        ),
        yaxis=dict(showgrid=False, zeroline=False, range=[0.54, 0.92]),
        plot_bgcolor='#F5F5F8',
    )

    fig.update_traces(textfont=dict(size=16))
    pareto_points = plot_table.loc[plot_table['Pareto']]
    x_pareto = pareto_points['Aggregated_Time'].values
    y_pareto = pareto_points['Quality_Score'].values
    sorted_indices = np.argsort(x_pareto)
    x_sorted = x_pareto[sorted_indices]
    y_sorted = y_pareto[sorted_indices]
    log_x_sorted = np.log10(x_sorted)
    interp = interp1d(log_x_sorted, y_sorted, kind='linear', fill_value='extrapolate')
    log_x_fit = np.linspace(0.7, 6, 100)
    y_fit = interp(log_x_fit)
    x_fit = np.power(10, log_x_fit)

    # Plot smooth interpolation
    fig.add_trace(
        go.Scatter(
            x=x_fit,
            y=y_fit,
            mode='lines',
            name='Pareto Frontier',
            line=dict(color='black', width=2),
        )
    )
    x_shade = np.concatenate([x_fit, x_fit[::-1]])
    y_shade = np.concatenate([y_fit, np.full_like(x_fit, min(y_fit))[::-1]])
    fig.add_trace(
        go.Scatter(
            x=x_shade,
            y=y_shade,
            fill='toself',
            fillcolor='rgba(0, 0, 54, 0.25)',
            line=dict(color='#000036'),
            hoverinfo='skip',
            showlegend=False,
        )
    )

    return fig


def upload_to_drive(file_path, file_id):
    """Upload a local file to a Google Drive folder.

    Args:
        file_path (str or Path): Path to the local file to upload.
        file_id (str): Google Drive file ID.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f'File not found: {file_path}')

    creds_dict = json.loads(os.environ['PYDRIVE_CREDENTIALS'])
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


def upload_results(
    aws_access_key_id, aws_secret_access_key, folder_infos, s3_client, bucket, prefix, github_env
):
    """Upload benchmark results to S3, GDrive, and save locally."""
    folder_name = folder_infos['folder_name']
    run_date = folder_infos['date']
    result_explorer = SDGymResultsExplorer(
        OUTPUT_DESTINATION_AWS,
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
    df_to_plot = get_df_to_plot(results)
    figure = generate_graph(df_to_plot)
    local_export_dir = os.environ.get('GITHUB_LOCAL_RESULTS_DIR')
    temp_dir = None
    if not local_export_dir:
        temp_dir = tempfile.mkdtemp()
        local_export_dir = temp_dir

    Path(local_export_dir).mkdir(parents=True, exist_ok=True)
    local_file_path = str(Path(local_export_dir) / RESULT_FILENAME)
    s3_key = f'{prefix}{RESULT_FILENAME}'
    s3_client.download_file(bucket, s3_key, local_file_path)
    datas = {
        'Wins': summary,
        f'{run_date}_Detailed_results': results,
        f'{run_date}_plot_data': df_to_plot,
        f'{run_date}_plot_image': figure,
    }
    local_results_writer.write_xlsx(datas, local_file_path)
    upload_to_drive((local_file_path), SDGYM_FILE_ID)
    s3_client.upload_file(local_file_path, bucket, s3_key)
    write_uploaded_marker(s3_client, bucket, prefix, folder_name)
    if temp_dir:
        shutil.rmtree(temp_dir)


def main():
    """Main function to upload benchmark results."""
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    folder_infos, s3_client, bucket, prefix = get_result_folder_name_and_s3_vars(
        aws_access_key_id, aws_secret_access_key
    )
    github_env = os.getenv('GITHUB_ENV')
    if upload_already_done(s3_client, bucket, prefix, folder_infos['folder_name']):
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
    )


if __name__ == '__main__':
    main()
