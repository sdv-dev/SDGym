"""Script to run a benchmark and upload results to S3."""

import json
import os
import tempfile
from importlib.resources import files

import yaml
from botocore.exceptions import ClientError

from sdgym._benchmark.benchmark import (
    _benchmark_multi_table_compute_gcp,
    _benchmark_single_table_compute_gcp,
)
from sdgym._benchmark_launcher.benchmark_config import BenchmarkConfig, _deep_merge
from sdgym.run_benchmark.utils import (
    KEY_DATE_FILE,
    _parse_args,
    get_result_folder_name,
)
from sdgym.s3 import get_s3_client, parse_s3_path

_YAML_PKG = 'sdgym._benchmark_launcher'
MODALITY_TO_CONFIG_FILE = {
    'single_table': 'benchmark_single_table.yaml',
    'multi_table': 'benchmark_multi_table.yaml',
}


def _load_yaml_resource(filename: str) -> dict:
    resource = files(_YAML_PKG).joinpath(filename)
    with resource.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


_METHODS = {
    ('single_table', 'gcp'): _benchmark_single_table_compute_gcp,
    ('multi_table', 'gcp'): _benchmark_multi_table_compute_gcp,
}


def append_benchmark_run(
    output_destination: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    date_str: str,
    modality: str,
):
    """Append a new benchmark run to the benchmark dates file in S3."""
    s3_client = get_s3_client(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    bucket, prefix = parse_s3_path(output_destination)
    key = f'{prefix}{modality}/{KEY_DATE_FILE}'
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        body = obj['Body'].read().decode('utf-8')
        data = json.loads(body)
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            data = {'runs': []}
        else:
            raise RuntimeError(f'Failed to read {KEY_DATE_FILE} from S3: {e}')

    data['runs'].append({'date': date_str, 'folder_name': get_result_folder_name(date_str)})
    data['runs'] = sorted(data['runs'], key=lambda x: x['date'])

    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data).encode('utf-8'),
    )


def _resolve_modality_config(modality):
    """Method that resolves the config for a modality and save it into a tmp yaml file."""
    base_config = _load_yaml_resource('benchmark_base.yaml')
    modality_config = _load_yaml_resource(MODALITY_TO_CONFIG_FILE[modality])
    merged_config = _deep_merge(base_config, modality_config)
    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml')
    try:
        yaml.safe_dump(merged_config, tmp)
        tmp.flush()
        return tmp.name
    finally:
        tmp.close()


def _get_config(modality):
    yaml_path = _resolve_modality_config(modality)
    try:
        config = BenchmarkConfig.load_from_yaml(yaml_path)
    finally:
        os.unlink(yaml_path)

    config.validate()
    return config


def main():
    """Main function to run the benchmark."""
    args = _parse_args()
    modality = args.modality
    config = _get_config(modality)
    config.run()


if __name__ == '__main__':
    main()
