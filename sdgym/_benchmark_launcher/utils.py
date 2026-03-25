"""Utilities for benchmark launcher."""

import json
import os
import uuid
from datetime import datetime
from importlib.resources import files

import yaml

from sdgym._benchmark.benchmark import (
    _benchmark_multi_table_compute_gcp,
    _benchmark_single_table_compute_gcp,
)
from sdgym.s3 import parse_s3_path

_YAML_PKG = 'sdgym._benchmark_launcher'
MODALITY_TO_CONFIG_FILE = {
    'single_table': 'benchmark_single_table.yaml',
    'multi_table': 'benchmark_multi_table.yaml',
}
CONFIG_KEYS = {
    'modality',
    'method_params',
    'credentials_filepath',
    'compute',
    'instance_jobs',
}
_METHODS = {
    ('single_table', 'gcp'): _benchmark_single_table_compute_gcp,
    ('multi_table', 'gcp'): _benchmark_multi_table_compute_gcp,
}
_AWS_CREDENTIAL_KEYS = (
    'AWS_ACCESS_KEY_ID',
    'AWS_SECRET_ACCESS_KEY',
)

_SDV_ENTERPRISE_CREDENTIAL_KEYS = (
    'SDV_ENTERPRISE_USERNAME',
    'SDV_ENTERPRISE_LICENSE_KEY',
)

_GCP_SERVICE_ACCOUNT_REQUIRED_KEYS = (
    'type',
    'project_id',
    'private_key_id',
    'private_key',
    'client_email',
    'client_id',
    'auth_uri',
    'token_uri',
    'auth_provider_x509_cert_url',
    'client_x509_cert_url',
)

_GCP_SERVICE_ACCOUNT_JSON = 'GCP_SERVICE_ACCOUNT_JSON'
_GCP_SERVICE_ACCOUNT_JSON_FILEPATH = 'GCP_SERVICE_ACCOUNT_JSON_FILEPATH'


def _load_merged_modality_config(modality):
    """Load and merge the base and modality-specific benchmark configs."""
    base_config = _load_yaml_resource('benchmark_base.yaml')
    modality_config = _load_yaml_resource(MODALITY_TO_CONFIG_FILE[modality])
    return _deep_merge(base_config, modality_config)


def _resolve_modality_config(modality):
    """Resolve the launchable benchmark config for a modality."""
    merged_config = _load_merged_modality_config(modality)
    return {key: value for key, value in merged_config.items() if key in CONFIG_KEYS}


def _resolve_datasets(datasets_spec):
    """Resolve the list of datasets to run on based on the 'datasets' specification in the config.

    The 'datasets' specification can be either:
      - ["adult", "census"]  (already final)
      - {"include": [...], "exclude": [...]}  (compute final)
    """
    if isinstance(datasets_spec, list):
        return list(datasets_spec)

    if isinstance(datasets_spec, dict):
        include = datasets_spec.get('include', [])
        exclude = set(datasets_spec.get('exclude', []))

        return [dataset for dataset in include if dataset not in exclude]

    raise ValueError(f"'datasets' must be a list or dict. Found: {type(datasets_spec)}")


def _load_yaml_resource(filename: str) -> dict:
    resource = files(_YAML_PKG).joinpath(filename)
    with resource.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _env(name):
    if not name:
        return None
    value = os.getenv(name)

    return value if value not in (None, '') else None


def _deep_merge(base, override):
    """Recursively merge override into base (override wins)."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)

        else:
            result[key] = value

    return result


def generate_ids(prefixes):
    """Generate a unique identifier for the benchmark instance.

    This method creates a unique identifier by combining the modality, the compute
    service and the last part of a UUID4 composed by 6 characters.

    Args:
        prefixes (list of str):
            A list of prefixes to include in the benchmark ID.

    Returns:
        ID:
            A unique identifier for this benchmark instance.
    """
    unique_id = ''.join(str(uuid.uuid4()).split('-'))[-6:]
    return '_'.join(prefixes + [unique_id])


def _load_json_file(path):
    """Load JSON from a file path."""
    with open(path, encoding='utf-8') as file:
        return json.load(file)


def _get_gcp_credentials_from_env():
    """Build resolved GCP credentials from environment variables."""
    json_content = _env(_GCP_SERVICE_ACCOUNT_JSON)
    if json_content:
        return json.loads(json_content)

    json_filepath = _env(_GCP_SERVICE_ACCOUNT_JSON_FILEPATH)
    if json_filepath:
        return _load_json_file(json_filepath)

    return {key: _env(f'GCP_{key.upper()}') for key in _GCP_SERVICE_ACCOUNT_REQUIRED_KEYS}


def _get_env_credentials():
    """Build resolved credentials from environment variables."""
    return {
        'aws': {key: _env(key) for key in _AWS_CREDENTIAL_KEYS},
        'sdv_enterprise': {key: _env(key) for key in _SDV_ENTERPRISE_CREDENTIAL_KEYS},
        'gcp': _get_gcp_credentials_from_env(),
    }


def _lowercase_keys(data):
    if isinstance(data, dict):
        return {str(key).lower(): _lowercase_keys(value) for key, value in data.items()}

    return data


def resolve_credentials(credentials_filepath=None):
    """Resolve credentials from environment variables and, optionally, a credentials file.

    Environment variables are loaded first. If a credentials file is provided, any values
    defined in that file override the corresponding environment variables.

    As a result, values from the credentials file take precedence. Environment variables
    are only used for credentials that are missing from the file.
    """
    env_credentials = _get_env_credentials()
    if credentials_filepath is None:
        return _lowercase_keys(env_credentials)

    file_credentials = _load_json_file(credentials_filepath)
    return _lowercase_keys(_deep_merge(env_credentials, file_credentials))


def _add_dataset_suffix(dataset):
    """Return the dataset folder name used in artifact paths."""
    today = datetime.today().strftime('%m_%d_%Y')
    return f'{dataset}_{today}'


def _get_top_folder_prefix(output_destination, modality):
    """Return the top folder prefix used for benchmark artifacts."""
    _, key_prefix = parse_s3_path(output_destination)
    today = datetime.today().strftime('%m_%d_%Y')
    modality_prefix = '/'.join([part for part in [key_prefix.rstrip('/'), modality] if part])
    return f'{modality_prefix}/SDGym_results_{today}'


def _get_synthetic_data_extension(modality):
    """Return the synthetic data file extension for the given modality."""
    return 'zip' if modality == 'multi_table' else 'csv'


def _build_job_artifact_keys(artifact_key_prefix, artifact_dataset, artifact_synthesizer, modality):
    """Build the expected artifact keys for a benchmark job."""
    job_prefix = f'{artifact_key_prefix.rstrip("/")}/{artifact_dataset}/{artifact_synthesizer}'
    synthetic_data_extension = _get_synthetic_data_extension(modality)

    benchmark_result_key = f'{job_prefix}/{artifact_synthesizer}_benchmark_result.csv'
    synthetic_data_key = (
        f'{job_prefix}/{artifact_synthesizer}_synthetic_data.{synthetic_data_extension}'
    )
    synthesizer_key = f'{job_prefix}/{artifact_synthesizer}.pkl'

    return benchmark_result_key, synthetic_data_key, synthesizer_key
