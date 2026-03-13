"""Utilities for benchmark launcher."""

import json
import os
import uuid
from importlib.resources import files

import yaml

from sdgym._benchmark.benchmark import (
    _benchmark_multi_table_compute_gcp,
    _benchmark_single_table_compute_gcp,
)

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


def _resolve_modality_config(modality):
    base_config = _load_yaml_resource('benchmark_base.yaml')
    modality_config = _load_yaml_resource(MODALITY_TO_CONFIG_FILE[modality])
    merged_config = _deep_merge(base_config, modality_config)
    resolved_dict = {key: value for key, value in merged_config.items() if key in CONFIG_KEYS}

    return resolved_dict


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


def generate_benchmark_id(benchmark_launcher):
    """Generate a unique identifier for the benchmark instance.

    This method creates a unique identifier by combining the modality, the compute
    service and the last part of a UUID4 composed by 6 characters.

    Args:
        benchmark_launcher (BenchmarkLauncher):
            An SDGym benchmark launcher.

    Returns:
        ID:
            A unique identifier for this benchmark instance.
    """
    modality = benchmark_launcher.modality
    compute_service = benchmark_launcher.compute_service
    unique_id = ''.join(str(uuid.uuid4()).split('-'))[-6:]

    return f'{modality}_{compute_service}_{unique_id}'


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
    env_credentials = _get_env_credentials()
    if credentials_filepath is None:
        return _lowercase_keys(env_credentials)

    file_credentials = _load_json_file(credentials_filepath)
    return _lowercase_keys(_deep_merge(env_credentials, file_credentials))
