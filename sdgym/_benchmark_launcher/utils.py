"""Utilities for benchmark launcher."""

import uuid
from importlib.resources import files

import yaml

from sdgym._benchmark.benchmark import (
    _benchmark_multi_table_compute_gcp,
    _benchmark_single_table_compute_gcp,
)
from sdgym._benchmark_launcher._validation import _get_credentials, _validate_resolved_credentials

_YAML_PKG = 'sdgym._benchmark_launcher'
MODALITY_TO_CONFIG_FILE = {
    'single_table': 'benchmark_single_table.yaml',
    'multi_table': 'benchmark_multi_table.yaml',
}
CONFIG_KEYS = {
    'modality',
    'method_params',
    'credentials',
    'compute',
    'instance_jobs',
}
_METHODS = {
    ('single_table', 'gcp'): _benchmark_single_table_compute_gcp,
    ('multi_table', 'gcp'): _benchmark_multi_table_compute_gcp,
}


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


def _deep_merge(base, override):
    """Recursively merge override into base (override wins)."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)

        else:
            result[key] = value
    return result


def resolve_credentials(credentials_config):
    """Resolve credentials dict from config."""
    credentials = _get_credentials(credentials_config)
    errors = _validate_resolved_credentials(credentials)
    if errors:
        raise ValueError('Invalid resolved credentials:\n- ' + '\n- '.join(errors))

    return credentials


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
