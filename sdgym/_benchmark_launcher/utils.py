"""Utilities for benchmark launcher."""

import json
import os
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
import yaml
from importlib.resources import files

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

@contextmanager
def resolved_credential_filepath(credentials_config, build_credentials_dict_fn):
    """Yields a credential_filepath to use.

    - If credentials_config defines credential_filepath_env and it's set, yield that path.
    - Else build a dict (from env var names in YAML), write temp JSON, yield temp path,
      then delete it on exit.
    """
    created_tmp_path = None
    env_name = (credentials_config or {}).get('credential_filepath_env')
    if env_name:
        existing_path = os.getenv(env_name)
        if existing_path:
            yield existing_path
            return

    credentials_dict = build_credentials_dict_fn(credentials_config)
    tmp = NamedTemporaryFile(mode='w', delete=False, suffix='.json')
    try:
        json.dump(credentials_dict, tmp)
        tmp.flush()
        tmp.close()
        try:
            os.chmod(tmp.name, 0o600)
        except Exception:
            pass

        created_tmp_path = tmp.name
        yield created_tmp_path
    finally:
        if created_tmp_path:
            try:
                os.remove(created_tmp_path)
            except FileNotFoundError:
                pass


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

def _resolve_modality_config(modality):
    base_config = _load_yaml_resource('benchmark_base.yaml')
    modality_config = _load_yaml_resource(MODALITY_TO_CONFIG_FILE[modality])
    merged_config = _deep_merge(base_config, modality_config)
    resolved_dict = {
        key: value for key, value in merged_config.items() if key in CONFIG_KEYS
    }

    return resolved_dict


def _env(env_name):
    if not env_name:
        return None
    value = os.getenv(env_name)
    return value if value != '' else None
