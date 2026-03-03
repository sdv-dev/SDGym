"""Build a ready-to-run benchmark configuration from YAML files."""

from __future__ import annotations

import json
from copy import deepcopy

import yaml

from sdgym._benchmark.benchmark import (
    _benchmark_multi_table_compute_gcp,
    _benchmark_single_table_compute_gcp,
)
from sdgym._benchmark_launcher._validation import (
    _validate_credentials_config,
    _validate_jobs,
    _validate_method_params,
    _validate_modality,
    _validate_structure,
)
from sdgym._benchmark_launcher.utils import _resolve_datasets, resolve_credentials
from sdgym.errors import BenchmarkConfigError

_METHODS = {
    ('single_table', 'gcp'): _benchmark_single_table_compute_gcp,
    ('multi_table', 'gcp'): _benchmark_multi_table_compute_gcp,
}
CONFIG_KEYS = frozenset([
    'modality',
    'method_params',
    'credentials',
    'compute',
    'instance_jobs',
])


class BenchmarkConfig:
    """Build and validate benchmark configs."""

    _KEYS = CONFIG_KEYS
    _CREDENTIAL_KEYS = {
        'aws': {'access_key_id_env', 'secret_access_key_env'},
        'gcp': {'service_account_json_env', 'project_id_env', 'zone_env'},
        'sdv': {'username_env', 'license_key_env'},
    }
    _CREDENTIAL_VALID_KEYS = frozenset({'credential_filepath'} | _CREDENTIAL_KEYS.keys())

    def __init__(self):
        self.modality = None
        self.method_params = None
        self.credentials_config = {}
        self.compute = {
            'service': None,
        }
        self.instance_jobs = []
        self._is_validated = False

    def to_dict(self):
        """Return a python ``dict`` representation of the ``BenchmarkConfig``."""
        config = {}
        for key in self._KEYS:
            value = getattr(self, f'{key}', None)
            if value is not None:
                config[key] = value

        return deepcopy(config)

    def __repr__(self):
        """Pretty print the ``BenchmarkConfig``."""
        printed = json.dumps(self.to_dict(), indent=4)
        return printed

    def validate(self):
        """Validate that the BenchmarkConfig is well-formed and can be run."""
        errors = []
        errors.append(_validate_structure(self))
        errors.append(_validate_method_params(self.method_params))
        errors.append(_validate_credentials_config(self.credentials_config))
        errors.append(_validate_jobs(self.instance_jobs))
        errors.append(_validate_modality(self.modality))
        if any(errors):
            errors = [error for error in errors if error is not None]
            message = 'BenchmarkConfig validation failed with the following errors:\n - '
            message += '\n - '.join(errors)
            raise BenchmarkConfigError(message)

        self._is_validated = True

    def _run(self):
        method_to_run = _METHODS[(self.modality, self.compute.get('service'))]
        credentials = resolve_credentials(self.credentials_config)

        for instance_job in self.instance_jobs:
            sdv_datasets = _resolve_datasets(instance_job['datasets'])
            method_to_run(
                synthesizers=instance_job['synthesizers'],
                sdv_datasets=sdv_datasets,
                credentials=credentials,
                compute_config=self.compute,
                **self.method_params,
            )

    def run(self):
        """Run the BenchmarkConfig: validate it and then execute the specified benchmark method."""
        if not self._is_validated:
            self.validate()
            self._is_validated = True

        self._run()

    def _validate_no_extra_keys(self, config_dict):
        """Validate that the config dictionary does not contain extra keys."""
        extra_keys = set(config_dict.keys()).difference(self._KEYS)
        if extra_keys:
            extra_keys = "', '".join(sorted(extra_keys))
            valid_keys = "', '".join(sorted(self._KEYS))
            raise ValueError(
                f"The config dictionary contains extra keys: '{extra_keys}'. "
                f"Valid keys are: '{valid_keys}'."
            )

    @classmethod
    def load_from_dict(cls, config_dict):
        """Load the BenchmarkConfig from a dict."""
        instance = cls()
        instance._validate_no_extra_keys(config_dict)
        instance.modality = config_dict.get('modality')
        instance.method_params = config_dict.get('method_params', {})
        instance.credentials_config = config_dict.get('credentials', {})
        instance.compute = config_dict.get('compute', {})
        instance.instance_jobs = config_dict.get('instance_jobs', [])

        return instance

    @classmethod
    def load_from_yaml(cls, filepath):
        """Load a config from a YAML file."""
        instance = cls()
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)

        instance._validate_no_extra_keys(config_dict)
        instance.modality = config_dict.get('modality')
        instance.method_params = config_dict.get('method_params', {})
        instance.credentials_config = config_dict.get('credentials', {})
        instance.compute = config_dict.get('compute', {})
        instance.instance_jobs = config_dict.get('instance_jobs', [])

        return instance

    def save_to_yaml(self, filepath):
        """Save the BenchmarkConfig in a YAML file."""
        config_dict = {
            'modality': self.modality,
            'method_params': self.method_params,
            'credentials': self.credentials_config,
            'compute': self.compute,
            'instance_jobs': self.instance_jobs,
        }
        with open(filepath, 'w') as file:
            yaml.dump(config_dict, file)
