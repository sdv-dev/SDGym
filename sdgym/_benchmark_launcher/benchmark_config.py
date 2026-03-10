"""Define the BenchmarkConfig class, which represents the configuration for a benchmark."""

import json
from copy import deepcopy

import yaml

from sdgym._benchmark_launcher._validation import (
    _format_sectioned_errors,
    _validate_credential_locations,
    _validate_instance_jobs,
    _validate_method_params,
    _validate_structure,
)
from sdgym._benchmark_launcher.utils import _METHODS
from sdgym.errors import BenchmarkConfigError

CONFIG_KEYS = frozenset([
    'modality',
    'method_params',
    'credential_locations',
    'compute',
    'instance_jobs',
])


class BenchmarkConfig:
    """BenchmarkConfig class.

    This class represents the configuration for a benchmark. It can be loaded from a YAML file
    or a dictionary and provides methods for validation and conversion to different formats.
    The expected structure of the config is as follows:
    {
        'modality': 'single_table' or 'multi_table',
        'method_params': dict of parameters to pass to the benchmark method (e.g. timeout),
        'credentials': dict specifying how to resolve credentials (e.g. from env vars or a file),
        'compute': dict specifying the compute configuration (e.g. service: 'gcp'),
        'instance_jobs': list of dicts, each specifying a combination of synthesizers and datasets:
            [
                {
                    'synthesizers': ['synthesizer1', 'synthesizer2'],
                    'datasets': ['dataset1', 'dataset2'] or {'include': [...], 'exclude': [...]}
                },
                ...
            ]
    }
    """

    _CREDENTIAL_KEYS = {
        'aws': {'access_key_id_env', 'secret_access_key_env'},
        'gcp': {'service_account_json_env', 'project_id_env', 'zone_env'},
        'sdv': {'username_env', 'license_key_env'},
    }
    _CREDENTIAL_VALID_KEYS = frozenset({'credential_filepath'} | _CREDENTIAL_KEYS.keys())

    def __init__(self):
        self.modality = None
        self.method_params = None
        self.credential_locations = {}
        self.compute = {'service': None}
        self.instance_jobs = []
        self._is_validated = False

    def to_dict(self):
        """Return a python ``dict`` representation of the ``BenchmarkConfig``."""
        config = {}
        for key in CONFIG_KEYS:
            value = getattr(self, f'{key}', None)
            if value is not None:
                config[key] = value

        return deepcopy(config)

    def __str__(self):
        """Pretty print the ``BenchmarkConfig``."""
        printed = json.dumps(self.to_dict(), indent=4)
        return printed

    def validate(self):
        method_to_run = _METHODS[(self.modality, self.compute.get('service'))]
        errors = _validate_structure(self)
        if errors:
            raise BenchmarkConfigError(_format_sectioned_errors({'structure': errors}))

        section_errors = {
            'method_params': _validate_method_params(self.method_params, method_to_run),
            'credential_locations': _validate_credential_locations(self.credential_locations),
            'instance_jobs': _validate_instance_jobs(self.instance_jobs),
        }
        if any(section_errors.values()):
            raise BenchmarkConfigError(_format_sectioned_errors(section_errors))

        self._is_validated = True

    def _validate_no_extra_keys(self, config_dict):
        """Validate that the config dictionary does not contain extra keys."""
        extra_keys = set(config_dict.keys()).difference(CONFIG_KEYS)
        if extra_keys:
            extra_keys = "', '".join(sorted(extra_keys))
            valid_keys = "', '".join(sorted(CONFIG_KEYS))
            raise ValueError(
                f"The config dictionary contains extra keys: '{extra_keys}'. "
                f"Valid keys are: '{valid_keys}'."
            )

    @classmethod
    def load_from_dict(cls, config_dict):
        """Load the BenchmarkConfig from a dict."""
        instance = cls()
        instance._validate_no_extra_keys(config_dict)
        for attribute_name, attribute_value in config_dict.items():
            setattr(instance, attribute_name, attribute_value)

        return instance

    @classmethod
    def load_from_yaml(cls, filepath):
        """Load a config from a YAML file."""
        instance = cls()
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)

        instance._validate_no_extra_keys(config_dict)
        for attribute_name, attribute_value in config_dict.items():
            setattr(instance, attribute_name, attribute_value)

        return instance

    def save_to_yaml(self, filepath):
        """Save the BenchmarkConfig in a YAML file."""
        config_dict = self.to_dict()
        with open(filepath, 'w') as file:
            yaml.dump(config_dict, file)
