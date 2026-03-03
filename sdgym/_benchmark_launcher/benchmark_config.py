"""Build a ready-to-run benchmark configuration from YAML files."""

from __future__ import annotations

import json
import os

import yaml
from copy import deepcopy
from sdgym._benchmark.benchmark import (
    _benchmark_multi_table_compute_gcp,
    _benchmark_single_table_compute_gcp,
)
from sdgym._benchmark_launcher.utils import (
    _env,
    _resolve_datasets,
    resolved_credential_filepath,
)
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
    _CREDENTIAL_VALID_KEYS = frozenset(
        {"credential_filepath", "credential_filepath_env"} | _CREDENTIAL_KEYS.keys()
    )

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

    def _validate_structure(self):
        """Validate the overall structure of the config (keys and types)."""
        errors = []
        if self.modality not in ['single_table', 'multi_table']:
            errors.append(
                f"Invalid modality '{self.modality}'. Must be 'single_table' or 'multi_table'."
            )

        if not isinstance(self.method_params, dict):
            errors.append(f"'method_params' must be a dict. Found: {type(self.method_params)}")

        if not isinstance(self.credentials_config, dict):
            errors.append(f"'credentials' must be a dict. Found: {type(self.credentials_config)}")

        if not isinstance(self.compute, dict):
            errors.append(f"'compute' must be a dict. Found: {type(self.compute)}")
        elif 'service' not in self.compute or self.compute['service'] not in ['gcp']:  # We only support GCP for now, AWS support is coming soon
            errors.append(
                f"'compute.service' must be either 'aws' or 'gcp'. Found: "
                f'{self.compute.get("service")}'
            )

        return '\n'.join(errors) if errors else None

    def _validate_modality(modality):
        if modality not in ['single_table', 'multi_table']:
            raise BenchmarkConfigError(
                f"Invalid modality '{modality}'. Must be 'single_table' or 'multi_table'."
            )

    def _validate_method_params(self):
        """Validate the method parameters."""
        return

    def _resolve_credentials_filepath(self):
        """Return a credentials filepath if provided (directly or via env). Otherwise None."""
        cred_config = self.credentials_config or {}

        direct_path = cred_config.get("credential_filepath")
        if direct_path:
            return direct_path

        env_name = cred_config.get("credential_filepath_env")
        if env_name:
            return _env(env_name)  # env value is the filepath (or None)

        return None

    def _validate_credentials_config_shape(self):
        """Validate basic credentials config shape/types (no env lookups yet)."""
        cred_config = self.credentials_config or {}
        errors = []

        unknown = set(cred_config.keys()) - set(self._CREDENTIAL_VALID_KEYS)
        if unknown:
            errors.append(f"credentials: unknown top-level keys: {sorted(unknown)}")

        for section in ("aws", "gcp", "sdv"):
            if section in cred_config and not isinstance(cred_config[section], dict):
                errors.append(
                    f"credentials.{section}: must be a dict. Found: {type(cred_config[section])}"
                )

        return errors

    def _validate_credentials_file(self, filepath):
        """Validate that a credentials file exists, is JSON, and is readable."""
        errors = []
        if not os.path.isfile(filepath):
            errors.append(f'credentials file not found: {filepath}')
            return errors

        if not filepath.endswith('.json'):
            errors.append(f'credentials file must be a .json file: {filepath}')

        try:
            with open(filepath, 'r') as f:
                json.load(f)
        except json.JSONDecodeError:
            errors.append(f'credentials file is not valid JSON: {filepath}')
        except OSError as e:
            errors.append(f'credentials file could not be read ({filepath}): {e}')

        return errors

    def _validate_env_section(self, section, section_config, *, required=False):
        """Validate a single provider block that contains env var names."""
        errors = []
        expected = self._CREDENTIAL_KEYS.get(section, set())

        if section_config is None:
            if required:
                errors.append(f"credentials.{section}: section is required but missing.")
            return errors

        if not isinstance(section_config, dict):
            errors.append(f"credentials.{section}: must be a dict.")
            return errors

        keys = set(section_config.keys())
        missing = expected - keys
        extra = keys - expected
        if missing:
            errors.append(f"credentials.{section}: missing keys: {sorted(missing)}")
        if extra:
            errors.append(f"credentials.{section}: unknown keys: {sorted(extra)}")

        for key in expected & keys:
            env_name = section_config.get(key)
            if not isinstance(env_name, str) or not env_name:
                errors.append(
                    f'credentials.{section}.{key}: must be a non-empty env var name string.'
                )
                continue

            env_value = _env(env_name)
            if env_value is None or env_value == '':
                errors.append(
                    f"Environment variable '{env_name}' (referenced by credentials.{section}.{key})"
                    ' is not set or empty.'
                )

            if section == 'gcp' and key == 'service_account_json_env' and env_value:
                try:
                    json.loads(env_value)
                except json.JSONDecodeError:
                    errors.append(
                        f"Environment variable '{env_name}' must contain valid JSON "
                        '(GCP service account).'
                    )

        return errors

    def _validate_credentials(self):
        """Validate credentials.

        File mode if credential_filepath OR credential_filepath_env is present in YAML.
        Env mode otherwise.
        """
        errors = []
        cred_config = self.credentials_config or {}

        errors.extend(self._validate_credentials_config_shape())

        file_mode = bool(
            cred_config.get("credential_filepath") or cred_config.get("credential_filepath_env")
        )

        if file_mode:
            # Direct path takes precedence
            direct_path = cred_config.get("credential_filepath")
            if direct_path:
                errors.extend(self._validate_credentials_file(direct_path))
                return "\n".join(errors) if errors else None

            # Env-provided path must exist
            env_name = cred_config.get("credential_filepath_env")
            filepath = _env(env_name) if env_name else None
            if not filepath:
                errors.append(
                    f"credentials.credential_filepath_env is set to '{env_name}', but that env var is not set or empty."
                )
                return "\n".join(errors)

            errors.extend(self._validate_credentials_file(filepath))
            return "\n".join(errors) if errors else None

        # Env mode (only if no file keys exist)
        errors.extend(self._validate_env_section("aws", cred_config.get("aws"), required=True))
        errors.extend(self._validate_env_section("gcp", cred_config.get("gcp"), required=True))
        errors.extend(self._validate_env_section("sdv", cred_config.get("sdv"), required=False))

        return "\n".join(errors) if errors else None

    def _validate_jobs(self):
        error_message = (
            "Each job in 'instance_jobs' must be a dict with 'synthesizers' (list of strings) "
            "and 'datasets' (list of strings or dict with 'include' and optional 'exclude')."
        )
        invalid_jobs = []
        for job in self.instance_jobs:
            if 'datasets' not in job or 'synthesizers' not in job:
                invalid_jobs.append(job)
                continue

            if not isinstance(job['synthesizers'], list) or not all(
                isinstance(s, str) for s in job['synthesizers']
            ):
                invalid_jobs.append(job)
                continue

            if not (isinstance(job['datasets'], list) or isinstance(job['datasets'], dict)):
                invalid_jobs.append(job)
                continue

            if isinstance(job['datasets'], list) and not all(
                isinstance(d, str) for d in job['datasets']
            ):
                invalid_jobs.append(job)
                continue

            if isinstance(job['datasets'], dict):
                if ('include' not in job['datasets']) or (
                    not isinstance(job['datasets']['include'], list)
                    or not all(isinstance(d, str) for d in job['datasets']['include'])
                ):
                    invalid_jobs.append(job)
                    continue

                if 'exclude' in job['datasets']:
                    if (not isinstance(job['datasets']['exclude'], list)) or not all(
                        isinstance(d, str) for d in job['datasets']['exclude']
                    ):
                        invalid_jobs.append(job)
                        continue

        if invalid_jobs:
            invalid_jobs = '\n'.join(str(job) for job in invalid_jobs)
            error_message = f'{error_message}\n Invalid jobs: {invalid_jobs}'
            return error_message

    def validate(self):
        """Validate that the BenchmarkConfig is well-formed and can be run."""
        errors = []
        errors.append(self._validate_structure())
        errors.append(self._validate_method_params())
        errors.append(self._validate_credentials())
        errors.append(self._validate_jobs())
        if any(errors):
            errors = [error for error in errors if error is not None]
            message = 'BenchmarkConfig validation failed with the following errors:\n - '
            message += '\n - '.join(errors)
            raise BenchmarkConfigError(message)

        self._is_validated = True

    def _get_credentials_dict(self, credentials_config):
        """Build the credentials dict from env vars referenced in the `credentials_config`."""
        config = credentials_config or {}
        aws_config = config.get('aws', {}) or {}
        gcp_config = config.get('gcp', {}) or {}
        sdv_config = config.get('sdv', {}) or {}

        aws_access_key_id = _env(aws_config.get('access_key_id_env'))
        aws_secret_access_key = _env(aws_config.get('secret_access_key_env'))
        gcp_sa_json_raw = _env(gcp_config.get('service_account_json_env'))
        gcp_sa_obj = {}
        if gcp_sa_json_raw:
            try:
                gcp_sa_obj = json.loads(gcp_sa_json_raw)
            except json.JSONDecodeError:
                gcp_sa_obj = {}

        gcp_project = _env(gcp_config.get('project_id_env'))
        gcp_zone = _env(gcp_config.get('zone_env'))

        sdv_username = _env(sdv_config.get('username_env'))
        sdv_license_key = _env(sdv_config.get('license_key_env'))
        credentials = {
            'aws': {
                'aws_access_key_id': aws_access_key_id,
                'aws_secret_access_key': aws_secret_access_key,
            },
            'gcp': {
                **gcp_sa_obj,
                'gcp_project': gcp_project,
                'gcp_zone': gcp_zone,
            },
            'sdv': {
                'username': sdv_username,
                'license_key': sdv_license_key,
            },
        }

        return credentials

    def _run(self):
        method_to_run = _METHODS.get((self.modality, self.compute.get('service')))
        with resolved_credential_filepath(
            self.credentials_config, self._get_credentials_dict
        ) as cred_path:
            for instance_job in self.instance_jobs:
                sdv_datasets = _resolve_datasets(instance_job['datasets'])
                method_to_run(
                    synthesizers=instance_job['synthesizers'],
                    sdv_datasets=sdv_datasets,
                    credential_filepath=cred_path,
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
