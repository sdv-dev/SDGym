"""Build a ready-to-run benchmark configuration from YAML files."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from tempfile import NamedTemporaryFile

import yaml

from sdgym._benchmark.benchmark import (
    _benchmark_multi_table_compute_gcp,
    _benchmark_single_table_compute_gcp,
)
from sdgym.benchmark import benchmark_multi_table_aws, benchmark_single_table_aws

_METHODS = {
    ('single_table', 'gcp'): _benchmark_single_table_compute_gcp,
    ('multi_table', 'gcp'): _benchmark_multi_table_compute_gcp,
    ('single_table', 'aws'): benchmark_single_table_aws,
    ('multi_table', 'aws'): benchmark_multi_table_aws,
}


@contextmanager
def resolved_credential_filepath(credentials_config, build_credentials_dict_fn):
    """Yields a credential_filepath to use.

    - If credentials_config defines credential_filepath_env and it's set, yield that path.
    - Else build a dict (from env var names in YAML), write temp JSON, yield temp path,
      then delete it on exit.
    """
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

        yield tmp.name
    finally:
        try:
            os.remove(tmp.name)
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


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)

        else:
            result[key] = value
    return result


class BenchmarkConfig:
    """Build and validate benchmark configs."""

    def __init__(self):
        self.modality = None
        self.method_params = None
        self.credentials_config = {}
        self.compute = {
            'service': None,
        }
        self.instance_jobs = []
        self._is_validated = False

    @classmethod
    def load_from_yaml(cls, filepath):
        """Load a config from a YAML file."""
        instance = cls()
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)

        instance.modality = config_dict.get('modality')
        instance.method_params = config_dict.get('method_params', {})
        instance.credentials_config = config_dict.get('credentials', {})
        instance.compute = config_dict.get('compute', {})
        instance.instance_jobs = config_dict.get('instance_jobs', [])

        return instance

    def _get_credentials_dict(self):
        """Build a credentials dict from env vars specified in self.credentials_config."""
        creds = self.credentials_config
        gcp_json = os.getenv(creds.get('gcp_service_account_json_env', ''))
        credentials = {
            'aws': {
                'aws_access_key_id': os.getenv(creds.get('aws_access_key_id_env', '')),
                'aws_secret_access_key': os.getenv(creds.get('aws_secret_access_key_env', '')),
            },
            'gcp': {
                **json.loads(gcp_json),
                'gcp_project': os.getenv(creds.get('gcp_project_id_env', '')),
                'gcp_zone': os.getenv(creds.get('gcp_zone_env', '')),
            },
            'sdv': {
                'username': os.getenv(creds.get('sdv_username_env', '')),
                'license_key': os.getenv(creds.get('sdv_license_key_env', '')),
            },
        }

        return credentials

    def _validate_method_params(self):
        """Validate the method parameters."""

    def _validate_credentials(self):
        """Validate that the necessary credentials are available in the environment."""
        raise NotImplementedError

    def _validate_jobs(self):
        error_message = (
            "Each job in 'instance_jobs' must be a dict with 'synthesizers' (list of strings) "
            "and 'datasets' (list of strings or dict with 'include' and optional 'exclude')."
        )
        for job in self.instance_jobs:
            if 'datasets' not in job or 'synthesizers' not in job:
                raise ValueError(error_message)

            if not isinstance(job['synthesizers'], list) or not all(
                isinstance(s, str) for s in job['synthesizers']
            ):
                raise ValueError(error_message)

            if not (isinstance(job['datasets'], list) or isinstance(job['datasets'], dict)):
                raise ValueError(error_message)

            if isinstance(job['datasets'], list) and not all(
                isinstance(d, str) for d in job['datasets']
            ):
                raise ValueError(error_message)

            if isinstance(job['datasets'], dict):
                if ('include' not in job['datasets']) or (
                    not isinstance(job['datasets']['include'], list)
                    or not all(isinstance(d, str) for d in job['datasets']['include'])
                ):
                    raise ValueError(error_message)

                if 'exclude' in job['datasets']:
                    if (not isinstance(job['datasets']['exclude'], list)) or not all(
                        isinstance(d, str) for d in job['datasets']['exclude']
                    ):
                        raise ValueError(error_message)

    def validate(self):
        """Validate that the BenchmarkConfig is well-formed and can be run."""
        self._validate_method_params()
        self._validate_credentials()
        self._validate_jobs()

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

    def save(self):
        """Save the BenchmarkConfig."""
        raise NotImplementedError

    def load(self):
        """Load a BenchmarkConfig."""
        raise NotImplementedError
