"""Build a ready-to-run benchmark configuration from YAML files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import os

import yaml

from sdgym._benchmark.benchmark import _benchmark_single_table_compute_gcp, _benchmark_multi_table_compute_gcp
from sdgym.benchmark import benchmark_single_table_aws, benchmark_multi_table_aws
from sdgym.s3 import get_s3_client, parse_s3_path

_METHODS = {
    ('single_table', 'gcp'): _benchmark_single_table_compute_gcp,
    ('multi_table', 'gcp'): _benchmark_multi_table_compute_gcp,
    ('single_table', 'aws'): benchmark_single_table_aws,
    ('multi_table', 'aws'): benchmark_multi_table_aws,
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    result = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        
        else:
            result[key] = value
    return result


class BenchmarkConfig:
    """Build and validate benchmark configs."""

    def __init__(self):
        self.modality = None
        self.method_params = None
        self.credentials = {}
        self.compute = {
            'service': None,
        }
        self.instance_jobs = []

    @classmethod
    def load_from_yaml(cls, filepath):
        """Load a config from a YAML file."""
        instance = cls()
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
    
        instance.modality = config_dict.get('modality')
        instance.method_params = config_dict.get('method_params', {})
        instance.credentials = config_dict.get('credentials', {})
        instance.compute = config_dict.get('compute', {})
        instance.instance_jobs = config_dict.get('instance_jobs', [])

        return instance

    def _validate_method_params(self):
        raise NotImplementedError

    def _validate_credentials(self):
        raise NotImplementedError

    def _validate_jobs(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def _load_credentials(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    @classmethod
    def default(cls) -> 'BenchmarkConfig':
        return cls(launcher_dir=Path(__file__).resolve().parent)

    def _load_yaml(self, filepath: str | Path) -> Dict[str, Any]:
        path = Path(filepath)
        if not path.is_absolute():
            path = self.launcher_dir / path

        if not path.exists():
            raise FileNotFoundError(f'Config file not found: {path}')

        data = yaml.safe_load(path.read_text())
        if data is None:
            return {}

        if not isinstance(data, dict):
            raise ValueError(f'YAML config must be a mapping/dict: {path}')

        return data

    def load_credentials(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve *_env keys into concrete credential values in-place."""
        creds = config.get('credentials', {})
        if not isinstance(creds, dict):
            raise ValueError('Invalid credentials section (must be a dict).')

        def _resolve_env(env_key: str, target_key: str) -> None:
            env_name = creds.get(env_key)
            if not isinstance(env_name, str) or not env_name:
                raise ValueError(f'Missing/invalid credentials.{env_key}.')
            value = os.getenv(env_name)
            if not value:
                raise ValueError(f'Missing environment variable: {env_name} (required by credentials.{env_key}).')
            creds[target_key] = value

        _resolve_env('credential_filepath_env', 'credential_filepath')
        _resolve_env('aws_access_key_id_env', 'aws_access_key_id')
        _resolve_env('aws_secret_access_key_env', 'aws_secret_access_key')

        # Optional: keep the *_env keys, or delete them to avoid confusion
        # for k in ('credential_filepath_env', 'aws_access_key_id_env', 'aws_secret_access_key_env'):
        #     creds.pop(k, None)

        config['credentials'] = creds
        return config


    def create_config(self, config_filepath: str, base_config: Optional[str | dict] = None) -> Dict[str, Any]:
        """Create a valid and ready-to-use benchmark configuration.

        Args:
            config_filepath: Path to modality yaml config (e.g. benchmark_single_table.yaml)
            base_config: None -> use base_benchmark.yaml.
                        str -> path to base yaml.
                        dict -> base config dict.

        Returns:
            Fully built config dict ready for run_benchmark.py
        """
        if base_config is None:
            base = self._load_yaml('base_benchmark.yaml')
        elif isinstance(base_config, str):
            base = self._load_yaml(base_config)
        elif isinstance(base_config, dict):
            base = dict(base_config)
        else:
            raise TypeError('base_config must be None, a filepath string, or a dict.')

        override = self._load_yaml(config_filepath)
        merged = _deep_merge(base, override)

        built = self._normalize(merged)
        self._validate_basic(built)
        self.load_credentials(built)
        # Optional/“ideal” validations can be enabled once you’re ready:
        # self._validate_s3_access(built)

        return built

    def _normalize(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize to a runner-friendly structure."""
        modality = config.get('modality')
        jobs_cfg = config.get('jobs', {})

        datasets = jobs_cfg.get('datasets')
        instance_jobs = jobs_cfg.get('instance_jobs')

        # Produce a flat list of job dicts, each with datasets+synthesizers
        jobs: list[dict] = []
        if isinstance(instance_jobs, list):
            for idx, item in enumerate(instance_jobs):
                if not isinstance(item, dict):
                    raise ValueError(f'jobs.instance_jobs[{idx}] must be a mapping/dict.')
                job = dict(item)
                # Allow per-job datasets override, otherwise inherit shared datasets
                job.setdefault('datasets', datasets)
                jobs.append(job)

        config['modality'] = modality
        config['jobs'] = jobs
        return config

    def _validate_basic(self, config: Dict[str, Any]) -> None:
        """Basic validation: names/value/type and required keys."""
        # modality
        modality = config.get('modality')
        if modality not in {'single_table', 'multi_table'}:
            raise ValueError()

        # output
        output = config.get('output', {})
        dest = output.get('destination')
        if not isinstance(dest, str) or not dest:
            raise ValueError('Missing/invalid output.destination (must be a non-empty string).')

        # timeout
        timeout = config.get('timeout_seconds')
        if not isinstance(timeout, int) or timeout <= 0:
            raise ValueError('Missing/invalid timeout_seconds (must be a positive int).')

        # compute
        compute = config.get('compute', {})
        service = compute.get('service')
        if service not in {'gcp', 'aws'}:
            raise ValueError()

        # credentials env var names
        creds = config.get('credentials', {})
        for key in ('credential_filepath_env', 'aws_access_key_id_env', 'aws_secret_access_key_env'):
            if key not in creds or not isinstance(creds[key], str) or not creds[key]:
                raise ValueError(f'Missing/invalid credentials.{key} (must be a non-empty string).')

        # jobs
        jobs = config.get('jobs')
        if not isinstance(jobs, list) or not jobs:
            raise ValueError('No jobs found. Ensure jobs.instance_jobs is a non-empty list.')

        for i, job in enumerate(jobs):
            synths = job.get('synthesizers')
            dsets = job.get('datasets')
            if not isinstance(synths, list) or not synths or not all(isinstance(x, str) for x in synths):
                raise ValueError(f'jobs[{i}].synthesizers must be a non-empty list of strings.')
            if not isinstance(dsets, list) or not dsets or not all(isinstance(x, str) for x in dsets):
                raise ValueError(f'jobs[{i}].datasets must be a non-empty list of strings.')

    def _validate_s3_access(self, config: Dict[str, Any]) -> None:
        """Optional: validate S3 read/write access using provided AWS env var names."""
        dest = config['output']['destination']
        if not dest.startswith('s3://'):
            return

        creds = config['credentials']
        import os

        key_env = creds['aws_access_key_id_env']
        secret_env = creds['aws_secret_access_key_env']
        aws_access_key_id = os.getenv(key_env)
        aws_secret_access_key = os.getenv(secret_env)

        if not aws_access_key_id or not aws_secret_access_key:
            raise ValueError(
                f'Missing AWS credentials in env vars: {key_env}, {secret_env} '
                '(required to validate S3 access).'
            )

        s3 = get_s3_client(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        bucket, prefix = parse_s3_path(dest)

        test_key = f'{prefix.rstrip('/')}/_sdgym_write_test'
        s3.put_object(Bucket=bucket, Key=test_key, Body=b'ok')
        s3.get_object(Bucket=bucket, Key=test_key)
        s3.delete_object(Bucket=bucket, Key=test_key)
