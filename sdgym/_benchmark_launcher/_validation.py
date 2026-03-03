import json
import os

_REQUIRED_SECTIONS = {'aws', 'gcp'}
_OPTIONAL_SECTIONS = {'sdv'}
_ALLOWED_SECTIONS = _REQUIRED_SECTIONS | _OPTIONAL_SECTIONS
_GCP_SA_REQUIRED_KEYS = {'type', 'project_id', 'private_key', 'client_email', 'token_uri'}
_CREDENTIAL_SECTION_SCHEMA = {
    'aws': {
        'env_to_var': {
            'access_key_id_env': 'aws_access_key_id',
            'secret_access_key_env': 'aws_secret_access_key',
        },
        'required': {'aws_access_key_id', 'aws_secret_access_key'},
        'optional': set(),
    },
    'gcp': {
        'env_to_var': {
            'service_account_json_env': 'service_account_json',
            'project_id_env': 'gcp_project',
            'zone_env': 'gcp_zone',
        },
        'required': {'gcp_project', 'gcp_zone'},
        'optional': {'service_account_json'},
    },
    'sdv': {
        'env_to_var': {
            'username_env': 'username',
            'license_key_env': 'license_key',
        },
        'required': set(),
        'optional': {'username', 'license_key'},
    },
}


def _validate_structure(benchmark_config):
    """Validate the overall structure of the config (keys and types)."""
    errors = []
    if benchmark_config.modality not in ['single_table', 'multi_table']:
        errors.append(
            f"Invalid modality '{benchmark_config.modality}'. Must be 'single_table'"
            " or 'multi_table'."
        )

    if not isinstance(benchmark_config.method_params, dict):
        errors.append(
            f"'method_params' must be a dict. Found: {type(benchmark_config.method_params)}"
        )

    if not isinstance(benchmark_config.credentials_config, dict):
        errors.append(
            f"'credentials' must be a dict. Found: {type(benchmark_config.credentials_config)}"
        )

    if not isinstance(benchmark_config.compute, dict):
        errors.append(f"'compute' must be a dict. Found: {type(benchmark_config.compute)}")
    elif 'service' not in benchmark_config.compute or benchmark_config.compute['service'] not in [
        'gcp'
    ]:
        errors.append(
            f"'compute.service' must be either 'aws' or 'gcp'. Found: "
            f'{benchmark_config.compute.get("service")}'
        )

    return '\n'.join(errors) if errors else None


def _validate_modality(modality):
    """Validate that the modality is valid."""
    if modality not in ['single_table', 'multi_table']:
        return f"modality: Invalid modality '{modality}'. Must be 'single_table' or 'multi_table'."


def _validate_method_params(method_params):
    """Validate the method parameters."""
    errors = []
    if not isinstance(method_params, dict):
        return f'method_params: must be a dict. Found: {type(method_params)}'

    errors.append('la')


def _validate_jobs(instance_jobs):
    error_message = (
        "Each job in 'instance_jobs' must be a dict with 'synthesizers' (list of strings) "
        "and 'datasets' (list of strings or dict with 'include' and optional 'exclude')."
    )
    invalid_jobs = []
    for job in instance_jobs:
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


def _env(name: str | None) -> str | None:
    if not name:
        return None
    value = os.getenv(name)
    return value if value not in (None, '') else None


def _validate_credentials_config_structure(credentials_config):
    """Validate credential config structure.

    This validates:
    - either file mode (credential_filepath exists + readable json dict)
    - or env mode (required provider sections present; expected *_env keys present;
      referenced env vars are set)
    """
    errors = []
    if not isinstance(credentials_config, dict):
        return [f"'credentials' must be a dict. Found: {type(credentials_config)}"]

    allowed_top = {'credential_filepath'} | _ALLOWED_SECTIONS
    unknown_top = set(credentials_config.keys()) - allowed_top
    if unknown_top:
        errors.append(f'credentials: unknown top-level keys: {sorted(unknown_top)}')

    filepath = credentials_config.get('credential_filepath')
    if filepath is not None:
        if not isinstance(filepath, str) or not filepath:
            errors.append('credentials.credential_filepath must be a non-empty string path.')
            return errors

        if not os.path.isfile(filepath):
            errors.append(f'credentials file not found: {filepath}')
            return errors

        try:
            with open(filepath, 'r') as f:
                payload = json.load(f)
        except json.JSONDecodeError:
            errors.append(f'credentials file is not valid JSON: {filepath}')
            return errors
        except OSError as e:
            errors.append(f'credentials file could not be read ({filepath}): {e}')
            return errors

        if not isinstance(payload, dict):
            errors.append('credentials file JSON must be a dict at the top level.')
            return errors

        return errors

    for section in _REQUIRED_SECTIONS:
        if section not in credentials_config:
            errors.append(f'credentials.{section}: section is required but missing.')
            continue
        if not isinstance(credentials_config[section], dict):
            errors.append(
                f'credentials.{section}: must be a dict. Found: {type(credentials_config[section])}'
            )

    for section in ('sdv',):
        if section in credentials_config and not isinstance(credentials_config[section], dict):
            errors.append(
                f'credentials.{section}: must be a dict. Found: {type(credentials_config[section])}'
            )

    for section, schema in _CREDENTIAL_SECTION_SCHEMA.items():
        section_config = credentials_config.get(section)
        if section_config is None:
            continue
        if not isinstance(section_config, dict):
            continue

        expected_env_keys = set(schema['env_to_var'].keys())
        actual_keys = set(section_config.keys())

        missing_env_keys = expected_env_keys - actual_keys
        extra_keys = actual_keys - expected_env_keys
        if missing_env_keys:
            errors.append(f'credentials.{section}: missing keys: {sorted(missing_env_keys)}')
        if extra_keys:
            errors.append(f'credentials.{section}: unknown keys: {sorted(extra_keys)}')

        for env_key, canon_key in schema['env_to_var'].items():
            if env_key not in section_config:
                continue

            env_var_name = section_config.get(env_key)
            if not isinstance(env_var_name, str) or not env_var_name:
                errors.append(f'credentials.{section}.{env_key}: must be a non-empty env var name.')
                continue

            env_val = _env(env_var_name)
            if env_val is None:
                errors.append(
                    f"Environment variable '{env_var_name}' (for credentials.{section}.{env_key}) "
                    'is not set or empty.'
                )
                continue

            if section == 'gcp' and env_key == 'service_account_json_env':
                try:
                    json.loads(env_val)
                except json.JSONDecodeError:
                    errors.append(
                        f"Environment variable '{env_var_name}' must contain valid JSON "
                        '(GCP service account).'
                    )

    return errors


def _get_credentials(credentials_config):
    """Get resolved credentials dict."""
    config = credentials_config or {}
    filepath = config.get('credential_filepath')
    if filepath:
        with open(filepath, 'r') as f:
            raw = json.load(f)

        result = {section: (raw.get(section) or {}) for section in _ALLOWED_SECTIONS}
        for section in _ALLOWED_SECTIONS:
            if not isinstance(result[section], dict):
                result[section] = {}

        return result

    resolved = {section: {} for section in _ALLOWED_SECTIONS}
    for section, schema in _CREDENTIAL_SECTION_SCHEMA.items():
        section_config = config.get(section) or {}
        if not isinstance(section_config, dict):
            continue

        for env_key, canon_key in schema['env_to_var'].items():
            env_var_name = section_config.get(env_key)
            resolved[section][canon_key] = (
                _env(env_var_name) if isinstance(env_var_name, str) else None
            )

    gcp = resolved.get('gcp', {})
    sa_json = gcp.pop('service_account_json', None)
    if sa_json:
        sa_obj = json.loads(sa_json)
        keep = {'gcp_project': gcp.get('gcp_project'), 'gcp_zone': gcp.get('gcp_zone')}
        resolved['gcp'] = {**sa_obj, **keep}

    resolved['sdv'].setdefault('username', None)
    resolved['sdv'].setdefault('license_key', None)

    return resolved


def _validate_resolved_credentials(credentials):
    """Validate the resolved credentials dict (actual values, canonical keys)."""
    errors = []
    if not isinstance(credentials, dict):
        return ['credentials must be a dict.']

    unknown_sections = set(credentials.keys()) - _ALLOWED_SECTIONS
    if unknown_sections:
        errors.append(f'credentials has unknown sections: {sorted(unknown_sections)}')

    missing_sections = _REQUIRED_SECTIONS - set(credentials.keys())
    if missing_sections:
        errors.append(f'credentials missing required sections: {sorted(missing_sections)}')

    for section in _ALLOWED_SECTIONS & set(credentials.keys()):
        sec = credentials.get(section)
        if not isinstance(sec, dict):
            errors.append(f'credentials["{section}"] must be a dict.')
            continue

    for section, schema in _CREDENTIAL_SECTION_SCHEMA.items():
        sec = credentials.get(section, {})
        if not isinstance(sec, dict):
            continue

        required = set(schema['required'])
        optional = set(schema['optional'])
        allowed = required | optional
        if section == 'sdv':
            username = sec.get('username')
            license_key = sec.get('license_key')
            if (username in (None, '')) and (license_key in (None, '')):
                continue

            if username in (None, ''):
                errors.append(
                    "credentials['sdv']['username'] is required when SDV credentials are provided."
                )
            if license_key in (None, ''):
                errors.append(
                    "credentials['sdv']['license_key'] is required when SDV credentials"
                    ' are provided.'
                )
            continue

        missing = required - set(sec.keys())
        if missing:
            errors.append(f"credentials['{section}'] missing keys: {sorted(missing)}")

        for k in required:
            if sec.get(k) in (None, ''):
                errors.append(f"credentials['{section}']['{k}'] is missing or empty.")

        if section == 'aws':
            extra = set(sec.keys()) - allowed
            if extra:
                errors.append(f"credentials['aws'] has unknown keys: {sorted(extra)}")

        # For GCP, we allow many extra keys from service account JSON,
        # but we still require core SA fields.
        if section == 'gcp':
            for k in _GCP_SA_REQUIRED_KEYS:
                if sec.get(k) in (None, ''):
                    errors.append(f"credentials['gcp']['{k}'] is missing or empty.")

    return errors


def _validate_credentials_config(credentials_config):
    """Validate credentials config end-to-end."""
    errors = _validate_credentials_config_structure(credentials_config)
    if errors:
        return '\n'.join(errors)

    credentials = _get_credentials(credentials_config)
    errors = _validate_resolved_credentials(credentials)
    return '\n'.join(errors) if errors else None
