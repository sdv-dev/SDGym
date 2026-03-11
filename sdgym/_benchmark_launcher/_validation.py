import inspect
import json
import os
from urllib.parse import urlparse

_REQUIRED_SECTIONS = {'aws', 'gcp'}
_OPTIONAL_SECTIONS = {'sdv_enterprise'}
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
    'sdv_enterprise': {
        'env_to_var': {
            'username_env': 'username',
            'license_key_env': 'license_key',
        },
        'required': set(),
        'optional': {'username', 'license_key'},
    },
}
_INJECTED_PARAMS = {'credentials', 'synthesizers', 'sdv_datasets', 'compute_config'}


def _as_errors(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v]

    return [str(value)]


def _format_sectioned_errors(section_errors):
    parts = ['BenchmarkConfig validation failed:\n']
    for section, raw in section_errors.items():
        errs = _as_errors(raw)
        if not errs:
            continue
        parts.append(f'[{section}]')
        parts.extend([f'- {e}' for e in errs])
        parts.append('')

    return '\n'.join(parts).rstrip()


def _env(name):
    if not name:
        return None
    value = os.getenv(name)

    return value if value not in (None, '') else None


def _get_credentials(credential_locations):
    """Get resolved credentials dict."""
    config = credential_locations or {}
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

    resolved['sdv_enterprise'].setdefault('username', None)
    resolved['sdv_enterprise'].setdefault('license_key', None)

    return resolved


def _validate_structure(config):
    errors = []
    if config.modality not in ('single_table', 'multi_table'):
        errors.append(
            f"modality: must be 'single_table' or 'multi_table'. Found: {config.modality!r}"
        )

    expected_types = {
        'method_params': dict,
        'credential_locations': dict,
        'compute': dict,
        'instance_jobs': list,
    }
    for key, expected_type in expected_types.items():
        value = getattr(config, key, None)
        if value is None:
            errors.append(f'{key}: is a required section but missing.')
        elif not isinstance(value, expected_type):
            errors.append(f'{key}: must be a {expected_type.__name__}. Found: {type(value)}')

    compute = getattr(config, 'compute', None)
    if isinstance(compute, dict):
        service = compute.get('service')
        if service not in ('gcp',):
            errors.append(f"compute.service: must be 'gcp'. Found: {service!r}")

    return errors


def _validate_method_params(method_params, method_to_run):
    errors = []
    output_destination = method_params.get('output_destination')
    if not isinstance(output_destination, str) or not output_destination:
        errors.append(
            'method_params.output_destination: is required and must be a non-empty string.'
        )
    else:
        parsed = urlparse(output_destination)
        if parsed.scheme != 's3':
            errors.append(
                'method_params.output_destination: must be an S3 URI like "s3://bucket/prefix/".'
            )
        elif not output_destination.endswith('/'):
            errors.append('method_params.output_destination: should end with "/".')

    timeout = method_params.get('timeout')
    if timeout is not None:
        if not isinstance(timeout, int):
            errors.append(
                f'method_params.timeout: must be int seconds. Found: {timeout!r} ({type(timeout)})'
            )
        elif timeout <= 0:
            errors.append('method_params.timeout: must be > 0.')

    for key in ('compute_quality_score', 'compute_diagnostic_score', 'compute_privacy_score'):
        value = method_params.get(key)
        if value is not None and not isinstance(value, bool):
            errors.append(f'method_params.{key}: must be bool. Found: {value!r} ({type(value)})')

    sig = inspect.signature(method_to_run)
    required = {
        parameter.name
        for parameter in sig.parameters.values()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    required_from_yaml = required - _INJECTED_PARAMS
    missing = required_from_yaml - set(method_params)
    if missing:
        errors.append(
            f'method_params: missing required parameters for {method_to_run.__name__}:'
            f' {sorted(missing)}'
        )

    illegal = _INJECTED_PARAMS & set(method_params)
    if illegal:
        errors.append(
            f'method_params: must not define injected parameters {sorted(illegal)} '
            f'(resolved from credentials/instance_jobs).'
        )

    return errors


def _validate_instance_jobs(instance_jobs):
    error_message = (
        "Each job in 'instance_jobs' must be a dict with 'synthesizers' (list of strings) "
        "and 'datasets' (list of strings or dict with 'include' and optional 'exclude')."
    )
    invalid_jobs = []
    for job in instance_jobs:
        if not isinstance(job, dict):
            invalid_jobs.append(job)
            continue

        if 'datasets' not in job or 'synthesizers' not in job:
            invalid_jobs.append(job)
            continue

        synthesizers = job['synthesizers']
        if not isinstance(synthesizers, list) or not all(isinstance(s, str) for s in synthesizers):
            invalid_jobs.append(job)
            continue

        datasets = job['datasets']
        if isinstance(datasets, list):
            if not all(isinstance(d, str) for d in datasets):
                invalid_jobs.append(job)
            continue

        if isinstance(datasets, dict):
            include = datasets.get('include')
            exclude = datasets.get('exclude')

            if not isinstance(include, list) or not all(isinstance(d, str) for d in include):
                invalid_jobs.append(job)
                continue

            if exclude is not None and (
                not isinstance(exclude, list) or not all(isinstance(d, str) for d in exclude)
            ):
                invalid_jobs.append(job)
            continue

        invalid_jobs.append(job)

    if not invalid_jobs:
        return []

    invalid_jobs_str = '\n'.join(str(job) for job in invalid_jobs)

    return [f'{error_message}\nInvalid jobs:\n{invalid_jobs_str}']


def _validate_credential_locations_structure(credential_locations):
    errors = []
    allowed_top = {'credential_filepath'} | _ALLOWED_SECTIONS
    unknown = set(credential_locations) - allowed_top
    if unknown:
        errors.append(f'credentials: unknown top-level keys: {sorted(unknown)}')

    filepath = credential_locations.get('credential_filepath')
    if filepath is not None:
        if not isinstance(filepath, str) or not filepath:
            return errors + [
                'credential_locations.credential_filepath: must be a non-empty string.'
            ]
        elif not os.path.isfile(filepath):
            return errors + [
                f'credential_locations.credential_filepath: file not found: {filepath}'
            ]

        try:
            with open(filepath, 'r') as f:
                cred_dict = json.load(f)
        except Exception as e:
            return errors + [
                f'credential_locations.credential_filepath: invalid JSON: ({filepath}): {e}'
            ]

        if not isinstance(cred_dict, dict):
            return errors + ['credentials file JSON must be a dict at the top level.']

        for section in _ALLOWED_SECTIONS & set(cred_dict):
            if not isinstance(cred_dict.get(section), dict):
                errors.append(f'credentials file section "{section}" must be a dict.')

        return errors

    for section in _REQUIRED_SECTIONS:
        if section not in credential_locations:
            errors.append(f'credential_locations.{section}: section is required but missing.')
        elif not isinstance(credential_locations[section], dict):
            errors.append(
                f'credential_locations.{section}: must be a dict. Found: '
                f'{type(credential_locations[section])}'
            )

    if 'sdv_enterprise' in credential_locations and not isinstance(
        credential_locations['sdv_enterprise'], dict
    ):
        errors.append(
            f'credential_locations.sdv_enterprise: must be a dict. Found: {
                type(credential_locations["sdv"])
            }'
        )

    for section, schema in _CREDENTIAL_SECTION_SCHEMA.items():
        section_cfg = credential_locations.get(section)
        if section_cfg is None:
            continue
        if not isinstance(section_cfg, dict):
            continue

        expected = set(schema['env_to_var'])
        actual = set(section_cfg)
        missing = expected - actual
        extra = actual - expected
        if missing:
            errors.append(f'credential_locations.{section}: missing keys: {sorted(missing)}')
        if extra:
            errors.append(f'credential_locations.{section}: unknown keys: {sorted(extra)}')

        for env_key in expected & actual:
            env_var = section_cfg.get(env_key)
            if not isinstance(env_var, str) or not env_var:
                errors.append(
                    f'credential_locations.{section}.{env_key}: must be a non-empty env var name.'
                )
                continue

            value = _env(env_var)
            if value is None:
                errors.append(
                    f"Environment variable '{env_var}' (for credential_locations."
                    f'{section}.{env_key}) is not set or empty.'
                )
                continue

            if section == 'gcp' and env_key == 'service_account_json_env':
                try:
                    json.loads(value)
                except json.JSONDecodeError:
                    errors.append(
                        f"Environment variable '{env_var}' must contain valid JSON (GCP "
                        'service account).'
                    )

    return sorted(errors)


def _validate_resolved_credentials(credentials):
    errors = []
    unknown_sections = set(credentials) - _ALLOWED_SECTIONS
    if unknown_sections:
        errors.append(f'credentials: unknown sections: {sorted(unknown_sections)}')

    missing_sections = _REQUIRED_SECTIONS - set(credentials)
    if missing_sections:
        errors.append(f'credentials: missing required sections: {sorted(missing_sections)}')

    for section in _ALLOWED_SECTIONS & set(credentials):
        section_dict = credentials.get(section)
        if not isinstance(section_dict, dict):
            errors.append(f"credentials['{section}'] must be a dict.")
            continue

    for section, schema in _CREDENTIAL_SECTION_SCHEMA.items():
        section_dict = credentials.get(section, {})
        if not isinstance(section_dict, dict):
            continue

        if section == 'sdv_enterprise':
            username = section_dict.get('username')
            licence_key = section_dict.get('license_key')
            if (username in (None, '')) and (licence_key in (None, '')):
                continue
            if username in (None, ''):
                errors.append(
                    "credential_locations['sdv_enterprise']['username'] is required when SDV"
                    ' credentials are provided.'
                )
            if licence_key in (None, ''):
                errors.append(
                    "credential_locations['sdv_enterprise']['license_key'] is required when SDV"
                    ' credentials are provided.'
                )
            continue

        for key in schema['required']:
            if key not in section_dict:
                errors.append(f'credentials["{section}"] missing key: "{key}"')
            elif section_dict.get(key) in (None, ''):
                errors.append(f'credentials["{section}"]["{key}"] is missing or empty.')

        if section == 'aws':
            allowed = set(schema['required']) | set(schema['optional'])
            extra = set(section_dict) - allowed
            if extra:
                errors.append(f'credentials["aws"] has unknown keys: {sorted(extra)}')

        if section == 'gcp':
            for key in _GCP_SA_REQUIRED_KEYS:
                if section_dict.get(key) in (None, ''):
                    errors.append(f'credentials["gcp"]["{key}"] is missing or empty.')

    return sorted(errors)


def _validate_credential_locations(credential_locations):
    errors = _validate_credential_locations_structure(credential_locations)
    if errors:
        return errors

    credentials = _get_credentials(credential_locations)
    return _validate_resolved_credentials(credentials)
