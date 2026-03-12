import inspect
from urllib.parse import urlparse

from sdgym._benchmark_launcher.utils import (
    _AWS_CREDENTIAL_KEYS,
    _GCP_SERVICE_ACCOUNT_REQUIRED_KEYS,
    resolve_credentials,
)

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


def _validate_structure(config):
    errors = []
    if config.modality not in ('single_table', 'multi_table'):
        errors.append(
            f"modality: must be 'single_table' or 'multi_table'. Found: {config.modality!r}"
        )

    if config.credentials_filepath is not None and not isinstance(config.credentials_filepath, str):
        errors.append('credentials_filepath must be a string or None.')

    expected_types = {
        'method_params': dict,
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

    return sorted(errors)


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


def _validate_resolved_credentials(credentials):
    errors = []
    aws = credentials.get('aws', {})
    if not isinstance(aws, dict):
        errors.append('credentials["aws"] must be a dict.')
    else:
        if any(aws.values()):
            for key in _AWS_CREDENTIAL_KEYS:
                if aws.get(key) in (None, ''):
                    errors.append(f'credentials["aws"]["{key}"] is missing or empty.')

    sdv = credentials.get('sdv_enterprise', {})
    if not isinstance(sdv, dict):
        errors.append('credentials["sdv_enterprise"] must be a dict.')
    else:
        username = sdv.get('SDV_ENTERPRISE_USERNAME')
        license_key = sdv.get('SDV_ENTERPRISE_LICENSE_KEY')
        if username or license_key:
            if not username:
                errors.append(
                    "credentials['sdv_enterprise']['SDV_ENTERPRISE_USERNAME'] "
                    'is required when SDV Enterprise credentials are provided.'
                )
            if not license_key:
                errors.append(
                    "credentials['sdv_enterprise']['SDV_ENTERPRISE_LICENSE_KEY'] "
                    'is required when SDV Enterprise credentials are provided.'
                )

    gcp = credentials.get('gcp', {})
    if not isinstance(gcp, dict):
        errors.append('credentials["gcp"] must be a dict.')
    else:
        if gcp:
            for key in _GCP_SERVICE_ACCOUNT_REQUIRED_KEYS:
                if gcp.get(key) in (None, ''):
                    errors.append(f'credentials["gcp"]["{key}"] is missing or empty.')

    return sorted(errors)


def _validate_credentials(credentials_filepath):
    if credentials_filepath is not None and not isinstance(credentials_filepath, str):
        return ['credentials_filepath: must be a string path to the credentials file or None.']

    credentials = resolve_credentials(credentials_filepath)
    return _validate_resolved_credentials(credentials)
