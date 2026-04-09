from sdgym._benchmark_launcher.utils import (
    _AWS_CREDENTIAL_KEYS,
    _GCP_COMPUTE_REQUIRED_KEYS,
    _GCP_SERVICE_ACCOUNT_REQUIRED_KEYS,
    _is_unique_string_list,
    resolve_credentials,
)

_INJECTED_PARAMS = {
    'credentials',
    'synthesizers',
    'sdv_datasets',
    'compute',
    'output_destination',
}


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
        if service is None:
            errors.append('compute.service: is required but missing.')

    return sorted(errors)


def _validate_compute_gcp(compute):
    errors = []
    for key in _GCP_COMPUTE_REQUIRED_KEYS:
        if not compute.get(key):
            errors.append(f'compute.{key} is required for GCP compute.')

    gpu_count = int(compute.get('gpu_count') or 0)
    if gpu_count > 0 and not compute.get('gpu_type'):
        errors.append('compute.gpu_type is required when compute.gpu_count > 0.')

    return sorted(errors)


def _validate_compute(compute):
    service = compute.get('service')
    if service == 'gcp':
        return _validate_compute_gcp(compute)

    return [f"compute.service: must be 'gcp'. Found: {service!r}"]


def _validate_method_params(method_params, method_to_run):
    errors = []
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

    illegal = _INJECTED_PARAMS & set(method_params)
    if illegal:
        errors.append(
            f'method_params: must not define injected parameters {sorted(illegal)} '
            f'(resolved from credentials/instance_jobs).'
        )

    return errors


def _validate_instance_jobs(instance_jobs):
    error_message = (
        "Each job in 'instance_jobs' must be a dict with an 'output_destination' (string), "
        "'synthesizers' (list of unique strings), and 'datasets' (list of unique strings or "
        "dict with 'include' and optional 'exclude')."
    )
    invalid_jobs = []
    for job in instance_jobs:
        if not isinstance(job, dict):
            invalid_jobs.append(job)
            continue

        if 'datasets' not in job or 'synthesizers' not in job or 'output_destination' not in job:
            invalid_jobs.append(job)
            continue

        synthesizers = job['synthesizers']
        if not _is_unique_string_list(synthesizers):
            invalid_jobs.append(job)
            continue

        output_destination = job['output_destination']
        if not isinstance(output_destination, str) or not output_destination:
            invalid_jobs.append(job)
            continue

        datasets = job['datasets']
        if isinstance(datasets, list):
            if not _is_unique_string_list(datasets):
                invalid_jobs.append(job)
            continue

        if isinstance(datasets, dict):
            include = datasets.get('include')
            exclude = datasets.get('exclude')
            if not _is_unique_string_list(include):
                invalid_jobs.append(job)
                continue

            if exclude is not None and not _is_unique_string_list(exclude):
                invalid_jobs.append(job)
            continue

        invalid_jobs.append(job)

    if not invalid_jobs:
        return []

    invalid_jobs_str = '\n'.join(str(job) for job in invalid_jobs)

    return [f'{error_message}\nInvalid jobs:\n{invalid_jobs_str}']


def _validate_aws_credentials(credentials):
    errors = []
    aws = credentials.get('aws', {})
    if not isinstance(aws, dict):
        errors.append("credentials['aws'] must be a dict.")
    else:
        if any(aws.values()):
            for key in _AWS_CREDENTIAL_KEYS:
                key = key.lower()
                if aws.get(key) in (None, ''):
                    errors.append(f"credentials['aws']['{key}'] is missing or empty.")

    return sorted(errors)


def _validate_sdv_enterprise_credentials(credentials):
    errors = []
    sdv = credentials.get('sdv_enterprise', {})
    if not isinstance(sdv, dict):
        errors.append("credentials['sdv_enterprise'] must be a dict.")
    else:
        username = sdv.get('sdv_enterprise_username')
        license_key = sdv.get('sdv_enterprise_license_key')
        message = (
            "credentials['sdv_enterprise'] require both 'sdv_enterprise_username' and "
            "'sdv_enterprise_license_key' to be provided and non-empty if any SDV Enterprise"
            ' credential is provided.'
        )
        if bool(username) != bool(license_key):
            errors.append(message)

    return sorted(errors)


def _validate_gcp_credentials(credentials):
    errors = []
    gcp = credentials.get('gcp', {})
    if not isinstance(gcp, dict):
        errors.append("credentials['gcp'] must be a dict.")
    else:
        if gcp:
            for key in _GCP_SERVICE_ACCOUNT_REQUIRED_KEYS:
                if gcp.get(key) in (None, ''):
                    errors.append(f"credentials['gcp']['{key}'] is missing or empty.")

    return sorted(errors)


def _validate_resolved_credentials(credentials):
    errors = []
    errors.extend(_validate_aws_credentials(credentials))
    errors.extend(_validate_sdv_enterprise_credentials(credentials))
    errors.extend(_validate_gcp_credentials(credentials))

    return sorted(errors)


def _validate_credentials(credentials_filepath):
    if credentials_filepath is not None and not isinstance(credentials_filepath, str):
        return ['credentials_filepath: must be a string path to the credentials file or None.']

    credentials = resolve_credentials(credentials_filepath)
    return _validate_resolved_credentials(credentials)
