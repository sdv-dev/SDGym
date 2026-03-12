"""Unit tests for the benchmark launcher validation."""

from types import SimpleNamespace
from unittest.mock import patch

from sdgym._benchmark_launcher._validation import (
    _as_errors,
    _format_sectioned_errors,
    _validate_credentials,
    _validate_instance_jobs,
    _validate_method_params,
    _validate_resolved_credentials,
    _validate_structure,
)


def test__as_errors_with_none():
    """Test `_as_errors` returns an empty list for None."""
    # Setup
    value = None

    # Run
    errors = _as_errors(value)

    # Assert
    assert errors == []


def test__as_errors_with_list():
    """Test `_as_errors` converts list items to strings and drops falsy values."""
    # Setup
    value = ['error 1', None, '', 123]

    # Run
    errors = _as_errors(value)

    # Assert
    assert errors == ['error 1', '123']


def test__as_errors_with_scalar():
    """Test `_as_errors` wraps a scalar value in a list."""
    # Setup
    value = 123

    # Run
    errors = _as_errors(value)

    # Assert
    assert errors == ['123']


def test__format_sectioned_errors():
    """Test `_format_sectioned_errors` formats non-empty sections."""
    # Setup
    section_errors = {
        'structure': ['bad modality', 'missing compute'],
        'credentials': [],
        'instance_jobs': 'invalid job',
    }

    # Run
    formatted = _format_sectioned_errors(section_errors)

    # Assert
    expected = (
        'BenchmarkConfig validation failed:\n\n'
        '[structure]\n'
        '- bad modality\n'
        '- missing compute\n\n'
        '[instance_jobs]\n'
        '- invalid job'
    )
    assert formatted == expected


def test__validate_structure_valid():
    """Test `_validate_structure` returns no errors for a valid config."""
    # Setup
    config = SimpleNamespace(
        modality='single_table',
        method_params={'output_destination': 's3://bucket/prefix/'},
        credentials_filepath='creds.json',
        compute={'service': 'gcp'},
        instance_jobs=[],
    )

    # Run
    errors = _validate_structure(config)

    # Assert
    assert errors == []


def test__validate_structure_invalid():
    """Test `_validate_structure` returns errors for invalid config structure."""
    # Setup
    config = SimpleNamespace(
        modality='bad_modality',
        method_params=[],
        credentials_filepath=None,
        compute={'service': 'aws'},
        instance_jobs={},
    )

    # Run
    errors = _validate_structure(config)

    # Assert
    assert errors == [
        "compute.service: must be 'gcp'. Found: 'aws'",
        "instance_jobs: must be a list. Found: <class 'dict'>",
        "method_params: must be a dict. Found: <class 'list'>",
        "modality: must be 'single_table' or 'multi_table'. Found: 'bad_modality'",
    ]


def test__validate_method_params_valid():
    """Test `_validate_method_params` returns no errors for valid params."""

    # Setup
    def method_to_run(output_destination, timeout=10, credentials=None, synthesizers=None):
        return None

    method_params = {
        'output_destination': 's3://bucket/prefix/',
        'timeout': 60,
        'compute_quality_score': True,
        'compute_diagnostic_score': False,
        'compute_privacy_score': True,
    }

    # Run
    errors = _validate_method_params(method_params, method_to_run)

    # Assert
    assert errors == []


def test__validate_method_params_invalid():
    """Test `_validate_method_params` returns errors for invalid params."""

    # Setup
    def method_to_run(output_destination, required_param, credentials=None, synthesizers=None):
        return None

    method_params = {
        'output_destination': 'not-an-s3-uri',
        'timeout': 0,
        'compute_quality_score': 'yes',
        'credentials': 'forbidden',
    }

    # Run
    errors = _validate_method_params(method_params, method_to_run)

    # Assert
    assert errors == [
        'method_params.output_destination: must be an S3 URI like "s3://bucket/prefix/".',
        'method_params.timeout: must be > 0.',
        "method_params.compute_quality_score: must be bool. Found: 'yes' (<class 'str'>)",
        "method_params: missing required parameters for method_to_run: ['required_param']",
        "method_params: must not define injected parameters ['credentials'] "
        '(resolved from credentials/instance_jobs).',
    ]


def test__validate_method_params_requires_trailing_slash():
    """Test `_validate_method_params` requires output_destination to end with slash."""

    # Setup
    def method_to_run(output_destination, credentials=None):
        return None

    method_params = {'output_destination': 's3://bucket/prefix'}

    # Run
    errors = _validate_method_params(method_params, method_to_run)

    # Assert
    assert errors == ['method_params.output_destination: should end with "/".']


def test__validate_method_params_timeout_must_be_int():
    """Test `_validate_method_params` validates timeout type."""

    # Setup
    def method_to_run(output_destination, credentials=None):
        return None

    method_params = {
        'output_destination': 's3://bucket/prefix/',
        'timeout': '60',
    }

    # Run
    errors = _validate_method_params(method_params, method_to_run)

    # Assert
    assert errors == ["method_params.timeout: must be int seconds. Found: '60' (<class 'str'>)"]


def test__validate_instance_jobs_valid():
    """Test `_validate_instance_jobs` returns no errors for valid jobs."""
    # Setup
    instance_jobs = [
        {
            'synthesizers': ['synth1', 'synth2'],
            'datasets': ['adult', 'census'],
        },
        {
            'synthesizers': ['synth3'],
            'datasets': {
                'include': ['adult', 'census'],
                'exclude': ['adult'],
            },
        },
    ]

    # Run
    errors = _validate_instance_jobs(instance_jobs)

    # Assert
    assert errors == []


def test__validate_instance_jobs_invalid():
    """Test `_validate_instance_jobs` returns an error for invalid jobs."""
    # Setup
    instance_jobs = [
        'not-a-dict',
        {'synthesizers': ['synth1']},
        {'synthesizers': 'not-a-list', 'datasets': ['adult']},
        {'synthesizers': ['synth1'], 'datasets': [1, 2]},
        {'synthesizers': ['synth1'], 'datasets': {'include': 'adult'}},
        {'synthesizers': ['synth1'], 'datasets': {'include': ['adult'], 'exclude': 'census'}},
    ]

    # Run
    errors = _validate_instance_jobs(instance_jobs)

    # Assert
    assert len(errors) == 1
    assert "Each job in 'instance_jobs' must be a dict" in errors[0]
    assert 'not-a-dict' in errors[0]
    assert "{'synthesizers': ['synth1']}" in errors[0]


def test__validate_resolved_credentials_valid():
    """Test `_validate_resolved_credentials` returns no errors for valid credentials."""
    # Setup
    credentials = {
        'aws': {
            'AWS_ACCESS_KEY_ID': 'AKIA',
            'AWS_SECRET_ACCESS_KEY': 'SECRET',
        },
        'sdv_enterprise': {
            'SDV_ENTERPRISE_USERNAME': 'user',
            'SDV_ENTERPRISE_LICENSE_KEY': 'license',
        },
        'gcp': {
            'type': 'service_account',
            'project_id': 'my-project',
            'private_key_id': 'private-key-id',
            'private_key': 'private-key',
            'client_email': 'test@example.com',
            'client_id': 'client-id',
            'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
            'token_uri': 'https://oauth2.googleapis.com/token',
            'auth_provider_x509_cert_url': 'https://www.googleapis.com/oauth2/v1/certs',
            'client_x509_cert_url': (
                'https://www.googleapis.com/robot/v1/metadata/x509/test@example.com'
            ),
        },
    }

    # Run
    errors = _validate_resolved_credentials(credentials)

    # Assert
    assert errors == []


def test__validate_resolved_credentials_invalid_section_types():
    """Test `_validate_resolved_credentials` validates section types."""
    # Setup
    credentials = {
        'aws': 'bad',
        'sdv_enterprise': 'bad',
        'gcp': 'bad',
    }

    # Run
    errors = _validate_resolved_credentials(credentials)

    # Assert
    assert errors == [
        'credentials["aws"] must be a dict.',
        'credentials["gcp"] must be a dict.',
        'credentials["sdv_enterprise"] must be a dict.',
    ]


def test__validate_resolved_credentials_missing_aws_key():
    """Test `_validate_resolved_credentials` validates missing AWS credentials."""
    # Setup
    credentials = {
        'aws': {
            'AWS_ACCESS_KEY_ID': 'AKIA',
            'AWS_SECRET_ACCESS_KEY': None,
        },
        'sdv_enterprise': {},
        'gcp': {},
    }

    # Run
    errors = _validate_resolved_credentials(credentials)

    # Assert
    assert errors == ['credentials["aws"]["AWS_SECRET_ACCESS_KEY"] is missing or empty.']


def test__validate_resolved_credentials_partial_sdv_enterprise():
    """Test `_validate_resolved_credentials` requires both SDV Enterprise fields together."""
    # Setup
    credentials = {
        'aws': {},
        'sdv_enterprise': {
            'SDV_ENTERPRISE_USERNAME': 'user',
            'SDV_ENTERPRISE_LICENSE_KEY': None,
        },
        'gcp': {},
    }

    # Run
    errors = _validate_resolved_credentials(credentials)

    # Assert
    assert errors == [
        'credentials["sdv_enterprise"]["SDV_ENTERPRISE_LICENSE_KEY"] '
        'is required when SDV Enterprise credentials are provided.'
    ]


def test__validate_resolved_credentials_missing_gcp_keys():
    """Test `_validate_resolved_credentials` validates missing GCP service account keys."""
    # Setup
    credentials = {
        'aws': {},
        'sdv_enterprise': {},
        'gcp': {
            'type': 'service_account',
            'project_id': 'my-project',
            'private_key_id': None,
            'private_key': None,
            'client_email': 'test@example.com',
            'client_id': 'client-id',
            'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
            'token_uri': 'https://oauth2.googleapis.com/token',
            'auth_provider_x509_cert_url': 'https://www.googleapis.com/oauth2/v1/certs',
            'client_x509_cert_url': None,
        },
    }

    # Run
    errors = _validate_resolved_credentials(credentials)

    # Assert
    assert errors == [
        'credentials["gcp"]["client_x509_cert_url"] is missing or empty.',
        'credentials["gcp"]["private_key"] is missing or empty.',
        'credentials["gcp"]["private_key_id"] is missing or empty.',
    ]


@patch('sdgym._benchmark_launcher._validation.resolve_credentials')
def test__validate_credentials_invalid_filepath_type(mock_resolve_credentials):
    """Test `_validate_credentials` rejects non-string credentials_filepath."""
    # Setup
    credentials_filepath = 123

    # Run
    errors = _validate_credentials(credentials_filepath)

    # Assert
    mock_resolve_credentials.assert_not_called()
    assert errors == [
        'credentials_filepath: must be a string path to the credentials file or None.'
    ]


@patch('sdgym._benchmark_launcher._validation.resolve_credentials')
def test__validate_credentials_returns_resolved_validation_errors(mock_resolve_credentials):
    """Test `_validate_credentials` validates resolved credentials."""
    # Setup
    credentials_filepath = 'creds.json'
    mock_resolve_credentials.return_value = {
        'aws': {
            'AWS_ACCESS_KEY_ID': 'AKIA',
            'AWS_SECRET_ACCESS_KEY': None,
        },
        'sdv_enterprise': {},
        'gcp': {},
    }

    # Run
    errors = _validate_credentials(credentials_filepath)

    # Assert
    mock_resolve_credentials.assert_called_once_with(credentials_filepath)
    assert errors == ['credentials["aws"]["AWS_SECRET_ACCESS_KEY"] is missing or empty.']


@patch('sdgym._benchmark_launcher._validation.resolve_credentials')
def test__validate_credentials_returns_no_errors_when_valid(mock_resolve_credentials):
    """Test `_validate_credentials` returns no errors for valid credentials."""
    # Setup
    credentials_filepath = None
    mock_resolve_credentials.return_value = {
        'aws': {
            'AWS_ACCESS_KEY_ID': 'AKIA',
            'AWS_SECRET_ACCESS_KEY': 'SECRET',
        },
        'sdv_enterprise': {},
        'gcp': {},
    }

    # Run
    errors = _validate_credentials(credentials_filepath)

    # Assert
    mock_resolve_credentials.assert_called_once_with(credentials_filepath)
    assert errors == []
