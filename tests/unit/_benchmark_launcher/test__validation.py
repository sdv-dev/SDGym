"""Unit tests for the benchmark launcher validation functions."""

import json
from unittest.mock import Mock, call, patch

from sdgym._benchmark_launcher._validation import (
    _as_errors,
    _env,
    _format_sectioned_errors,
    _get_credentials,
    _validate_credentials_config,
    _validate_credentials_config_structure,
    _validate_instance_jobs,
    _validate_method_params,
    _validate_resolved_credentials,
    _validate_structure,
)


class TestBenchmarkLauncherValidation:
    def test__as_errors_returns_empty_for_none(self):
        """Test `_as_errors` returns an empty list for None."""
        # Setup
        value = None

        # Run
        result = _as_errors(value)

        # Assert
        assert result == []

    def test__as_errors_filters_list_and_wraps_string(self):
        """Test `_as_errors` filters list values and wraps strings."""
        # Setup
        list_value = ['a', '', None, 2]
        str_value = 'hello'

        # Run
        list_result = _as_errors(list_value)
        str_result = _as_errors(str_value)

        # Assert
        assert list_result == ['a', '2']
        assert str_result == ['hello']

    def test__format_sectioned_errors(self):
        """Test the `_format_sectioned_errors` method."""
        # Setup
        section_errors = {'instance_jobs': "Each job in 'instance_jobs' must be valid."}
        expected_message = (
            'BenchmarkConfig validation failed:\n'
            '\n'
            '[instance_jobs]\n'
            "- Each job in 'instance_jobs' must be valid."
        )

        # Run
        rendered = _format_sectioned_errors(section_errors)

        # Assert
        assert rendered == expected_message

    def test__format_sectioned_errors_skips_empty_sections(self):
        """Test `_format_sectioned_errors` skips sections with no errors."""
        # Setup
        section_errors = {'structure': [], 'instance_jobs': ['bad job'], 'credentials': None}
        expected_message = 'BenchmarkConfig validation failed:\n\n[instance_jobs]\n- bad job'

        # Run
        rendered = _format_sectioned_errors(section_errors)

        # Assert
        assert rendered == expected_message

    @patch('sdgym._benchmark_launcher._validation.os.getenv')
    def test__env(self, mock_getenv):
        """Test the `_env` method."""

        # Setup
        def getenv_side_effect(key):
            return {
                'MY_VAR': 'value',
                'EMPTY_VAR': '',
            }.get(key, None)

        mock_getenv.side_effect = getenv_side_effect

        # Run
        value = _env('MY_VAR')
        empty = _env('EMPTY_VAR')
        missing = _env('MISSING_VAR')

        # Assert
        mock_getenv.assert_has_calls([call('MY_VAR'), call('EMPTY_VAR'), call('MISSING_VAR')])
        assert value == 'value'
        assert empty is None
        assert missing is None

    def test__get_credentials_file_mode(self, tmp_path):
        """Test `_get_credentials` returns credentials from a file."""
        # Setup
        credential_file = tmp_path / 'creds.json'
        expected_credentials = {
            'aws': {'aws_access_key_id': 'AKIA', 'aws_secret_access_key': 'SECRET'},
            'gcp': {
                'type': 'service_account',
                'project_id': 'sa-project',
                'private_key': 'KEY',
                'client_email': 'x@y.z',
                'token_uri': 'https://oauth2.googleapis.com/token',
                'gcp_project': 'my-project',
                'gcp_zone': 'zone',
            },
            'sdv': {'username': 'u', 'license_key': 'k'},
        }
        credential_file.write_text(json.dumps(expected_credentials))
        credentials_config = {'credential_filepath': str(credential_file)}

        # Run
        creds = _get_credentials(credentials_config)

        # Assert
        assert creds == expected_credentials

    @patch('sdgym._benchmark_launcher._validation.os.getenv')
    def test__get_credentials_env_mode(self, mock_getenv):
        """Test `_get_credentials` with environment variable."""
        # Setup
        service_account = {
            'type': 'service_account',
            'project_id': 'sa-project',
            'private_key': 'KEY',
            'client_email': 'x@y.z',
            'token_uri': 'https://oauth2.googleapis.com/token',
        }

        def getenv_side_effect(key):
            return {
                'AWS_ACCESS_KEY_ID': 'AKIA',
                'AWS_SECRET_ACCESS_KEY': 'SECRET',
                'GCP_SA_JSON': json.dumps(service_account),
                'GCP_PROJECT': 'my-project',
                'GCP_ZONE': 'zone',
            }.get(key, None)

        mock_getenv.side_effect = getenv_side_effect
        credentials_config = {
            'aws': {
                'access_key_id_env': 'AWS_ACCESS_KEY_ID',
                'secret_access_key_env': 'AWS_SECRET_ACCESS_KEY',
            },
            'gcp': {
                'service_account_json_env': 'GCP_SA_JSON',
                'project_id_env': 'GCP_PROJECT',
                'zone_env': 'GCP_ZONE',
            },
        }
        expected_credentials = {
            'aws': {'aws_access_key_id': 'AKIA', 'aws_secret_access_key': 'SECRET'},
            'gcp': {
                'type': 'service_account',
                'project_id': 'sa-project',
                'private_key': 'KEY',
                'client_email': 'x@y.z',
                'token_uri': 'https://oauth2.googleapis.com/token',
                'gcp_project': 'my-project',
                'gcp_zone': 'zone',
            },
            'sdv': {'username': None, 'license_key': None},
        }

        # Run
        credentials = _get_credentials(credentials_config)

        # Assert
        mock_getenv.assert_has_calls([
            call('AWS_ACCESS_KEY_ID'),
            call('AWS_SECRET_ACCESS_KEY'),
            call('GCP_SA_JSON'),
            call('GCP_PROJECT'),
            call('GCP_ZONE'),
        ])
        assert credentials == expected_credentials

    def test__validate_structure_valid(self):
        """Test `_validate_structure` returns empty list for valid config."""
        # Setup
        config = Mock()
        config.modality = 'single_table'
        config.method_params = {}
        config.credentials_config = {}
        config.compute = {'service': 'gcp'}
        config.instance_jobs = []

        # Run
        errors = _validate_structure(config)

        # Assert
        assert errors == []

    def test__validate_structure_invalid(self):
        """Test `_validate_structure` returns errors for invalid config."""
        # Setup
        config = Mock()
        config.modality = 'bad'
        config.method_params = []
        config.credentials_config = 'nope'
        config.compute = {'service': 'aws'}
        config.instance_jobs = {}
        expected_errors = [
            "modality: must be 'single_table' or 'multi_table'. Found: 'bad'",
            "method_params: must be a dict. Found: <class 'list'>",
            "credentials_config: must be a dict. Found: <class 'str'>",
            "instance_jobs: must be a list. Found: <class 'dict'>",
            "compute.service: must be 'gcp'. Found: 'aws'",
        ]

        # Run
        errors = _validate_structure(config)

        # Assert
        assert errors == expected_errors

    def test__validate_method_params_valid(self):
        """Test `_validate_method_params` returns empty list for valid method_params."""

        # Setup
        def method_to_run(
            output_destination,
            credentials,
            required_param,
            compute_config=None,
            synthesizers=None,
            sdv_datasets=None,
        ):
            return None

        method_params = {
            'output_destination': 's3://bucket/prefix/',
            'timeout': 3600,
            'required_param': 'value',
            'compute_quality_score': True,
            'compute_diagnostic_score': False,
            'compute_privacy_score': False,
        }

        # Run
        errors = _validate_method_params(method_params, method_to_run)

        # Assert
        assert errors == []

    def test__validate_method_params_errors(self):
        """Test `_validate_method_params` when method_params contain errors."""

        # Setup
        def method_to_run(output_destination, credentials, required_param, compute_config=None):
            return None

        method_params = {
            'credentials': {'aws': {}},
            'exta_param': 'value',
        }
        expected_errors = [
            'method_params.output_destination: is required and must be a non-empty string.',
            "method_params: missing required parameters for method_to_run: ['output_destination', "
            "'required_param']",
            "method_params: must not define injected parameters ['credentials'] (resolved "
            'from credentials/instance_jobs).',
        ]

        # Run
        errors = _validate_method_params(method_params, method_to_run)

        # Assert
        assert errors == expected_errors

    def test__validate_instance_jobs_valid(self):
        """Test `_validate_instance_jobs` returns empty list for valid jobs."""
        # Setup
        instance_jobs = [
            {'synthesizers': ['GaussianCopulaSynthesizer'], 'datasets': ['adult']},
            {
                'synthesizers': ['CTGANSynthesizer'],
                'datasets': {'include': ['adult'], 'exclude': ['alarm']},
            },
        ]

        # Run
        errors = _validate_instance_jobs(instance_jobs)

        # Assert
        assert errors == []

    def test__validate_instance_jobs_invalid(self):
        """Test `_validate_instance_jobs` returns one aggregated error for invalid jobs."""
        # Setup
        instance_jobs = [
            {'datasets': ['adult']},
            'not_a_dict',
        ]
        expected_error = [
            "Each job in 'instance_jobs' must be a dict with 'synthesizers' (list of strings) and "
            "'datasets' (list of strings or dict with 'include' and optional 'exclude').\nInvalid"
            " jobs:\n{'datasets': ['adult']}\nnot_a_dict"
        ]

        # Run
        errors = _validate_instance_jobs(instance_jobs)

        # Assert
        assert errors == expected_error

    def test__validate_credentials_config_structure_file_valid(self, tmp_path):
        """Test `_validate_credentials_config_structure` returns empty list in file mode."""
        # Setup
        credential_file = tmp_path / 'creds.json'
        credential_file.write_text(json.dumps({'aws': {}, 'gcp': {}}))
        credentials_config = {'credential_filepath': str(credential_file)}

        # Run
        errors = _validate_credentials_config_structure(credentials_config)

        # Assert
        assert errors == []

    def test__validate_credentials_config_structure_env_missing_required_section(self):
        """Test `_validate_credentials_config_structure` reports missing required gcp section."""
        # Setup
        credentials_config = {
            'aws': {
                'access_key_id_env': 'AWS_ACCESS_KEY_ID',
                'secret_access_key_env': 'AWS_SECRET_ACCESS_KEY',
            },
        }
        expected_errors = [
            'credentials.gcp: section is required but missing.',
        ]

        # Run
        errors = _validate_credentials_config_structure(credentials_config)

        # Assert
        assert errors == expected_errors

    def test__validate_resolved_credentials_valid(self):
        """Test `_validate_resolved_credentials` with valid credentials."""
        # Setup
        credentials = {
            'aws': {'aws_access_key_id': 'AKIA', 'aws_secret_access_key': 'SECRET'},
            'gcp': {
                'type': 'service_account',
                'project_id': 'sa-project',
                'private_key': 'KEY',
                'client_email': 'x@y.z',
                'token_uri': 'https://oauth2.googleapis.com/token',
                'gcp_project': 'my-project',
                'gcp_zone': 'zone',
            },
            'sdv': {'username': None, 'license_key': None},
        }

        # Run
        errors = _validate_resolved_credentials(credentials)

        # Assert
        assert errors == []

    def test__validate_resolved_credentials_missing_required_fields(self):
        """Test `_validate_resolved_credentials` catches missing required fields."""
        # Setup
        credentials = {'aws': {}, 'gcp': {}, 'sdv': {'username': 'u'}}
        expected_errors = [
            'credentials["aws"] missing key: "aws_access_key_id"',
            'credentials["aws"] missing key: "aws_secret_access_key"',
            'credentials["gcp"] missing key: "gcp_project"',
            'credentials["gcp"] missing key: "gcp_zone"',
            'credentials["gcp"]["client_email"] is missing or empty.',
            'credentials["gcp"]["private_key"] is missing or empty.',
            'credentials["gcp"]["project_id"] is missing or empty.',
            'credentials["gcp"]["token_uri"] is missing or empty.',
            'credentials["gcp"]["type"] is missing or empty.',
            "credentials['sdv']['license_key'] is required when SDV credentials are provided.",
        ]

        # Run
        errors = _validate_resolved_credentials(credentials)

        # Assert
        assert errors == expected_errors

    @patch('sdgym._benchmark_launcher._validation.os.getenv')
    def test__validate_credentials_config_end_to_end_env_valid(self, mock_getenv):
        """Test `_validate_credentials_config` returns empty list for valid env credentials."""
        # Setup
        service_account = {
            'type': 'service_account',
            'project_id': 'sa-project',
            'private_key': 'KEY',
            'client_email': 'x@y.z',
            'token_uri': 'https://oauth2.googleapis.com/token',
        }

        def getenv_side_effect(key):
            return {
                'AWS_ACCESS_KEY_ID': 'AKIA',
                'AWS_SECRET_ACCESS_KEY': 'SECRET',
                'GCP_SA_JSON': json.dumps(service_account),
                'GCP_PROJECT': 'my-project',
                'GCP_ZONE': 'zone',
            }.get(key, None)

        mock_getenv.side_effect = getenv_side_effect
        credentials_config = {
            'aws': {
                'access_key_id_env': 'AWS_ACCESS_KEY_ID',
                'secret_access_key_env': 'AWS_SECRET_ACCESS_KEY',
            },
            'gcp': {
                'service_account_json_env': 'GCP_SA_JSON',
                'project_id_env': 'GCP_PROJECT',
                'zone_env': 'GCP_ZONE',
            },
        }

        # Run
        errors = _validate_credentials_config(credentials_config)

        # Assert
        assert errors == []

    def test__validate_credentials_config_end_to_end_file_invalid(self, tmp_path):
        """Test `_validate_credentials_config` returns errors for invalid credentials file."""
        # Setup
        credential_file = tmp_path / 'creds.json'
        credential_file.write_text(json.dumps({'aws': {}, 'gcp': {}}))
        credentials_config = {'credential_filepath': str(credential_file)}
        expected_errors = [
            'credentials["aws"] missing key: "aws_access_key_id"',
            'credentials["aws"] missing key: "aws_secret_access_key"',
            'credentials["gcp"] missing key: "gcp_project"',
            'credentials["gcp"] missing key: "gcp_zone"',
            'credentials["gcp"]["client_email"] is missing or empty.',
            'credentials["gcp"]["private_key"] is missing or empty.',
            'credentials["gcp"]["project_id"] is missing or empty.',
            'credentials["gcp"]["token_uri"] is missing or empty.',
            'credentials["gcp"]["type"] is missing or empty.',
        ]

        # Run
        errors = _validate_credentials_config(credentials_config)

        # Assert
        assert errors == expected_errors
