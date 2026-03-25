"""Unit tests for the benchmark launcher utils."""

import json
from unittest.mock import MagicMock, Mock, call, patch

import pytest

from sdgym._benchmark_launcher.utils import (
    MODALITY_TO_CONFIG_FILE,
    _add_dataset_suffix,
    _build_job_artifact_keys,
    _deep_merge,
    _env,
    _get_env_credentials,
    _get_gcp_credentials_from_env,
    _get_synthetic_data_extension,
    _get_top_folder_prefix,
    _load_json_file,
    _load_merged_modality_config,
    _load_yaml_resource,
    _resolve_datasets,
    _resolve_modality_config,
    generate_ids,
    resolve_credentials,
)


@patch('sdgym._benchmark_launcher.utils._load_yaml_resource')
def test__load_merged_modality_config_calls_load_yaml_resource(mock_load_yaml):
    """Test `_load_merged_modality_config` loads and merges base and modality configs."""
    # Setup
    modality = 'single_table'
    base_config = {'a': 1, 'b': {'x': 1}}
    modality_config = {'b': {'y': 2}, 'c': 3}
    mock_load_yaml.side_effect = [base_config, modality_config]

    # Run
    merged = _load_merged_modality_config(modality)

    # Assert
    mock_load_yaml.assert_has_calls([
        call('benchmark_base.yaml'),
        call(MODALITY_TO_CONFIG_FILE[modality]),
    ])
    assert merged == {'a': 1, 'b': {'x': 1, 'y': 2}, 'c': 3}


@patch('sdgym._benchmark_launcher.utils._load_merged_modality_config')
def test__resolve_modality_config_mock(mock_load_modality_config):
    """Test `_resolve_modality_config` merges configs and filters to CONFIG_KEYS."""
    # Setup
    modality = 'single_table'
    base = {
        'method_params': {'timeout': 1},
        'extra': 'drop',
        'compute': {'service': 'gcp'},
        'credentials_filepath': {},
    }
    mock_load_modality_config.return_value = base

    # Run
    config = _resolve_modality_config(modality)

    # Assert
    mock_load_modality_config.assert_called_once_with(modality)
    assert config == {
        'method_params': {'timeout': 1},
        'credentials_filepath': {},
        'compute': {'service': 'gcp'},
    }


@pytest.mark.parametrize('modality', ['single_table', 'multi_table'])
@patch('sdgym._benchmark_launcher.utils._load_yaml_resource')
def test__resolve_modality_config_filters_to_config_keys(mock_load_yaml, modality):
    """Test `_resolve_modality_config` merges configs and filters to CONFIG_KEYS."""
    # Setup
    base = {
        'method_params': {'timeout': 1},
        'extra': 'drop',
        'compute': {'service': 'gcp'},
        'credentials_filepath': {},
    }
    modality_dict = {
        'modality': modality,
        'method_params': {'timeout': 999, 'other_param': 2},
        'instance_jobs': [{'synthesizers': ['A'], 'datasets': ['d1']}],
        'extra': 'keep',
        'another': 'drop',
    }
    expected = {
        'modality': modality,
        'method_params': {'timeout': 999, 'other_param': 2},
        'credentials_filepath': {},
        'compute': {'service': 'gcp'},
        'instance_jobs': [{'synthesizers': ['A'], 'datasets': ['d1']}],
    }

    mock_load_yaml.side_effect = [base, modality_dict]

    # Run
    resolved = _resolve_modality_config(modality)

    # Assert
    mock_load_yaml.assert_has_calls([
        call('benchmark_base.yaml'),
        call(MODALITY_TO_CONFIG_FILE[modality]),
    ])
    assert resolved == expected


def test__resolve_datasets_include_exclude():
    """Test `_resolve_datasets` resolves include/exclude correctly."""
    # Setup
    datasets_spec = {'include': ['adult', 'census', 'intrusion'], 'exclude': ['intrusion']}

    # Run
    resolved = _resolve_datasets(datasets_spec)

    # Assert
    assert resolved == ['adult', 'census']


def test__resolve_datasets_list():
    """Test `_resolve_datasets` returns a copy when given a list."""
    # Setup
    datasets_spec = ['adult', 'census']

    # Run
    resolved = _resolve_datasets(datasets_spec)

    # Assert
    assert resolved == ['adult', 'census']
    assert resolved is not datasets_spec


def test__resolve_datasets_raises_for_invalid_type():
    """Test `_resolve_datasets` raises for invalid types."""
    # Setup
    datasets_spec = 'adult'

    # Run / Assert
    with pytest.raises(ValueError, match='must be a list or dict'):
        _resolve_datasets(datasets_spec)


@patch('sdgym._benchmark_launcher.utils.files')
@patch('sdgym._benchmark_launcher.utils.yaml.safe_load', return_value={'a': 1})
def test__load_yaml_resource_calls_safe_load(mock_safe_load, mock_files):
    """Test `_load_yaml_resource` loads YAML via safe_load."""
    # Setup
    file_handle = Mock()
    open_cm = MagicMock()
    open_cm.__enter__.return_value = file_handle
    open_cm.__exit__.return_value = False
    resource_file = Mock()
    resource_file.open.return_value = open_cm
    resource = Mock()
    resource.joinpath.return_value = resource_file
    mock_files.return_value = resource

    # Run
    loaded = _load_yaml_resource('my.yaml')

    # Assert
    resource.joinpath.assert_called_once_with('my.yaml')
    resource_file.open.assert_called_once_with('r', encoding='utf-8')
    mock_safe_load.assert_called_once_with(file_handle)
    assert loaded == {'a': 1}


def test__deep_merge_recursive_override_wins():
    """Test `_deep_merge` recursively merges dicts and override wins."""
    # Setup
    base = {'a': 1, 'nested': {'x': 1, 'y': 2}}
    override = {'nested': {'y': 999, 'z': 3}, 'b': 2}

    # Run
    merged = _deep_merge(base, override)

    # Assert
    assert merged == {'a': 1, 'b': 2, 'nested': {'x': 1, 'y': 999, 'z': 3}}
    assert base == {'a': 1, 'nested': {'x': 1, 'y': 2}}


@patch('sdgym._benchmark_launcher.utils.os.getenv')
def test__env(mock_getenv):
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


def test__load_json_file(tmp_path):
    """Test `_load_json_file` loads JSON content from disk."""
    # Setup
    filepath = tmp_path / 'credentials.json'
    expected = {'aws': {'AWS_ACCESS_KEY_ID': 'AKIA'}}
    filepath.write_text(json.dumps(expected))

    # Run
    loaded = _load_json_file(filepath)

    # Assert
    assert loaded == expected


@patch('sdgym._benchmark_launcher.utils._env')
@patch('sdgym._benchmark_launcher.utils._load_json_file')
def test__get_gcp_credentials_from_env_uses_json_filepath(mock_load_json_file, mock_env):
    """Test `_get_gcp_credentials_from_env` loads JSON from filepath env var."""
    # Setup
    expected = {'type': 'service_account', 'project_id': 'my-project'}
    mock_load_json_file.return_value = expected

    def env_side_effect(name):
        return {
            'GCP_SERVICE_ACCOUNT_JSON_FILEPATH': '/tmp/gcp.json',
        }.get(name)

    mock_env.side_effect = env_side_effect

    # Run
    credentials = _get_gcp_credentials_from_env()

    # Assert
    mock_env.assert_has_calls([
        call('GCP_SERVICE_ACCOUNT_JSON_FILEPATH'),
    ])
    mock_load_json_file.assert_called_once_with('/tmp/gcp.json')
    assert credentials == expected


@patch('sdgym._benchmark_launcher.utils._env')
def test__get_gcp_credentials_from_env_uses_json(mock_env):
    """Test `_get_gcp_credentials_from_env` loads JSON from content env var."""
    # Setup
    service_account = {'type': 'service_account', 'project_id': 'my-project'}

    def env_side_effect(name):
        return {
            'GCP_SERVICE_ACCOUNT_JSON': json.dumps(service_account),
        }.get(name)

    mock_env.side_effect = env_side_effect

    # Run
    credentials = _get_gcp_credentials_from_env()

    # Assert
    mock_env.assert_has_calls([
        call('GCP_SERVICE_ACCOUNT_JSON'),
    ])
    assert credentials == service_account


@patch('sdgym._benchmark_launcher.utils._env')
def test__get_gcp_credentials_from_env_uses_individual_env_vars(mock_env):
    """Test `_get_gcp_credentials_from_env` builds credentials from individual env vars."""
    # Setup
    values = {
        'GCP_SERVICE_ACCOUNT_JSON_FILEPATH': None,
        'GOOGLE_APPLICATION_CREDENTIALS': None,
        'GCP_TYPE': 'service_account',
        'GCP_PROJECT_ID': 'my-project',
        'GCP_PRIVATE_KEY_ID': 'private-key-id',
        'GCP_PRIVATE_KEY': 'private-key',
        'GCP_CLIENT_EMAIL': 'test@example.com',
        'GCP_CLIENT_ID': 'client-id',
        'GCP_AUTH_URI': 'https://accounts.google.com/o/oauth2/auth',
        'GCP_TOKEN_URI': 'https://oauth2.googleapis.com/token',
        'GCP_AUTH_PROVIDER_X509_CERT_URL': 'https://www.googleapis.com/oauth2/v1/certs',
        'GCP_CLIENT_X509_CERT_URL': 'https://www.googleapis.com/robot/v1/metadata/x509/test',
    }
    mock_env.side_effect = values.get

    expected = {
        'type': 'service_account',
        'project_id': 'my-project',
        'private_key_id': 'private-key-id',
        'private_key': 'private-key',
        'client_email': 'test@example.com',
        'client_id': 'client-id',
        'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
        'token_uri': 'https://oauth2.googleapis.com/token',
        'auth_provider_x509_cert_url': 'https://www.googleapis.com/oauth2/v1/certs',
        'client_x509_cert_url': 'https://www.googleapis.com/robot/v1/metadata/x509/test',
    }

    # Run
    credentials = _get_gcp_credentials_from_env()

    # Assert
    assert credentials == expected


@patch('sdgym._benchmark_launcher.utils._env')
@patch('sdgym._benchmark_launcher.utils._get_gcp_credentials_from_env')
def test__get_env_credentials(mock_get_gcp_credentials_from_env, mock_env):
    """Test `_get_env_credentials` builds credentials for all services."""
    # Setup
    mock_get_gcp_credentials_from_env.return_value = {'type': 'service_account'}

    def env_side_effect(name):
        return {
            'AWS_ACCESS_KEY_ID': 'AKIA',
            'AWS_SECRET_ACCESS_KEY': 'SECRET',
            'SDV_ENTERPRISE_USERNAME': 'user',
            'SDV_ENTERPRISE_LICENSE_KEY': 'license',
        }.get(name)

    mock_env.side_effect = env_side_effect
    expected = {
        'aws': {
            'AWS_ACCESS_KEY_ID': 'AKIA',
            'AWS_SECRET_ACCESS_KEY': 'SECRET',
        },
        'sdv_enterprise': {
            'SDV_ENTERPRISE_USERNAME': 'user',
            'SDV_ENTERPRISE_LICENSE_KEY': 'license',
        },
        'gcp': {'type': 'service_account'},
    }

    # Run
    credentials = _get_env_credentials()

    # Assert
    mock_env.assert_has_calls([
        call('AWS_ACCESS_KEY_ID'),
        call('AWS_SECRET_ACCESS_KEY'),
        call('SDV_ENTERPRISE_USERNAME'),
        call('SDV_ENTERPRISE_LICENSE_KEY'),
    ])
    mock_get_gcp_credentials_from_env.assert_called_once_with()
    assert credentials == expected


@patch('sdgym._benchmark_launcher.utils._get_env_credentials')
def test_resolve_credentials_without_filepath_returns_env_credentials(mock_get_env_credentials):
    """Test `resolve_credentials` returns env credentials when filepath is None."""
    # Setup
    expected = {
        'aws': {'aws_access_key_id': 'AKIA', 'aws_secret_access_key': 'SECRET'},
        'sdv_enterprise': {
            'sdv_enterprise_username': 'user',
            'sdv_enterprise_license_key': 'license',
        },
        'gcp': {'type': 'service_account', 'project_id': 'my-project'},
    }
    mock_get_env_credentials.return_value = expected

    # Run
    credentials = resolve_credentials(None)

    # Assert
    mock_get_env_credentials.assert_called_once_with()
    assert credentials == expected


@patch('sdgym._benchmark_launcher.utils._load_json_file')
@patch('sdgym._benchmark_launcher.utils._get_env_credentials')
def test_resolve_credentials_with_filepath_deep_merges_file_over_env(
    mock_get_env_credentials,
    mock_load_json_file,
):
    """Test `resolve_credentials` deep merges file credentials over env credentials."""
    # Setup
    env_credentials = {
        'aws': {
            'AWS_ACCESS_KEY_ID': 'ENV_AKIA',
            'AWS_SECRET_ACCESS_KEY': 'ENV_SECRET',
        },
        'sdv_enterprise': {
            'SDV_ENTERPRISE_USERNAME': None,
            'SDV_ENTERPRISE_LICENSE_KEY': 'ENV_LICENSE',
        },
        'gcp': {
            'type': 'service_account',
            'project_id': 'env-project',
            'private_key_id': 'env-key-id',
            'private_key': 'env-private-key',
            'client_email': 'env@example.com',
            'client_id': 'env-client-id',
            'auth_uri': 'env-auth-uri',
            'token_uri': 'env-token-uri',
            'auth_provider_x509_cert_url': 'env-auth-cert-url',
            'client_x509_cert_url': 'env-client-cert-url',
        },
    }
    file_credentials = {
        'aws': {
            'AWS_ACCESS_KEY_ID': 'FILE_AKIA',
        },
        'sdv_enterprise': {
            'SDV_ENTERPRISE_USERNAME': 'file-user',
        },
        'gcp': {
            'project_id': 'file-project',
            'client_email': 'file@example.com',
        },
    }
    expected = {
        'aws': {
            'aws_access_key_id': 'FILE_AKIA',
            'aws_secret_access_key': 'ENV_SECRET',
        },
        'sdv_enterprise': {
            'sdv_enterprise_username': 'file-user',
            'sdv_enterprise_license_key': 'ENV_LICENSE',
        },
        'gcp': {
            'type': 'service_account',
            'project_id': 'file-project',
            'private_key_id': 'env-key-id',
            'private_key': 'env-private-key',
            'client_email': 'file@example.com',
            'client_id': 'env-client-id',
            'auth_uri': 'env-auth-uri',
            'token_uri': 'env-token-uri',
            'auth_provider_x509_cert_url': 'env-auth-cert-url',
            'client_x509_cert_url': 'env-client-cert-url',
        },
    }
    credentials_filepath = 'credentials.json'
    mock_get_env_credentials.return_value = env_credentials
    mock_load_json_file.return_value = file_credentials

    # Run
    credentials = resolve_credentials(credentials_filepath)

    # Assert
    mock_get_env_credentials.assert_called_once_with()
    mock_load_json_file.assert_called_once_with(credentials_filepath)
    assert credentials == expected


def test_resolve_credentials_file_mode(tmp_path):
    """Test `resolve_credentials` returns credentials from a file merged over env defaults."""
    # Setup
    credential_file = tmp_path / 'credentials.json'
    file_credentials = {
        'aws': {
            'AWS_ACCESS_KEY_ID': 'FILE_AKIA',
            'AWS_SECRET_ACCESS_KEY': 'FILE_SECRET',
        },
        'gcp': {
            'type': 'service_account',
            'project_id': 'file-project',
        },
        'sdv_enterprise': {
            'SDV_ENTERPRISE_USERNAME': 'file-user',
            'SDV_ENTERPRISE_LICENSE_KEY': 'file-license',
        },
    }
    credential_file.write_text(json.dumps(file_credentials))
    expected_credentials = {
        'aws': {'aws_access_key_id': 'FILE_AKIA', 'aws_secret_access_key': 'FILE_SECRET'},
        'sdv_enterprise': {
            'sdv_enterprise_username': 'file-user',
            'sdv_enterprise_license_key': 'file-license',
        },
        'gcp': {
            'type': 'service_account',
            'project_id': 'file-project',
            'private_key_id': None,
            'private_key': None,
            'client_email': None,
            'client_id': None,
            'auth_uri': None,
            'token_uri': None,
            'auth_provider_x509_cert_url': None,
            'client_x509_cert_url': None,
        },
    }

    # Run
    credentials = resolve_credentials(str(credential_file))

    # Assert
    assert credentials == expected_credentials


@patch('sdgym._benchmark_launcher.utils._env')
def test_resolve_credentials_env_mode(mock_env):
    """Test `resolve_credentials` returns credentials from environment variables."""
    # Setup
    values = {
        'AWS_ACCESS_KEY_ID': 'AKIA',
        'AWS_SECRET_ACCESS_KEY': 'SECRET',
        'SDV_ENTERPRISE_USERNAME': 'user',
        'SDV_ENTERPRISE_LICENSE_KEY': 'license',
        'GCP_SERVICE_ACCOUNT_JSON_FILEPATH': None,
        'GOOGLE_APPLICATION_CREDENTIALS': None,
        'GCP_TYPE': 'service_account',
        'GCP_PROJECT_ID': 'my-project',
        'GCP_PRIVATE_KEY_ID': 'private-key-id',
        'GCP_PRIVATE_KEY': 'private-key',
        'GCP_CLIENT_EMAIL': 'test@example.com',
        'GCP_CLIENT_ID': 'client-id',
        'GCP_AUTH_URI': 'https://accounts.google.com/o/oauth2/auth',
        'GCP_TOKEN_URI': 'https://oauth2.googleapis.com/token',
        'GCP_AUTH_PROVIDER_X509_CERT_URL': 'https://www.googleapis.com/oauth2/v1/certs',
        'GCP_CLIENT_X509_CERT_URL': 'https://www.googleapis.com/robot/v1/metadata/x509/test',
    }
    mock_env.side_effect = values.get

    expected_credentials = {
        'aws': {
            'aws_access_key_id': 'AKIA',
            'aws_secret_access_key': 'SECRET',
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
            'client_x509_cert_url': 'https://www.googleapis.com/robot/v1/metadata/x509/test',
        },
        'sdv_enterprise': {
            'sdv_enterprise_username': 'user',
            'sdv_enterprise_license_key': 'license',
        },
    }

    # Run
    credentials = resolve_credentials(None)

    # Assert
    assert credentials == expected_credentials


def test_generate_ids_returns_unique_ids():
    """Test `generate_ids` returns unique IDs."""
    # Setup
    components = ['COMPONENT1', 'COMPONENT2']

    # Run
    id1 = generate_ids(components)
    id2 = generate_ids(components)

    # Assert
    assert id1 != id2
    assert id1.startswith('COMPONENT1_COMPONENT2_')
    assert id2.startswith('COMPONENT1_COMPONENT2_')


@patch('sdgym._benchmark_launcher.utils.datetime')
def test__add_dataset_suffix(mock_datetime):
    """Test the `_add_dataset_suffix` method."""
    # Setup
    mock_datetime.today.return_value.strftime.return_value = '03_25_2026'

    # Run
    result = _add_dataset_suffix('adult')

    # Assert
    assert result == 'adult_03_25_2026'


@patch('sdgym._benchmark_launcher.utils.datetime')
@patch('sdgym._benchmark_launcher.utils.parse_s3_path')
def test__get_top_folder_prefix(mock_parse_s3_path, mock_datetime):
    """Test the `_get_top_folder_prefix` method."""
    # Setup
    mock_parse_s3_path.return_value = ('my-bucket', 'Debug/Issue_570/')
    mock_datetime.today.return_value.strftime.return_value = '03_25_2026'

    # Run
    result = _get_top_folder_prefix(
        output_destination='s3://my-bucket/Debug/Issue_570/',
        modality='single_table',
    )

    # Assert
    mock_parse_s3_path.assert_called_once_with('s3://my-bucket/Debug/Issue_570/')
    assert result == 'Debug/Issue_570/single_table/SDGym_results_03_25_2026'


@patch('sdgym._benchmark_launcher.utils.datetime')
@patch('sdgym._benchmark_launcher.utils.parse_s3_path')
def test__get_top_folder_prefix_empty_key_prefix(mock_parse_s3_path, mock_datetime):
    """Test `_get_top_folder_prefix` when the output destination has no key prefix."""
    # Setup
    mock_parse_s3_path.return_value = ('my-bucket', '')
    mock_datetime.today.return_value.strftime.return_value = '03_25_2026'

    # Run
    result = _get_top_folder_prefix(
        output_destination='s3://my-bucket',
        modality='single_table',
    )

    # Assert
    assert result == 'single_table/SDGym_results_03_25_2026'


@pytest.mark.parametrize(
    ('modality', 'expected'),
    [
        ('single_table', 'csv'),
        ('multi_table', 'zip'),
    ],
)
def test__get_synthetic_data_extension(modality, expected):
    """Test the `_get_synthetic_data_extension` method."""
    # Run
    result = _get_synthetic_data_extension(modality)

    # Assert
    assert result == expected


@pytest.mark.parametrize(
    ('artifact_key_prefix', 'artifact_dataset', 'artifact_synthesizer', 'modality', 'expected'),
    [
        (
            'Debug/Issue_570/single_table/SDGym_results_03_25_2026',
            'adult_03_25_2026',
            'CTGANSynthesizer',
            'single_table',
            (
                'Debug/Issue_570/single_table/SDGym_results_03_25_2026/'
                'adult_03_25_2026/CTGANSynthesizer/CTGANSynthesizer_benchmark_result.csv',
                'Debug/Issue_570/single_table/SDGym_results_03_25_2026/'
                'adult_03_25_2026/CTGANSynthesizer/CTGANSynthesizer_synthetic_data.csv',
                'Debug/Issue_570/single_table/SDGym_results_03_25_2026/'
                'adult_03_25_2026/CTGANSynthesizer/CTGANSynthesizer.pkl',
            ),
        ),
        (
            'Debug/Issue_570/multi_table/SDGym_results_03_25_2026/',
            'fake_hotels_03_25_2026',
            'HMASynthesizer(1)',
            'multi_table',
            (
                'Debug/Issue_570/multi_table/SDGym_results_03_25_2026/'
                'fake_hotels_03_25_2026/HMASynthesizer(1)/'
                'HMASynthesizer(1)_benchmark_result.csv',
                'Debug/Issue_570/multi_table/SDGym_results_03_25_2026/'
                'fake_hotels_03_25_2026/HMASynthesizer(1)/'
                'HMASynthesizer(1)_synthetic_data.zip',
                'Debug/Issue_570/multi_table/SDGym_results_03_25_2026/'
                'fake_hotels_03_25_2026/HMASynthesizer(1)/HMASynthesizer(1).pkl',
            ),
        ),
    ],
)
def test__build_job_artifact_keys(
    artifact_key_prefix,
    artifact_dataset,
    artifact_synthesizer,
    modality,
    expected,
):
    """Test the `_build_job_artifact_keys` method."""
    # Run
    result = _build_job_artifact_keys(
        artifact_key_prefix=artifact_key_prefix,
        artifact_dataset=artifact_dataset,
        artifact_synthesizer=artifact_synthesizer,
        modality=modality,
    )

    # Assert
    assert result == expected
