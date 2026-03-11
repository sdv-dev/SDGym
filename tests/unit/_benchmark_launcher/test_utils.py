"""Unit tests for the benchmark launcher utils."""

from unittest.mock import MagicMock, Mock, call, patch

import pytest

from sdgym._benchmark_launcher.utils import (
    MODALITY_TO_CONFIG_FILE,
    _deep_merge,
    _load_yaml_resource,
    _resolve_datasets,
    _resolve_modality_config,
    resolve_credentials,
)


@pytest.mark.parametrize('modality', ['single_table', 'multi_table'])
@patch('sdgym._benchmark_launcher.utils._load_yaml_resource')
def test__resolve_modality_config_filters_to_config_keys(mock_load_yaml, modality):
    """Test `_resolve_modality_config` merges configs and filters to CONFIG_KEYS."""
    # Setup
    base = {
        'method_params': {'timeout': 1},
        'extra': 'drop',
        'compute': {'service': 'gcp'},
        'credential_locations': {},
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
        'credential_locations': {},
        'compute': {'service': 'gcp'},
        'instance_jobs': [{'synthesizers': ['A'], 'datasets': ['d1']}],
    }

    mock_load_yaml.side_effect = [base, modality_dict]

    # Run
    resolved = _resolve_modality_config('single_table')

    # Assert
    mock_load_yaml.assert_has_calls([
        call('benchmark_base.yaml'),
        call(MODALITY_TO_CONFIG_FILE['single_table']),
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


@patch(
    'sdgym._benchmark_launcher.utils._get_credentials',
    return_value={'aws': {}, 'gcp': {}, 'sdv_enterprise': {}},
)
@patch('sdgym._benchmark_launcher.utils._validate_resolved_credentials', return_value=[])
def test_resolve_credentials_returns_credentials_when_valid(mock_validate, mock_get_credentials):
    """Test `resolve_credentials` returns credentials when validation passes."""
    # Setup
    credential_locations = {'credential_filepath': 'creds.json'}

    # Run
    credentials = resolve_credentials(credential_locations)

    # Assert
    mock_get_credentials.assert_called_once_with(credential_locations)
    mock_validate.assert_called_once_with({'aws': {}, 'gcp': {}, 'sdv_enterprise': {}})
    assert credentials == {'aws': {}, 'gcp': {}, 'sdv_enterprise': {}}


@patch('sdgym._benchmark_launcher.utils._get_credentials', return_value={'aws': {}, 'gcp': {}})
@patch('sdgym._benchmark_launcher.utils._validate_resolved_credentials', return_value=['bad'])
def test_resolve_credentials_raises_when_invalid(mock_validate, mock_get_credentials):
    """Test `resolve_credentials` raises ValueError when resolved credentials are invalid."""
    # Setup
    credential_locations = {'credential_filepath': 'creds.json'}

    # Run
    with pytest.raises(ValueError, match='Invalid resolved credentials'):
        resolve_credentials(credential_locations)

    # Assert
    mock_get_credentials.assert_called_once_with(credential_locations)
    mock_validate.assert_called_once()
