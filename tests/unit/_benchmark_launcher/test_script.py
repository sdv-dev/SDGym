"""Unit tests for the benchmark launcher script."""

from argparse import Namespace
from unittest.mock import Mock, patch

import pytest

from sdgym._benchmark_launcher.script import (
    _build_instance_jobs,
    _get_default_datasets_and_synthesizers,
    _parse_args,
    _validate_args,
    build_config_from_args,
    build_dict_from_args,
    launch_from_args,
)
from sdgym.run_benchmark.utils import OUTPUT_DESTINATION_AWS


@patch('sdgym._benchmark_launcher.script.argparse.ArgumentParser.parse_args')
def test__parse_args_calls_parse_args(mock_parse_args):
    """Test `_parse_args` method."""
    # Setup
    expected = Mock()
    mock_parse_args.return_value = expected

    # Run
    args = _parse_args()

    # Assert
    mock_parse_args.assert_called_once_with()
    assert args is expected


def test__validate_args_with_config_filepath():
    """Test `_validate_args` with a config_filepath."""
    # Setup
    args = Namespace(
        config_filepath='config.yaml',
        modality=None,
        datasets=None,
        synthesizers=None,
        timeout=None,
        output_destination=None,
    )

    # Run
    _validate_args(args)

    # Assert
    assert True


@pytest.mark.parametrize(
    'param,value,message',
    [
        ('modality', None, "'--modality' is required when '--config-filepath' is not provided."),
        (
            'output_destination',
            None,
            ("'--output-destination' is required when '--config-filepath' is not provided."),
        ),
        (
            'output_destination',
            OUTPUT_DESTINATION_AWS,
            (
                f"'--output-destination' cannot be {OUTPUT_DESTINATION_AWS!r} that is reserved"
                ' for internal benchmarks'
            ),
        ),
    ],
)
def test__validate_args(param, value, message):
    """Test `_validate_args` raises appropriate errors."""
    # Setup
    args = Namespace(
        config_filepath=None,
        modality='single_table',
        datasets='adult',
        synthesizers='CTGANSynthesizer',
        timeout=None,
        output_destination='s3://sdgym-benchmark/Debug/test/',
    )
    setattr(args, param, value)

    # Run and Assert
    with pytest.raises(ValueError, match=message):
        _validate_args(args)


def test__build_instance_jobs():
    """Test `_build_instance_jobs` builds one instance job per dataset/synthesizer pair."""
    # Setup
    datasets = ['adult', 'alarm']
    synthesizers = ['CTGANSynthesizer', 'TVAESynthesizer']
    output_destination = 's3://bucket/prefix/'

    # Run
    instance_jobs = _build_instance_jobs(datasets, synthesizers, output_destination)

    # Assert
    assert instance_jobs == [
        {
            'output_destination': 's3://bucket/prefix/',
            'synthesizers': ['CTGANSynthesizer'],
            'datasets': ['adult'],
        },
        {
            'output_destination': 's3://bucket/prefix/',
            'synthesizers': ['TVAESynthesizer'],
            'datasets': ['adult'],
        },
        {
            'output_destination': 's3://bucket/prefix/',
            'synthesizers': ['CTGANSynthesizer'],
            'datasets': ['alarm'],
        },
        {
            'output_destination': 's3://bucket/prefix/',
            'synthesizers': ['TVAESynthesizer'],
            'datasets': ['alarm'],
        },
    ]


@patch('sdgym._benchmark_launcher.script._load_merged_modality_config')
def test__get_default_datasets_and_synthesizers(mock_load_merged_modality_config):
    """Test `_get_default_datasets_and_synthesizers` returns default datasets and synthesizers."""
    # Setup
    output_destination = 's3://bucket/prefix/'
    mock_load_merged_modality_config.return_value = {
        'instance_jobs': [
            {
                'datasets': ['adult', 'alarm'],
                'synthesizers': ['CTGANSynthesizer'],
                'output_destination': output_destination,
            },
            {
                'datasets': ['census'],
                'synthesizers': ['TVAESynthesizer'],
                'output_destination': output_destination,
            },
        ],
    }

    # Run
    datasets, synthesizers = _get_default_datasets_and_synthesizers('single_table')

    # Assert
    mock_load_merged_modality_config.assert_called_once_with('single_table')
    assert datasets == ['adult', 'alarm', 'census']
    assert synthesizers == ['CTGANSynthesizer', 'TVAESynthesizer']


@patch('sdgym._benchmark_launcher.script._resolve_modality_config')
def test_build_dict_from_args_uses_default_modality_config(mock_resolve_modality_config):
    """Test `build_dict_from_args` uses the default modality config when all selectors are None."""
    # Setup
    args = Namespace(
        timeout=60,
        datasets=None,
        synthesizers=None,
        modality='single_table',
        output_destination='s3://bucket/prefix/',
    )
    mock_resolve_modality_config.return_value = {
        'instance_jobs': [{'datasets': ['adult'], 'synthesizers': ['CTGANSynthesizer']}],
    }

    # Run
    config = build_dict_from_args(args)

    # Assert
    mock_resolve_modality_config.assert_called_once_with('single_table')
    assert config == {
        'instance_jobs': [
            {
                'datasets': ['adult'],
                'synthesizers': ['CTGANSynthesizer'],
                'output_destination': 's3://bucket/prefix/',
            }
        ],
        'method_params': {'timeout': 60},
    }


@patch('sdgym._benchmark_launcher.script._build_instance_jobs')
@patch('sdgym._benchmark_launcher.script._get_default_datasets_and_synthesizers')
def test_build_dict_from_args_uses_defaults_for_missing_values(
    mock_get_default_datasets_and_synthesizers,
    mock_build_instance_jobs,
):
    """Test `build_dict_from_args` fills missing values with defaults."""
    # Setup
    args = Namespace(
        timeout=None,
        datasets=None,
        synthesizers=['CTGANSynthesizer'],
        modality='single_table',
        output_destination='s3://bucket/prefix/',
    )
    mock_get_default_datasets_and_synthesizers.return_value = (
        ['adult', 'alarm'],
        ['TVAESynthesizer'],
    )
    mock_build_instance_jobs.return_value = [{'some': 'job'}]

    # Run
    config = build_dict_from_args(args)

    # Assert
    mock_get_default_datasets_and_synthesizers.assert_called_once_with('single_table')
    mock_build_instance_jobs.assert_called_once_with(
        datasets=['adult', 'alarm'],
        synthesizers=['CTGANSynthesizer'],
        output_destination='s3://bucket/prefix/',
    )
    assert config == {
        'method_params': {},
        'instance_jobs': [{'some': 'job'}],
    }


@patch('sdgym._benchmark_launcher.script._build_instance_jobs')
def test_build_dict_from_args_builds_expected_override_dict(mock_build_instance_jobs):
    """Test `build_dict_from_args` builds the expected config override dict."""
    # Setup
    args = Namespace(
        modality='single_table',
        datasets=['adult', 'alarm'],
        synthesizers=['CTGANSynthesizer', 'TVAESynthesizer'],
        timeout=3600,
        output_destination='s3://sdgym-benchmark/Debug/test/',
    )
    mock_build_instance_jobs.return_value = [
        {
            'synthesizers': ['CTGANSynthesizer'],
            'datasets': ['adult'],
            'output_destination': 's3://sdgym-benchmark/Debug/test/',
        },
        {
            'synthesizers': ['TVAESynthesizer'],
            'datasets': ['adult'],
            'output_destination': 's3://sdgym-benchmark/Debug/test/',
        },
    ]

    # Run
    config = build_dict_from_args(args)

    # Assert
    mock_build_instance_jobs.assert_called_once_with(
        datasets=['adult', 'alarm'],
        synthesizers=['CTGANSynthesizer', 'TVAESynthesizer'],
        output_destination='s3://sdgym-benchmark/Debug/test/',
    )
    assert config == {
        'method_params': {
            'timeout': 3600,
        },
        'instance_jobs': [
            {
                'synthesizers': ['CTGANSynthesizer'],
                'datasets': ['adult'],
                'output_destination': 's3://sdgym-benchmark/Debug/test/',
            },
            {
                'synthesizers': ['TVAESynthesizer'],
                'datasets': ['adult'],
                'output_destination': 's3://sdgym-benchmark/Debug/test/',
            },
        ],
    }


@patch('sdgym._benchmark_launcher.script._build_instance_jobs', return_value=[])
def test_build_dict_from_args_without_timeout(mock_build_instance_jobs):
    """Test `build_dict_from_args` omits timeout when it is not provided."""
    # Setup
    args = Namespace(
        modality='single_table',
        datasets=['adult'],
        synthesizers=['CTGANSynthesizer'],
        timeout=None,
        output_destination='s3://sdgym-benchmark/Debug/test/',
    )

    # Run
    config = build_dict_from_args(args)

    # Assert
    mock_build_instance_jobs.assert_called_once_with(
        datasets=['adult'],
        synthesizers=['CTGANSynthesizer'],
        output_destination='s3://sdgym-benchmark/Debug/test/',
    )
    assert config == {
        'method_params': {},
        'instance_jobs': [],
    }


@patch('sdgym._benchmark_launcher.script.BenchmarkConfig.load_from_dict')
@patch('sdgym._benchmark_launcher.script._deep_merge')
@patch('sdgym._benchmark_launcher.script.build_dict_from_args')
@patch('sdgym._benchmark_launcher.script._resolve_modality_config')
def test_build_config_from_args_mock(
    mock_resolve_modality_config,
    mock_build_dict_from_args,
    mock_deep_merge,
    mock_load_from_dict,
):
    """Test `build_config_from_args` resolves, merges and loads the config."""
    # Setup
    args = Namespace(modality='single_table')
    base_dict = {'base': 'config'}
    args_dict = {'override': 'config'}
    merged_dict = {'merged': 'config'}
    config = Mock()

    mock_resolve_modality_config.return_value = base_dict
    mock_build_dict_from_args.return_value = args_dict
    mock_deep_merge.return_value = merged_dict
    mock_load_from_dict.return_value = config

    # Run
    result = build_config_from_args(args)

    # Assert
    mock_resolve_modality_config.assert_called_once_with('single_table')
    mock_build_dict_from_args.assert_called_once_with(args)
    mock_deep_merge.assert_called_once_with(base_dict, args_dict)
    mock_load_from_dict.assert_called_once_with(merged_dict)
    assert result is config


@patch('sdgym._benchmark_launcher.script.BenchmarkLauncher')
@patch('sdgym._benchmark_launcher.script.BenchmarkConfig.load_from_yaml')
@patch('sdgym._benchmark_launcher.script._validate_args')
@patch('sdgym._benchmark_launcher.script._parse_args')
def test_launch_from_args_uses_yaml_config_when_config_filepath_is_provided(
    mock_parse_args,
    mock_validate_args,
    mock_load_from_yaml,
    mock_benchmark_launcher,
):
    """Test `launch_from_args` loads YAML when config filepath is provided."""
    # Setup
    args = Namespace(config_filepath='config.yaml')
    config = Mock()
    launcher = Mock()

    mock_parse_args.return_value = args
    mock_load_from_yaml.return_value = config
    mock_benchmark_launcher.return_value = launcher

    # Run
    launch_from_args()

    # Assert
    mock_parse_args.assert_called_once_with()
    mock_validate_args.assert_called_once_with(args)
    mock_load_from_yaml.assert_called_once_with('config.yaml')
    mock_benchmark_launcher.assert_called_once_with(config)
    launcher.launch.assert_called_once_with()


@patch('sdgym._benchmark_launcher.script.BenchmarkLauncher')
@patch('sdgym._benchmark_launcher.script.build_config_from_args')
@patch('sdgym._benchmark_launcher.script._validate_args')
@patch('sdgym._benchmark_launcher.script._parse_args')
def test_launch_from_args_builds_config_when_config_filepath_is_not_provided(
    mock_parse_args,
    mock_validate_args,
    mock_build_config_from_args,
    mock_benchmark_launcher,
):
    """Test `launch_from_args` method."""
    # Setup
    args = Namespace(config_filepath=None)
    config = Mock()
    launcher = Mock()

    mock_parse_args.return_value = args
    mock_build_config_from_args.return_value = config
    mock_benchmark_launcher.return_value = launcher

    # Run
    launch_from_args()

    # Assert
    mock_parse_args.assert_called_once_with()
    mock_validate_args.assert_called_once_with(args)
    mock_build_config_from_args.assert_called_once_with(args)
    mock_benchmark_launcher.assert_called_once_with(config)
    launcher.launch.assert_called_once_with()
