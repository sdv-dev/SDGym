"""Unit tests for the benchmark launcher script."""

import re
from argparse import Namespace
from unittest.mock import Mock, patch

import pytest

from sdgym._benchmark_launcher.script import (
    _build_instance_jobs,
    _get_default_datasets_and_synthesizers,
    _instance_job_size,
    _parse_args,
    _parse_csv,
    _split_instance_jobs,
    _split_list,
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


def test__parse_csv_splits_and_strips_values():
    """Test `_parse_csv` splits comma-separated values and strips whitespace."""
    # Setup
    value = 'adult, alarm ,census ,, intrusion '

    # Run
    parsed = _parse_csv(value)
    empty = _parse_csv('')

    # Assert
    assert parsed == ['adult', 'alarm', 'census', 'intrusion']
    assert empty is None


def test__validate_args_with_config_filepath():
    """Test `_validate_args` with a config_filepath."""
    # Setup
    args = Namespace(
        config_filepath='config.yaml',
        modality=None,
        datasets=None,
        synthesizers=None,
        num_instances=None,
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
        ('num_instances', 0, "'--num-instances' must be greater than or equal to 1."),
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
        num_instances=1,
        timeout=None,
        output_destination='s3://sdgym-benchmark/Debug/test/',
    )
    setattr(args, param, value)

    # Run and Assert
    with pytest.raises(ValueError, match=message):
        _validate_args(args)


@pytest.mark.parametrize(
    'values,expected_left,expected_right',
    [
        (['a', 'b', 'c', 'd'], ['a', 'b'], ['c', 'd']),
        (['a', 'b', 'c'], ['a'], ['b', 'c']),
        (['a', 'b'], ['a'], ['b']),
    ],
)
def test__split_list(values, expected_left, expected_right):
    """Test `_split_list` method."""
    # Run
    left, right = _split_list(values)

    # Assert
    assert left == expected_left
    assert right == expected_right


def test__instance_job_size_returns_number_of_atomic_jobs():
    """Test `_instance_job_size` returns synthesizers x datasets."""
    # Setup
    instance_job = {
        'synthesizers': ['CTGANSynthesizer', 'TVAESynthesizer'],
        'datasets': ['adult', 'alarm', 'census'],
    }

    # Run
    size = _instance_job_size(instance_job)

    # Assert
    assert size == 6


def test__split_instance_jobs_synthesizers():
    """Test `_split_instance_jobs` first splits synthesizers when possible."""
    # Setup
    instance_job = {
        'synthesizers': ['CTGANSynthesizer', 'TVAESynthesizer', 'GaussianCopulaSynthesizer'],
        'datasets': ['adult', 'alarm'],
    }

    # Run
    split_jobs = _split_instance_jobs(instance_job)

    # Assert
    assert split_jobs == [
        {
            'synthesizers': ['CTGANSynthesizer'],
            'datasets': ['adult', 'alarm'],
        },
        {
            'synthesizers': ['TVAESynthesizer', 'GaussianCopulaSynthesizer'],
            'datasets': ['adult', 'alarm'],
        },
    ]


def test__split_instance_jobs_splits_datasets():
    """Test `_split_instance_jobs` splits datasets when only one synthesizer exists."""
    # Setup
    instance_job = {
        'synthesizers': ['CTGANSynthesizer'],
        'datasets': ['adult', 'alarm', 'census'],
    }

    # Run
    split_jobs = _split_instance_jobs(instance_job)

    # Assert
    assert split_jobs == [
        {
            'synthesizers': ['CTGANSynthesizer'],
            'datasets': ['adult'],
        },
        {
            'synthesizers': ['CTGANSynthesizer'],
            'datasets': ['alarm', 'census'],
        },
    ]


def test__split_instance_jobs_error():
    """Test `_split_instance_jobs` raises an error when it cannot be split further."""
    # Setup
    instance_job = {
        'synthesizers': ['CTGANSynthesizer'],
        'datasets': ['adult'],
    }
    expected_message = re.escape('Cannot split the instance job any further.')

    # Run and Assert
    with pytest.raises(ValueError, match=expected_message):
        _split_instance_jobs(instance_job)


def test__build_instance_jobs():
    """Test `_build_instance_jobs` method."""
    # Setup
    datasets = ['adult', 'alarm']
    synthesizers = ['CTGANSynthesizer', 'TVAESynthesizer', 'GaussianCopulaSynthesizer']
    num_instances = 3

    # Run
    instance_jobs = _build_instance_jobs(datasets, synthesizers, num_instances)

    # Assert
    assert instance_jobs == [
        {
            'synthesizers': ['CTGANSynthesizer'],
            'datasets': ['adult', 'alarm'],
        },
        {
            'synthesizers': ['TVAESynthesizer'],
            'datasets': ['adult', 'alarm'],
        },
        {
            'synthesizers': ['GaussianCopulaSynthesizer'],
            'datasets': ['adult', 'alarm'],
        },
    ]


def test__build_instance_jobs_warns_and_caps_num_instances():
    """Test `_build_instance_jobs` warns and caps num_instances to the maximum."""
    # Setup
    datasets = ['adult', 'alarm']
    synthesizers = ['CTGANSynthesizer']
    num_instances = 3
    expected_message = re.escape(
        'num_instances is too high for the number of synthesizers and datasets. '
        'Maximum number of instances is 2. Setting num_instances to 2.'
    )

    # Run
    with pytest.warns(UserWarning, match=expected_message):
        instance_jobs = _build_instance_jobs(datasets, synthesizers, num_instances)

    # Assert
    assert instance_jobs == [
        {
            'synthesizers': ['CTGANSynthesizer'],
            'datasets': ['adult'],
        },
        {
            'synthesizers': ['CTGANSynthesizer'],
            'datasets': ['alarm'],
        },
    ]


@patch('sdgym._benchmark_launcher.script._resolve_modality_config')
def test__get_default_datasets_and_synthesizers(mock_resolve_modality_config):
    """Test `_get_default_datasets_and_synthesizers` returns default datasets and synthesizers."""
    # Setup
    mock_resolve_modality_config.return_value = {
        'instance_jobs': [
            {'datasets': ['adult', 'alarm'], 'synthesizers': ['CTGANSynthesizer']},
            {'datasets': ['census'], 'synthesizers': ['TVAESynthesizer']},
        ]
    }

    # Run
    datasets, synthesizers = _get_default_datasets_and_synthesizers('single_table')

    # Assert
    mock_resolve_modality_config.assert_called_once_with('single_table')
    assert datasets == ['adult', 'alarm', 'census']
    assert synthesizers == ['CTGANSynthesizer', 'TVAESynthesizer']


@patch('sdgym._benchmark_launcher.script._build_instance_jobs')
@patch('sdgym._benchmark_launcher.script._parse_csv')
def test_build_dict_from_args_builds_expected_override_dict(
    mock_parse_csv, mock_build_instance_jobs
):
    """Test `build_dict_from_args` builds the expected config override dict."""
    # Setup
    args = Namespace(
        modality='single_table',
        datasets='adult,alarm',
        synthesizers='CTGANSynthesizer,TVAESynthesizer',
        num_instances=2,
        timeout=3600,
        output_destination='s3://sdgym-benchmark/Debug/test/',
    )
    mock_parse_csv.side_effect = [
        ['adult', 'alarm'],
        ['CTGANSynthesizer', 'TVAESynthesizer'],
    ]
    mock_build_instance_jobs.return_value = [
        {'synthesizers': ['CTGANSynthesizer'], 'datasets': ['adult', 'alarm']},
        {'synthesizers': ['TVAESynthesizer'], 'datasets': ['adult', 'alarm']},
    ]

    # Run
    config = build_dict_from_args(args)

    # Assert
    assert mock_parse_csv.call_count == 2
    mock_parse_csv.assert_any_call('adult,alarm')
    mock_parse_csv.assert_any_call('CTGANSynthesizer,TVAESynthesizer')
    mock_build_instance_jobs.assert_called_once_with(
        datasets=['adult', 'alarm'],
        synthesizers=['CTGANSynthesizer', 'TVAESynthesizer'],
        num_instances=2,
    )
    assert config == {
        'method_params': {
            'timeout': 3600,
            'output_destination': 's3://sdgym-benchmark/Debug/test/',
        },
        'instance_jobs': [
            {'synthesizers': ['CTGANSynthesizer'], 'datasets': ['adult', 'alarm']},
            {'synthesizers': ['TVAESynthesizer'], 'datasets': ['adult', 'alarm']},
        ],
    }


@patch('sdgym._benchmark_launcher.script._build_instance_jobs', return_value=[])
@patch('sdgym._benchmark_launcher.script._parse_csv', side_effect=[['adult'], ['CTGANSynthesizer']])
def test_build_dict_from_args_without_timeout(mock_parse_csv, mock_build_instance_jobs):
    """Test `build_dict_from_args` omits timeout when it is not provided."""
    # Setup
    args = Namespace(
        modality='single_table',
        datasets='adult',
        synthesizers='CTGANSynthesizer',
        num_instances=1,
        timeout=None,
        output_destination='s3://sdgym-benchmark/Debug/test/',
    )

    # Run
    config = build_dict_from_args(args)

    # Assert
    mock_parse_csv.assert_any_call('adult')
    mock_parse_csv.assert_any_call('CTGANSynthesizer')
    mock_build_instance_jobs.assert_called_once_with(
        datasets=['adult'],
        synthesizers=['CTGANSynthesizer'],
        num_instances=1,
    )
    assert config == {
        'method_params': {
            'output_destination': 's3://sdgym-benchmark/Debug/test/',
        },
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
