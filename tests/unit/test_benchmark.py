import json
import logging
import re
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from unittest.mock import ANY, MagicMock, Mock, call, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from sdgym.benchmark import (
    _add_adjusted_scores,
    _check_write_permissions,
    _create_sdgym_script,
    _directory_exists,
    _ensure_uniform_included,
    _fill_adjusted_scores_with_none,
    _format_output,
    _generate_job_args_list,
    _get_metainfo_increment,
    _handle_deprecated_parameters,
    _import_and_validate_synthesizers,
    _setup_output_destination,
    _setup_output_destination_aws,
    _update_metainfo_file,
    _validate_aws_inputs,
    _validate_output_destination,
    _write_metainfo_file,
    benchmark_multi_table,
    benchmark_multi_table_aws,
    benchmark_single_table,
    benchmark_single_table_aws,
)
from sdgym.result_writer import LocalResultsWriter
from sdgym.s3 import S3_REGION
from sdgym.synthesizers import MultiTableUniformSynthesizer, UniformSynthesizer


class FakeSynth:
    """Simple fake synthesizer with a configurable modality flag."""

    def __init__(self, modality):
        self._MODALITY_FLAG = modality


@pytest.mark.parametrize('modality', ['single_table', 'multi_table'])
@patch('sdgym.benchmark.get_duplicates', return_value=[])
@patch('sdgym.benchmark.get_synthesizers')
def test__import_and_validate_synthesizers_valid(
    mock_get_synthesizers, mock_get_duplicates, modality
):
    """Test that `_import_and_validate_synthesizers` returns the `get_synthesizers` values."""
    # Setup
    fake_synth = FakeSynth(modality)

    mock_get_synthesizers.return_value = [{'name': 'FakeSynth', 'synthesizer': fake_synth}]

    # Run
    result = _import_and_validate_synthesizers(
        synthesizers=['FakeSynth'],
        custom_synthesizers=None,
        modality=modality,
    )

    # Assert
    assert result == mock_get_synthesizers.return_value
    mock_get_synthesizers.assert_called_once_with(['FakeSynth'])


@pytest.mark.parametrize('modality', ['single_table', 'multi_table'])
@patch('sdgym.benchmark.get_duplicates', return_value=[])
@patch('sdgym.benchmark.get_synthesizers')
def test__import_and_validate_synthesizers_mismatched_modality(
    mock_get_synthesizers, mock_get_duplicates, modality
):
    """Test `_import_and_validate_synthesizers` raises  ValueError.

    Test to ensure that if a synthesizer's modality does not match the expected one a ValueError
    is being raised.
    """
    # Setup
    wrong_modality = 'multi_table' if modality == 'single_table' else 'single_table'

    # Dynamically create a class named BadSynth
    FakeWrong = type('BadSynth', (), {'_MODALITY_FLAG': wrong_modality})
    fake_wrong = FakeWrong()

    mock_get_synthesizers.return_value = [{'name': 'BadSynth', 'synthesizer': fake_wrong}]

    expected_message = (
        f"Synthesizers must be of modality '{modality}'. "
        f"Found these synthesizers that don't match: BadSynth"
    )

    # Run and Assert
    with pytest.raises(ValueError, match=expected_message):
        _import_and_validate_synthesizers(
            synthesizers=['BadSynth'],
            custom_synthesizers=[],
            modality=modality,
        )


@pytest.mark.parametrize('modality', ['single_table', 'multi_table'])
@patch('sdgym.benchmark.get_duplicates', return_value=['DupSynth'])
@patch('sdgym.benchmark.get_synthesizers')
def test__import_and_validate_synthesizers_duplicates(
    mock_get_synthesizers, mock_get_duplicates, modality
):
    """Test `_import_and_validate_synthesizers` raises a ValueError when duplicate values."""
    # Setup
    fake_synth = FakeSynth(modality)
    mock_get_synthesizers.return_value = [{'name': 'DupSynth', 'synthesizer': fake_synth}]

    expected_message = re.escape(
        'Synthesizers must be unique. Please remove repeated values in the provided synthesizers.'
    )

    # Run and Assert
    with pytest.raises(ValueError, match=expected_message):
        _import_and_validate_synthesizers(
            synthesizers=['DupSynth'],
            custom_synthesizers=['DupSynth'],
            modality=modality,
        )


@pytest.mark.parametrize('modality', ['single_table', 'multi_table'])
@patch('sdgym.benchmark.get_duplicates', return_value=[])
@patch('sdgym.benchmark.get_synthesizers')
def test__import_and_validate_synthesizers_none_inputs(
    mock_get_synthesizers, mock_get_duplicates, modality
):
    """Test `_import_and_validate_synthesizers` with empty lists."""
    # Run
    result = _import_and_validate_synthesizers(
        synthesizers=None,
        custom_synthesizers=None,
        modality=modality,
    )

    # Assert
    assert result == mock_get_synthesizers.return_value
    mock_get_synthesizers.assert_called_once_with([])


@patch('sdgym.benchmark.os.path')
def test_output_file_exists(path_mock):
    """Test the benchmark function when the output path already exists."""
    # Setup
    path_mock.exists.return_value = True
    output_filepath = 's3://test_output.csv'

    # Run and assert
    with pytest.raises(
        ValueError,
        match='test_output.csv already exists. Please provide a file that does not already exist.',
    ):
        benchmark_single_table(
            synthesizers=['DataIdentity', 'ColumnSynthesizer', 'UniformSynthesizer'],
            sdv_datasets=['student_placements'],
            output_filepath=output_filepath,
        )


@patch('sdgym.benchmark.boto3.client')
@patch('sdgym.benchmark.LOGGER')
def test__get_metainfo_increment_aws(mock_logger, mock_client):
    """Test the `_get_metainfo_increment` method when looking for files on aws."""
    # Setup
    # Mock S3 response with existing metainfo files
    mock_client.list_objects_v2.side_effect = [
        ValueError('Simulated S3 error'),
        {
            'Contents': [
                {'Key': 'SDGym_results_10_01_2023/metainfo.yaml'},
                {'Key': 'SDGym_results_10_01_2023/metainfo(1).yaml'},
                {'Key': 'SDGym_results_10_01_2023/metainfo(2).yaml'},
            ]
        },
        {'Contents': []},
    ]
    top_folder = 's3://my-bucket/SDGym_results_10_01_2023'
    s3_client = mock_client

    # Run
    result_1 = _get_metainfo_increment(top_folder, s3_client)
    mock_logger.info.assert_called_once_with('No metainfo file found, starting from increment (0)')
    result_2 = _get_metainfo_increment(top_folder, s3_client)
    result_3 = _get_metainfo_increment(top_folder, s3_client)

    # Assert
    assert result_1 == 0
    assert result_2 == 3
    assert result_3 == 0


@patch('sdgym.benchmark.LOGGER')
def test__get_metainfo_increment_local(mock_logger, tmp_path):
    """Test the `_get_metainfo_increment` method when looking for local files."""
    # Setup
    top_folder = tmp_path / 'SDGym_results_10_01_2023'

    # Run
    result_1 = _get_metainfo_increment(top_folder)
    mock_logger.info.assert_called_once_with('No metainfo file found, starting from increment (0)')
    top_folder.mkdir(parents=True, exist_ok=True)
    result_2 = _get_metainfo_increment(top_folder)
    (top_folder / 'metainfo.yaml').touch()
    (top_folder / 'metainfo(1).yaml').touch()
    (top_folder / 'metainfo(2).yaml').touch()
    result_3 = _get_metainfo_increment(top_folder)

    # Assert
    assert result_1 == 0
    assert result_2 == 0
    assert result_3 == 3


@patch('sdgym.benchmark.tqdm.tqdm')
@patch('sdgym.benchmark._handle_deprecated_parameters')
def test_benchmark_single_table_deprecated_params(mock_handle_deprecated, tqdm_mock):
    """Test that the benchmarking function updates the progress bar on one line."""
    # Setup
    scores_mock = MagicMock()
    scores_mock.__iter__.return_value = [
        pd.DataFrame({
            'Synthesizer': ['UniformSynthesizer'],
            'Dataset': ['student_placements'],
            'Train_Time': [1.0],
            'Sample_Time': [1.0],
        })
    ]
    tqdm_mock.return_value = scores_mock

    # Run
    benchmark_single_table(
        synthesizers=['UniformSynthesizer'],
        sdv_datasets=['student_placements'],
        show_progress=True,
    )

    # Assert
    mock_handle_deprecated.assert_called_once_with(None, None, None, False, None)
    tqdm_mock.assert_called_once_with(ANY, total=1, position=0, leave=True)


@patch('sdgym.benchmark._score')
@patch('sdgym.benchmark.multiprocessing')
def test_benchmark_single_table_with_timeout(mock_multiprocessing, mock__score):
    """Test that benchmark runs with timeout."""
    # Setup
    mocked_process = mock_multiprocessing.Process.return_value
    manager = mock_multiprocessing.Manager.return_value
    manager_dict = {'timeout': True, 'error': 'Synthesizer Timeout'}
    manager.__enter__.return_value.dict.return_value = manager_dict

    # Run
    scores = benchmark_single_table(
        synthesizers=['UniformSynthesizer'],
        sdv_datasets=['student_placements'],
        timeout=1,
    )

    # Assert
    mocked_process.start.assert_called_once_with()
    mocked_process.join.assert_called_once_with(1)
    mocked_process.terminate.assert_called_once_with()
    expected_scores = pd.DataFrame({
        'Synthesizer': {0: 'UniformSynthesizer'},
        'Dataset': {0: 'student_placements'},
        'Dataset_Size_MB': {0: None},
        'Train_Time': {0: None},
        'Peak_Memory_MB': {0: None},
        'Synthesizer_Size_MB': {0: None},
        'Sample_Time': {0: None},
        'Evaluate_Time': {0: None},
        'Diagnostic_Score': {0: None},
        'Quality_Score': {0: None},
        'Privacy_Score': {0: None},
        'error': {0: 'Synthesizer Timeout'},
        'Adjusted_Total_Time': {0: None},
        'Adjusted_Quality_Score': {0: None},
    })
    pd.testing.assert_frame_equal(scores, expected_scores, check_dtype=False)


@patch('sdgym.benchmark.boto3.client')
def test__directory_exists(mock_client):
    # Setup
    mock_client.return_value.list_objects_v2.return_value = {
        'Contents': [
            {
                'Key': 'example.txt',
                'ETag': '"1234567890abcdef1234567890abcdef"',
                'Size': 1024,
                'StorageClass': 'STANDARD',
            },
            {
                'Key': 'example_folder/',
                'ETag': '"0987654321fedcba0987654321fedcba"',
                'Size': 0,
                'StorageClass': 'STANDARD',
            },
        ],
        'CommonPrefixes': [
            {'Prefix': 'example_folder/subfolder1/'},
            {'Prefix': 'example_folder/subfolder2/'},
        ],
    }

    # Run and Assert
    assert _directory_exists('bucket', 'file_path/mock.csv')

    # Setup Failure
    mock_client.return_value.list_objects_v2.return_value = {}

    # Run and Assert
    assert not _directory_exists('bucket', 'file_path/mock.csv')


def test__check_write_permissions():
    """Test the `_check_write_permissions` function."""
    # Setup
    mock_client = Mock()

    # Run and Assert
    assert _check_write_permissions(mock_client, 'bucket')
    mock_client.put_object.side_effect = Exception('Simulated error')
    assert not _check_write_permissions(mock_client, 'bucket')


@patch('sdgym.benchmark._directory_exists')
@patch('sdgym.benchmark._check_write_permissions')
@patch('sdgym.benchmark.boto3.session.Session')
@patch('sdgym.benchmark._create_instance_on_ec2')
def test_run_ec2_flag(create_ec2_mock, session_mock, mock_write_permissions, mock_directory_exists):
    """Test that the benchmarking function updates the progress bar on one line."""
    # Setup
    create_ec2_mock.return_value = MagicMock()
    session_mock.get_credentials.return_value = MagicMock()
    mock_write_permissions.return_value = True
    mock_directory_exists.return_value = True

    # Run
    benchmark_single_table(run_on_ec2=True, output_filepath='s3://BucketName/path')

    # Assert
    create_ec2_mock.assert_called_once()

    # Run
    with pytest.raises(
        ValueError, match=r'In order to run on EC2, please provide an S3 folder output.'
    ):
        benchmark_single_table(run_on_ec2=True)

    # Assert
    create_ec2_mock.assert_called_once()

    # Run
    with pytest.raises(
        ValueError,
        match=r"""Invalid S3 path format.
                         Expected 's3://<bucket_name>/<path_to_file>'.""",
    ):
        benchmark_single_table(run_on_ec2=True, output_filepath='Wrong_Format')

    # Assert
    create_ec2_mock.assert_called_once()

    # Setup for failure in permissions
    mock_write_permissions.return_value = False

    # Run
    with pytest.raises(ValueError, match=r'No write permissions allowed for the bucket.'):
        benchmark_single_table(run_on_ec2=True, output_filepath='s3://BucketName/path')

    # Setup for failure in directory exists
    mock_write_permissions.return_value = True
    mock_directory_exists.return_value = False

    # Run
    with pytest.raises(ValueError, match=r'Directories in mock/path do not exist'):
        benchmark_single_table(run_on_ec2=True, output_filepath='s3://BucketName/mock/path')


@pytest.mark.parametrize(
    'modality,uniform_string',
    [('single_table', 'UniformSynthesizer'), ('multi_table', 'MultiTableUniformSynthesizer')],
)
def test__ensure_uniform_included_adds_uniform(modality, uniform_string, caplog):
    """Test that UniformSynthesizer gets added to the synthesizers list."""
    # Setup
    synthesizers = ['GaussianCopulaSynthesizer']
    expected_message = f'Adding {uniform_string} to the list of synthesizers.'

    # Run
    with caplog.at_level(logging.INFO):
        _ensure_uniform_included(synthesizers, modality)

    # Assert
    assert synthesizers == ['GaussianCopulaSynthesizer', uniform_string]
    assert any(expected_message in record.message for record in caplog.records)


@pytest.mark.parametrize(
    'modality,uniform_class',
    [('single_table', UniformSynthesizer), ('multi_table', MultiTableUniformSynthesizer)],
)
def test__ensure_uniform_included_detects_uniform_class(modality, uniform_class, caplog):
    """Test that the synthesizers list is unchanged if UniformSynthesizer class present."""
    # Setup
    synthesizers = [uniform_class, 'GaussianCopulaSynthesizer']
    expected_message = f'Adding {uniform_class} to the list of synthesizers.'

    # Run
    with caplog.at_level(logging.INFO):
        _ensure_uniform_included(synthesizers, modality)

    # Assert
    assert synthesizers == [uniform_class, 'GaussianCopulaSynthesizer']
    assert all(expected_message not in record.message for record in caplog.records)


@pytest.mark.parametrize(
    'modality,uniform_string',
    [('single_table', 'UniformSynthesizer'), ('multi_table', 'MultiTableUniformSynthesizer')],
)
def test__ensure_uniform_included_detects_uniform_string(modality, uniform_string, caplog):
    """Test that the synthesizers list is unchanged if UniformSynthesizer string present."""
    # Setup
    synthesizers = ['UniformSynthesizer', 'GaussianCopulaSynthesizer']
    expected_message = 'Adding UniformSynthesizer to list of synthesizers.'

    # Run
    with caplog.at_level(logging.INFO):
        _ensure_uniform_included(synthesizers, 'single_table')

    # Assert
    assert synthesizers == ['UniformSynthesizer', 'GaussianCopulaSynthesizer']
    assert all(expected_message not in record.message for record in caplog.records)


@patch('sdgym.benchmark._directory_exists')
@patch('sdgym.benchmark._check_write_permissions')
@patch('sdgym.benchmark.boto3.session.Session')
def test__create_sdgym_script(session_mock, mock_write_permissions, mock_directory_exists):
    """Test that the created SDGym script contains the expected values."""
    # Setup
    session_mock.get_credentials.return_value = MagicMock()
    test_params = {
        'synthesizers': ['GaussianCopulaSynthesizer', 'CTGANSynthesizer'],
        'custom_synthesizers': None,
        'sdv_datasets': [
            'adult',
            'alarm',
            'census',
            'child',
            'expedia_hotel_logs',
            'insurance',
            'intrusion',
            'news',
            'covtype',
        ],
        'limit_dataset_size': True,
        'compute_quality_score': False,
        'compute_privacy_score': False,
        'compute_diagnostic_score': False,
        'sdmetrics': None,
        'timeout': 600,
        'output_filepath': 's3://sdgym-results/address_comments.csv',
        'detailed_results_folder': None,
        'additional_datasets_folder': 'Details/',
        'show_progress': False,
        'multi_processing_config': None,
        'dummy': True,
    }
    mock_write_permissions.return_value = True
    mock_directory_exists.return_value = True

    # Run
    result = _create_sdgym_script(test_params, 's3://Bucket/Filepath')

    # Assert
    assert 'synthesizers=["GaussianCopulaSynthesizer", "CTGANSynthesizer"]' in result
    assert 'detailed_results_folder=None' in result
    assert "additional_datasets_folder='Details/'" in result
    assert 'multi_processing_config=None' in result
    assert 'sdmetrics=None' in result
    assert 'timeout=600' in result
    assert 'compute_quality_score=False' in result
    assert 'compute_diagnostic_score=False' in result
    assert 'compute_privacy_score=False' in result
    assert 'import boto3' in result


def test__format_output():
    """Test the method ``_format_output`` and confirm that metrics are properly computed."""
    # Setup
    mock_dataframe = pd.DataFrame([])
    mock_output = {
        'timeout': False,
        'dataset_size': 3.907452,
        'synthetic_data': mock_dataframe,
        'train_time': 267.028721,
        'sample_time': 1.039627,
        'synthesizer_size': 0.936981,
        'peak_memory': 127.729832,
        'diagnostic_score': 1.0,
        'quality_score': 0.881,
        'privacy_score': 0.588,
        'quality_score_time': 1.0,
        'diagnostic_score_time': 3.0,
        'privacy_score_time': 4.0,
        'scores': [
            {
                'metric': 'NewRowSynthesis',
                'error': None,
                'score': 0.998,
                'normalized_score': 0.998,
                'metric_time': 6.0,
            },
            {
                'metric': 'NewMetric',
                'error': None,
                'score': 0.998,
                'normalized_score': 0.998,
                'metric_time': 5.0,
            },
        ],
    }

    # Run
    scores = _format_output(mock_output, 'mock_name', 'mock_dataset', True, True, True, False)

    # Assert
    expected_scores = pd.DataFrame({
        'Synthesizer': ['mock_name'],
        'Dataset': ['mock_dataset'],
        'Dataset_Size_MB': [mock_output.get('dataset_size')],
        'Train_Time': [mock_output.get('train_time')],
        'Peak_Memory_MB': [mock_output.get('peak_memory')],
        'Synthesizer_Size_MB': [mock_output.get('synthesizer_size')],
        'Sample_Time': [mock_output.get('sample_time')],
        'Evaluate_Time': [19.0],
        'Diagnostic_Score': [1.0],
        'Quality_Score': [0.881],
        'Privacy_Score': [0.588],
        'NewRowSynthesis': [0.998],
        'NewMetric': [0.998],
    })
    pd.testing.assert_frame_equal(scores, expected_scores)


def test__handle_deprecated_parameters():
    """Test the ``_handle_deprecated_parameters`` function."""
    # Setup
    output_filepath = 's3://BucketName/path'
    detailed_results_folder = 'mock/path'
    multi_processing_config = {'num_processes': 4}
    run_on_ec2 = True
    base_warning = (
        "are deprecated in the 'benchmark_single_table' function. For saving results, "
        "please use the 'output_destination' parameter. For running SDGym remotely on AWS "
        "please use the 'benchmark_single_table_aws' method."
    )
    base_error = (
        "parameter is deprecated and cannot be used together with 'output_destination'. "
        "Please use only 'output_destination' to specify the output path."
    )

    # Expected messages
    expected_warning_1 = "Parameters 'detailed_results_folder', 'output_filepath' " + base_warning
    expected_warning_2 = (
        "Parameters 'detailed_results_folder', 'multi_processing_config', "
        "'output_filepath', 'run_on_ec2' " + base_warning
    )
    expected_error_1 = f"The 'output_filepath' {base_error}"
    expected_error_2 = f"The 'detailed_results_folder' {base_error}"

    # Run and Assert
    _handle_deprecated_parameters(None, None, None, False, None)
    with pytest.warns(FutureWarning, match=expected_warning_1):
        _handle_deprecated_parameters(output_filepath, detailed_results_folder, None, False, None)

    with pytest.warns(FutureWarning, match=expected_warning_2):
        _handle_deprecated_parameters(
            output_filepath, detailed_results_folder, multi_processing_config, run_on_ec2, None
        )

    with pytest.raises(ValueError, match=expected_error_1):
        _handle_deprecated_parameters(
            output_filepath, None, multi_processing_config, run_on_ec2, 'output_destination'
        )

    with pytest.raises(ValueError, match=expected_error_2):
        _handle_deprecated_parameters(
            None, detailed_results_folder, multi_processing_config, run_on_ec2, 'output_destination'
        )


def test__validate_output_destination(tmp_path):
    """Test the `_validate_output_destination` function."""
    # Setup
    wrong_type = 12345
    aws_destination = 's3://valid-bucket/path/to/file'
    valid_destination = tmp_path / 'valid-destination'
    err_1 = re.escape(
        'The `output_destination` parameter must be a string representing the output path.'
    )
    err_2 = re.escape(
        'The `output_destination` parameter cannot be an S3 path. '
        'Please use `benchmark_single_table_aws` instead.'
    )

    # Run and Assert
    _validate_output_destination(str(valid_destination))
    with pytest.raises(ValueError, match=err_1):
        _validate_output_destination(wrong_type)

    with pytest.raises(ValueError, match=err_2):
        _validate_output_destination(aws_destination)


@patch('sdgym.benchmark._validate_aws_inputs')
def test__validate_output_destination_with_aws_access_key_ids(mock_validate):
    """Test the `_validate_output_destination` function with AWS keys."""
    # Setup
    output_destination = 's3://my-bucket/path/to/file'
    aws_access_key_ids = {
        'aws_access_key_id': 'mock_access_key',
        'aws_secret_access_key': 'mock_secret_key',
    }

    # Run
    _validate_output_destination(output_destination, aws_access_key_ids)

    # Assert
    mock_validate.assert_called_once_with(
        output_destination,
        aws_access_key_ids['aws_access_key_id'],
        aws_access_key_ids['aws_secret_access_key'],
    )


def test__setup_output_destination_none():
    """If output_destination is None, the function should return an empty dict."""
    # Setup
    synthesizers = ['GaussianCopulaSynthesizer', 'CTGANSynthesizer']
    datasets = ['adult', 'census']

    # Run
    result = _setup_output_destination(None, synthesizers, datasets, 'single_table')

    # Assert
    assert result == {}


def test__setup_output_destination_single_table(tmp_path):
    """Test the `_setup_output_destination` function with `single_table` modality."""
    # Setup
    output_destination = tmp_path / 'output_destination'
    synthesizers = ['GaussianCopulaSynthesizer', 'CTGANSynthesizer']
    datasets = ['adult', 'census']
    today = datetime.today().strftime('%m_%d_%Y')
    base_path = output_destination / 'single_table' / f'SDGym_results_{today}'

    # Run
    result = _setup_output_destination(output_destination, synthesizers, datasets, 'single_table')

    # Assert
    expected = {
        dataset: {
            synth: {
                'synthesizer': str(base_path / f'{dataset}_{today}' / synth / f'{synth}.pkl'),
                'synthetic_data': str(
                    base_path / f'{dataset}_{today}' / synth / f'{synth}_synthetic_data.csv'
                ),
                'benchmark_result': str(
                    base_path / f'{dataset}_{today}' / synth / f'{synth}_benchmark_result.csv'
                ),
                'metainfo': str(base_path / 'metainfo.yaml'),
                'results': str(base_path / 'results.csv'),
            }
            for synth in synthesizers
        }
        for dataset in datasets
    }

    assert json.loads(json.dumps(result)) == expected


def test__setup_output_destination_multi_table(tmp_path):
    """Test the `_setup_output_destination` function with `multi_table` modality."""
    # Setup
    output_destination = tmp_path / 'output_destination'
    synthesizers = ['HMASynthesizer']
    datasets = ['NBA', 'financial']
    today = datetime.today().strftime('%m_%d_%Y')
    base_path = output_destination / 'multi_table' / f'SDGym_results_{today}'

    # Run
    result = _setup_output_destination(output_destination, synthesizers, datasets, 'multi_table')

    # Assert
    expected = {
        dataset: {
            synth: {
                'synthesizer': str(base_path / f'{dataset}_{today}' / synth / f'{synth}.pkl'),
                'synthetic_data': str(
                    base_path / f'{dataset}_{today}' / synth / f'{synth}_synthetic_data.zip'
                ),
                'benchmark_result': str(
                    base_path / f'{dataset}_{today}' / synth / f'{synth}_benchmark_result.csv'
                ),
                'metainfo': str(base_path / 'metainfo.yaml'),
                'results': str(base_path / 'results.csv'),
            }
            for synth in synthesizers
        }
        for dataset in datasets
    }

    assert json.loads(json.dumps(result)) == expected


@patch('sdgym.benchmark.datetime')
def test__write_metainfo_file(mock_datetime, tmp_path):
    """Test the `_write_metainfo_file` method."""
    # Setup
    output_destination = tmp_path / 'SDGym_results_06_26_2025'
    output_destination.mkdir()
    mock_datetime.today.return_value.strftime.return_value = '06_26_2025'
    file_name = {'metainfo': f'{output_destination}/metainfo.yaml'}
    result_writer = LocalResultsWriter()
    jobs = [
        ({'name': 'GaussianCopulaSynthesizer'}, 'adult', None, file_name),
        ({'name': 'CTGANSynthesizer'}, 'census', None, None),
    ]
    expected_jobs = [['adult', 'GaussianCopulaSynthesizer'], ['census', 'CTGANSynthesizer']]
    synthesizers = [
        {'name': 'GaussianCopulaSynthesizer'},
        {'name': 'CTGANSynthesizer'},
        {'name': 'RealTabFormerSynthesizer'},
    ]

    # Run
    _write_metainfo_file(synthesizers, jobs, 'single_table', result_writer)

    # Assert
    with open(file_name['metainfo'], 'r') as file:
        metainfo_data = yaml.safe_load(file)

    assert Path(file_name['metainfo']).exists()
    assert metainfo_data['run_id'] == 'run_06_26_2025_0'
    assert metainfo_data['starting_date'] == '06_26_2025'
    assert metainfo_data['jobs'] == expected_jobs
    assert metainfo_data['sdgym_version'] == version('sdgym')
    assert metainfo_data['sdv_version'] == version('sdv')
    assert metainfo_data['realtabformer_version'] == version('realtabformer')
    assert metainfo_data['completed_date'] is None
    assert metainfo_data['modality'] == 'single_table'


@patch('sdgym.benchmark.datetime')
def test__update_metainfo_file(mock_datetime, tmp_path):
    """Test the `_update_metainfo_file` method."""
    # Setup
    output_destination = tmp_path / 'SDGym_results_06_25_2025'
    output_destination.mkdir()
    metadata = {'run_id': 'run_06_25_2025_1', 'starting_date': '06_25_2025', 'completed_date': None}
    metainfo_file = output_destination / 'metainfo.yaml'
    run_id = 'run_06_25_2025_1'
    result_writer = LocalResultsWriter()
    mock_datetime.today.return_value.strftime.return_value = '06_26_2025'
    with open(metainfo_file, 'w') as file:
        yaml.dump(metadata, file)

    # Run
    _update_metainfo_file(metainfo_file, result_writer)

    # Assert
    with open(metainfo_file, 'r') as file:
        metainfo_data = yaml.safe_load(file)
        assert metainfo_data['completed_date'] == '06_26_2025'
        assert metainfo_data['starting_date'] == '06_25_2025'
        assert metainfo_data['run_id'] == run_id


@pytest.mark.parametrize('modality', ['single_table', 'multi_table'])
@patch('sdgym.benchmark._get_metainfo_increment')
def test_setup_output_destination_aws(mock_get_metainfo_increment, modality):
    """Test the `_setup_output_destination_aws` function."""
    # Setup
    output_destination = 's3://my-bucket/results'
    synthesizers = ['GaussianCopulaSynthesizer', 'CTGANSynthesizer']
    datasets = ['Dataset1', 'Dataset2']
    s3_client_mock = Mock()
    mock_get_metainfo_increment.return_value = 0
    data_suffix = '_synthetic_data.csv' if modality == 'single_table' else '_synthetic_data.zip'

    # Run
    paths = _setup_output_destination_aws(
        output_destination, synthesizers, datasets, modality, s3_client_mock
    )

    # Assert
    today = datetime.today().strftime('%m_%d_%Y')
    bucket_name = 'my-bucket'
    top_folder = f'results/{modality}/SDGym_results_{today}'
    expected_calls = [call(Bucket=bucket_name, Key=top_folder + '/')]
    mock_get_metainfo_increment.assert_called_once_with(
        f's3://{bucket_name}/{top_folder}', s3_client_mock
    )
    for dataset in datasets:
        dataset_folder = f'{top_folder}/{dataset}_{today}'
        expected_calls.append(call(Bucket=bucket_name, Key=dataset_folder + '/'))
        for synth in synthesizers:
            synth_folder = f'{dataset_folder}/{synth}'
            expected_calls.append(call(Bucket=bucket_name, Key=synth_folder + '/'))

    s3_client_mock.put_object.assert_has_calls(expected_calls, any_order=True)
    for dataset in datasets:
        for synth in synthesizers:
            assert 'synthesizer' in paths[dataset][synth]
            assert paths[dataset][synth]['synthesizer'] == (
                f's3://{bucket_name}/{top_folder}/{dataset}_{today}/{synth}/{synth}.pkl'
            )
            assert 'synthetic_data' in paths[dataset][synth]
            assert paths[dataset][synth]['synthetic_data'] == (
                f's3://{bucket_name}/{top_folder}/{dataset}_{today}/{synth}/{synth}{data_suffix}'
            )
            assert 'benchmark_result' in paths[dataset][synth]
            assert paths[dataset][synth]['benchmark_result'] == (
                f's3://{bucket_name}/{top_folder}/{dataset}_{today}/{synth}/{synth}_benchmark_result.csv'
            )
            assert 'metainfo' in paths[dataset][synth]
            assert paths[dataset][synth]['metainfo'] == (
                f's3://{bucket_name}/{top_folder}/metainfo.yaml'
            )
            assert 'results' in paths[dataset][synth]
            assert paths[dataset][synth]['results'] == (
                f's3://{bucket_name}/{top_folder}/results.csv'
            )


@patch('sdgym.benchmark.boto3.client')
@patch('sdgym.benchmark._check_write_permissions')
@patch('sdgym.benchmark.Config')
def test_validate_aws_inputs_valid(mock_config, mock_check_write_permissions, mock_boto3_client):
    """Test `_validate_aws_inputs` with valid inputs and credentials."""
    # Setup
    config_mock = Mock()
    mock_config.return_value = config_mock
    valid_url = 's3://my-bucket/some/path'
    s3_client_mock = Mock()
    mock_boto3_client.return_value = s3_client_mock
    mock_check_write_permissions.return_value = True

    # Run
    result = _validate_aws_inputs(
        output_destination=valid_url, aws_access_key_id='AKIA...', aws_secret_access_key='SECRET'
    )

    # Assert
    mock_boto3_client.assert_called_once_with(
        's3',
        aws_access_key_id='AKIA...',
        aws_secret_access_key='SECRET',
        region_name=S3_REGION,
        config=config_mock,
    )
    s3_client_mock.head_bucket.assert_called_once_with(Bucket='my-bucket')
    mock_check_write_permissions.assert_called_once_with(s3_client_mock, 'my-bucket')
    assert result == s3_client_mock


def test_validate_aws_inputs_invalid():
    """Test `_validate_aws_inputs` raises ValueError for invalid inputs."""
    # Setup
    invalid_url_type = 123
    invalid_url_no_s3 = 'https://my-bucket/path'
    invalid_url_empty_bucket = 's3://'

    # Run and Assert
    with pytest.raises(
        ValueError,
        match=re.escape(
            'The `output_destination` parameter must be a string representing the S3 URL.'
        ),
    ):
        _validate_aws_inputs(invalid_url_type, None, None)

    with pytest.raises(
        ValueError,
        match=re.escape("'output_destination' must be an S3 URL starting with 's3://'. "),
    ):
        _validate_aws_inputs(invalid_url_no_s3, None, None)

    with pytest.raises(ValueError, match=re.escape(f'Invalid S3 URL: {invalid_url_empty_bucket}')):
        _validate_aws_inputs(invalid_url_empty_bucket, None, None)


@patch('sdgym.benchmark.boto3.client')
@patch('sdgym.benchmark._check_write_permissions')
def test_validate_aws_inputs_permission_error(mock_check_write_permissions, mock_boto3_client):
    """Test `_validate_aws_inputs` raises PermissionError when write permission is missing."""
    valid_url = 's3://my-bucket/some/path'
    s3_client_mock = Mock()
    mock_boto3_client.return_value = s3_client_mock
    mock_check_write_permissions.return_value = False

    with pytest.raises(
        PermissionError,
        match=re.escape(
            'No write permissions for the S3 bucket: my-bucket. '
            'Please check your AWS credentials or bucket policies.'
        ),
    ):
        _validate_aws_inputs(valid_url, None, None)


@patch('sdgym.benchmark._import_and_validate_synthesizers')
@patch('sdgym.benchmark._validate_output_destination')
@patch('sdgym.benchmark._generate_job_args_list')
@patch('sdgym.benchmark._run_on_aws')
def test_benchmark_single_table_aws(
    mock_run_on_aws,
    mock_generate_job_args_list,
    mock_validate_output_destination,
    mock__import_and_validate_synthesizers,
):
    """Test `benchmark_single_table_aws` method."""
    # Setup
    output_destination = 's3://sdgym-benchmark/Debug/Issue_414_test_3'
    synthesizers = ['GaussianCopulaSynthesizer', 'TVAESynthesizer']
    datasets = ['adult', 'census']
    aws_access_key_id = '12345'
    aws_secret_access_key = '67890'
    mock_validate_output_destination.return_value = 's3_client_mock'
    mock_generate_job_args_list.return_value = 'job_args_list_mock'
    mock__import_and_validate_synthesizers.return_value = synthesizers

    # Run
    benchmark_single_table_aws(
        output_destination=output_destination,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        synthesizers=synthesizers,
        sdv_datasets=datasets,
    )

    # Assert
    assert 'UniformSynthesizer' in synthesizers
    mock_validate_output_destination.assert_called_once_with(
        output_destination,
        aws_keys={
            'aws_access_key_id': aws_access_key_id,
            'aws_secret_access_key': aws_secret_access_key,
        },
    )
    mock_generate_job_args_list.assert_called_once_with(
        limit_dataset_size=False,
        sdv_datasets=datasets,
        additional_datasets_folder=None,
        sdmetrics=None,
        timeout=None,
        output_destination=output_destination,
        compute_quality_score=True,
        compute_diagnostic_score=True,
        compute_privacy_score=True,
        synthesizers=synthesizers,
        detailed_results_folder=None,
        s3_client='s3_client_mock',
        modality='single_table',
    )
    mock_run_on_aws.assert_called_once_with(
        output_destination=output_destination,
        synthesizers=synthesizers,
        s3_client='s3_client_mock',
        job_args_list='job_args_list_mock',
        aws_access_key_id='12345',
        aws_secret_access_key='67890',
    )


@patch('sdgym.benchmark._import_and_validate_synthesizers')
@patch('sdgym.benchmark._validate_output_destination')
@patch('sdgym.benchmark._generate_job_args_list')
@patch('sdgym.benchmark._run_on_aws')
def test_benchmark_single_table_aws_synthesizers_none(
    mock_run_on_aws,
    mock_generate_job_args_list,
    mock_validate_output_destination,
    mock__import_and_validate_synthesizers,
):
    """Test `benchmark_single_table_aws` includes UniformSynthesizer if omitted."""
    # Setup
    output_destination = 's3://sdgym-benchmark/Debug/Issue_414_test_3'
    synthesizers = None
    datasets = ['adult', 'census']
    aws_access_key_id = '12345'
    aws_secret_access_key = '67890'
    mock_validate_output_destination.return_value = 's3_client_mock'
    mock_generate_job_args_list.return_value = 'job_args_list_mock'
    mock__import_and_validate_synthesizers.return_value = ['UniformSynthesizer']

    # Run
    benchmark_single_table_aws(
        output_destination=output_destination,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        synthesizers=synthesizers,
        sdv_datasets=datasets,
    )

    # Assert
    mock_validate_output_destination.assert_called_once_with(
        output_destination,
        aws_keys={
            'aws_access_key_id': aws_access_key_id,
            'aws_secret_access_key': aws_secret_access_key,
        },
    )
    mock_generate_job_args_list.assert_called_once_with(
        limit_dataset_size=False,
        sdv_datasets=datasets,
        additional_datasets_folder=None,
        sdmetrics=None,
        timeout=None,
        output_destination=output_destination,
        compute_quality_score=True,
        compute_diagnostic_score=True,
        compute_privacy_score=True,
        detailed_results_folder=None,
        synthesizers=['UniformSynthesizer'],
        s3_client='s3_client_mock',
        modality='single_table',
    )
    mock_run_on_aws.assert_called_once_with(
        output_destination=output_destination,
        synthesizers=['UniformSynthesizer'],
        s3_client='s3_client_mock',
        job_args_list='job_args_list_mock',
        aws_access_key_id='12345',
        aws_secret_access_key='67890',
    )


def test__add_adjusted_scores_no_failures():
    """Test _add_adjusted_scores with no errors present."""
    # Setup
    scores = pd.DataFrame({
        'Synthesizer': ['GaussianCopulaSynthesizer', 'UniformSynthesizer'],
        'Dataset': ['dataset1', 'dataset1'],
        'Train_Time': [1.0, 0.5],
        'Sample_Time': [2.0, 0.25],
        'Quality_Score': [1.0, 0.5],
    })
    expected = pd.DataFrame({
        'Synthesizer': ['GaussianCopulaSynthesizer', 'UniformSynthesizer'],
        'Dataset': ['dataset1', 'dataset1'],
        'Train_Time': [1.0, 0.5],
        'Sample_Time': [2.0, 0.25],
        'Quality_Score': [1.0, 0.5],
        'Adjusted_Total_Time': [3.5, 1.25],
        'Adjusted_Quality_Score': [1.0, 0.5],
    })

    # Run
    _add_adjusted_scores(scores, 10.0)

    # Assert
    pd.testing.assert_frame_equal(scores, expected)


def test__fill_adjusted_scores_with_none():
    """Test the `_fill_adjusted_scores_with_none` method."""
    # Setup
    scores_1 = pd.DataFrame({
        'Synthesizer': ['Synth1', 'Synth2'],
        'Dataset': ['dataset1', 'dataset1'],
        'Train_Time': [1.0, None],
        'Sample_Time': [2.0, 3.0],
        'Quality_Score': [0.8, None],
    })
    scores_2 = pd.DataFrame({
        'Synthesizer': ['Synth1', 'Synth2'],
        'Dataset': ['dataset1', 'dataset1'],
        'Train_Time': [1.0, None],
        'Sample_Time': [2.0, 3.0],
    })
    expected = pd.DataFrame({
        'Synthesizer': ['Synth1', 'Synth2'],
        'Dataset': ['dataset1', 'dataset1'],
        'Train_Time': [1.0, None],
        'Sample_Time': [2.0, 3.0],
        'Quality_Score': [0.8, None],
        'Adjusted_Total_Time': None,
        'Adjusted_Quality_Score': None,
    })

    # Run
    result_1 = _fill_adjusted_scores_with_none(scores_1)
    result_2 = _fill_adjusted_scores_with_none(scores_2)

    # Assert
    pd.testing.assert_frame_equal(result_1, expected)
    pd.testing.assert_frame_equal(
        result_2, expected.drop(columns=['Quality_Score', 'Adjusted_Quality_Score'])
    )


def test__add_adjusted_scores_timeout():
    """Test _add_adjusted_scores when a synthesizer times out."""
    # Setup
    scores = pd.DataFrame({
        'Synthesizer': ['GaussianCopulaSynthesizer', 'UniformSynthesizer'],
        'Dataset': ['dataset1', 'dataset1'],
        'Train_Time': [np.nan, 0.5],
        'Sample_Time': [np.nan, 0.25],
        'Quality_Score': [np.nan, 0.5],
        'error': ['Synthesizer Timeout', np.nan],
    })
    expected = pd.DataFrame({
        'Synthesizer': ['GaussianCopulaSynthesizer', 'UniformSynthesizer'],
        'Dataset': ['dataset1', 'dataset1'],
        'Train_Time': [np.nan, 0.5],
        'Sample_Time': [np.nan, 0.25],
        'Quality_Score': [np.nan, 0.5],
        'error': ['Synthesizer Timeout', np.nan],
        'Adjusted_Total_Time': [10.75, 1.25],
        'Adjusted_Quality_Score': [0.5, 0.5],
    })

    # Run
    _add_adjusted_scores(scores, 10.0)

    # Assert
    assert scores.equals(expected)


def test__add_adjusted_scores_errors():
    """Test _add_adjusted_scores when synthesizers error out."""
    # Setup
    scores = pd.DataFrame({
        'Synthesizer': ['ErrorOnTrain', 'ErrorOnSample', 'ErrorAfterSample', 'UniformSynthesizer'],
        'Dataset': ['dataset1', 'dataset1', 'dataset1', 'dataset1'],
        'Train_Time': [np.nan, 1.0, 1.0, 0.5],
        'Sample_Time': [np.nan, np.nan, 2.0, 0.25],
        'Quality_Score': [np.nan, np.nan, np.nan, 0.5],
        'error': ['ValueError', 'RuntimeError', 'KeyError', np.nan],
    })
    expected = pd.DataFrame({
        'Synthesizer': ['ErrorOnTrain', 'ErrorOnSample', 'ErrorAfterSample', 'UniformSynthesizer'],
        'Dataset': ['dataset1', 'dataset1', 'dataset1', 'dataset1'],
        'Train_Time': [np.nan, 1.0, 1.0, 0.5],
        'Sample_Time': [np.nan, np.nan, 2.0, 0.25],
        'Quality_Score': [np.nan, np.nan, np.nan, 0.5],
        'error': ['ValueError', 'RuntimeError', 'KeyError', np.nan],
        'Adjusted_Total_Time': [0.75, 1.75, 3.75, 1.25],
        'Adjusted_Quality_Score': [0.5, 0.5, 0.5, 0.5],
    })

    # Run
    _add_adjusted_scores(scores, 10.0)

    # Assert
    assert scores.equals(expected)


def test__add_adjusted_scores_missing_fallback():
    """Test _add_adjusted_scores with the UniformSynthesizer row missing."""
    # Setup
    scores = pd.DataFrame({
        'Synthesizer': ['GaussianCopulaSynthesizer', 'CTGANSynthesizer'],
        'Train_Time': [1.0, 3.0],
        'Sample_Time': [2.0, 1.0],
        'Quality_Score': [1.0, 0.8],
    })
    expected = pd.DataFrame({
        'Synthesizer': ['GaussianCopulaSynthesizer', 'CTGANSynthesizer'],
        'Train_Time': [1.0, 3.0],
        'Sample_Time': [2.0, 1.0],
        'Quality_Score': [1.0, 0.8],
        'Adjusted_Total_Time': [None, None],
        'Adjusted_Quality_Score': [None, None],
    })

    # Run
    _add_adjusted_scores(scores, 10.0)

    # Assert
    assert scores.equals(expected)


@pytest.mark.parametrize('modality', ['single_table', 'multi_table'])
@patch('sdgym.benchmark.get_dataset_paths')
def test__generate_job_args_list_local_root_additional_folder(
    get_dataset_paths_mock,
    tmp_path,
    modality,
):
    """Local additional_datasets_folder should point to root/single_table."""
    # Setup
    local_root = tmp_path / 'my_root'
    local_root.mkdir()
    dataset_path = tmp_path / 'my_root' / modality / 'datasetA'
    get_dataset_paths_mock.return_value = [dataset_path]

    # Run
    _generate_job_args_list(
        limit_dataset_size=False,
        sdv_datasets=None,
        additional_datasets_folder=str(local_root),
        sdmetrics=None,
        detailed_results_folder=None,
        timeout=None,
        output_destination=None,
        compute_quality_score=False,
        compute_diagnostic_score=False,
        compute_privacy_score=False,
        synthesizers=[],
        s3_client=None,
        modality=modality,
    )

    # Assert
    get_dataset_paths_mock.assert_called_once_with(
        modality=modality,
        bucket=str(local_root / modality),
        s3_client=None,
    )


@patch('sdgym.benchmark.get_dataset_paths')
@patch('sdgym.benchmark._setup_output_destination')
@patch('sdgym.benchmark.load_dataset')
def test__generate_job_args_list_s3_root_additional_folder(
    mock_load_dataset,
    mock__setup_output_destination,
    get_dataset_paths_mock,
):
    """S3 additional_datasets_folder should point to the root path."""
    # Setup
    s3_root = 's3://my-bucket/custom-datasets'
    dataset_path = Path('/dummy/single_table/datasetA')
    get_dataset_paths_mock.return_value = [dataset_path]
    s3_client = Mock()
    mock__setup_output_destination.return_value = {}
    mock_load_dataset.return_value = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

    # Run
    _generate_job_args_list(
        limit_dataset_size=False,
        sdv_datasets=None,
        additional_datasets_folder=s3_root,
        sdmetrics=None,
        detailed_results_folder=None,
        timeout=None,
        output_destination=None,
        compute_quality_score=False,
        compute_diagnostic_score=False,
        compute_privacy_score=False,
        synthesizers=[{'name': 'GaussianCopulaSynthesizer'}],
        s3_client=s3_client,
        modality='single_table',
    )

    # Assert
    get_dataset_paths_mock.assert_called_once_with(
        modality='single_table',
        bucket=s3_root,
        s3_client=s3_client,
    )
    mock__setup_output_destination.assert_called_once_with(
        None,
        ['GaussianCopulaSynthesizer'],
        ['datasetA'],
        modality='single_table',
        s3_client=s3_client,
    )
    mock_load_dataset.assert_called_once_with(
        'single_table', dataset_path, limit_dataset_size=False, s3_client=s3_client
    )


def test_benchmark_single_table_no_warning_uniform_synthesizer(recwarn):
    """Test that no UserWarning is raised when running `UniformSynthesizer`."""
    # Setup
    expected_result = pd.DataFrame({
        'Synthesizer': {0: 'UniformSynthesizer'},
        'Dataset': {0: 'fake_hotel_guests'},
    })

    # Run
    result = benchmark_single_table(
        synthesizers=['UniformSynthesizer'],
        custom_synthesizers=None,
        sdv_datasets=['fake_hotel_guests'],
        additional_datasets_folder=None,
        limit_dataset_size=False,
        timeout=None,
        show_progress=False,
        compute_diagnostic_score=True,
        compute_quality_score=True,
        compute_privacy_score=False,
        sdmetrics=None,
    )

    # Assert
    warnings_text = ' '.join(str(w.message) for w in recwarn)
    assert 'is incompatible with transformer' not in warnings_text
    pd.testing.assert_frame_equal(result[expected_result.columns], expected_result)


@patch('sdgym.benchmark._import_and_validate_synthesizers')
@patch('sdgym.benchmark._update_metainfo_file')
@patch('sdgym.benchmark._write_metainfo_file')
@patch('sdgym.benchmark._run_jobs')
@patch('sdgym.benchmark._generate_job_args_list')
@patch('sdgym.benchmark.LocalResultsWriter')
@patch('sdgym.benchmark._validate_output_destination')
def test_benchmark_multi_table_with_jobs(
    mock__validate_output_destination,
    mock_LocalResultsWriter,
    mock__generate_job_args_list,
    mock__run_jobs,
    mock__write_metainfo_file,
    mock__update_metainfo_file,
    mock__import_and_validate_synthesizers,
):
    """Test that `benchmark_multi_table` runs jobs and updates metainfo when there are job args."""
    # Setup
    fake_scores = pd.DataFrame({'a': [1]})
    mock__run_jobs.return_value = fake_scores
    job_args = ('arg1', 'arg2', {'metainfo': 'meta.yaml'})
    mock__generate_job_args_list.return_value = [job_args]
    expected_valid_synthesizers = ['HMASynthesizer', 'MultiTableUniformSynthesizer', 'CustomSynth']

    mock__import_and_validate_synthesizers.return_value = expected_valid_synthesizers
    # Run
    scores = benchmark_multi_table(
        synthesizers=['HMASynthesizer'],
        custom_synthesizers=['CustomSynth'],
        sdv_datasets=['dataset1'],
        additional_datasets_folder='extra',
        limit_dataset_size=True,
        compute_quality_score=True,
        compute_diagnostic_score=True,
        timeout=10,
        output_destination='output_dir',
        show_progress=True,
    )

    # Assert
    mock__validate_output_destination.assert_called_once_with('output_dir')
    mock_LocalResultsWriter.assert_called_once_with()
    mock__generate_job_args_list.assert_called_once_with(
        limit_dataset_size=True,
        sdv_datasets=['dataset1'],
        additional_datasets_folder='extra',
        sdmetrics=None,
        detailed_results_folder=None,
        timeout=10,
        output_destination='output_dir',
        compute_quality_score=True,
        compute_diagnostic_score=True,
        compute_privacy_score=None,
        synthesizers=expected_valid_synthesizers,
        s3_client=None,
        modality='multi_table',
    )
    mock__write_metainfo_file.assert_called_once()
    mock__run_jobs.assert_called_once_with(
        multi_processing_config=None,
        job_args_list=[job_args],
        show_progress=True,
        result_writer=mock_LocalResultsWriter.return_value,
    )
    mock__update_metainfo_file.assert_called_once_with(
        'meta.yaml',
        mock_LocalResultsWriter.return_value,
    )
    pd.testing.assert_frame_equal(scores, fake_scores)
    mock__import_and_validate_synthesizers.assert_called_once_with(
        ['HMASynthesizer', 'MultiTableUniformSynthesizer'], ['CustomSynth'], 'multi_table'
    )


@patch('sdgym.benchmark._import_and_validate_synthesizers')
@patch('sdgym.benchmark._write_metainfo_file')
@patch('sdgym.benchmark._validate_output_destination')
def test_benchmark_multi_table_no_jobs(
    mock__validate_output_destination,
    mock__write_metainfo_file,
    mock__import_and_validate_synthesizers,
):
    """Test that benchmark_multi_table returns empty dataframe when there are no job args."""
    # Setup
    empty_scores = pd.DataFrame({
        'Synthesizer': [],
        'Dataset': [],
        'Dataset_Size_MB': [],
        'Train_Time': [],
        'Peak_Memory_MB': [],
        'Synthesizer_Size_MB': [],
        'Sample_Time': [],
        'Evaluate_Time': [],
        'Adjusted_Total_Time': [],
        'Diagnostic_Score': [],
    })

    # Run
    scores = benchmark_multi_table(
        synthesizers=[],
        custom_synthesizers=None,
        sdv_datasets=None,
        additional_datasets_folder=None,
        limit_dataset_size=False,
        compute_quality_score=False,
        compute_diagnostic_score=True,
        timeout=None,
        output_destination=None,
        show_progress=False,
    )

    # Assert
    mock__validate_output_destination.assert_called_once_with(None)
    mock__write_metainfo_file.assert_called_once()
    pd.testing.assert_frame_equal(scores, empty_scores)


@patch('sdgym.benchmark._validate_output_destination')
@patch('sdgym.benchmark._generate_job_args_list')
@patch('sdgym.benchmark._run_on_aws')
@patch('sdgym.benchmark._import_and_validate_synthesizers')
def test_benchmark_multi_table_aws(
    mock_import_and_validate_synthesizers,
    mock_run_on_aws,
    mock_generate_job_args_list,
    mock_validate_output_destination,
):
    """Test `benchmark_multi_table_aws` method."""
    # Setup
    output_destination = 's3://sdgym-benchmark/Debug/Issue_487_test_1'
    synthesizers = ['HMASynthesizer']
    datasets = ['financial', 'NBA']
    aws_access_key_id = '12345'
    aws_secret_access_key = '67890'
    mock_validate_output_destination.return_value = 's3_client_mock'
    mock_generate_job_args_list.return_value = 'job_args_list_mock'
    mock_import_and_validate_synthesizers.return_value = synthesizers

    # Run
    benchmark_multi_table_aws(
        output_destination=output_destination,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        synthesizers=synthesizers,
        sdv_datasets=datasets,
    )

    # Assert
    assert 'MultiTableUniformSynthesizer' in synthesizers
    mock_validate_output_destination.assert_called_once_with(
        output_destination,
        aws_keys={
            'aws_access_key_id': aws_access_key_id,
            'aws_secret_access_key': aws_secret_access_key,
        },
    )
    mock_import_and_validate_synthesizers.assert_called_once_with(
        synthesizers=synthesizers,
        custom_synthesizers=None,
        modality='multi_table',
    )
    mock_generate_job_args_list.assert_called_once_with(
        limit_dataset_size=False,
        sdv_datasets=datasets,
        additional_datasets_folder=None,
        sdmetrics=None,
        timeout=None,
        output_destination=output_destination,
        compute_quality_score=True,
        compute_diagnostic_score=True,
        compute_privacy_score=None,
        synthesizers=synthesizers,
        detailed_results_folder=None,
        s3_client='s3_client_mock',
        modality='multi_table',
    )
    mock_run_on_aws.assert_called_once_with(
        output_destination=output_destination,
        synthesizers=synthesizers,
        s3_client='s3_client_mock',
        job_args_list='job_args_list_mock',
        aws_access_key_id='12345',
        aws_secret_access_key='67890',
    )


@patch('sdgym.benchmark._validate_output_destination')
@patch('sdgym.benchmark._generate_job_args_list')
@patch('sdgym.benchmark._import_and_validate_synthesizers')
@patch('sdgym.benchmark._get_empty_dataframe')
def test_benchmark_multi_table_aws_no_jobs(
    mock_get_empty_dataframe,
    mock_import_and_validate_synthesizers,
    mock_generate_job_args_list,
    mock_validate_output_destination,
):
    """Test `benchmark_multi_table_aws` method."""
    # Setup
    output_destination = 's3://sdgym-benchmark/Debug/Issue_487_test_1'
    synthesizers = ['HMASynthesizer']
    datasets = ['financial', 'NBA']
    aws_access_key_id = '12345'
    aws_secret_access_key = '67890'
    mock_validate_output_destination.return_value = 's3_client_mock'
    mock_generate_job_args_list.return_value = []
    mock_import_and_validate_synthesizers.return_value = synthesizers
    expected_result = pd.DataFrame({'Dataset': []})
    mock_get_empty_dataframe.return_value = expected_result

    # Run
    result = benchmark_multi_table_aws(
        output_destination=output_destination,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        synthesizers=synthesizers,
        sdv_datasets=datasets,
    )

    # Assert
    pd.testing.assert_frame_equal(result, expected_result)
    assert 'MultiTableUniformSynthesizer' in synthesizers
    mock_validate_output_destination.assert_called_once_with(
        output_destination,
        aws_keys={
            'aws_access_key_id': aws_access_key_id,
            'aws_secret_access_key': aws_secret_access_key,
        },
    )
    mock_import_and_validate_synthesizers.assert_called_once_with(
        synthesizers=synthesizers,
        custom_synthesizers=None,
        modality='multi_table',
    )
    mock_generate_job_args_list.assert_called_once_with(
        limit_dataset_size=False,
        sdv_datasets=datasets,
        additional_datasets_folder=None,
        sdmetrics=None,
        timeout=None,
        output_destination=output_destination,
        compute_quality_score=True,
        compute_diagnostic_score=True,
        compute_privacy_score=None,
        synthesizers=synthesizers,
        detailed_results_folder=None,
        s3_client='s3_client_mock',
        modality='multi_table',
    )
    mock_get_empty_dataframe.assert_called_once_with(
        compute_quality_score=True,
        compute_diagnostic_score=True,
        compute_privacy_score=None,
        sdmetrics=None,
    )
