import json
from datetime import datetime, timezone
from unittest.mock import Mock, call, patch

import pytest
from botocore.exceptions import ClientError

from sdgym._benchmark.benchmark import (
    _benchmark_multi_table_compute_gcp,
    _benchmark_single_table_compute_gcp,
)
from sdgym.run_benchmark.run_benchmark import (
    MULTI_TABLE_DATASETS,
    SINGLE_TABLE_DATASETS,
    _get_benchmark_setup,
    append_benchmark_run,
    main,
)
from sdgym.run_benchmark.utils import (
    OUTPUT_DESTINATION_AWS,
)


@patch('sdgym.run_benchmark.run_benchmark.get_s3_client')
@patch('sdgym.run_benchmark.run_benchmark.parse_s3_path')
@patch('sdgym.run_benchmark.run_benchmark.get_result_folder_name')
def test_append_benchmark_run(mock_get_result_folder_name, mock_parse_s3_path, mock_get_s3_client):
    """Test the `append_benchmark_run` method."""
    # Setup
    aws_access_key_id = 'my_access_key'
    aws_secret_access_key = 'my_secret_key'
    date = '2023-10-01'
    mock_get_result_folder_name.return_value = 'SDGym_results_10_01_2023'
    mock_parse_s3_path.return_value = ('my-bucket', 'my-prefix/')
    mock_s3_client = Mock()
    benchmark_date = {
        'runs': [
            {'date': '2023-09-30', 'folder_name': 'SDGym_results_09_30_2023'},
        ]
    }
    mock_get_s3_client.return_value = mock_s3_client
    mock_s3_client.get_object.return_value = {
        'Body': Mock(read=lambda: json.dumps(benchmark_date).encode('utf-8'))
    }
    expected_data = {
        'runs': [
            {'date': '2023-09-30', 'folder_name': 'SDGym_results_09_30_2023'},
            {'date': date, 'folder_name': 'SDGym_results_10_01_2023'},
        ]
    }

    # Run
    append_benchmark_run(aws_access_key_id, aws_secret_access_key, date)

    # Assert
    mock_get_s3_client.assert_called_once_with(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    mock_parse_s3_path.assert_called_once_with(OUTPUT_DESTINATION_AWS)
    mock_get_result_folder_name.assert_called_once_with(date)
    mock_s3_client.get_object.assert_called_once_with(
        Bucket='my-bucket', Key='my-prefix/single_table/_BENCHMARK_DATES.json'
    )
    mock_s3_client.put_object.assert_called_once_with(
        Bucket='my-bucket',
        Key='my-prefix/single_table/_BENCHMARK_DATES.json',
        Body=json.dumps(expected_data).encode('utf-8'),
    )


@patch('sdgym.run_benchmark.run_benchmark.get_s3_client')
@patch('sdgym.run_benchmark.run_benchmark.parse_s3_path')
@patch('sdgym.run_benchmark.run_benchmark.get_result_folder_name')
def test_append_benchmark_run_new_file(
    mock_get_result_folder_name, mock_parse_s3_path, mock_get_s3_client
):
    """Test the `append_benchmark_run` with a new file."""
    # Setup
    aws_access_key_id = 'my_access_key'
    aws_secret_access_key = 'my_secret_key'
    date = '2023-10-01'
    mock_get_result_folder_name.return_value = 'SDGym_results_10_01_2023'
    mock_parse_s3_path.return_value = ('my-bucket', 'my-prefix/')
    mock_s3_client = Mock()
    mock_get_s3_client.return_value = mock_s3_client
    mock_s3_client.get_object.side_effect = ClientError(
        {'Error': {'Code': 'NoSuchKey'}}, 'GetObject'
    )
    expected_data = {
        'runs': [
            {'date': date, 'folder_name': 'SDGym_results_10_01_2023'},
        ]
    }

    # Run
    append_benchmark_run(aws_access_key_id, aws_secret_access_key, date)

    # Assert
    mock_get_s3_client.assert_called_once_with(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    mock_parse_s3_path.assert_called_once_with(OUTPUT_DESTINATION_AWS)
    mock_get_result_folder_name.assert_called_once_with(date)
    mock_s3_client.get_object.assert_called_once_with(
        Bucket='my-bucket', Key='my-prefix/single_table/_BENCHMARK_DATES.json'
    )
    mock_s3_client.put_object.assert_called_once_with(
        Bucket='my-bucket',
        Key='my-prefix/single_table/_BENCHMARK_DATES.json',
        Body=json.dumps(expected_data).encode('utf-8'),
    )


def test__get_benchmark_setup_single_table():
    """Test the `_get_benchmark_setup` method for single_table modality."""
    # Setup
    gan_not_excluded = [
        dataset for dataset in SINGLE_TABLE_DATASETS if dataset not in ['covtype', 'intrusion']
    ]
    expected_job_split = [
        (['ColumnSynthesizer', 'GaussianCopulaSynthesizer'], SINGLE_TABLE_DATASETS),
        (['TVAESynthesizer'], SINGLE_TABLE_DATASETS),
        (['SegmentSynthesizer'], SINGLE_TABLE_DATASETS),
        (['XGCSynthesizer'], SINGLE_TABLE_DATASETS),
        (['BootstrapSynthesizer'], SINGLE_TABLE_DATASETS),
        (['CTGANSynthesizer'], gan_not_excluded),
        (['CopulaGANSynthesizer'], gan_not_excluded),
        (['RealTabFormerSynthesizer'], ['adult', 'alarm', 'child', 'insurance', 'news']),
        (['RealTabFormerSynthesizer'], ['covtype']),
        (['RealTabFormerSynthesizer'], ['intrusion']),
        (['RealTabFormerSynthesizer'], ['expedia_hotel_logs']),
        (['RealTabFormerSynthesizer'], ['census']),
        (['CTGANSynthesizer'], ['covtype']),
        (['CopulaGANSynthesizer'], ['covtype']),
        (['CTGANSynthesizer'], ['intrusion']),
        (['CopulaGANSynthesizer'], ['intrusion']),
    ]

    # Run
    setup = _get_benchmark_setup('single_table')

    # Assert
    assert setup['method'] == _benchmark_single_table_compute_gcp
    assert len(setup['job_split']) == len(expected_job_split) == 16
    for expected, actual in zip(expected_job_split, setup['job_split']):
        assert expected[0] == actual[0]
        assert set(expected[1]) == set(actual[1])


def test__get_benchmark_setup_multi_table():
    """Test the `_get_benchmark_setup` method for multi_table modality."""
    # Setup
    hma_not_excluded = [
        dataset
        for dataset in MULTI_TABLE_DATASETS
        if dataset
        not in [
            'Accidents',
            'AustralianFootball',
            'Countries',
            'MuskSmall',
            'NBA',
            'OMOP_CDM_dayz',
            'PremierLeague',
            'SalesDB',
            'airbnb-simplified',
            'imdb_ijs',
            'legalActs',
            'SAP',
            'imdb_MovieLens',
        ]
    ]
    expected_job_split = [
        (['HSASynthesizer', 'IndependentSynthesizer'], MULTI_TABLE_DATASETS),
        (['HMASynthesizer'], hma_not_excluded),
        (['HMASynthesizer'], ['Accidents']),
        (['HMASynthesizer'], ['AustralianFootball']),
        (['HMASynthesizer'], ['Countries']),
        (['HMASynthesizer'], ['MuskSmall']),
        (['HMASynthesizer'], ['NBA']),
        (['HMASynthesizer'], ['OMOP_CDM_dayz']),
        (['HMASynthesizer'], ['PremierLeague']),
        (['HMASynthesizer'], ['SalesDB']),
        (['HMASynthesizer'], ['airbnb-simplified']),
        (['HMASynthesizer'], ['imdb_ijs']),
        (['HMASynthesizer'], ['legalActs']),
        (['HMASynthesizer'], ['SAP']),
        (['HMASynthesizer'], ['imdb_MovieLens']),
    ]

    # Run
    setup = _get_benchmark_setup('multi_table')

    # Assert
    assert setup['method'] == _benchmark_multi_table_compute_gcp
    assert len(setup['job_split']) == len(expected_job_split) == 15
    for expected, actual in zip(expected_job_split, setup['job_split']):
        assert expected[0] == actual[0]
        assert set(expected[1]) == set(actual[1])


@pytest.mark.parametrize(
    'modality, benchmark_setup',
    [
        (
            'single_table',
            {
                'method': Mock(name='mock_single_method'),
                'job_split': [
                    (['GaussianCopulaSynthesizer', 'ColumnSynthesizer'], ['dataset1', 'dataset2']),
                    (['TVAESynthesizer'], ['dataset3', 'dataset2']),
                ],
            },
        ),
        (
            'multi_table',
            {
                'method': Mock(name='mock_multi_method'),
                'job_split': [
                    (['HSASynthesizer', 'IndependentSynthesizer'], ['datasetA', 'datasetB']),
                    (['HMASynthesizer'], ['datasetC', 'datasetD']),
                ],
            },
        ),
    ],
)
@patch('sdgym.run_benchmark.run_benchmark.post_benchmark_launch_message')
@patch('sdgym.run_benchmark.run_benchmark.append_benchmark_run')
@patch('sdgym.run_benchmark.run_benchmark.os.getenv')
@patch('sdgym.run_benchmark.run_benchmark._parse_args')
@patch('sdgym.run_benchmark.run_benchmark._get_benchmark_setup')
def test_main(
    mock_get_benchmark_setup,
    mock_parse_args,
    mock_getenv,
    mock_append_benchmark_run,
    mock_post_benchmark_launch_message,
    modality,
    benchmark_setup,
):
    """Test the `main` function with both single_table and multi_table modalities."""
    # Setup
    mock_get_benchmark_setup.return_value = benchmark_setup
    mock_parse_args.return_value = Mock(modality=modality)
    mock_getenv.side_effect = lambda key: {
        'AWS_ACCESS_KEY_ID': 'my_access_key',
        'AWS_SECRET_ACCESS_KEY': 'my_secret_key',
        'CREDENTIALS_FILEPATH': '/path/to/creds.json',
    }.get(key)
    date = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    # Run
    main()

    # Assert
    expected_calls = [
        call(
            output_destination=OUTPUT_DESTINATION_AWS,
            credential_filepath='/path/to/creds.json',
            synthesizers=group[0],
            sdv_datasets=group[1],
            timeout=345600,
        )
        for group in benchmark_setup['job_split']
    ]
    benchmark_setup['method'].assert_has_calls(expected_calls)
    mock_get_benchmark_setup.assert_called_once_with(modality)
    mock_append_benchmark_run.assert_called_once_with(
        'my_access_key',
        'my_secret_key',
        date,
        modality=modality,
    )
    mock_post_benchmark_launch_message.assert_called_once_with(
        date,
        compute_service='GCP',
        modality=modality,
    )
