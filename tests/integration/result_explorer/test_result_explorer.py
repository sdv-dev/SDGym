import shutil
import time
from pathlib import Path

import pandas as pd
import pytest
import yaml
from sdv.single_table import TVAESynthesizer

from sdgym import ResultsExplorer
from sdgym.benchmark import benchmark_single_table
from sdgym.result_explorer.result_handler import SUMMARY_COLUMNS


def test_end_to_end_local(tmp_path):
    """Test the ResultsExplorer end-to-end with local paths."""
    # Setup
    output_destination = tmp_path / 'benchmark_output'
    result_explorer_path = output_destination / 'single_table'
    benchmark_single_table(
        output_destination=str(output_destination),
        synthesizers=['GaussianCopulaSynthesizer', 'TVAESynthesizer'],
        sdv_datasets=['expedia_hotel_logs', 'fake_companies'],
    )
    today = time.strftime('%m_%d_%Y')

    # Run
    result_explorer = ResultsExplorer(str(result_explorer_path), modality='single_table')
    runs = result_explorer.list()
    results = result_explorer.load_results(results_folder_name=runs[0])
    metainfo = result_explorer.load_metainfo(results_folder_name=runs[0])
    synthetic_data = result_explorer.load_synthetic_data(
        results_folder_name=runs[0],
        dataset_name='expedia_hotel_logs',
        synthesizer_name='GaussianCopulaSynthesizer',
    )
    synthetic_data_fake_companies = result_explorer.load_synthetic_data(
        results_folder_name=runs[0],
        dataset_name='fake_companies',
        synthesizer_name='GaussianCopulaSynthesizer',
    )
    synthesizer = result_explorer.load_synthesizer(
        results_folder_name=runs[0],
        dataset_name='fake_companies',
        synthesizer_name='TVAESynthesizer',
    )
    assert isinstance(synthesizer, TVAESynthesizer)
    new_synthetic_data = synthesizer.sample(num_rows=10)

    # Assert
    expected_results = pd.read_csv(f'{result_explorer_path}/SDGym_results_{today}/results.csv')
    pd.testing.assert_frame_equal(results, expected_results)
    assert metainfo[f'run_{today}_0']['jobs'] == [
        ['expedia_hotel_logs', 'GaussianCopulaSynthesizer'],
        ['expedia_hotel_logs', 'TVAESynthesizer'],
        ['expedia_hotel_logs', 'UniformSynthesizer'],
        ['fake_companies', 'GaussianCopulaSynthesizer'],
        ['fake_companies', 'TVAESynthesizer'],
        ['fake_companies', 'UniformSynthesizer'],
    ]
    expected_run = f'SDGym_results_{today}'
    assert runs == [expected_run]
    assert isinstance(synthetic_data, pd.DataFrame)
    assert isinstance(synthesizer, TVAESynthesizer)
    assert set(new_synthetic_data.columns) == set(synthetic_data_fake_companies.columns)
    assert new_synthetic_data.shape[0] == 10


@pytest.mark.parametrize(
    'dataset_names, synthesizer_names, summary, expected_columns',
    [
        (
            ['fake_hotels'],
            ['HMASynthesizer'],
            True,
            SUMMARY_COLUMNS,
        ),
        (['fake_hotels'], None, False, None),
        (None, ['HMASynthesizer'], False, None),
        (None, None, False, None),
        (None, None, True, SUMMARY_COLUMNS),
    ],
)
def test_load_results_with_filters(dataset_names, synthesizer_names, summary, expected_columns):
    """Test loading results with dataset and synthesizer and summary filters."""
    # Setup
    output_destination = 'tests/integration/result_explorer/_benchmark_results/'
    result_explorer = ResultsExplorer(output_destination, modality='multi_table')
    expected_results = pd.read_csv(
        'tests/integration/result_explorer/_benchmark_results/multi_table/'
        'SDGym_results_12_02_2025/results.csv',
    )
    expected_columns = set(expected_columns) if expected_columns else set(expected_results.columns)
    all_dataset_names = expected_results['Dataset'].unique().tolist()
    all_synthesizer_names = expected_results['Synthesizer'].unique().tolist()

    # Run
    results = result_explorer.load_results(
        results_folder_name='SDGym_results_12_02_2025',
        dataset_names=dataset_names,
        synthesizer_names=synthesizer_names,
        summary=summary,
    )

    # Assert
    expected_datasets = set(dataset_names) if dataset_names is not None else set(all_dataset_names)
    expected_synthesizers = (
        set(synthesizer_names) if synthesizer_names is not None else set(all_synthesizer_names)
    )
    assert set(results['Dataset']) == expected_datasets
    assert set(results['Synthesizer']) == expected_synthesizers
    assert set(results.columns) == expected_columns
    dataset_mask = (
        expected_results['Dataset'].isin(dataset_names)
        if dataset_names is not None
        else pd.Series(True, index=expected_results.index)
    )
    synth_mask = (
        expected_results['Synthesizer'].isin(synthesizer_names)
        if synthesizer_names is not None
        else pd.Series(True, index=expected_results.index)
    )
    filtered_expected = expected_results[dataset_mask & synth_mask][results.columns]
    pd.testing.assert_frame_equal(
        results.sort_values(['Dataset', 'Synthesizer']).reset_index(drop=True),
        filtered_expected.sort_values(['Dataset', 'Synthesizer']).reset_index(drop=True),
    )


def test_summarize():
    """Test the `summarize` method."""
    # Setup
    output_destination = 'tests/integration/result_explorer/_benchmark_results/'
    result_explorer = ResultsExplorer(output_destination, modality='single_table')

    # Run
    summary, results = result_explorer.summarize(results_folder_name='SDGym_results_03_01_2026')

    # Assert
    expected_summary = pd.DataFrame({
        'Synthesizer': [
            'BootstrapSynthesizer',
            'CTGANSynthesizer',
            'ColumnSynthesizer',
            'CopulaGANSynthesizer',
            'GaussianCopulaSynthesizer',
            'RealTabFormerSynthesizer',
            'SegmentSynthesizer',
            'TVAESynthesizer',
            'UniformSynthesizer',
            'XGCSynthesizer',
        ],
        '03_01_2026 - # datasets: 9 - sdgym version: 0.13.1': [1, 0, 2, 1, 6, 5, 5, 7, 1, 4],
        '02_01_2026 - # datasets: 9 - sdgym version: 0.13.0': [
            '-',
            2.0,
            3.0,
            1.0,
            5.0,
            5.0,
            '-',
            4.0,
            1.0,
            '-',
        ],
        '01_01_2026 - # datasets: 9 - sdgym version: 0.12.1': [
            '-',
            2.0,
            3.0,
            1.0,
            5.0,
            2.0,
            '-',
            4.0,
            1.0,
            '-',
        ],
    })

    expected_results = (
        pd
        .read_csv(
            'tests/integration/result_explorer/_benchmark_results/'
            'expected_result_integration_test_single_table.csv',
        )
        .sort_values(by=['Dataset', 'Synthesizer'])
        .reset_index(drop=True)
    )
    columns_to_compare = [
        'Synthesizer',
        'Dataset',
        'Adjusted_Total_Time',
        'Adjusted_Quality_Score',
        'Win',
    ]
    results_to_compare = (
        results[columns_to_compare]
        .sort_values(by=['Dataset', 'Synthesizer'])
        .reset_index(drop=True)
    )
    expected_results_to_compare = (
        expected_results[columns_to_compare]
        .sort_values(by=['Dataset', 'Synthesizer'])
        .reset_index(drop=True)
    )
    expected_results_to_compare['Win'] = expected_results_to_compare['Win'].astype(
        results_to_compare['Win'].dtype
    )
    pd.testing.assert_frame_equal(summary, expected_summary)
    pd.testing.assert_frame_equal(
        results_to_compare,
        expected_results_to_compare,
        check_dtype=False,
    )


def test_summarize_multi_table():
    """Test summarize works under the multi_table subfolder."""
    # Setup
    output_destination = 'tests/integration/result_explorer/_benchmark_results/'
    result_explorer = ResultsExplorer(output_destination, modality='multi_table')

    # Run
    summary, results = result_explorer.summarize(results_folder_name='SDGym_results_12_02_2025')

    # Assert
    expected_summary = pd.DataFrame({
        'Synthesizer': ['HMASynthesizer', 'MultiTableUniformSynthesizer'],
        '12_02_2025 - # datasets: 1 - sdgym version: 0.11.2.dev0': [0, 0],
    })
    expected_results = (
        pd
        .read_csv(
            'tests/integration/result_explorer/_benchmark_results/multi_table/'
            'SDGym_results_12_02_2025/results.csv',
        )
        .sort_values(by=['Dataset', 'Synthesizer'])
        .reset_index(drop=True)
    )
    expected_results['Win'] = [0, 0]
    pd.testing.assert_frame_equal(summary, expected_summary)
    pd.testing.assert_frame_equal(results, expected_results)


def test_list_and_load_results_multi_table(tmp_path):
    """Test listing and loading results under multi_table subfolder."""
    # Setup
    run_folder = 'SDGym_results_12_02_2025'
    src_root = 'tests/integration/result_explorer/_benchmark_results/multi_table/' + run_folder
    dst_root = tmp_path / 'benchmark_output' / 'multi_table' / run_folder
    shutil.copytree(src_root, dst_root)

    explorer = ResultsExplorer(str(tmp_path / 'benchmark_output'), modality='multi_table')

    # Run
    runs = explorer.list()
    assert runs == [run_folder]
    loaded_results = (
        explorer
        .load_results(results_folder_name=runs[0])
        .sort_values(by=['Dataset', 'Synthesizer'])
        .reset_index(drop=True)
    )
    metainfo = explorer.load_metainfo(results_folder_name=runs[0])

    # Assert
    expected_results = (
        pd
        .read_csv(dst_root / 'results.csv')
        .sort_values(by=['Dataset', 'Synthesizer'])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(loaded_results, expected_results)
    assert isinstance(metainfo, dict) and len(metainfo) >= 1


def test_loading_last_run_results_by_default():
    """Test that the last run results are loaded when no folder name is provided."""
    # Setup
    output_destination = 'tests/integration/result_explorer/_benchmark_results/'
    result_explorer = ResultsExplorer(output_destination, modality='single_table')
    results_dir = Path(output_destination) / 'single_table' / 'SDGym_results_03_01_2026'

    metainfo_paths = sorted(results_dir.glob('metainfo*.yaml'))
    expected_metainfo = {}
    for metainfo_path in metainfo_paths:
        with open(metainfo_path, 'r') as f:
            raw_yaml = yaml.safe_load(f)

        run_id = raw_yaml['run_id']
        expected_metainfo[run_id] = {
            key: value for key, value in raw_yaml.items() if key != 'run_id'
        }

    result_paths = sorted(results_dir.glob('results*.csv'))
    expected_results = (
        pd
        .concat(
            [pd.read_csv(result_path) for result_path in result_paths],
            ignore_index=True,
        )
        .sort_values(['Dataset', 'Synthesizer'])
        .reset_index(drop=True)
    )

    # Run
    results = result_explorer.load_results()
    metainfo = result_explorer.load_metainfo()

    # Assert
    results = results.sort_values(['Dataset', 'Synthesizer']).reset_index(drop=True)
    assert metainfo == expected_metainfo
    pd.testing.assert_frame_equal(results, expected_results)
