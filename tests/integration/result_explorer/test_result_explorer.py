import shutil
import time

import pandas as pd
import yaml
from sdv.single_table import TVAESynthesizer

from sdgym import ResultsExplorer
from sdgym.benchmark import benchmark_single_table


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
    results = result_explorer.load_results(runs[0])
    metainfo = result_explorer.load_metainfo(runs[0])
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


def test_summarize():
    """Test the `summarize` method."""
    # Setup
    output_destination = 'tests/integration/result_explorer/_benchmark_results/'
    result_explorer = ResultsExplorer(output_destination, modality='single_table')

    # Run
    summary, results = result_explorer.summarize('SDGym_results_10_11_2024')

    # Assert
    expected_summary = pd.DataFrame({
        'Synthesizer': ['CTGANSynthesizer', 'CopulaGANSynthesizer', 'TVAESynthesizer'],
        '10_11_2024 - # datasets: 9 - sdgym version: 0.9.1': [6, 4, 5],
        '05_10_2024 - # datasets: 9 - sdgym version: 0.8.0': [4, 4, 5],
        '04_05_2024 - # datasets: 9 - sdgym version: 0.7.0': [5, 3, 5],
    })
    expected_results = (
        pd
        .read_csv(
            'tests/integration/result_explorer/_benchmark_results/single_table/'
            'SDGym_results_10_11_2024/results.csv',
        )
        .sort_values(by=['Dataset', 'Synthesizer'])
        .reset_index(drop=True)
    )
    expected_results['Win'] = expected_results['Win'].astype('int64')
    pd.testing.assert_frame_equal(summary, expected_summary)
    pd.testing.assert_frame_equal(results, expected_results)


def test_summarize_multi_table():
    """Test summarize works under the multi_table subfolder."""
    # Setup
    output_destination = 'tests/integration/result_explorer/_benchmark_results/'
    result_explorer = ResultsExplorer(output_destination, modality='multi_table')

    # Run
    summary, results = result_explorer.summarize('SDGym_results_12_02_2025')

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
        .load_results(runs[0])
        .sort_values(by=['Dataset', 'Synthesizer'])
        .reset_index(drop=True)
    )
    metainfo = explorer.load_metainfo(runs[0])

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
    metainfo_path = f'{output_destination}single_table/SDGym_results_12_17_2024/metainfo.yaml'
    with open(metainfo_path, 'r') as f:
        raw_yaml = yaml.safe_load(f)

    run_id = raw_yaml.get('run_id')
    expected_metainfo = {run_id: {k: v for k, v in raw_yaml.items() if k != 'run_id'}}

    # Run
    results = result_explorer.load_results()
    metainfo = result_explorer.load_metainfo()

    # Assert
    assert metainfo == expected_metainfo
    expected_results = pd.read_csv(
        'tests/integration/result_explorer/_benchmark_results/single_table/'
        'SDGym_results_12_17_2024/results.csv',
    )
    pd.testing.assert_frame_equal(results, expected_results)
