import time

import pandas as pd
from sdv.single_table import TVAESynthesizer

from sdgym import ResultsExplorer
from sdgym.benchmark import benchmark_single_table


def test_end_to_end_local(tmp_path):
    """Test the ResultsExplorer end-to-end with local paths."""
    # Setup
    output_destination = str(tmp_path / 'benchmark_output')
    benchmark_single_table(
        output_destination=output_destination,
        synthesizers=['GaussianCopulaSynthesizer', 'TVAESynthesizer'],
        sdv_datasets=['expedia_hotel_logs', 'fake_companies'],
    )
    today = time.strftime('%m_%d_%Y')

    # Run
    result_explorer = ResultsExplorer(output_destination)
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
    expected_results = pd.read_csv(f'{output_destination}/SDGym_results_{today}/results.csv')
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
    output_destination = 'tests/integration/result_explorer/_benchmark_results'
    result_explorer = ResultsExplorer(output_destination)

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
        pd.read_csv(
            'tests/integration/result_explorer/_benchmark_results/'
            'SDGym_results_10_11_2024/results.csv',
        )
        .sort_values(by=['Dataset', 'Synthesizer'])
        .reset_index(drop=True)
    )
    expected_results['Win'] = expected_results['Win'].astype('int64')
    pd.testing.assert_frame_equal(summary, expected_summary)
    pd.testing.assert_frame_equal(results, expected_results)
