import time

import pandas as pd
from sdv.single_table import TVAESynthesizer

from sdgym.benchmark import benchmark_single_table
from sdgym.sdgym_result_explorer import SDGymResultsExplorer


def test_end_to_end_local(tmp_path):
    """Test the SDGymResultsExplorer end-to-end with local paths."""
    # Setup
    output_destination = str(tmp_path / 'benchmark_output')
    benchmark_single_table(
        output_destination=output_destination,
        synthesizers=['GaussianCopulaSynthesizer', 'TVAESynthesizer'],
        sdv_datasets=['expedia_hotel_logs', 'fake_companies'],
    )
    today = time.strftime('%m_%d_%Y')

    # Run
    result_explorer = SDGymResultsExplorer(output_destination)
    runs = result_explorer.list()
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
    new_synthetic_data = synthesizer.sample(num_rows=10)

    # Assert
    expected_run = f'SDGym_results_{today}'
    assert runs == [expected_run]
    assert isinstance(synthetic_data, pd.DataFrame)
    assert isinstance(synthesizer, TVAESynthesizer)
    assert set(new_synthetic_data.columns) == set(synthetic_data_fake_companies.columns)
    assert new_synthetic_data.shape[0] == 10


def test_summarize():
    """Test the `summarize` method."""
    # Setup
    output_destination = 'tests/integration/sdgym_result_explorer/_benchmark_results'
    result_explorer = SDGymResultsExplorer(output_destination)

    # Run
    summary = result_explorer.summarize('SDGym_results_10_11_2024')

    # Assert
    expected_results = pd.DataFrame({
        '10_11_2024 - # datasets: 9 - sdgym version: 0.9.1': [6, 4, 5],
        '05_10_2024 - # datasets: 9 - sdgym version: 0.8.0': [4, 4, 5],
        '04_05_2024 - # datasets: 9 - sdgym version: 0.7.0': [5, 3, 5],
        'Synthesizer': ['CTGANSynthesizer', 'CopulaGANSynthesizer', 'TVAESynthesizer'],
    })
    expected_results = expected_results.set_index('Synthesizer')
    pd.testing.assert_frame_equal(summary, expected_results)
