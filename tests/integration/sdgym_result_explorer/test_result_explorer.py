import os
import time

import pandas as pd
from sdv.single_table import TVAESynthesizer

from sdgym.benchmark import benchmark_single_table
from sdgym.sdgym_result_explorer import SDGymResultsExplorer


def test_end_to_end_local_and_s3(tmp_path):
    """Test the SDGymResultsExplorer end-to-end with both local and S3 paths."""
    # Setup
    aws_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    today = time.strftime('%m_%d_%Y')
    if aws_key_id is None or aws_secret_key is None:
        run_on_s3 = False
        output_destination = str(tmp_path / 'benchmark_output')
        benchmark_single_table(
            output_destination=output_destination,
            synthesizers=['GaussianCopulaSynthesizer', 'TVAESynthesizer'],
            sdv_datasets=['expedia_hotel_logs', 'fake_companies'],
        )
    else:
        run_on_s3 = True
        output_destination = 's3://sdgym-benchmark/Debug/Issue_414_test_5'

    # Run
    result_explorer = SDGymResultsExplorer(
        output_destination, aws_access_key_id=aws_key_id, aws_secret_access_key=aws_secret_key
    )
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
    expected_run = f'SDGym_results_{today}' if not run_on_s3 else 'SDGym_results_07_07_2025'
    assert runs == [expected_run]
    assert isinstance(synthetic_data, pd.DataFrame)
    assert isinstance(synthesizer, TVAESynthesizer)
    assert set(new_synthetic_data.columns) == set(synthetic_data_fake_companies.columns)
    assert new_synthetic_data.shape[0] == 10
