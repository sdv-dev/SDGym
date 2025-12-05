from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from sdgym import DatasetExplorer
from sdgym.dataset_explorer import SUMMARY_OUTPUT_COLUMNS


@pytest.mark.parametrize('modality', ['single_table', 'multi_table'])
def test_end_to_end_dataset_explorer(modality, tmp_path):
    """Integration test for DatasetExplorer end-to-end workflow."""
    # Setup
    de = DatasetExplorer()
    output_filepath = Path(tmp_path) / 'datasets_summary.csv'

    # Run
    dataset_summary = de.summarize_datasets(modality=modality, output_filepath=output_filepath)

    # Assert
    assert isinstance(dataset_summary, pd.DataFrame)
    assert not dataset_summary.empty
    assert output_filepath.exists()
    assert len(dataset_summary) > 1
    assert list(dataset_summary.columns) == SUMMARY_OUTPUT_COLUMNS
    loaded_summary = pd.read_csv(output_filepath)
    pd.testing.assert_frame_equal(loaded_summary, dataset_summary)


@pytest.mark.parametrize('modality', ['single_table', 'multi_table'])
def test_dataset_explorer_empty_bucket_warns_and_returns_header_only(modality, tmp_path):
    """When no datasets are present, warn and return header-only table and write CSV."""
    # Setup
    de = DatasetExplorer(s3_url='s3://my_bucket/')
    output_filepath = Path(tmp_path) / f'datasets_summary_empty_{modality}.csv'

    with patch('sdgym.dataset_explorer._get_available_datasets') as mock_get:
        mock_get.return_value = pd.DataFrame([])

        expected_message = (
            f"The provided S3 URL 's3://my_bucket/' does not contain any datasets "
            f"of modality '{modality}'."
        )

        # Run
        with pytest.warns(UserWarning, match=expected_message):
            frame = de.summarize_datasets(modality=modality, output_filepath=str(output_filepath))

    # Assert
    assert isinstance(frame, pd.DataFrame)
    assert frame.empty
    assert output_filepath.exists()
    assert list(frame.columns) == SUMMARY_OUTPUT_COLUMNS
    loaded_summary = pd.read_csv(output_filepath)
    pd.testing.assert_frame_equal(loaded_summary, frame)
