from pathlib import Path

import pandas as pd
import pytest

from sdgym import DatasetExplorer

SUMMARY_OUTPUT_COLUMNS = [
    'Dataset',
    'Datasize_Size_MB',
    'Num_Tables',
    'Total_Num_Columns',
    'Total_Num_Columns_Categorical',
    'Total_Num_Columns_Numerical',
    'Total_Num_Columns_Datetime',
    'Total_Num_Columns_PII',
    'Total_Num_Columns_ID_NonKey',
    'Max_Num_Columns_Per_Table',
    'Total_Num_Rows',
    'Max_Num_Rows_Per_Table',
    'Num_Relationships',
    'Max_Schema_Depth',
    'Max_Schema_Branch',
]


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
