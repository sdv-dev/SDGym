from unittest.mock import patch

import pandas as pd

from sdgym.summary import make_summary_spreadsheet

@patch('sdgym.summary.preprocess')
@patch('sdgym.summary.errors_summary')
@patch('sdgym.summary.summarize')
@patch('sdgym.summary.pd.ExcelWriter')
@patch('sdgym.summary.add_sheet')
def test_make_summary_spreadsheet(add_sheet_mock, excel_writer_mock, summarize_mock,
                                  errors_summary_mock, preprocess_mock):
    """Test the ``sdgym.summary.make_summary_spreadsheet`` function.

    The ``make_summary_spreadsheet`` function is expected to extract the correct
    columns from the input file and add them to the correct sheets. It should
    then save the file as an .xslx.

    Input:
    - file path to results csv

    Side effect:
    - Saves excel sheets
    """
    # Setup
    data = pd.DataFrame({
        'total': [2, 2],
        'solved': [2, 1],
        'coverage': ['2 / 2', '1 / 2'],
        'coverage_perc': [1.0, 0.5],
        'time': [100, 200],
        'best': [2, 0],
        'beat_uniform': [2, 1],
        'beat_independent': [2, 1],
        'beat_clbn': [2, 1],
        'beat_privbn': [2, 1],
        'timeout': [0, 1],
        'memory_error': [0, 0],
        'errors': [0, 1],
        'metric_errors': [0, 0],
        'avg score': [0.9, 0.45]
    })
    preprocessed_data = pd.DataFrame({'modality': ['single-table']})
    errors = pd.DataFrame({
        'synth1': [0],
        'synth2': [1],
        'error': ['RuntimeError: error.']
    })
    preprocess_mock.return_value = preprocessed_data
    summarize_mock.return_value = data
    errors_summary_mock.return_value = errors
    add_sheet_mock.return_value = None

    # Run
    make_summary_spreadsheet('file_path')

    # Assert
    expected_summary = pd.DataFrame({
        'coverage %': [1.0, 0.5],
        'avg time': [100, 200],
        'avg score': [0.9, 0.45]
    })
    expected_summary.index.name = ''
    expected_quality = pd.DataFrame({
        'total': [2, 2],
        'solved': [2, 1],
        'best': [2, 0],
        'beat_uniform': [2, 1],
        'beat_independent': [2, 1],
        'beat_clbn': [2, 1],
        'beat_privbn': [2, 1]
    })
    expected_quality.index.name = ''
    expected_performance = pd.DataFrame({'time': [100, 200]})
    expected_performance.index.name = ''
    expected_errors = pd.DataFrame({
        'total': [2, 2],
        'solved': [2, 1],
        'coverage': ['2 / 2', '1 / 2'],
        'coverage_perc': [1.0, 0.5],
        'timeout': [0, 1],
        'memory_error': [0, 0],
        'errors': [0, 1],
        'metric_errors': [0, 0]
    })
    expected_errors.index.name = ''
    add_sheet_calls = add_sheet_mock.mock_calls
    assert len(add_sheet_calls) == 5
    pd.testing.assert_frame_equal(add_sheet_calls[0][1][0], expected_summary)
    pd.testing.assert_frame_equal(add_sheet_calls[1][1][0], expected_quality)
    pd.testing.assert_frame_equal(add_sheet_calls[2][1][0], expected_performance)
    pd.testing.assert_frame_equal(add_sheet_calls[3][1][0], expected_errors)
    pd.testing.assert_frame_equal(add_sheet_calls[4][1][0], errors)
