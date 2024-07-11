import io
from unittest.mock import patch

import pandas as pd

from sdgym.cli.summary import make_summary_spreadsheet


@patch('sdgym.cli.summary.write_file')
@patch('sdgym.cli.summary.read_csv')
@patch('sdgym.cli.summary.preprocess')
@patch('sdgym.cli.summary.errors_summary')
@patch('sdgym.cli.summary.summarize')
@patch('sdgym.cli.summary.pd.ExcelWriter')
@patch('sdgym.cli.summary.add_sheet')
def test_make_summary_spreadsheet(
    add_sheet_mock,
    excel_writer_mock,
    summarize_mock,
    errors_summary_mock,
    preprocess_mock,
    read_csv_mock,
    write_file_mock,
):
    """Test the ``sdgym.cli.summary.make_summary_spreadsheet`` function.

    The ``make_summary_spreadsheet`` function is expected to extract the correct
    columns from the input file and add them to the correct sheets. It should
    then use the ``aws_key`` and ``aws_secret`` provided to call ``sdgym.s3.write_file``
    and save the output document.

    Input:
    - file path to results csv.
    - output file path
    - asw key
    - aws secret

    Side effect:
    - Saves excel sheets. The ``ExcelWriter`` should also write to a ``BytesIO``
    object that is saved using ``sdgym.s3.write_file``.
    """
    # Setup
    data = pd.DataFrame(
        {
            'total': [2, 2],
            'solved': [2, 1],
            'coverage': ['2 / 2', '1 / 2'],
            'coverage_perc': [1.0, 0.5],
            'time': [100, 200],
            'best': [2, 0],
            'best_time': [1, 0],
            'second_best_time': [0, 1],
            'third_best_time': [0, 0],
            'beat_uniform': [2, 1],
            'beat_column': [2, 1],
            'beat_clbn': [2, 1],
            'beat_privbn': [2, 1],
            'timeout': [0, 1],
            'memory_error': [0, 0],
            'errors': [0, 1],
            'metric_errors': [0, 0],
            'avg score': [0.9, 0.45],
        },
        index=['synth1', 'synth2'],
    )
    preprocessed_data = pd.DataFrame({'modality': ['single-table']})
    errors = pd.DataFrame({'synth1': [0], 'synth2': [1], 'error': ['RuntimeError: error.']})
    preprocess_mock.return_value = preprocessed_data
    summarize_mock.return_value = data
    errors_summary_mock.return_value = errors

    # Run
    make_summary_spreadsheet('file_path.csv', 'output_path.xlsx', None, None, None)

    # Assert
    expected_summary = pd.DataFrame(
        {
            'coverage %': [1.0, 0.5],
            '# of Wins': [1, 0],
            '# of 2nd best': [0, 1],
            '# of 3rd best': [0, 0],
            'library': ['Other', 'Other'],
        },
        index=['synth1', 'synth2'],
    )
    expected_summary.index.name = ''
    expected_quality = pd.DataFrame(
        {
            'total': [2, 2],
            'solved': [2, 1],
            'best': [2, 0],
            'beat_uniform': [2, 1],
            'beat_column': [2, 1],
            'beat_clbn': [2, 1],
            'beat_privbn': [2, 1],
        },
        index=['synth1', 'synth2'],
    )
    expected_quality.index.name = ''
    expected_performance = pd.DataFrame({'time': [100, 200]}, index=['synth1', 'synth2'])
    expected_performance.index.name = ''
    expected_errors = pd.DataFrame(
        {
            'total': [2, 2],
            'solved': [2, 1],
            'coverage': ['2 / 2', '1 / 2'],
            'coverage_perc': [1.0, 0.5],
            'timeout': [0, 1],
            'memory_error': [0, 0],
            'errors': [0, 1],
            'metric_errors': [0, 0],
        },
        index=['synth1', 'synth2'],
    )
    expected_errors.index.name = ''
    add_sheet_calls = add_sheet_mock.mock_calls
    read_csv_mock.assert_called_once_with('file_path.csv', None, None)
    assert isinstance(excel_writer_mock.mock_calls[0][1][0], io.BytesIO)
    excel_writer_mock.return_value.save.assert_called_once()
    assert len(add_sheet_calls) == 5
    pd.testing.assert_frame_equal(add_sheet_calls[0][1][0], expected_summary)
    pd.testing.assert_frame_equal(add_sheet_calls[1][1][0], expected_quality)
    pd.testing.assert_frame_equal(add_sheet_calls[2][1][0], expected_performance)
    pd.testing.assert_frame_equal(add_sheet_calls[3][1][0], expected_errors)
    pd.testing.assert_frame_equal(add_sheet_calls[4][1][0], errors)
    write_file_mock.assert_called_once_with(b'', 'output_path.xlsx', None, None)


@patch('sdgym.cli.summary.write_file')
@patch('sdgym.cli.summary._add_summary')
@patch('sdgym.cli.summary.read_csv')
@patch('sdgym.cli.summary.preprocess')
@patch('sdgym.cli.summary.errors_summary')
@patch('sdgym.cli.summary.summarize')
@patch('sdgym.cli.summary.pd.ExcelWriter')
@patch('sdgym.cli.summary.add_sheet')
def test_make_summary_spreadsheet_no_output_path(
    add_sheet_mock,
    excel_writer_mock,
    summarize_mock,
    errors_summary_mock,
    preprocess_mock,
    read_csv_mock,
    add_summary_mock,
    write_file_mock,
):
    """Test the ``sdgym.cli.summary.make_summary_spreadsheet`` function.

    The ``make_summary_spreadsheet`` function is expected to use the
    input file path to create the output file path if none is provided.

    Input:
    - file path to results csv
    - No output file path
    - aws key
    - aws secret

    Side effect:
    - The ``sdgym.s3.write_file`` method should be called with the correct
    output file path name.
    """
    # Setup

    preprocess_mock.return_value = pd.DataFrame({'modality': ['single-table']})
    summarize_mock.return_value = pd.DataFrame()
    errors_summary_mock.return_value = pd.DataFrame()
    add_sheet_mock.return_value = None

    # Run
    make_summary_spreadsheet('file_path.csv', None, None, None, None)

    # Assert
    read_csv_mock.assert_called_once_with('file_path.csv', None, None)
    assert isinstance(excel_writer_mock.mock_calls[0][1][0], io.BytesIO)
    excel_writer_mock.return_value.save.assert_called_once()
    add_summary_mock.assert_called_once()
    write_file_mock.assert_called_once_with(b'', 'file_path.xlsx', None, None)
