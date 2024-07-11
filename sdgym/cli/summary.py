"""Functions to summarize the sdgym.run output."""

import io
import re

import numpy as np
import pandas as pd

from sdgym.cli.utils import read_csv
from sdgym.s3 import write_file

KNOWN_ERRORS = (
    ('Synthesizer Timeout', 'timeout'),
    ('MemoryError', 'memory_error'),
)

MODALITY_BASELINES = {
    'single-table': ['Uniform', 'Column', 'CLBN', 'PrivBN'],
    'multi-table': ['Uniform', 'Independent'],
    'timeseries': [],
}

LIBRARIES = {
    'SDV': ['ctgan', 'copulagan', 'gaussiancopula', 'tvae', 'hma1', 'par'],
    'YData': ['dragan', 'vanillagan', 'wgan'],
}


def preprocess(data):
    """Preprocess data."""
    if isinstance(data, str):
        data = pd.read_csv(data)

    grouped = data.groupby(['Synthesizer', 'Dataset'])
    bydataset = grouped.mean()
    data = bydataset.reset_index()

    if 'error' in data.columns:
        errors = data.error.fillna('')
        for message, column in KNOWN_ERRORS:
            data[column] = errors.str.contains(message)
            data.loc[data[column], 'error'] = np.nan

    return data


def _coverage(data):
    total = len(data.Dataset.unique())
    scores = data.groupby('Synthesizer').apply(lambda x: x.Quality_Score.notna().sum())
    coverage_perc = scores / total
    coverage_str = scores.astype(str) + f' / {total}'
    return coverage_perc, coverage_str


def _mean_score(data):
    return data.groupby('Synthesizer').Quality_Score.mean()


def _best(data, rank, field, ascending):
    ranks = data.groupby('Dataset').rank(method='dense', ascending=ascending)[field] == rank
    return ranks.groupby(data.Synthesizer).sum()


def _seconds(data):
    return data.groupby('Synthesizer').Train_Time.mean().round()


def _synthesizer_beat_baseline(synthesizer_data, baseline_scores):
    synthesizer_scores = synthesizer_data.set_index('Dataset').Quality_Score
    synthesizer_scores = synthesizer_scores.reindex(baseline_scores.index)
    return (synthesizer_scores.fillna(-np.inf) >= baseline_scores.fillna(-np.inf)).sum()


def _beat_baseline(data, baseline_scores):
    return data.groupby('Synthesizer').apply(
        _synthesizer_beat_baseline, baseline_scores=baseline_scores
    )


def summarize(data, baselines=(), datasets=None):
    """Obtain an overview of the performance of each synthesizer.

    Optionally compare the synthesizers with the indicated baselines or analyze
    only some of the datasets.

    Args:
        data (pandas.DataFrame):
            Table in the ``sdgym.run`` output format.
        baselines (list-like):
            Names of the synthesizers to use as baselines to compare to.
        datasets (list-like):
            Names of the datasets to summarize.

    Returns:
        pandas.DataFrame
    """
    if datasets is not None:
        data = data[data.Dataset.isin(datasets)]

    baselines_data = data[data.Synthesizer.isin(baselines)]
    data = data[~data.Synthesizer.isin(baselines)]
    no_identity = data[data.Synthesizer != 'DataIdentity']

    coverage_perc, coverage_str = _coverage(data)
    solved = data.groupby('Synthesizer').apply(lambda x: x.Quality_Score.notna().sum())

    results = {
        'total': len(data.Dataset.unique()),
        'solved': solved,
        'coverage': coverage_str,
        'coverage_perc': coverage_perc,
        'time': _seconds(data),
        'best': _best(no_identity, 1, 'Quality_Score', False),
        'avg score': _mean_score(data),
        'best_time': _best(no_identity, 1, 'Train_Time', True),
        'second_best_time': _best(no_identity, 2, 'Train_Time', True),
        'third_best_time': _best(no_identity, 3, 'Train_Time', True),
    }

    for baseline in baselines:
        baseline_data = baselines_data[baselines_data.Synthesizer == baseline]
        baseline_scores = baseline_data.set_index('Dataset').Quality_Score
        results[f'beat_{baseline.lower()}'] = _beat_baseline(data, baseline_scores)

    if 'error' in data.columns:
        grouped = data.groupby('Synthesizer')
        for _, error_column in KNOWN_ERRORS:
            results[error_column] = grouped[error_column].sum()

        results['errors'] = grouped.error.apply(lambda x: x.notna().sum())
        total_errors = results['errors']
        results['metric_errors'] = results['total'] - results['solved'] - total_errors

    return pd.DataFrame(results)


def _error_counts(data):
    if 'error' in data.columns:
        return data.error.value_counts()
    return 0


def errors_summary(data):
    """Obtain a summary of the most frequent errors.

    The output is a table that contains the error strings as index,
    the synthesizer names as columns and the number of times each
    synthesizer had that error as values.

    An additional column called ``all`` is also included with the
    overall count of errors across all the synthesizers. The table
    is sorted descendingly based on this column.

    Args:
        data (pandas.DataFrame):
            Table in the ``sdgym.run`` output format.

    Returns:
        pandas.DataFrame
    """
    if 'error' in data.columns:
        all_errors = pd.DataFrame(_error_counts(data)).rename(columns={'error': 'all'})
        synthesizer_errors = data.groupby('Synthesizer').apply(_error_counts).pivot_table(level=0)
        for synthesizer, errors in synthesizer_errors.items():
            all_errors[synthesizer] = errors.fillna(0).astype(int)

        return all_errors

    return pd.DataFrame()


def add_sheet(dfs, name, writer, cell_fmt, index_fmt, header_fmt):
    """Add sheet."""
    startrow = 0
    widths = [0]
    if not isinstance(dfs, dict):
        dfs = {None: dfs}

    for df_name, df in dfs.items():
        df = df.fillna('N/E').sort_index().reset_index()
        startrow += bool(df_name)
        df.to_excel(writer, sheet_name=name, startrow=startrow + 1, index=False, header=False)

        worksheet = writer.sheets[name]

        if df_name:
            worksheet.write(startrow - 1, 0, df_name, index_fmt)
            widths[0] = max(widths[0], len(df_name))

        for idx, column in enumerate(df.columns):
            worksheet.write(startrow, idx, column, header_fmt)
            if df.empty:
                width = len(column) + 1
            else:
                width = max(len(column), *df[column].astype(str).str.len()) + 1

            if len(widths) > idx:
                widths[idx] = max(widths[idx], width)
            else:
                widths.append(width)

        startrow += len(df) + 2

    for idx, width in enumerate(widths):
        fmt = cell_fmt if idx else index_fmt
        worksheet.set_column(idx, idx, width + 1, fmt)


def _find_library(synthesizer):
    for library, library_synthesizers in LIBRARIES.items():
        for library_synthesizer in library_synthesizers:
            if library_synthesizer in synthesizer.lower():
                return library

    return None


def _add_summary_libraries(summary_data):
    summary_data['library'] = summary_data.index.map(_find_library)
    summary_data['library'] = summary_data['library'].fillna('Other')
    return summary_data


def _add_summary(data, modality, baselines, writer):
    total_summary = summarize(data, baselines=baselines)

    summary = total_summary[
        [
            'coverage_perc',
            'best_time',
            'second_best_time',
            'third_best_time',
        ]
    ].rename(
        {
            'coverage_perc': 'coverage %',
            'best_time': '# of Wins',
            'second_best_time': '# of 2nd best',
            'third_best_time': '# of 3rd best',
        },
        axis=1,
    )
    summary = summary.drop(index='Identity', errors='ignore')
    summary = _add_summary_libraries(summary)

    beat_baseline_headers = ['beat_' + b.lower() for b in baselines]
    quality = total_summary[['total', 'solved', 'best'] + beat_baseline_headers]
    performance = total_summary[['time']]
    error_details = errors_summary(data)
    error_summary = total_summary[
        [
            'total',
            'solved',
            'coverage',
            'coverage_perc',
            'timeout',
            'memory_error',
            'errors',
            'metric_errors',
        ]
    ]
    summary.index.name = ''
    quality.index.name = ''
    performance.index.name = ''
    error_details.index.name = ''
    error_summary.index.name = ''

    cell_fmt = writer.book.add_format({'font_name': 'Roboto', 'font_size': '11', 'align': 'right'})
    index_fmt = writer.book.add_format({
        'font_name': 'Roboto',
        'font_size': '11',
        'bold': True,
        'align': 'center',
    })
    header_fmt = writer.book.add_format({
        'font_name': 'Roboto',
        'font_size': '11',
        'bold': True,
        'align': 'right',
    })

    add_sheet(summary, f'Summary ({modality})', writer, cell_fmt, index_fmt, header_fmt)
    add_sheet(quality, f'Quality ({modality})', writer, cell_fmt, index_fmt, header_fmt)
    add_sheet(performance, f'Performance ({modality})', writer, cell_fmt, index_fmt, header_fmt)
    add_sheet(
        error_summary, f'Errors Summary ({modality})', writer, cell_fmt, index_fmt, header_fmt
    )
    add_sheet(error_details, f'Errors Detail ({modality})', writer, cell_fmt, index_fmt, header_fmt)


def make_summary_spreadsheet(
    results_csv_path, output_path=None, baselines=None, aws_key=None, aws_secret=None
):
    """Create a spreadsheet document organizing information from results.

    This function creates a ``.xlsx`` file containing information from
    the results of running ``sdgym.benchmark_single_table``. The file contains five sheets
    for each modality: summary, quality, performance, error summary and error details.

    Args:
        results_csv_path (str):
            Path to the csv file containing the results.
        output_path (str):
            Path constaining where to store the output spreadsheet.
            Defaults to {results_csv_path}.xlsx.
        baselines (dict):
            Optional dict mapping modalities to a list of baseline
            model names. If not provided, a default dict is used.
    """
    results = read_csv(results_csv_path, aws_key, aws_secret)
    data = preprocess(results)
    baselines = baselines or MODALITY_BASELINES
    output_path = output_path or re.sub('.csv$', '.xlsx', results_csv_path)
    output = io.BytesIO()
    writer = pd.ExcelWriter(output)

    for modality, df in data.groupby('modality'):
        modality_baselines = baselines[modality]
        _add_summary(df, modality, modality_baselines, writer)

    writer.save()
    write_file(output.getvalue(), output_path, aws_key, aws_secret)
