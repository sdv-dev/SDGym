"""Functions to summarize the sdgym.run output."""

import numpy as np
import pandas as pd

KNOWN_ERRORS = (
    ('Synthesizer Timeout', 'timeout'),
    ('MemoryError', 'memory_error'),
)


def preprocess(data):
    if isinstance(data, str):
        data = pd.read_csv(data)

    del data['run_id']
    del data['iteration']

    grouped = data.groupby(['synthesizer', 'dataset', 'modality'])
    bydataset = grouped.mean()
    model_errors = grouped.error.first()[bydataset.metric_time.isnull()].fillna('')
    bydataset['error'] = model_errors
    data = bydataset.reset_index()

    errors = data.error.fillna('')
    for message, column in KNOWN_ERRORS:
        data[column] = errors.str.contains(message)
        data.loc[data[column], 'error'] = np.nan

    return data


def _coverage(data):
    total = len(data.dataset.unique())
    scores = data.groupby('synthesizer').apply(lambda x: x.score.notnull().sum())
    coverage_perc = scores / total
    coverage_str = (scores.astype(str) + f' / {total}')
    return coverage_perc, coverage_str


def _mean_score(data):
    return data.groupby('synthesizer').score.mean()


def _best(data):
    ranks = data.groupby('dataset').rank(method='min', ascending=False)['score'] == 1
    return ranks.groupby(data.synthesizer).sum()


def _wins(data):
    for synthesizer in data.synthesizer.unique():
        solved = data[(data.synthesizer == synthesizer) & data.score.notnull()]
        solved_data = hma1[hma1.dataset.isin(solved.dataset.unique())].copy()
        solved_data['rank'] = solved_data.groupby('dataset').score.rank(method='min', ascending=False)
        wins = solved_data[solved_data.synthesizer == synthesizer]['rank'] == 1
        wins.sum(), wins.mean()


def _seconds(data):
    return data.groupby('synthesizer').model_time.mean().round()


def _synthesizer_beat_baseline(synthesizer_data, baseline_scores):
    synthesizer_scores = synthesizer_data.set_index('dataset').score
    synthesizer_scores = synthesizer_scores.reindex(baseline_scores.index)
    return (synthesizer_scores.fillna(-np.inf) >= baseline_scores.fillna(-np.inf)).sum()


def _beat_baseline(data, baseline_scores):
    return data.groupby('synthesizer').apply(
        _synthesizer_beat_baseline, baseline_scores=baseline_scores)


def summarize(data, baselines=(), datasets=None):
    """Obtain an overview of the performance of each synthesizer.

    Optionally compare the synthesizers with the indicated baselines or analyze
    only some o the datasets.

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
        data = data[data.dataset.isin(datasets)]

    baselines_data = data[data.synthesizer.isin(baselines)]
    data = data[~data.synthesizer.isin(baselines)]
    no_identity = data[data.synthesizer != 'Identity']

    coverage_perc, coverage_str = _coverage(data)
    solved = data.groupby('synthesizer').apply(lambda x: x.score.notnull().sum())

    results = {
        'total': len(data.dataset.unique()),
        'solved': solved,
        'coverage': coverage_str,
        'coverage_perc': coverage_perc,
        'time': _seconds(data),
        'best': _best(no_identity),
    }
    for baseline in baselines:
        baseline_data = baselines_data[baselines_data.synthesizer == baseline]
        baseline_scores = baseline_data.set_index('dataset').score
        results[f'beat_{baseline.lower()}'] = _beat_baseline(data, baseline_scores)

    grouped = data.groupby('synthesizer')
    for _, error_column in KNOWN_ERRORS:
        results[error_column] = grouped[error_column].sum()

    results['errors'] = grouped.error.apply(lambda x: x.notnull().sum())
    total_errors = results['errors'] + results['memory_error'] + results['timeout']
    results['metric_errors'] = results['total'] - results['solved'] - total_errors

    return pd.DataFrame(results)


def _error_counts(data):
    return data.error.value_counts()


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
    all_errors = pd.DataFrame(_error_counts(data)).rename(columns={'error': 'all'})
    synthesizer_errors = data.groupby('synthesizer').apply(_error_counts).unstack(level=0)
    for synthesizer, errors in synthesizer_errors.items():
        all_errors[synthesizer] = errors.fillna(0).astype(int)

    return all_errors
