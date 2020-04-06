import logging
import os
import types
from datetime import datetime

import pandas as pd

from sdgym.data import load_dataset
from sdgym.evaluate import compute_scores
from sdgym.synthesizers import BaseSynthesizer

LOGGER = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(__file__)
LEADERBOARD_PATH = os.path.join(BASE_DIR, 'leaderboard.csv')

DEFAULT_DATASETS = [
    "adult",
    "alarm",
    "asia",
    "census",
    "child",
    "covtype",
    "credit",
    "grid",
    "gridr",
    "insurance",
    "intrusion",
    "mnist12",
    "mnist28",
    "news",
    "ring"
]


def compute_benchmark(synthesizer, datasets=DEFAULT_DATASETS, iterations=3):
    """Compute the scores of a synthesizer over a list of datasets.

    The results are returned in a raw format as a ``pandas.DataFrame`` containing:
        - One row for each dataset+scoring method (for example, a classifier)
        - One column for each computed metric
        - The columns:
            - dataset
            - distance
            - name (of the scoring method)
            - iteration

    For example, evaluating a synthesizer on the ``adult`` and ``asia`` datasets with 2
    iterations produces a table similar to this::

        dataset             name  iter  distance  accuracy    f1  syn_likelihood  test_likelihood
          adult  DecisionTree...     0       0.0      0.79  0.65             NaN              NaN
          adult      AdaBoost...     0       0.0      0.85  0.67             NaN              NaN
          adult      Logistic...     0       0.0      0.79  0.66             NaN              NaN
          adult           MLP...     0       0.0      0.84  0.67             NaN              NaN
          adult  DecisionTree...     1       0.0      0.80  0.66             NaN              NaN
          adult      AdaBoost...     1       0.0      0.86  0.68             NaN              NaN
          adult      Logistic...     1       0.0      0.79  0.65             NaN              NaN
          adult           MLP...     1       0.0      0.84  0.64             NaN              NaN
           asia     Bayesian ...     0       0.0       NaN   NaN           -2.23            -2.24
           asia     Bayesian ...     1       0.0       NaN   NaN           -2.23            -2.24
    """
    results = list()
    for dataset_name in datasets:
        LOGGER.info('Evaluating dataset %s', dataset_name)
        train, test, meta, categoricals, ordinals = load_dataset(dataset_name, benchmark=True)

        for iteration in range(iterations):
            try:
                synthesized = synthesizer(train, categoricals, ordinals)
                scores = compute_scores(train, test, synthesized, meta)
                scores['dataset'] = dataset_name
                scores['iteration'] = iteration
                results.append(scores)
            except Exception:
                LOGGER.exception('Error computing scores for %s on dataset %s - iteration %s',
                                 _get_synthesizer_name(synthesizer), dataset_name, iteration)

    return pd.concat(results, sort=False)


def _dataset_summary(grouped_df):
    dataset = grouped_df.name
    scores = grouped_df.mean().dropna()
    scores.index = dataset + '/' + scores.index

    return scores


def _summarize_scores(scores):
    """Computes a summary of the scores obtained by a synthesizer.

    The raw scores returned by the ``compute_benchmark`` function are summarized
    by grouping them by dataset and computing the average.

    The results are then put in a ``pandas.Series`` object with one value per
    dataset and metric.

    As an example, the summary of a synthesizer that has been evaluated on the
    ``adult`` and the ``asia`` datasets produces the following output::

        adult/accuracy          0.8765
        adult/f1_micro          0.7654
        adult/f1_macro          0.7654
        asia/syn_likelihood    -2.5364
        asia/test_likelihood   -2.4321
        dtype: float64

    Args:
        scores (pandas.DataFrame):
            Raw Scores dataframe as returned by the ``compute_benchmark`` function.

    Returns:
        pandas.Series:
            Summarized scores series in the format described above.
    """
    scores = scores.drop(['distance', 'iteration', 'name'], axis=1, errors='ignore')

    grouped = scores.groupby('dataset').apply(_dataset_summary)
    if isinstance(grouped, pd.Series):
        # If more than one dataset, grouped result is a series
        # with a multilevel index.
        return grouped.droplevel(0)

    # Otherwise, if there is only one dataset, it is DataFrame
    return grouped.iloc[0]


def _get_synthesizer_name(synthesizer):
    """Get the name of the synthesizer function or class.

    If the given synthesizer is a function, return its name.
    If it is a method, return the name of the class to which
    the method belongs.

    Args:
        synthesizer (function or method):
            The synthesizer function or method.

    Returns:
        str:
            Name of the function or the class to which the method belongs.
    """
    if isinstance(synthesizer, types.MethodType):
        synthesizer_name = synthesizer.__self__.__class__.__name__
    else:
        synthesizer_name = synthesizer.__name__

    return synthesizer_name


def _get_synthesizers(synthesizers):
    """Get the dict of synthesizers from the input value.

    If the input is a synthesizer or an iterable of synthesizers, get their names
    and put them on a dict.

    Args:
        synthesizers (function, class, list, tuple or dict):
            A synthesizer (function or method or class) or an iterable of synthesizers
            or a dict containing synthesizer names as keys and synthesizers as values.

    Returns:
        dict[str, function]:
            dict containing synthesizer names as keys and function as values.

    Raises:
        TypeError:
            if neither a synthesizer or an iterable or a dict is passed.
    """
    if callable(synthesizers):
        synthesizers = {_get_synthesizer_name(synthesizers): synthesizers}
    if isinstance(synthesizers, (list, tuple)):
        synthesizers = {
            _get_synthesizer_name(synthesizer): synthesizer
            for synthesizer in synthesizers
        }
    elif not isinstance(synthesizers, dict):
        raise TypeError('`synthesizers` can only be a function, a class, a list or a dict')

    for name, synthesizer in synthesizers.items():
        # If the synthesizer is one of the SDGym Synthesizer classes,
        # create and instance and replace it with its fit_sample method.
        if isinstance(synthesizer, type) and issubclass(synthesizer, BaseSynthesizer):
            synthesizers[name] = synthesizer().fit_sample

    return synthesizers


def benchmark(synthesizers, datasets=DEFAULT_DATASETS, iterations=3, add_leaderboard=True,
              leaderboard_path=LEADERBOARD_PATH, replace_existing=True):
    """Compute the benchmark scores for the synthesizers and return a leaderboard.

    The ``synthesizers`` object can either be a single synthesizer or, an iterable of
    synthesizers or a dict containing synthesizer names as keys and synthesizers as values.

    If ``add_leaderboard`` is ``True``, append the obtained scores to the leaderboard
    stored in the ``lederboard_path``. By default, the leaderboard used is the one which
    is included in the package, which contains the scores obtained by the SDGym Synthesizers.

    If ``replace_existing`` is ``True`` and any of the given synthesizers already existed
    in the leaderboard, the old rows are dropped.

    Args:
        synthesizers (function, class, list, tuple or dict):
            The synthesizer or synthesizers to evaluate. It can be a single synthesizer
            (function or method or class), or an iterable of synthesizers, or a dict
            containing synthesizer names as keys and synthesizers as values. If the input
            is not a dict, synthesizer names will be extracted from the given object.
        datasets (list[str]):
            Names of the datasets to use for the benchmark. Defaults to all the ones available.
        iterations (int):
            Number of iterations to perform over each dataset and synthesizer. Defaults to 3.
        add_leaderboard (bool):
            Whether to append the obtained scores to the previous leaderboard or not. Defaults
            to ``True``.
        leaderboard_path (str):
            Path to where the leaderboard is stored. Defaults to the leaderboard included
            with the package, which contains the scores obtained by the SDGym synthesizers.
        replace_existing (bool):
            Whether to replace old scores or keep them in the returned leaderboard. Defaults
            to ``True``.

    Returns:
        pandas.DataFrame:
            Table containing one row per synthesizer and one column for each dataset and metric.
    """
    synthesizers = _get_synthesizers(synthesizers)

    scores = list()
    for synthesizer_name, synthesizer in synthesizers.items():
        synthesizer_scores = compute_benchmark(synthesizer, datasets, iterations)
        summary_row = _summarize_scores(synthesizer_scores)
        summary_row.name = synthesizer_name
        scores.append(summary_row)

    leaderboard = pd.DataFrame(scores)
    leaderboard['timestamp'] = datetime.utcnow()

    if add_leaderboard:
        old_leaderboard = pd.read_csv(
            leaderboard_path,
            index_col=0,
            parse_dates=['timestamp']
        )[leaderboard.columns]
        if replace_existing:
            old_leaderboard.drop(labels=[leaderboard.index], errors='ignore', inplace=True)

        leaderboard = old_leaderboard.append(leaderboard, sort=False)

    return leaderboard
