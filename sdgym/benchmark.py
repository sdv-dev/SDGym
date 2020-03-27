import logging
import os

import pandas as pd

from sdgym.data import load_dataset
from sdgym.evaluate import evaluate_synthesizer

LOGGER = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(__file__)
SCORES_PATH = os.path.join(BASE_DIR, 'scores.csv')

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



def compute_benchmark(synthesizer, datasets=DEFAULT_DATASETS, repeat=3):
    """Compute the scores of a synthesizer over a list of datasts.

    The results are returned in a raw format as a ``pandas.DataFrame`` containing:
        - One row for each dataset+scoring method (for example, a classifier)
        - One column for each computed metric
        - The columns:
            - dataset
            - distance
            - name (of the scoring method)
            - iteration

    For example, evaluating a synthesizer on the ``adult`` dataset with 2 iterations produces
    the following table::

        dataset                    name  accuracy        f1  distance iteration
          adult  DecisionTreeClassifier    0.7901  0.653344       0.0         0
          adult      AdaBoostClassifier    0.8575  0.673988       0.0         0
          adult      LogisticRegression    0.7949  0.661160       0.0         0
          adult           MLPClassifier    0.8440  0.678351       0.0         0
          adult  DecisionTreeClassifier    0.8067  0.664584       0.0         1
          adult      AdaBoostClassifier    0.8617  0.680083       0.0         1
          adult      LogisticRegression    0.7923  0.659229       0.0         1
          adult           MLPClassifier    0.8466  0.642757       0.0         1

    """
    results = list()
    for name in datasets:
        LOGGER.info('Evaluating dataset %s', name)
        train, test, meta, categoricals, ordinals = load_dataset(name, benchmark=True)

        for iteration in range(repeat):
            scores = evaluate_synthesizer(synthesizer, train, test, meta, categoricals, ordinals)
            scores['dataset'] = name
            scores['iteration'] = iteration
            results.append(scores)

    return pd.concat(results, sort=False)


def summarize_scores(scores):
    """Computes a summary of the scores obtained by a synthesizer.

    The raw scores returned by the ``compute_benchmark`` function are summarized as follows:
        - Table is grouped by dataset and averaged
        - TBD

    For example, the summary of a synthesizer that has been evaluated on the ``adult`` and
    the ``asia`` datasets produces the following output::

        adult/accuracy          0.8765
        adult/f1_micro          0.7654
        adult/f1_macro          0.7654
        asia/syn_likelihood    -2.5364
        asia/test_likelihood   -2.4321
        dtype: float64
    """
    pass


def benchmark(synthesizer, dataset=DEFAULT_DATASETS, repeat=3, scores_path=SCORES_PATH):
    synthesizer_scores = compute_benchmark(synthesizer, datasets, repeat)
    summary_row = summarize_scores(synthesizer_scores)

    synthesizer_name = synthesizer.__name__
    summary_row.name = synthesizer_name

    scores = pd.read_csv(scores_path)
    scores.drop(labels=[synthesizer_name], errors='ignore', inplace=True)

    return scores.append(summary_row)
