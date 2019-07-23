import json
import logging
import os
import urllib

import numpy as np
import pandas as pd

from sdgym.constants import CATEGORICAL, ORDINAL
from sdgym.evaluate import evaluate

LOGGER = logging.getLogger(__name__)


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
BASE_URL = 'http://sdgym.s3.amazonaws.com/datasets/'
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


def _load_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def _load_file(filename, loader):
    local_path = os.path.join(DATA_PATH, filename)
    if not os.path.exists(local_path):
        os.makedirs(DATA_PATH, exist_ok=True)
        urllib.request.urlretrieve(BASE_URL + filename, local_path)

    return loader(local_path)


def _load_dataset(name):
    data = _load_file(name + '.npz', np.load)
    meta = _load_file(name + '.json', _load_json)

    return data['train'], data['test'], meta


def _get_columns(metadata):
    categorical_columns = list()
    ordinal_columns = list()
    for column_idx, column in enumerate(metadata['columns']):
        if column['type'] == CATEGORICAL:
            categorical_columns.append(column_idx)
        elif column['type'] == ORDINAL:
            ordinal_columns.append(column_idx)

    return categorical_columns, ordinal_columns


def benchmark(synthesizer, datasets=DEFAULT_DATASETS, repeat=3):
    results = list()
    for name in datasets:
        LOGGER.info('Evaluating dataset %s', name)
        train, test, metadata = _load_dataset(name)

        categorical_columns, ordinal_columns = _get_columns(metadata)

        for iteration in range(repeat):
            synthesized = synthesizer(train, categorical_columns, ordinal_columns)
            scores = evaluate(train, test, synthesized, metadata)
            scores['dataset'] = name
            scores['iter'] = iteration
            results.append(scores)

    return pd.concat(results)
