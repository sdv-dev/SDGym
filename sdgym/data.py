import json
import logging
import os
import urllib

import numpy as np

from sdgym.constants import CATEGORICAL, ORDINAL

LOGGER = logging.getLogger(__name__)

BASE_URL = 'http://sdgym.s3.amazonaws.com/datasets/'
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


def _load_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def _load_file(filename, loader):
    local_path = os.path.join(DATA_PATH, filename)
    if not os.path.exists(local_path):
        os.makedirs(DATA_PATH, exist_ok=True)
        url = BASE_URL + filename

        LOGGER.info('Downloading file %s to %s', url, local_path)
        urllib.request.urlretrieve(url, local_path)

    return loader(local_path)


def _get_columns(metadata):
    categorical_columns = list()
    ordinal_columns = list()
    for column_idx, column in enumerate(metadata['columns']):
        if column['type'] == CATEGORICAL:
            categorical_columns.append(column_idx)
        elif column['type'] == ORDINAL:
            ordinal_columns.append(column_idx)

    return categorical_columns, ordinal_columns


def load_dataset(name, benchmark=False):
    LOGGER.info('Loading dataset %s', name)
    data = _load_file(name + '.npz', np.load)
    meta = _load_file(name + '.json', _load_json)

    categorical_columns, ordinal_columns = _get_columns(meta)

    train = data['train']
    if benchmark:
        return train, data['test'], meta, categorical_columns, ordinal_columns

    return train, categorical_columns, ordinal_columns
