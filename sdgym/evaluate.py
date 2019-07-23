import glob
import logging

import numpy as np
import pandas as pd
from pomegranate import BayesianNetwork
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from sdgym.constants import CATEGORICAL, CONTINUOUS, ORDINAL

LOGGER = logging.getLogger(__name__)


_MODELS = {
    'binary_classification': [
        (DecisionTreeClassifier(max_depth=15, class_weight='balanced'),
            "Decision Tree (max_depth=20)"),
        (AdaBoostClassifier(), "Adaboost (estimator=50)"),
        (LogisticRegression(
            solver='lbfgs', n_jobs=2, class_weight='balanced', max_iter=50),
            "Logistic Regression"),
        (MLPClassifier((50, ), max_iter=50), "MLP (50)")
    ],
    'multiclass_classification': [
        (DecisionTreeClassifier(max_depth=30, class_weight='balanced'),
            "Decision Tree (max_depth=30)"),
        (MLPClassifier((100, ), max_iter=50), "MLP (100)"),
    ],
    'regression': [
        (LinearRegression(), "Linear Regression"),
        (MLPRegressor((100, ), max_iter=50), "MLP (100)")
    ]
}


class FeatureMaker:

    def __init__(self, metadata, label_column='label', label_type='int', sample=50000):
        self.columns = metadata['columns']
        self.label_column = label_column
        self.label_type = label_type
        self.sample = sample
        self.encoders = dict()

    def make_features(self, data):
        data = data.copy()
        np.random.shuffle(data)
        data = data[:self.sample]

        features = []
        labels = []

        for index, cinfo in enumerate(self.columns):
            col = data[:, index]
            if cinfo['name'] == self.label_column:
                if self.label_type == 'int':
                    labels = col.astype(int)
                elif self.label_type == 'float':
                    labels = col.astype(float)
                else:
                    assert 0, 'unkown label type'
                continue

            if cinfo['type'] == CONTINUOUS:
                cmin = cinfo['min']
                cmax = cinfo['max']
                if cmin >= 0 and cmax >= 1e3:
                    feature = np.log(np.maximum(col, 1e-2))

                else:
                    feature = (col - cmin) / (cmax - cmin) * 5

            elif cinfo['type'] == ORDINAL:
                feature = col

            else:
                if cinfo['size'] <= 2:
                    feature = col

                else:
                    encoder = self.encoders.get(index)
                    col = col.reshape(-1, 1)
                    if encoder:
                        feature = encoder.transform(col)
                    else:
                        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                        self.encoders[index] = encoder
                        feature = encoder.fit_transform(col)

            features.append(feature)

        features = np.column_stack(features)

        return features, labels


def _prepare_ml_problem(train, test, metadata):
    fm = FeatureMaker(metadata)
    x_train, y_train = fm.make_features(train)
    x_test, y_test = fm.make_features(test)

    return x_train, y_train, x_test, y_test, _MODELS[metadata['problem_type']]


def _evaluate_multi_classification(train, test, metadata):
    """Score classifiers using f1 score and the given train and test data.

    Args:
        x_train(numpy.ndarray):
        y_train(numpy.ndarray):
        x_test(numpy.ndarray):
        y_test(numpy):
        classifiers(list):

    Returns:
        list[dict]:


    """
    x_train, y_train, x_test, y_test, classifiers = _prepare_ml_problem(train, test, metadata)

    performance = []
    for clf, name in classifiers:
        LOGGER.info('Evaluating using multiclass classifier %s', name)
        unique_labels = np.unique(y_train)
        if len(unique_labels) == 1:
            pred = [unique_labels[0]] * len(x_test)
        else:
            clf.fit(x_train, y_train)
            pred = clf.predict(x_test)

        acc = accuracy_score(y_test, pred)
        macro_f1 = f1_score(y_test, pred, average='macro')
        micro_f1 = f1_score(y_test, pred, average='micro')

        performance.append(
            {
                "name": name,
                "accuracy": acc,
                "macro_f1": macro_f1,
                "micro_f1": micro_f1
            }
        )

    return performance


def _evaluate_binary_classification(train, test, metadata):
    x_train, y_train, x_test, y_test, classifiers = _prepare_ml_problem(train, test, metadata)

    performance = []
    for clf, name in classifiers:
        LOGGER.info('Evaluating using binary classifier %s', name)
        unique_labels = np.unique(y_train)
        if len(unique_labels) == 1:
            pred = [unique_labels[0]] * len(x_test)
        else:
            clf.fit(x_train, y_train)
            pred = clf.predict(x_test)

        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average='binary')

        performance.append(
            {
                "name": name,
                "accuracy": acc,
                "f1": f1
            }
        )

    return performance


def _evaluate_regression(train, test, metadata):
    x_train, y_train, x_test, y_test, regressors = _prepare_ml_problem(train, test, metadata)

    LOGGER.info('Evaluating using regression.')
    performance = []
    y_train = np.log(np.clip(y_train, 1, 20000))
    y_test = np.log(np.clip(y_test, 1, 20000))
    for clf, name in regressors:
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)

        r2 = r2_score(y_test, pred)

        performance.append(
            {
                "name": name,
                "r2": r2,
            }
        )

    return performance


BAYESIAN_PARAMETER = {
    'grid': 30,
    'gridr': 30,
    'ring': 10,
}


def _evaluate_gmm_likelihood(train, test, metadata):
    n = BAYESIAN_PARAMETER.get(metadata['problem_type'])

    gmm = GaussianMixture(n, covariance_type='diag')
    gmm.fit(test)
    l1 = gmm.score(train)

    gmm.fit(train)
    l2 = gmm.score(test)

    return [{
        "name": "default",
        "syn_likelihood": l1,
        "test_likelihood": l2,
    }]


def _mapper(data, metadata):
    data_t = []
    for row in data:
        row_t = []
        for id_, info in enumerate(metadata['columns']):
            row_t.append(info['i2s'][int(row[id_])])

        data_t.append(row_t)

    return data_t


def _evaluate_bayesian_likelihood(dataset, train, test, metadata):
    LOGGER.info('Evaluating using bayesian likehood.')
    struct = glob.glob("data/*/{}_structure.json".format(dataset))
    assert len(struct) == 1
    bn1 = BayesianNetwork.from_json(struct[0])

    train_mapped = _mapper(train, metadata)
    test_mapped = _mapper(test, metadata)
    prob = []
    for item in train_mapped:
        try:
            prob.append(bn1.probability(item))
        except Exception:
            prob.append(1e-8)

    l1 = np.mean(np.log(np.asarray(prob) + 1e-8))

    bn2 = BayesianNetwork.from_structure(train_mapped, bn1.structure)
    prob = []

    for item in test_mapped:
        try:
            prob.append(bn2.probability(item))
        except Exception:
            prob.append(1e-8)

    l2 = np.mean(np.log(np.asarray(prob) + 1e-8))

    return [{
        "name": "default",
        "syn_likelihood": l1,
        "test_likelihood": l2,
    }]


def _compute_distance(train, syn, metadata, sample=300):
    mask_d = np.zeros(len(metadata['columns']))

    for id_, info in enumerate(metadata['columns']):
        if info['type'] in [CATEGORICAL, ORDINAL]:
            mask_d[id_] = 1
        else:
            mask_d[id_] = 0

    std = np.std(train, axis=0) + 1e-6

    dis_all = []
    for i in range(min(sample, len(train))):
        current = syn[i]
        distance_d = (train - current) * mask_d > 0
        distance_d = np.sum(distance_d, axis=1)

        distance_c = (train - current) * (1 - mask_d) / 2 / std
        distance_c = np.sum(distance_c ** 2, axis=1)
        distance = np.sqrt(np.min(distance_c + distance_d))
        dis_all.append(distance)

    return np.mean(dis_all)


_EVALUATORS = {
    'bayesian_likelihood': _evaluate_bayesian_likelihood,
    'binary_classification': _evaluate_binary_classification,
    'gaussian_likelihood': _evaluate_gmm_likelihood,
    'multiclass_classification': _evaluate_multi_classification,
    'regression': _evaluate_regression,
}


def evaluate(train, test, synthesized_data, metadata):
    evaluator = _EVALUATORS[metadata['problem_type']]

    results = []
    for step, synth_data in enumerate(synthesized_data):
        performances = evaluator(synth_data, test, metadata)
        distance = _compute_distance(train, synth_data, metadata)

        for performance in performances:
            performance['step'] = step
            performance['distance'] = distance
            results.append(performance)

    return pd.DataFrame(results)