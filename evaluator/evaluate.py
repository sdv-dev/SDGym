import logging
import glob
import argparse
import os
import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import DecisionTreeRegressor as DTR

from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.ensemble import AdaBoostRegressor as ABR

from sklearn.linear_model import LogisticRegression as LRC
from sklearn.linear_model import LinearRegression as LRR

from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.neural_network import MLPRegressor as MLPR

from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.utils import shuffle

from utils import CATEGORICAL, CONTINUOUS, ORDINAL

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(
    description='Evaluate output of one synthesizer.')
parser.add_argument('--result', type=str, default='output/__result__',
                    help='result dir')
parser.add_argument('--force', dest='force',
                    action='store_true', help='overwrite result')
parser.set_defaults(force=False)
parser.add_argument('synthetic', type=str,
                    help='synthetic data folder')


def default_multi_classification(x_train, y_train, x_test, y_test, classifiers):
    performance = []
    for clf, name in classifiers:
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


def default_binary_classification(x_train, y_train, x_test, y_test, classifiers):
    performance = []
    for clf, name in classifiers:
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


def news_regression(x_train, y_train, x_test, y_test, regressors):
    performance = []
    y_train = np.log(np.clip(y_train, 0, 20000))
    y_test = np.log(np.clip(y_test, 0, 20000))
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


def make_features(data, meta, label_column='label', label_type='int', sample=50000):
    data = data.copy()
    np.random.shuffle(data)
    data = data[:sample]

    features = []
    labels = []

    for row in data:
        feature = []
        label = None
        for col, cinfo in zip(row, meta):
            if cinfo['name'] == 'label':
                if label_type == 'int':
                    label = int(col)
                elif label_type == 'float':
                    label = float(col)
                else:
                    assert 0, 'unkown label type'
                continue
            if cinfo['type'] == CONTINUOUS:
                if cinfo['min'] >= 0 and cinfo['max'] >= 1e3:
                    feature.append(np.log(max(col, 1e-2))) # log feature
                else:
                    feature.append((col - cinfo['min']) / (cinfo['max'] - cinfo['min']) * 5) #[0, 5]
            elif cinfo['type'] == ORDINAL:
                feature.append(col)
            else:
                if cinfo['size'] <= 2:
                    feature.append(col)
                else:
                    tmp = [0] * cinfo['size']
                    tmp[int(col)] = 1
                    feature += tmp
        features.append(feature)
        labels.append(label)

    return features, labels

def get_models(dataset):
    if dataset in ["mnist12", "mnist28"]:
        classifiers = [
            (DTC(max_depth=30, class_weight='balanced'), "Decision Tree (max_depth=30)"),
            (LRC(solver='lbfgs', n_jobs=2, multi_class="auto",
                 class_weight='balanced', max_iter=50), "Logistic Regression"),
            (MLPC((100, ), max_iter=50), "MLP (100)")
        ]
        return classifiers
    if dataset in ['adult']:
        classifiers = [
            (DTC(max_depth=15, class_weight='balanced'), "Decision Tree (max_depth=20)"),
            (ABC(), "Adaboost (estimator=50)"),
            (LRC(solver='lbfgs', n_jobs=2,
                 class_weight='balanced', max_iter=50), "Logistic Regression"),
            (MLPC((50, ), max_iter=50), "MLP (50)")
        ]
        return classifiers
    if dataset in ['census', 'credit']:
        classifiers = [
            (DTC(max_depth=30, class_weight='balanced'), "Decision Tree (max_depth=30)"),
            (ABC(), "Adaboost (estimator=50)"),
            (MLPC((100, ), max_iter=50), "MLP (100)"),
        ]
        return classifiers
    if dataset in ['intrusion', 'covtype']:
        classifiers = [
            (DTC(max_depth=30, class_weight='balanced'), "Decision Tree (max_depth=30)"),
            (MLPC((100, ), max_iter=50), "MLP (100)"),
        ]
        return classifiers
    if dataset in ['news']:
        regressors = [
            (LRR(), "Linear Regression"),
            (MLPR((100, ), max_iter=50), "MLP (100)")
        ]
        return regressors

    assert 0

def evalute_dataset(dataset, trainset, testset, meta):
    if dataset in ["mnist12", "mnist28", "covtype", "intrusion"]:
        x_train, y_train = make_features(trainset, meta)
        x_test, y_test = make_features(testset, meta)
        return default_multi_classification(x_train, y_train, x_test, y_test, get_models(dataset))

    elif dataset in ['credit', 'census', 'adult']:
        x_train, y_train = make_features(trainset, meta)
        x_test, y_test = make_features(testset, meta)
        return default_binary_classification(x_train, y_train, x_test, y_test, get_models(dataset))

    elif dataset in ['news']:
        x_train, y_train = make_features(trainset, meta)
        x_test, y_test = make_features(testset, meta)
        return news_regression(x_train, y_train, x_test, y_test, get_models(dataset))

    else:
        logging.warning("{} evaluation not defined.".format(dataset))
        assert 0


if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.exists(args.result):
        os.makedirs(args.result)

    result_file = "{}/{}.json".format(args.result,
                                      args.synthetic.replace('/', '\t').split()[-1])
    if os.path.exists(result_file):
        logging.warning("result file {} exists.".format(result_file))
        if args.force:
            logging.warning("overwrite {}.".format(result_file))
        else:
            exit()

    logging.info("use result file {}.".format(result_file))

    synthetic_folder = args.synthetic
    synthetic_files = glob.glob("{}/*.npz".format(synthetic_folder))

    results = []

    for synthetic_file in synthetic_files:
        # synthetic_file is like xxx/xxx/dataset_iter_step.npz
        # iter is the iteration of experiment
        # step is the learning steps of some synthesizer, 0 if no learning
        syn = np.load(synthetic_file)['syn']

        dataset_iter_step = synthetic_file.split('/')[-1]
        assert dataset_iter_step[-4:] == '.npz'
        dataset_iter_step = dataset_iter_step[:-4].split('_')

        dataset = dataset_iter_step[0]
        iter = int(dataset_iter_step[1])
        step = int(dataset_iter_step[2])

        data_filename = glob.glob("data/*/{}.npz".format(dataset))
        meta_filename = glob.glob("data/*/{}.json".format(dataset))

        if len(data_filename) != 1:
            logging.warning("Skip. Can't find dataset {}. ".format(dataset))
            continue

        if len(meta_filename) != 1:
            logging.warning("Skip. Can't find meta {}. ".format(dataset))
            continue

        data = np.load(data_filename[0])['test']
        with open(meta_filename[0]) as f:
            meta = json.load(f)

        logging.info("Evaluating {}".format(synthetic_file))
        performance = evalute_dataset(dataset, syn, data, meta)

        res = {
            "dataset": dataset,
            "iter": iter,
            "step": step,
            "performance": performance
        }

        results.append(res)

    with open(result_file, "w") as f:
        json.dump(results, f, sort_keys=True, indent=4, separators=(',', ': '))
