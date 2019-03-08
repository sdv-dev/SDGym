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

parser = argparse.ArgumentParser(description='Evaluate output of one synthesizer.')

parser.add_argument('--result', type=str, default='output/__result__',
                    help='result dir')
parser.add_argument('--force', dest='force', action='store_true', help='overwrite result')
parser.set_defaults(force=False)

parser.add_argument('synthetic', type=str,
                    help='synthetic data folder')


def default_multi_classification(x_train, y_train, x_test, y_test):
    classifiers = [
        (DTC(max_depth=10, class_weight='balanced'), "Decision Tree (max_depth=5)"),
        (DTC(max_depth=30, class_weight='balanced'), "Decision Tree (max_depth=30)"),
        (ABC(), "Adaboost (estimator=50)"),
        (LRC(solver='lbfgs', n_jobs=2, multi_class="auto", class_weight='balanced'), "Logistic Regression"),
        (MLPC((100, )), "MLP (100)"),
        (MLPC((100, 100)), "MLP (100, 100)")
    ]


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


def default_binary_classification(x_train, y_train, x_test, y_test):
    classifiers = [
        (DTC(max_depth=10, class_weight='balanced'), "Decision Tree (max_depth=5)"),
        (DTC(max_depth=30, class_weight='balanced'), "Decision Tree (max_depth=30)"),
        (ABC(), "Adaboost (estimator=50)"),
        (LRC(solver='lbfgs', n_jobs=2, multi_class="auto", class_weight='balanced'), "Logistic Regression"),
        (MLPC((100, )), "MLP (100)"),
        (MLPC((100, 100)), "MLP (100, 100)")
    ]


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


def default_regression(x_train, y_train, x_test, y_test):
    regressor = [
        (DTR(max_depth=10), "Decision Tree (max_depth=5)"),
        (DTR(max_depth=30), "Decision Tree (max_depth=30)"),
        (ABR(), "Adaboost (estimator=50)"),
        (LRR(), "Logistic Regression"),
        (MLPR((100, )), "MLP (100)"),
        (MLPR((100, 100)), "MLP (100, 100)")
    ]


    performance = []
    for clf, name in regressor:
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


def make_features(dataset, meta, label_column='label', sample=10000):
    dataset = dataset.copy()
    np.random.shuffle(dataset)
    dataset = dataset[:sample]

    features = []
    labels = []

    for row in dataset:
        feature = []
        label = None
        for col, cinfo in zip(row, meta):
            if cinfo['name'] == 'label':
                label = int(col)
                continue
            if cinfo['type'] in [CONTINUOUS, ORDINAL]:
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


def evalute_dataset(dataset, trainset, testset, meta):
    if dataset in ["mnist12", "mnist28", "covtype", "intrusion"]:
        x_train, y_train = make_features(trainset, meta)
        x_test, y_test = make_features(testset, meta)

        return default_multi_classification(x_train, y_train, x_test, y_test)
    elif dataset in ['credit', 'census', 'adult']:
        x_train, y_train = make_features(trainset, meta)
        x_test, y_test = make_features(testset, meta)

        return default_binary_classification(x_train, y_train, x_test, y_test)

    elif dataset in ['news']:
        x_train, y_train = make_features(trainset, meta)
        x_test, y_test = make_features(testset, meta)

        return default_regression(x_train, y_train, x_test, y_test)
    else:
        logging.warning("{} evaluation not defined.".format(dataset))
        assert 0


if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.exists(args.result):
        os.makedirs(args.result)

    result_file = "{}/{}.json".format(args.result, args.synthetic.replace('/', '\t').split()[-1])
    if os.path.exists(result_file):
        logging.warning("Skip. result file {} exists.".format(result_file))
        if args.force:
            logging.warning("overwrite {}.".format(result_file))
        else:
            exit()

    logging.info("use result file {}.".format(result_file))

    synthetic_folder = args.synthetic
    synthetic_files = glob.glob("{}/*.npz".format(synthetic_folder))


    results = []

    for synthetic_file in synthetic_files:
        syn = np.load(synthetic_file)['syn']

        info = synthetic_file.split('/')[-1]
        assert info[-4:] == '.npz'
        info = info[:-4].split('_')


        dataset = info[0]
        iter = int(info[1])
        step = int(info[2])


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
