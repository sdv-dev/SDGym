import logging
import glob
import argparse
import os
import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.linear_model import LogisticRegression as LRC
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import shuffle

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='Evaluate output of one synthesizer.')

parser.add_argument('--result', type=str, default='output/__result__',
                    help='result dir')
parser.add_argument('--synthetic', type=str, required=True,
                    help='synthetic data folder')


def default_multi_classification(x_train, y_train, x_test, y_test):
    N = 10000
    x_train, y_train = shuffle(x_train, y_train)
    x_train = x_train[:N]
    y_train = y_train[:N]

    classifiers = [
        (DTC(max_depth=10), "Decision Tree (max_depth=5)"),
        (DTC(max_depth=30), "Decision Tree (max_depth=5)"),
        (ABC(), "Adaboost (estimator=50)"),
        (LRC(n_jobs=2, "Logistic Regression"),
        (MLPC((100, )), "MLP (100)"),
        (MLPC((100, 100)), "MLP (100, 100)")
    ]


    performance = []
    for clf, name in classifiers:
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


def evalute_dataset(dataset, trainset, testset, meta):
    if dataset == "mnist28" or dataset == "mnist12":
        x_train = trainset[:, :-1]
        y_train = trainset[:, -1]

        x_test = testset[:, :-1]
        y_test = testset[:, -1]

        return default_multi_classification(x_train, y_train, x_test, y_test)
    else:
        assert 0


if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.exists(args.result):
        os.makedirs(args.result)

    result_file = "{}/{}.json".format(args.result, args.synthetic.replace('/', '\t').split()[-1])
    if os.path.exists(result_file):
        logging.warning("Skip. result file {} exists.".format(result_file))
        exit()

    logging.info("use result file {}.".format(result_file))

    synthetic_folder = args.synthetic
    synthetic_files = glob.glob("{}/*.npz".format(synthetic_folder))


    results = []

    print(synthetic_files)
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
