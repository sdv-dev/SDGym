# Generate census datasets

import os
import logging
import json
import numpy as np
import pandas as pd

from ..utils import CATEGORICAL, CONTINUOUS, ORDINAL, verify


output_dir = "data/real/"
temp_dir = "tmp/"


def project_table(data, meta):
    values = np.zeros(shape=data.shape, dtype='float32')

    for id_, info in enumerate(meta):
        if info['type'] == CONTINUOUS:
            values[:, id_] = data.iloc[:, id_].values.astype('float32')
        else:
            mapper = dict([(item, id) for id, item in enumerate(info['i2s'])])
            mapped = data.iloc[:, id_].apply(lambda x: mapper[x]).values
            values[:, id_] = mapped
            mapped = data.iloc[:, id_].apply(lambda x: mapper[x]).values
    return values


if __name__ == "__main__":
    try:
        os.mkdir(output_dir)
    except:
        pass

    try:
        os.mkdir(temp_dir)
    except:
        pass

    trainset = pd.read_csv("data/raw/census/census-income.data", dtype='str', header=-1)
    trainset = trainset.apply(lambda x: x.str.strip(' \t.'))
    testset = pd.read_csv("data/raw/census/census-income.test", dtype='str', header=-1)
    testset = testset.apply(lambda x: x.str.strip(' \t.'))
    trainset.drop([24], axis='columns', inplace=True) # drop instance weight
    testset.drop([24], axis='columns', inplace=True)

    col_type = [
        ("age", CONTINUOUS),
        ("class of worker", CATEGORICAL),
        ("detailed industry recode", CATEGORICAL),
        ("detailed occupation recode", CATEGORICAL),
        ("education", CATEGORICAL),
        ("wage per hour", CONTINUOUS),
        ("enroll in edu inst last wk", CATEGORICAL),
        ("marital stat", CATEGORICAL),
        ("major industry code", CATEGORICAL),
        ("major occupation code", CATEGORICAL),
        ("race", CATEGORICAL),
        ("hispanic origin", CATEGORICAL),
        ("sex", CATEGORICAL),
        ("member of a labor union", CATEGORICAL),
        ("reason for unemployment", CATEGORICAL),
        ("full or part time employment stat", CATEGORICAL),
        ("capital gains", CONTINUOUS),
        ("capital losses", CONTINUOUS),
        ("dividends from stocks", CONTINUOUS),
        ("tax filer stat", CATEGORICAL),
        ("region of previous residence", CATEGORICAL),
        ("state of previous residence", CATEGORICAL),
        ("detailed household and family stat", CATEGORICAL),
        ("detailed household summary in household", CATEGORICAL),
        ("migration code-change in msa", CATEGORICAL),
        ("migration code-change in reg", CATEGORICAL),
        ("migration code-move within reg", CATEGORICAL),
        ("live in this house 1 year ago", CATEGORICAL),
        ("migration prev res in sunbelt", CATEGORICAL),
        ("num persons worked for employer", CONTINUOUS),
        ("family members under 18", CATEGORICAL),
        ("country of birth father", CATEGORICAL),
        ("country of birth mother", CATEGORICAL),
        ("country of birth self", CATEGORICAL),
        ("citizenship", CATEGORICAL),
        ("own business or self employed", CATEGORICAL),
        ("fill inc questionnaire for veteran's admin", CATEGORICAL),
        ("veterans benefits", CATEGORICAL),
        ("weeks worked in year", CONTINUOUS),
        ("year", CATEGORICAL),
        ("label", CATEGORICAL)
    ]

    meta = []
    for id_, info in enumerate(col_type):
        if info[1] == CONTINUOUS:
            meta.append({
                "name": info[0],
                "type": info[1],
                "min": np.min(trainset.iloc[:, id_].values.astype('float')),
                "max": np.max(trainset.iloc[:, id_].values.astype('float'))
            })
        else:
            value_count = list(dict(trainset.iloc[:, id_].value_counts()).items())
            value_count = sorted(value_count, key=lambda x: -x[1])
            mapper = list(map(lambda x: x[0], value_count))
            meta.append({
                "name": info[0],
                "type": info[1],
                "size": len(mapper),
                "i2s": mapper
            })


    t_train = project_table(trainset, meta)
    t_test = project_table(testset, meta)

    name = "census"
    with open("{}/{}.json".format(output_dir, name), 'w') as f:
        json.dump(meta, f, sort_keys=True, indent=4, separators=(',', ': '))
    np.savez("{}/{}.npz".format(output_dir, name), train=t_train, test=t_test)

    verify("{}/{}.npz".format(output_dir, name),
            "{}/{}.json".format(output_dir, name))
