# Generate adult datasets

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
        s.mkdir(output_dir)
    except:
        pass

    try:
        os.mkdir(temp_dir)
    except:
        pass

    df = pd.read_csv("data/raw/adult/adult.data", dtype='str', header=-1)
    df = df.apply(lambda x: x.str.strip(' \t.'))

    col_type = [
        ("age", CONTINUOUS),
        ("workclass", CATEGORICAL),
        ("fnlwgt", CONTINUOUS),
        ("education", ORDINAL, ["Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th", "HS-grad", "Prof-school", "Assoc-voc", "Assoc-acdm", "Some-college", "Bachelors", "Masters", "Doctorate"]),
        ("education-num", CONTINUOUS),
        ("marital-status", CATEGORICAL),
        ("occupation", CATEGORICAL),
        ("relationship", CATEGORICAL),
        ("race", CATEGORICAL),
        ("sex", CATEGORICAL),
        ("capital-gain", CONTINUOUS),
        ("capital-loss", CONTINUOUS),
        ("hours-per-week", CONTINUOUS),
        ("native-country", CATEGORICAL),
        ("label", CATEGORICAL)
    ]

    meta = []
    for id_, info in enumerate(col_type):
        if info[1] == CONTINUOUS:
            meta.append({
                "name": info[0],
                "type": info[1],
                "min": np.min(df.iloc[:, id_].values.astype('float')),
                "max": np.max(df.iloc[:, id_].values.astype('float'))
            })
        else:
            if info[1] == CATEGORICAL:
                value_count = list(dict(df.iloc[:, id_].value_counts()).items())
                value_count = sorted(value_count, key=lambda x: -x[1])
                mapper = list(map(lambda x: x[0], value_count))
            else:
                mapper = info[2]

            meta.append({
                "name": info[0],
                "type": info[1],
                "size": len(mapper),
                "i2s": mapper
            })


    tdata = project_table(df, meta)

    np.random.seed(0)
    np.random.shuffle(tdata)

    t_train = tdata[:-10000]
    t_test = tdata[-10000:]

    name = "adult"
    with open("{}/{}.json".format(output_dir, name), 'w') as f:
        json.dump(meta, f, sort_keys=True, indent=4, separators=(',', ': '))
    np.savez("{}/{}.npz".format(output_dir, name), train=t_train, test=t_test)

    verify("{}/{}.npz".format(output_dir, name),
            "{}/{}.json".format(output_dir, name))
