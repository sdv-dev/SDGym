# Generate news datasets

import os
import logging
import json
import numpy as np
import pandas as pd

from ..utils import CATEGORICAL, CONTINUOUS, ORDINAL, verify


output_dir = "data/real/"
temp_dir = "tmp/"


if __name__ == "__main__":
    try:
        os.mkdir(output_dir)
    except:
        pass

    try:
        os.mkdir(temp_dir)
    except:
        pass

    df = pd.read_csv("data/raw/news/OnlineNewsPopularity.csv", dtype='str', header=0)
    df = df.apply(lambda x: x.str.strip(' \t.'))
    df.drop(['url', ' timedelta'], axis=1, inplace=True)

    meta = []
    for col_name in df.columns:
        if "is_" in col_name:
            meta.append({
                "name": col_name,
                "type": CATEGORICAL,
                "size": 2,
                "i2s": ['0', '1']
            })
        else:
            meta.append({
                "name": "label" if col_name.strip() == "shares" else col_name.strip(),
                "type": CONTINUOUS,
                "min": np.min(df[col_name].values.astype('float')),
                "max": np.max(df[col_name].values.astype('float'))
            })

    tdata = df.values.astype('float32')

    np.random.seed(0)
    np.random.shuffle(tdata)

    t_train = tdata[:-8000]
    t_test = tdata[-8000:]

    name = "news"
    with open("{}/{}.json".format(output_dir, name), 'w') as f:
        json.dump(meta, f, sort_keys=True, indent=4, separators=(',', ': '))
    np.savez("{}/{}.npz".format(output_dir, name), train=t_train, test=t_test)

    verify("{}/{}.npz".format(output_dir, name),
            "{}/{}.json".format(output_dir, name))
