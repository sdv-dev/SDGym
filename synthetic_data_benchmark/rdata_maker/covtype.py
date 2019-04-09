# Generate covtype datasets

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

    df = pd.read_csv("data/raw/covtype/covtype.data", dtype='str', header=-1)

    col_type = [
        ("Elevation", CONTINUOUS),
        ("Aspect", CONTINUOUS),
        ("Slope", CONTINUOUS),
        ("Horizontal_Distance_To_Hydrology", CONTINUOUS),
        ("Vertical_Distance_To_Hydrology", CONTINUOUS),
        ("Horizontal_Distance_To_Roadways", CONTINUOUS),
        ("Hillshade_9am", CONTINUOUS),
        ("Hillshade_Noon", CONTINUOUS),
        ("Hillshade_3pm", CONTINUOUS),
        ("Horizontal_Distance_To_Fire_Points", CONTINUOUS)
    ] + [
        ("Wilderness_Area_{}".format(i), CATEGORICAL) for i in range(4)
    ] + [
        ("Soil_Type{}".format(i), CATEGORICAL) for i in range(40)
    ] + [
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
            if info[0] == "label":
                mapper = ['1', '2', '3', '4', '5', '6', '7']
            else:
                mapper = ['0', '1']

            meta.append({
                "name": info[0],
                "type": info[1],
                "size": len(mapper),
                "i2s": mapper
            })


    tdata = df.values.astype('float32')
    tdata[:, -1] -= 1

    np.random.seed(0)
    np.random.shuffle(tdata)

    t_train = tdata[:-100000]
    t_test = tdata[-100000:]

    name = "covtype"
    with open("{}/{}.json".format(output_dir, name), 'w') as f:
        json.dump(meta, f, sort_keys=True, indent=4, separators=(',', ': '))
    np.savez("{}/{}.npz".format(output_dir, name), train=t_train, test=t_test)

    verify("{}/{}.npz".format(output_dir, name),
            "{}/{}.json".format(output_dir, name))
