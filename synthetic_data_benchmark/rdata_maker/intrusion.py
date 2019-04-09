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
        os.mkdir(output_dir)
    except:
        pass

    try:
        os.mkdir(temp_dir)
    except:
        pass

    df = pd.read_csv("data/raw/intrusion/kddcup.data_10_percent", dtype='str', header=-1)
    df = df.apply(lambda x: x.str.strip(' \t.'))

    label_mapping = {
        "back": "dos",
        "buffer_overflow": "u2r",
        "ftp_write": "r2l",
        "guess_passwd": "r2l",
        "imap": "r2l",
        "ipsweep": "probe",
        "land": "dos",
        "loadmodule": "u2r",
        "multihop": "r2l",
        "neptune": "dos",
        "nmap": "probe",
        "perl": "u2r",
        "phf": "r2l",
        "pod": "dos",
        "portsweep": "probe",
        "rootkit": "u2r",
        "satan": "probe",
        "smurf": "dos",
        "spy": "r2l",
        "teardrop": "dos",
        "warezclient": "r2l",
        "warezmaster": "r2l",
        "normal": "normal"
    }
    df.iloc[:, -1] = df.iloc[:, -1].apply(lambda x: label_mapping[x])

    df.drop([19], axis=1, inplace=True)

    col_type = [
        ("duration", CONTINUOUS),
        ("protocol_type", CATEGORICAL),
        ("service", CATEGORICAL),
        ("flag", CATEGORICAL),
        ("src_bytes", CONTINUOUS),
        ("dst_bytes", CONTINUOUS),
        ("land", CATEGORICAL),
        ("wrong_fragment", ORDINAL, ['0', '1', '2', '3']),
        ("urgent", ORDINAL, ['0', '1', '2', '3']),
        ("hot", CONTINUOUS),
        ("num_failed_logins", ORDINAL, ['0', '1', '2', '3', '4', '5']),
        ("logged_in", CATEGORICAL),
        ("num_compromised", CONTINUOUS),
        ("root_shell", CATEGORICAL),
        ("su_attempted", ORDINAL, ['0', '1', '2', '3']),
        ("num_root", CONTINUOUS),
        ("num_file_creations", CONTINUOUS),
        ("num_shells", ORDINAL, ['0', '1', '2']),
        ("num_access_files", ORDINAL, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']),
        # ("num_outbound_cmds", CONTINUOUS), # all zero, removed
        ("is_host_login", CATEGORICAL),
        ("is_guest_login", CATEGORICAL),
        ("count", CONTINUOUS),
        ("srv_count", CONTINUOUS),
        ("serror_rate", CONTINUOUS),
        ("srv_serror_rate", CONTINUOUS),
        ("rerror_rate", CONTINUOUS),
        ("srv_rerror_rate", CONTINUOUS),
        ("same_srv_rate", CONTINUOUS),
        ("diff_srv_rate", CONTINUOUS),
        ("srv_diff_host_rate", CONTINUOUS),
        ("dst_host_count", CONTINUOUS),
        ("dst_host_srv_count", CONTINUOUS),
        ("dst_host_same_srv_rate", CONTINUOUS),
        ("dst_host_diff_srv_rate", CONTINUOUS),
        ("dst_host_same_src_port_rate", CONTINUOUS),
        ("dst_host_srv_diff_host_rate", CONTINUOUS),
        ("dst_host_serror_rate", CONTINUOUS),
        ("dst_host_srv_serror_rate", CONTINUOUS),
        ("dst_host_rerror_rate", CONTINUOUS),
        ("dst_host_srv_rerror_rate", CONTINUOUS),
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

    t_train = tdata[:-100000]
    t_test = tdata[-100000:]

    name = "intrusion"
    with open("{}/{}.json".format(output_dir, name), 'w') as f:
        json.dump(meta, f, sort_keys=True, indent=4, separators=(',', ': '))
    np.savez("{}/{}.npz".format(output_dir, name), train=t_train, test=t_test)

    verify("{}/{}.npz".format(output_dir, name),
            "{}/{}.json".format(output_dir, name))
