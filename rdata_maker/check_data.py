import glob
import numpy as np
import json
from utils import CONTINUOUS, CATEGORICAL, ORDINAL

if __name__ == "__main__":
    inputs = glob.glob("data/real/*.npz")
    for file in inputs:
        data = np.load(file)['train']
        print(file, data.dtype)

        with open(file[:-3] + "json") as f:
            meta = json.load(f)

        minimum_unique = 100
        minimum_col = ""
        for id_, info in enumerate(meta):
            if info['type'] == CONTINUOUS:
                ct = len(np.unique(data[:, id_]))
                if ct < minimum_unique:
                    minimum_unique = ct
                    minimum_col = info['name']

        print("  ", minimum_col, minimum_unique)
