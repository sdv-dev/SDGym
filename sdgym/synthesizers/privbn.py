import os
import shutil
import subprocess

import numpy as np

from sdgym.constants import CATEGORICAL, ORDINAL
from sdgym.synthesizers.base import BaseSynthesizer
from sdgym.synthesizers.utils import Transformer


def try_mkdirs(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)


class PrivBNSynthesizer(BaseSynthesizer):
    """docstring for IdentitySynthesizer."""

    def __init__(self):
        assert os.path.exists("privbayes/privBayes.bin")

    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        self.data = data.copy()
        self.meta = Transformer.get_metadata(data, categorical_columns, ordinal_columns)

    def sample(self, n):
        try_mkdirs("__privbn_tmp/data")
        try_mkdirs("__privbn_tmp/log")
        try_mkdirs("__privbn_tmp/output")
        shutil.copy("privbayes/privBayes.bin", "__privbn_tmp/privBayes.bin")
        d_cols = []
        with open("__privbn_tmp/data/real.domain", "w") as f:
            for id_, info in enumerate(self.meta):
                if info['type'] in [CATEGORICAL, ORDINAL]:
                    print("D", end='', file=f)
                    counter = 0
                    for i in range(info['size']):
                        if i > 0 and i % 4 == 0:
                            counter += 1
                            print(" {", end='', file=f)
                        print("", i, end='', file=f)
                    print(" }" * counter, file=f)
                    d_cols.append(id_)
                else:
                    minn = info['min']
                    maxx = info['max']
                    d = (maxx - minn) * 0.03
                    minn = minn - d
                    maxx = maxx + d
                    print("C", minn, maxx, file=f)

        with open("__privbn_tmp/data/real.dat", "w") as f:
            n = len(self.data)
            np.random.shuffle(self.data)
            n = min(n, 50000)
            for i in range(n):
                row = self.data[i]
                for id_, col in enumerate(row):
                    if id_ in d_cols:
                        print(int(col), end=' ', file=f)

                    else:
                        print(col, end=' ', file=f)

                print(file=f)

        privbayes = os.path.realpath("__privbn_tmp/privBayes.bin")
        # subprocess.call([privbayes, "real", str(n), "1", "5"], cwd="__privbn_tmp")
        subprocess.call([privbayes, "real", str(n), "1", "10"], cwd="__privbn_tmp")

        return np.loadtxt("__privbn_tmp/output/syn_real_eps10_theta10_iter0.dat")
