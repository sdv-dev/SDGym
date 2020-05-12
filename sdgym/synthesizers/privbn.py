import logging
import os
import shutil
import subprocess
from datetime import datetime

import numpy as np

from sdgym.constants import CATEGORICAL, ORDINAL
from sdgym.synthesizers.base import BaseSynthesizer
from sdgym.synthesizers.utils import Transformer

LOGGER = logging.getLogger(__name__)


def try_mkdirs(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)


class PrivBNSynthesizer(BaseSynthesizer):
    """docstring for IdentitySynthesizer."""

    def __init__(self, theta=20, max_samples=25000):
        assert os.path.exists("privbayes/privBayes.bin")
        self.theta = theta
        self.max_samples = max_samples

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
            n = min(n, self.max_samples)
            for i in range(n):
                row = self.data[i]
                for id_, col in enumerate(row):
                    if id_ in d_cols:
                        print(int(col), end=' ', file=f)

                    else:
                        print(col, end=' ', file=f)

                print(file=f)

        privbayes = os.path.realpath("__privbn_tmp/privBayes.bin")
        arguments = [privbayes, "real", str(n), "1", str(self.theta)]
        LOGGER.info('Calling %s', ' '.join(arguments))
        start = datetime.utcnow()
        subprocess.call(arguments, cwd="__privbn_tmp")
        LOGGER.info('Elapsed %s', datetime.utcnow() - start)

        return np.loadtxt(
            "__privbn_tmp/output/syn_real_eps10_theta{}_iter0.dat".format(self.theta))
