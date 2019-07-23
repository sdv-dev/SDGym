from .synthesizer_base import SynthesizerBase, run
import numpy as np
from sklearn.mixture import GaussianMixture
from synthetic_data_benchmark.utils import CONTINUOUS, ORDINAL, CATEGORICAL
rng = np.random

class IndependentSynthesizer(SynthesizerBase):
    """docstring for IdentitySynthesizer."""

    def __init__(self, gmm_n):
        self.gmm_n = gmm_n

    def train(self, train_data):
        self.dtype = train_data.dtype

        self.models = []
        for id_, info in enumerate(self.meta):
            if info['type'] == CONTINUOUS:
                model = GaussianMixture(self.gmm_n)
                model.fit(train_data[:, [id_]])
                self.models.append(model)
            else:
                nomial = np.bincount(train_data[:, id_].astype('int'), minlength=info['size'])
                nomial = nomial / np.sum(nomial)
                self.models.append(nomial)


    def generate(self, n):
        data = np.zeros([n, len(self.meta)], self.dtype)

        for i, info in enumerate(self.meta):
            if info['type'] == 'continuous':
                x, _ = self.models[i].sample(n)
                rng.shuffle(x)
                data[:, i] = x.reshape([n])
                data[:, i] = data[:, i].clip(info['min'], info['max'])
            else:
                data[:, i] = np.random.choice(np.arange(info['size']), n, p=self.models[i])

        return [(0, data)]

    def init(self, meta, working_dir):
        self.meta = meta


if __name__ == "__main__":
    run(IndependentSynthesizer(5))
