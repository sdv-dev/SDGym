import numpy as np
from sklearn.mixture import GaussianMixture

from sdgym.constants import CONTINUOUS
from sdgym.synthesizers.base import BaseSynthesizer
from sdgym.synthesizers.utils import Transformer


class IndependentSynthesizer(BaseSynthesizer):
    """docstring for IdentitySynthesizer."""

    def __init__(self, gmm_n=5):
        self.gmm_n = gmm_n

    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        self.dtype = data.dtype
        self.meta = Transformer.get_metadata(data, categorical_columns, ordinal_columns)

        self.models = []
        for id_, info in enumerate(self.meta):
            if info['type'] == CONTINUOUS:
                model = GaussianMixture(self.gmm_n)
                model.fit(data[:, [id_]])
                self.models.append(model)
            else:
                nomial = np.bincount(data[:, id_].astype('int'), minlength=info['size'])
                nomial = nomial / np.sum(nomial)
                self.models.append(nomial)

    def sample(self, samples):
        data = np.zeros([samples, len(self.meta)], self.dtype)

        for i, info in enumerate(self.meta):
            if info['type'] == CONTINUOUS:
                x, _ = self.models[i].sample(samples)
                np.random.shuffle(x)
                data[:, i] = x.reshape([samples])
                data[:, i] = data[:, i].clip(info['min'], info['max'])
            else:
                size = len(self.models[i])
                data[:, i] = np.random.choice(np.arange(size), samples, p=self.models[i])

        return data
