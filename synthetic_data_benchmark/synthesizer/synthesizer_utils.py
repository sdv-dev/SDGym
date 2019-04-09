from sklearn.preprocessing import KBinsDiscretizer
from sklearn.mixture import GaussianMixture
import numpy as np

from ..utils import CATEGORICAL, ORDINAL, CONTINUOUS

class DiscretizeTransformer(object):
    """Discretize continuous columns into several bins.
    Transformation result is a int array."""
    def __init__(self, meta, n_bins):
        self.meta = meta
        self.c_index = [id for id, info in enumerate(meta) if info['type'] == CONTINUOUS]
        self.kbin_discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')

    def fit(self, data):
        if self.c_index == []:
            return
        self.kbin_discretizer.fit(data[:, self.c_index])

    def transform(self, data):
        if self.c_index == []:
            return data.astype('int')

        data_t = data.copy()
        data_t[:, self.c_index] = self.kbin_discretizer.transform(data[:, self.c_index])
        return data_t.astype('int')

    def inverse_transform(self, data):
        if self.c_index == []:
            return data

        data_t = data.copy().astype('float32')
        data_t[:, self.c_index] = self.kbin_discretizer.inverse_transform(data[:, self.c_index])
        return data_t

class GeneralTransformer(object):
    """
    Continuous and ordinal columns are normalized to [0, 1].
    Discrete columns are converted to a one-hot vector.
    """
    def __init__(self, meta):
        self.meta = meta
        self.output_dim = 0
        for info in self.meta:
            if info['type'] in [CONTINUOUS, ORDINAL]:
                self.output_dim += 1
            else:
                self.output_dim += info['size']

    def fit(self, data):
        pass

    def transform(self, data):
        data_t = []
        self.output_info = []
        for id_, info in enumerate(self.meta):
            col = data[:, id_]
            if info['type'] == CONTINUOUS:
                col = (col - (info['min'])) / (info['max'] - info['min'])
                data_t.append(col.reshape([-1, 1]))
                self.output_info.append((1, 'sigmoid'))
            elif info['type'] == ORDINAL:
                col = col / info['size']
                data_t.append(col.reshape([-1, 1]))
                self.output_info.append((1, 'sigmoid'))
            else:
                col_t = np.zeros([len(data), info['size']])
                col_t[np.arange(len(data)), col.astype('int32')] = 1
                data_t.append(col_t)
                self.output_info.append((info['size'], 'softmax'))
        return np.concatenate(data_t, axis=1)

    def inverse_transform(self, data):
        data_t = np.zeros([len(data), len(self.meta)])

        data = data.copy()
        for id_, info in enumerate(self.meta):
            if info['type'] == CONTINUOUS:
                current = data[:, 0]
                data = data[:, 1:]

                current = np.clip(current, 0, 1)
                data_t[:, id_] = current * (info['max'] - info['min']) + info['min']

            elif info['type'] == ORDINAL:
                current = data[:, 0]
                data = data[:, 1:]
                current = current * info['size']
                current = np.round(current).clip(0, info['size'] - 1)
                data_t[:, id_] = current
            else:
                current = data[:, :info['size']]
                data = data[:, info['size']:]
                data_t[:, id_] = np.argmax(current, axis=1)

        return data_t

class GMMTransformer(object):
    """
    Continuous and ordinal columns are modeled with a GMM.
        and then normalized to a scalor [0, 1] and a n_cluster dimensional vector.

    Discrete and ordinal columns are converted to a one-hot vector.
    """

    def __init__(self, meta, n_clusters=5):
        self.meta = meta
        self.n_clusters = n_clusters

    def fit(self, data):
        model = []

        self.output_info = []
        self.output_dim = 0
        for id_, info in enumerate(self.meta):
            if info['type'] == CONTINUOUS:
                gm = GaussianMixture(self.n_clusters)
                gm.fit(data[:, id_].reshape([-1, 1]))
                model.append(gm)
                self.output_info += [(1, 'mix'), (self.n_clusters, 'softmax')]
                self.output_dim += 1 + self.n_clusters
            else:
                model.append(None)
                self.output_info += [(info['size'], 'softmax')]
                self.output_dim += info['size']

        self.model = model

    def transform(self, data):
        values = []
        for id_, info in enumerate(self.meta):
            current = data[:, id_]
            if info['type'] == CONTINUOUS:
                current = current.reshape([-1, 1])

                means = self.model[id_].means_.reshape((1, self.n_clusters))
                stds = np.sqrt(self.model[id_].covariances_).reshape((1, self.n_clusters))
                features = (current - means) / (2 * stds)

                probs = self.model[id_].predict_proba(current.reshape([-1, 1]))
                argmax = np.argmax(probs, axis=1)
                idx = np.arange((len(features)))
                features = features[idx, argmax].reshape([-1, 1])

                features = np.clip(features, -.99, .99)

                values += [features, probs]
            else:
                col_t = np.zeros([len(data), info['size']])
                col_t[np.arange(len(data)), current.astype('int32')] = 1
                values.append(col_t)

        return np.concatenate(values, axis=1)

    def inverse_transform(self, data, sigmas):
        data_t = np.zeros([len(data), len(self.meta)])

        st = 0
        for id_, info in enumerate(self.meta):
            if info['type'] == CONTINUOUS:
                u = data[:, st]
                v = data[:, st+1:st+1+self.n_clusters]
                if sigmas is not None:
                    sig = sigmas[st]
                    u = np.random.normal(u, sig)
                u = np.clip(u, -1, 1)
                st += 1 + self.n_clusters
                means = self.model[id_].means_.reshape([-1])
                stds = np.sqrt(self.model[id_].covariances_).reshape([-1])
                p_argmax = np.argmax(v, axis=1)
                std_t = stds[p_argmax]
                mean_t = means[p_argmax]
                tmp = u * 2 * std_t  + mean_t
                data_t[:, id_] = tmp
            else:
                current = data[:, st:st+info['size']]
                st += info['size']
                data_t[:, id_] = np.argmax(current, axis=1)
        return data_t
