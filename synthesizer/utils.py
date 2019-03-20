from sklearn.preprocessing import KBinsDiscretizer
import numpy as np

CATEGORICAL = "categorical"
CONTINUOUS = "continuous"
ORDINAL = "ordinal"

class Discretizer(object):
    """docstring for Discretizer."""
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
            return data

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
