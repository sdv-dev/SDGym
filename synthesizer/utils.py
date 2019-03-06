from sklearn.preprocessing import KBinsDiscretizer

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
