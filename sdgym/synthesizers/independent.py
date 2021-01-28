import pandas as pd
from sdv.metadata import Table
from sklearn.mixture import GaussianMixture

from sdgym.synthesizers.base import MultiSingleTableBaseline


class Independent(MultiSingleTableBaseline):
    """Synthesizer that learns each column independently.

    Categorical columns are sampled using empirical frequencies.
    Continuous columns are learned and sampled using a GMM.
    """

    @staticmethod
    def _fit_sample(real_data, metadata):
        metadata = Table(metadata, dtype_transformers={'O': None, 'i': None})
        metadata.fit(real_data)
        transformed = metadata.transform(real_data)

        sampled = pd.DataFrame()
        length = len(real_data)
        for name, column in transformed.items():
            kind = column.dtype.kind
            if kind == 'O':
                values = column.sample(length, replace=True).values
            else:
                model = GaussianMixture(5)
                model.fit(column.values.reshape(-1, 1))
                values = model.sample(length)[0].ravel().clip(column.min(), column.max())

            sampled[name] = values

        return metadata.reverse_transform(sampled)
