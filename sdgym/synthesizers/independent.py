import pandas as pd
from sdv.metadata import Table
from sklearn.mixture import GaussianMixture

from sdgym.synthesizers.base import MultiSingleTableBaselineSynthesizer


class IndependentSynthesizer(MultiSingleTableBaselineSynthesizer):
    """Synthesizer that learns each column independently.

    Categorical columns are sampled using empirical frequencies.
    Continuous columns are learned and sampled using a GMM.
    """

    def _get_trained_synthesizer(self, real_data, metadata):
        metadata = Table(metadata, dtype_transformers={'O': None, 'i': None})
        metadata.fit(real_data)
        transformed = metadata.transform(real_data)
        self.length = len(real_data)

        gm_models = {}
        for name, column in transformed.items():
            kind = column.dtype.kind
            if kind != 'O':
                num_components = min(column.nunique(), 5)
                model = GaussianMixture(num_components)
                model.fit(column.values.reshape(-1, 1))
                gm_models[name] = model

        return (metadata, transformed, gm_models)

    def _sample_from_synthesizer(self, synthesizer, n_samples):
        metadata, transformed, gm_models = synthesizer
        sampled = pd.DataFrame()
        for name, column in transformed.items():
            kind = column.dtype.kind
            if kind == 'O':
                values = column.sample(self.length, replace=True).values
            else:
                model = gm_models.get(name)
                values = model.sample(self.length)[0].ravel().clip(column.min(), column.max())

            sampled[name] = values

        return metadata.reverse_transform(sampled)
