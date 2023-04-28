import pandas as pd
from rdt.hyper_transformer import HyperTransformer
from sklearn.mixture import GaussianMixture

from sdgym.synthesizers.base import SingleTableBaselineSynthesizer


class IndependentSynthesizer(SingleTableBaselineSynthesizer):
    """Synthesizer that learns each column independently.

    Columns are learned and sampled using a GMM.
    """

    def _get_trained_synthesizer(self, real_data, metadata):
        hyper_transformer = HyperTransformer()
        hyper_transformer.detect_initial_config(real_data)
        hyper_transformer.fit(real_data)
        transformed = hyper_transformer.transform(real_data)

        gm_models = {}
        for name, column in transformed.items():
            num_components = min(column.nunique(), 5)
            model = GaussianMixture(num_components)
            model.fit(column.values.reshape(-1, 1))
            gm_models[name] = model

        return (hyper_transformer, transformed, gm_models)

    def _sample_from_synthesizer(self, synthesizer, n_samples):
        hyper_transformer, transformed, gm_models = synthesizer
        sampled = pd.DataFrame()
        for name, column in transformed.items():
            model = gm_models.get(name)
            values = model.sample(n_samples)[0].ravel().clip(column.min(), column.max())

            sampled[name] = values

        return hyper_transformer.reverse_transform(sampled)
