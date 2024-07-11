"""ColumnSynthesizer module."""

import pandas as pd
from rdt.hyper_transformer import HyperTransformer
from sklearn.mixture import GaussianMixture

from sdgym.synthesizers.base import BaselineSynthesizer


class ColumnSynthesizer(BaselineSynthesizer):
    """Synthesizer that learns each column independently.

    Categorical columns are sampled using empirical frequencies.
    Continuous columns are learned and sampled using a GMM.
    """

    def _get_trained_synthesizer(self, real_data, metadata):
        hyper_transformer = HyperTransformer()
        hyper_transformer.detect_initial_config(real_data)

        # This is done to match the behavior of the synthesizer for SDGym <= 0.6.0
        columns_to_remove = [
            column_name for column_name, data in real_data.items() if data.dtype.kind in {'O', 'i'}
        ]
        hyper_transformer.remove_transformers(columns_to_remove)

        hyper_transformer.fit(real_data)
        transformed = hyper_transformer.transform(real_data)

        self.length = len(real_data)
        gm_models = {}
        for name, column in transformed.items():
            kind = column.dtype.kind
            if kind != 'O':
                num_components = min(column.nunique(), 5)
                model = GaussianMixture(num_components)
                model.fit(column.to_numpy().reshape(-1, 1))
                gm_models[name] = model

        return (hyper_transformer, transformed, gm_models)

    def _sample_from_synthesizer(self, synthesizer, n_samples):
        hyper_transformer, transformed, gm_models = synthesizer
        sampled = pd.DataFrame()
        for name, column in transformed.items():
            kind = column.dtype.kind
            if kind == 'O':
                values = column.sample(self.length, replace=True).to_numpy()
            else:
                model = gm_models.get(name)
                values = model.sample(self.length)[0].ravel().clip(column.min(), column.max())

            sampled[name] = values

        return hyper_transformer.reverse_transform(sampled)
