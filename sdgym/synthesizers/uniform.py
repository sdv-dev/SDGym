import numpy as np
import pandas as pd
from rdt.hyper_transformer import HyperTransformer

from sdgym.synthesizers.base import SingleTableBaselineSynthesizer


class UniformSynthesizer(SingleTableBaselineSynthesizer):
    """Synthesizer that samples each column using a Uniform distribution."""

    def _get_trained_synthesizer(self, real_data, metadata):
        hyper_transformer = HyperTransformer()  # TODO: Update this to match original synthesizer
        hyper_transformer.detect_initial_config(real_data)
        hyper_transformer.fit(real_data)
        transformed = hyper_transformer.transform(real_data)

        self.length = len(real_data)
        return (hyper_transformer, transformed)

    def _sample_from_synthesizer(self, synthesizer, n_samples):
        hyper_transformer, transformed = synthesizer
        sampled = pd.DataFrame()
        for name, column in transformed.items():
            kind = column.dtype.kind
            if kind == 'i':
                values = np.random.randint(column.min(), column.max() + 1, size=self.length)
            elif kind == 'O':
                values = np.random.choice(column.unique(), size=self.length)
            else:
                values = np.random.uniform(column.min(), column.max(), size=self.length)

            sampled[name] = values

        return hyper_transformer.reverse_transform(sampled)
