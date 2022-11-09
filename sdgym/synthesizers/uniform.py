import numpy as np
import pandas as pd
from sdv.metadata import Table

from sdgym.synthesizers.base import MultiSingleTableBaselineSynthesizer


class UniformSynthesizer(MultiSingleTableBaselineSynthesizer):
    """Synthesizer that samples each column using a Uniform distribution."""

    def _get_trained_synthesizer(self, real_data, metadata):
        metadata = Table(metadata, dtype_transformers={'O': None, 'i': None})
        metadata.fit(real_data)
        transformed = metadata.transform(real_data)
        self.length = len(real_data)
        return (metadata, transformed)

    def _sample_from_synthesizer(self, synthesizer, n_samples):
        metadata, transformed = synthesizer
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

        return metadata.reverse_transform(sampled)
