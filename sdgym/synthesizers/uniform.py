import numpy as np
import pandas as pd
from sdv.metadata import Table

from sdgym.synthesizers.base import MultiSingleTableBaseline


class Uniform(MultiSingleTableBaseline):
    """Synthesizer that samples each column using a Uniform distribution."""

    @staticmethod
    def _fit_sample(real_data, metadata):
        metadata = Table(metadata, dtype_transformers={'O': None, 'i': None})
        metadata.fit(real_data)
        transformed = metadata.transform(real_data)

        sampled = pd.DataFrame()
        length = len(real_data)
        for name, column in transformed.items():
            kind = column.dtype.kind
            if kind == 'i':
                values = np.random.randint(column.min(), column.max() + 1, size=length)
            elif kind == 'O':
                values = np.random.choice(column.unique(), size=length)
            else:
                values = np.random.uniform(column.min(), column.max(), size=length)

            sampled[name] = values

        return metadata.reverse_transform(sampled)
