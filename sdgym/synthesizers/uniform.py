"""UniformSynthesizer module."""

import logging

import numpy as np
import pandas as pd
from rdt.hyper_transformer import HyperTransformer

from sdgym.synthesizers.base import BaselineSynthesizer

LOGGER = logging.getLogger(__name__)


class UniformSynthesizer(BaselineSynthesizer):
    """Synthesizer that samples each column using a Uniform distribution."""

    def _get_trained_synthesizer(self, real_data, metadata):
        hyper_transformer = HyperTransformer()
        hyper_transformer.detect_initial_config(real_data)
        supported_sdtypes = hyper_transformer._get_supported_sdtypes()
        config = {}
        for column_name, column in metadata.columns.items():
            sdtype = column['sdtype']
            if sdtype in supported_sdtypes:
                config[column_name] = sdtype
            elif column.get('pii', False):
                config[column_name] = 'pii'
            else:
                LOGGER.info(
                    f'Column {column} sdtype: {sdtype} is not supported, '
                    f'defaulting to inferred type.'
                )

        hyper_transformer.update_sdtypes(config)
        # This is done to match the behavior of the synthesizer for SDGym <= 0.6.0
        columns_to_remove = [
            column_name
            for column_name, data in real_data.items()
            if data.dtype.kind in {'O', 'i', 'b'}
        ]
        hyper_transformer.remove_transformers(columns_to_remove)

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
            elif kind in ['O', 'b']:
                values = np.random.choice(column.unique(), size=self.length)
            else:
                values = np.random.uniform(column.min(), column.max(), size=self.length)

            sampled[name] = values

        return hyper_transformer.reverse_transform(sampled)
