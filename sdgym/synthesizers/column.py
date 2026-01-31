"""ColumnSynthesizer module."""

import logging

import pandas as pd
from rdt.hyper_transformer import HyperTransformer
from sdv.metadata import Metadata
from sklearn.mixture import GaussianMixture

from sdgym.synthesizers.base import BaselineSynthesizer

LOGGER = logging.getLogger(__name__)


class ColumnSynthesizer(BaselineSynthesizer):
    """Synthesizer that learns each column independently.

    Categorical columns are sampled using empirical frequencies.
    Continuous columns are learned and sampled using a GMM.
    """

    _MODALITY_FLAG = 'single_table'

    def _fit(self, data, metadata):
        hyper_transformer = HyperTransformer()
        hyper_transformer.detect_initial_config(data)
        supported_sdtypes = hyper_transformer._get_supported_sdtypes()
        config = {}
        if isinstance(metadata, Metadata):
            table_name = metadata._get_single_table_name()
            columns = metadata.tables[table_name].columns
        else:
            columns = metadata.columns

        for column_name, column in columns.items():
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
            column_name for column_name, data in data.items() if data.dtype.kind in {'O', 'i'}
        ]
        hyper_transformer.remove_transformers(columns_to_remove)

        hyper_transformer.fit(data)
        transformed = hyper_transformer.transform(data)

        self.length = len(data)
        gm_models = {}
        for name, column in transformed.items():
            kind = column.dtype.kind
            if kind != 'O':
                num_components = min(column.nunique(), 5)
                model = GaussianMixture(num_components)
                model.fit(column.to_numpy().reshape(-1, 1))
                gm_models[name] = model

        self.hyper_transformer = hyper_transformer
        self.transformed_data = transformed
        self.gm_models = gm_models

    def _sample_from_synthesizer(self, synthesizer, n_samples):
        hyper_transformer = synthesizer.hyper_transformer
        transformed = synthesizer.transformed_data
        gm_models = synthesizer.gm_models
        sampled = pd.DataFrame()
        for name, column in transformed.items():
            kind = column.dtype.kind
            if kind == 'O':
                values = column.sample(n_samples, replace=True).to_numpy()
            else:
                model = gm_models.get(name)
                values = model.sample(n_samples)[0].ravel().clip(column.min(), column.max())

            sampled[name] = values

        return hyper_transformer.reverse_transform(sampled)
