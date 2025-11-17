"""UniformSynthesizer module."""

import logging
import warnings

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
        table = next(iter(metadata.tables.values()))
        for column_name, column in table.columns.items():
            sdtype = column['sdtype']
            if sdtype in supported_sdtypes:
                config[column_name] = sdtype
            elif column.get('pii', False):
                config[column_name] = 'pii'
            else:
                LOGGER.info(
                    f'Column {column_name} sdtype: {sdtype} is not supported, '
                    f'defaulting to inferred type.'
                )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message='.*is incompatible with transformer.*',
                category=UserWarning,
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

        return (hyper_transformer, transformed)

    def _sample_from_synthesizer(self, synthesizer, n_samples):
        hyper_transformer, transformed = synthesizer
        sampled = pd.DataFrame()
        for name, column in transformed.items():
            kind = column.dtype.kind
            if kind == 'i':
                values = np.random.randint(column.min(), column.max() + 1, size=n_samples)
            elif kind in ['O', 'b']:
                values = np.random.choice(column.unique(), size=n_samples)
            else:
                values = np.random.uniform(column.min(), column.max(), size=n_samples)
            sampled[name] = values

        return hyper_transformer.reverse_transform(sampled)


class MultiTableUniformSynthesizer(BaselineSynthesizer):
    """Synthesizer that uses UniformSynthesizer for multi-table data."""

    _MODALITY_FLAG = 'multi_table'

    def __init__(self):
        super().__init__()
        self.num_rows_per_table = {}

    def _get_trained_synthesizer(self, data, metadata):
        """This function should train single table UniformSynthesizers on each table in the data.

        Args:
            data (dict):
                A dict mapping table name to table data.
            metadata (sdv.metadata.Metadata):
                The metadata

        Returns:
            A dict mapping table name to trained UniformSynthesizer instance.
        """
        synthesizers = {}
        for table_name, table_data in data.items():
            self.num_rows_per_table[table_name] = len(table_data)
            table_metadata = metadata.get_table_metadata(table_name)
            synthesizer = UniformSynthesizer()
            trained_synthesizer = synthesizer._get_trained_synthesizer(table_data, table_metadata)
            synthesizers[table_name] = trained_synthesizer

        return synthesizers

    def sample_from_synthesizer(self, synthesizer, scale=1.0):
        """Sample data from the provided synthesizer.

        Args:
            synthesizer (dict[table_name, UniformSynthesizer]):
                Dict mapping table name to trained single-table UniformSynthesizer.
                This is the output of `get_trained_synthesizer`.
            scale (float):
                The scale of data to sample.
                Defaults to 1.0.

        Returns:
            dict:  A dict mapping table name to the sampled data.
        """
        sampled_data = {}
        for table_name, table_synthesizer in synthesizer.items():
            n_samples = int(self.num_rows_per_table[table_name] * scale)
            sampled_table = UniformSynthesizer().sample_from_synthesizer(
                table_synthesizer, n_samples
            )
            sampled_data[table_name] = sampled_table

        return sampled_data
