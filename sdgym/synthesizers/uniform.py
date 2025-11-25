"""UniformSynthesizer module."""

import logging
import warnings

import numpy as np
import pandas as pd
from rdt.hyper_transformer import HyperTransformer

from sdgym.synthesizers.base import BaselineSynthesizer, MultiTableBaselineSynthesizer

LOGGER = logging.getLogger(__name__)


class UniformSynthesizer(BaselineSynthesizer):
    """Synthesizer that samples each column using a Uniform distribution."""

    _MODALITY_FLAG = 'single_table'

    def __init__(self):
        super().__init__()
        self.hyper_transformer = None
        self.transformed_data = None

    def _fit(self, data, metadata):
        """Fit the synthesizer to the data.

        Args:
            data (pd.DataFrame):
                The data to fit the synthesizer to.
            metadata (sdv.metadata.Metadata):
                The metadata describing the data.
        """
        hyper_transformer = HyperTransformer()
        hyper_transformer.detect_initial_config(data)
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
            for column_name, column_data in data.items()
            if column_data.dtype.kind in {'O', 'i', 'b'}
        ]
        hyper_transformer.remove_transformers(columns_to_remove)

        hyper_transformer.fit(data)
        transformed = hyper_transformer.transform(data)
        self.hyper_transformer = hyper_transformer
        self.transformed_data = transformed

    def _sample_from_synthesizer(self, synthesizer, n_samples):
        hyper_transformer = synthesizer.hyper_transformer
        transformed = synthesizer.transformed_data
        sampled = pd.DataFrame()
        for name, column in transformed.items():
            kind = column.dtype.kind
            if kind == 'i':
                values = np.random.randint(
                    int(column.min()), int(column.max()) + 1, size=n_samples, dtype=np.int64
                )
            elif kind in ['O', 'b']:
                values = np.random.choice(column.unique(), size=n_samples)
            else:
                values = np.random.uniform(column.min(), column.max(), size=n_samples)
            sampled[name] = values

        return hyper_transformer.reverse_transform(sampled)


class MultiTableUniformSynthesizer(MultiTableBaselineSynthesizer):
    """Multi-table Uniform Synthesizer.

    This synthesizer trains a UniformSynthesizer on each table in the multi-table dataset.
    It samples data from each table independently using the corresponding trained synthesizer.
    """

    def __init__(self):
        super().__init__()
        self.num_rows_per_table = {}
        self.table_synthesizers = {}

    def _fit(self, data, metadata):
        """Fit the synthesizer to the multi-table data.

        Args:
            data (dict):
                A dict mapping table name to table data.
            metadata (sdv.metadata.MultiTableMetadata):
                The multi-table metadata describing the data.
        """
        for table_name, table_data in data.items():
            table_metadata = metadata.get_table_metadata(table_name)
            synthesizer = UniformSynthesizer()
            synthesizer._fit(table_data, table_metadata)
            self.num_rows_per_table[table_name] = len(table_data)
            self.table_synthesizers[table_name] = synthesizer

    def _sample_from_synthesizer(self, synthesizer, scale):
        """Sample data from the provided synthesizer.

        Args:
            synthesizer (SDGym synthesizer):
                The synthesizer object to sample data from.
            scale (float):
                The scale of data to sample.
                Defaults to 1.0.

        Returns:
            dict:  A dict mapping table name to the sampled data.
        """
        sampled_data = {}
        for table_name, table_synthesizer in synthesizer.table_synthesizers.items():
            n_samples = int(synthesizer.num_rows_per_table[table_name] * scale)
            sampled_table = UniformSynthesizer().sample_from_synthesizer(
                table_synthesizer, n_samples=n_samples
            )
            sampled_data[table_name] = sampled_table

        return sampled_data
