import abc
import logging

import pandas as pd
import rdt

from sdgym.errors import UnsupportedDataset

LOGGER = logging.getLogger(__name__)


class BaselineSynthesizer(abc.ABC):
    """Base class for all the ``SDGym`` baselines."""

    MODALITIES = ()

    @classmethod
    def get_subclasses(cls, include_parents=False):
        """Recursively find subclasses of this Baseline.

        Args:
            include_parents (bool):
                Whether to include subclasses which are parents to
                other classes. Defaults to ``False``.
        """
        subclasses = dict()
        for child in cls.__subclasses__():
            grandchildren = child.get_subclasses(include_parents)
            subclasses.update(grandchildren)
            if include_parents or not grandchildren:
                subclasses[child.__name__] = child

        return subclasses

    @classmethod
    def get_baselines(cls):
        subclasses = cls.get_subclasses(include_parents=True)
        synthesizers = []
        for _, subclass in subclasses.items():
            if abc.ABC not in subclass.__bases__:
                synthesizers.append(subclass)

        return synthesizers

    def get_trained_synthesizer(self, data, metadata):
        """Get a synthesizer that has been trained on the provided data and metadata.

        Args:
            data (pandas.DataFrame or dict):
                The data to train on.
            metadata (sdv.Metadata):
                The metadata.

        Returns:
            obj:
                The synthesizer object
        """

    def sample_from_synthesizer(synthesizer, n_samples):
        """Sample data from the provided synthesizer.

        Args:
            synthesizer (obj):
                The synthesizer object to sample data from.
            n_samples (int):
                The number of samples to create.

        Returns:
            pandas.DataFrame or dict:
                The sampled data. If single-table, should be a DataFrame. If multi-table,
                should be a dict mapping table name to DataFrame.
        """


class SingleTableBaselineSynthesizer(BaselineSynthesizer, abc.ABC):
    """Base class for all the SingleTable Baselines.

    Subclasses can choose to implement ``_fit_sample``, which will
    always be called with DataFrames and Table metadata dicts, or
    to overwrite the ``fit_sample`` method, which may be called with
    either DataFrames and Table dicts, or with dicts of tables and
    dataset metadata dicts.
    """

    MODALITIES = ('single-table', )
    CONVERT_TO_NUMERIC = False

    def _get_transformed_trained_synthesizer(self, real_data, metadata):
        self.ht = rdt.HyperTransformer()
        columns_to_transform = list()
        fields_metadata = metadata['fields']
        self.id_fields = list()
        for field in fields_metadata:
            if fields_metadata.get(field).get('type') != 'id':
                columns_to_transform.append(field)
            else:
                self.id_fields.append(field)

        self.id_field_values = real_data[self.id_fields]

        self.ht.fit(real_data[columns_to_transform])
        transformed_data = self.ht.transform(real_data)
        return self._get_trained_synthesizer(transformed_data, metadata)

    def _get_reverse_transformed_samples(self, data):
        synthetic_data = self._sample_from_synthesizer(data)
        reverse_transformed_synthetic_data = self.ht.reverse_transform(synthetic_data)
        reverse_transformed_synthetic_data[self.id_fields] = self.id_field_values
        return reverse_transformed_synthetic_data

    def get_trained_synthesizer(self, data, metadata):
        """Get a synthesizer that has been trained on the provided data and metadata.

        Args:
            data (pandas.DataFrame):
                The data to train on.
            metadata (sdv.Metadata):
                The metadata.

        Returns:
            obj:
                The synthesizer object
        """
        return self._get_transformed_trained_synthesizer(data, metadata) if (
            self.CONVERT_TO_NUMERIC) else self._get_trained_synthesizer(data, metadata)

    def sample_from_synthesizer(self, synthesizer, n_samples):
        """Sample data from the provided synthesizer.

        Args:
            synthesizer (obj):
                The synthesizer object to sample data from.
            n_samples (int):
                The number of samples to create.

        Returns:
            pandas.DataFrame:
                The sampled data.
        """
        return self._get_reverse_transformed_samples(synthesizer, n_samples) if (
            self.CONVERT_TO_NUMERIC) else self._sample_from_synthesizer(synthesizer, n_samples)


class MultiSingleTableBaselineSynthesizer(BaselineSynthesizer, abc.ABC):
    """Base class for SingleTableBaselines that are used on multi table scenarios.

    These classes model and sample each table independently and then just
    randomly choose ids from the parent tables to form the relationships.
    """

    MODALITIES = ('multi-table', 'single-table')

    def get_trained_synthesizer(self, data, metadata):
        """Get the trained synthesizer.

        Args:
            data (dict):
                A dict mapping table name to table data.
            metadata (sdv.Metadata):
                The multi-table metadata.

        Returns:
            dict:
                A mapping of table name to synthesizers.
        """
        self.metadata = metadata
        synthesizers = {
            table_name: self._get_trained_synthesizer(table, metadata.get_table_meta(table_name))
            for table_name, table in data.items()
        }
        self.table_columns = {table_name: data[table_name].columns for table_name in data.keys()}

        return synthesizers

    def sample_from_synthesizer(self, synthesizers, n_samples):
        """Sample from the given synthesizers.

        Args:
            synthesizers (dict):
                A dict mapping table name to table synthesizer.
            n_samples (int):
                The number of samples.

        Returns:
            dict:
                A mapping of table name to sampled table data.
        """
        tables = {
            table_name: self._sample_from_synthesizer(synthesizer, n_samples)
            for table_name, synthesizer in synthesizers.items()
        }

        for table_name, table in tables.items():
            parents = self.metadata.get_parents(table_name)
            for parent_name in parents:
                parent = tables[parent_name]
                primary_key = self.metadata.get_primary_key(parent_name)
                foreign_keys = self.metadata.get_foreign_keys(parent_name, table_name)
                for foreign_key in foreign_keys:
                    foreign_key_values = parent[primary_key].sample(len(table), replace=True)
                    table[foreign_key] = foreign_key_values.values

            tables[table_name] = table[self.table_columns[table_name]]

        return tables


class LegacySingleTableBaselineSynthesizer(SingleTableBaselineSynthesizer, abc.ABC):
    """Single table baseline which passes ordinals and categoricals down.

    This class exists here to support the legacy baselines which do not operate
    on metadata and instead expect lists of categorical and ordinal columns.
    """

    MODALITIES = ('single-table', )

    def _get_columns(self, real_data, table_metadata):
        model_columns = []
        categorical_columns = []

        fields_meta = table_metadata['fields']

        for column in real_data.columns:
            field_meta = fields_meta[column]
            field_type = field_meta['type']
            if field_type == 'id':
                continue

            index = len(model_columns)
            if field_type == 'categorical':
                categorical_columns.append(index)

            model_columns.append(column)

        return model_columns, categorical_columns

    def get_trained_synthesizer(self, data, metadata):
        """Get the trained synthesizer.

        Args:
            data (dict):
                A dict mapping table name to table data.
            metadata (sdv.Metadata):
                The multi-table metadata.

        Returns:
            dict:
                A mapping of table name to synthesizers.
        """
        self.columns, self.categoricals = self._get_columns(data, metadata)
        data = data[self.columns]

        if self.categoricals:
            self.ht = rdt.HyperTransformer(default_data_type_transformers={
                'categorical': 'LabelEncodingTransformer',
            })
            self.ht.fit(data.iloc[:, self.categoricals])
            model_data = self.ht.transform(data)
        else:
            model_data = data

        self.model_columns = model_data.columns

        supported = set(model_data.select_dtypes(('number', 'bool')).columns)
        unsupported = set(model_data.columns) - supported
        if unsupported:
            unsupported_dtypes = model_data[unsupported].dtypes.unique().tolist()
            raise UnsupportedDataset(f'Unsupported dtypes {unsupported_dtypes}')

        nulls = model_data.isnull().any()
        if nulls.any():
            unsupported_columns = nulls[nulls].index.tolist()
            raise UnsupportedDataset(f'Null values found in columns {unsupported_columns}')

        LOGGER.info("Fitting %s", self.__class__.__name__)
        self.fit(model_data.to_numpy(), self.categoricals, ())

    def sample_from_synthesizer(self, synthesizer, n_samples):
        """Sample from the given synthesizers.

        Args:
            synthesizer:
                The table synthesizer.
            n_samples (int):
                The number of samples.

        Returns:
            dict:
                A mapping of table name to sampled table data.
        """
        sampled_data = self.sample(n_samples)
        sampled_data = pd.DataFrame(sampled_data, columns=self.model_columns)

        if self.categoricals:
            sampled_data = self.ht.reverse_transform(sampled_data)

        return sampled_data
