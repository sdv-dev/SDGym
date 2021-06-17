import logging

import pandas as pd
import rdt

from sdgym.errors import UnsupportedDataset

LOGGER = logging.getLogger(__name__)


class Baseline:
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

    def fit_sample(self, real_data, metadata):
        pass


class SingleTableBaseline(Baseline):
    """Base class for all the SingleTable Baselines.

    Subclasses can choose to implement ``_fit_sample``, which will
    always be called with DataFrames and Table metadata dicts, or
    to overwrite the ``fit_sample`` method, which may be called with
    either DataFrames and Table dicts, or with dicts of tables and
    dataset metadata dicts.
    """

    MODALITIES = ('single-table', )
    CONVERT_TO_NUMERIC = False

    def _transform_fit_sample(self, real_data, metadata):
        ht = rdt.HyperTransformer()
        columns_to_transform = list()
        fields_metadata = metadata['fields']
        id_fields = list()
        for field in fields_metadata:
            if fields_metadata.get(field).get('type') != 'id':
                columns_to_transform.append(field)
            else:
                id_fields.append(field)

        ht.fit(real_data[columns_to_transform])
        transformed_data = ht.transform(real_data)
        synthetic_data = self._fit_sample(transformed_data, metadata)
        reverse_transformed_synthetic_data = ht.reverse_transform(synthetic_data)
        reverse_transformed_synthetic_data[id_fields] = real_data[id_fields]
        return reverse_transformed_synthetic_data

    def fit_sample(self, real_data, metadata):
        _fit_sample = self._transform_fit_sample if self.CONVERT_TO_NUMERIC else self._fit_sample
        if isinstance(real_data, dict):
            return {
                table_name: _fit_sample(table, metadata.get_table_meta(table_name))
                for table_name, table in real_data.items()
            }

        return _fit_sample(real_data, metadata)


class MultiSingleTableBaseline(Baseline):
    """Base class for SingleTableBaselines that are used on multi table scenarios.

    These classes model and sample each table independently and then just
    randomly choose ids from the parent tables to form the relationships.
    """

    MODALITIES = ('multi-table', 'single-table')

    def fit_sample(self, real_data, metadata):
        if isinstance(real_data, dict):
            tables = {
                table_name: self._fit_sample(table, metadata.get_table_meta(table_name))
                for table_name, table in real_data.items()
            }

            for table_name, table in tables.items():
                parents = metadata.get_parents(table_name)
                for parent_name in parents:
                    parent = tables[parent_name]
                    primary_key = metadata.get_primary_key(parent_name)
                    foreign_keys = metadata.get_foreign_keys(parent_name, table_name)
                    length = len(table)
                    for foreign_key in foreign_keys:
                        foreign_key_values = parent[primary_key].sample(length, replace=True)
                        table[foreign_key] = foreign_key_values.values

                tables[table_name] = table[real_data[table_name].columns]

            return tables

        return self._fit_sample(real_data, metadata)


class LegacySingleTableBaseline(SingleTableBaseline):
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

    def _fit_sample(self, real_data, table_metadata):
        columns, categoricals = self._get_columns(real_data, table_metadata)
        real_data = real_data[columns]

        ht = rdt.HyperTransformer(dtype_transformers={
            'O': 'label_encoding',
        })
        ht.fit(real_data.iloc[:, categoricals])
        model_data = ht.transform(real_data)

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
        self.fit(model_data.to_numpy(), categoricals, ())

        LOGGER.info("Sampling %s", self.__class__.__name__)
        sampled_data = self.sample(len(model_data))
        sampled_data = pd.DataFrame(sampled_data, columns=columns)

        synthetic_data = real_data.copy()
        synthetic_data.update(ht.reverse_transform(sampled_data))
        return synthetic_data
