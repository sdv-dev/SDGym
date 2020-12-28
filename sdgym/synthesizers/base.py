import logging

import pandas as pd
import rdt

LOGGER = logging.getLogger(__name__)


class BaseSynthesizer:
    """Base class for all default synthesizers of ``SDGym``."""

    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        pass

    def sample(self, samples):
        pass

    def fit_sample(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        LOGGER.info("Fitting %s", self.__class__.__name__)
        self.fit(data, categorical_columns, ordinal_columns)

        LOGGER.info("Sampling %s", self.__class__.__name__)
        return self.sample(data.shape[0])


class Baseline:
    """Base class for all the ``SDGym`` baselines."""

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

    Sublcasses can choose to implement ``_fit_sample``, which will
    always be called with DataFrames and Table metadata dicts, or
    to overwrite the ``fit_sample`` method, which may be called with
    either DataFrames and Table dicts, or with dicts of tables and
    dataset metadata dicts.
    """

    def fit_sample(self, real_data, metadata):
        if isinstance(real_data, dict):
            return {
                table_name: self._fit_sample(table, metadata.get_table_meta(table_name))
                for table_name, table in real_data.items()
            }

        return self._fit_sample(real_data, metadata)


class LegacySingleTableBaseline(SingleTableBaseline):
    """Single table baseline which passes ordinals and categoricals down.

    This class exists here to support the legacy baselines which do not operate
    on metadata and instead expect lists of categorical and ordinal columns.
    """

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

        ht = rdt.HyperTransformer(dtype_transformers={
            'O': 'label_encoding',
            'M': None,
        })
        model_data = ht.fit_transform(real_data[columns])

        LOGGER.info("Fitting %s", self.__class__.__name__)
        self.fit(model_data.to_numpy(), categoricals, ())

        LOGGER.info("Sampling %s", self.__class__.__name__)
        sampled_data = self.sample(len(model_data))
        sampled_data = pd.DataFrame(sampled_data, columns=columns)

        synthetic_data = real_data.copy()
        synthetic_data.update(ht.reverse_transform(sampled_data))
        return synthetic_data
