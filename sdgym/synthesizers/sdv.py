import pandas as pd
from sdv import tabular

from sdgym.synthesizers.base import BaseSynthesizer
from sdgym.synthesizers.utils import select_device


class SDVTabular(BaseSynthesizer):

    _MODEL = None
    _MODEL_KWARGS = None

    def fit(self, data, categorical_columns, ordinal_columns):
        data = pd.DataFrame(data).astype('float64')
        field_types = {
            str(column): (
                {
                    'type': 'categorical'
                }
                if column in categorical_columns else
                {
                    'type': 'numerical',
                    'subtype': 'integer'
                }
                if column in ordinal_columns else
                {
                    'type': 'numerical',
                    'subtype': 'float'
                }
            )
            for column in data.columns
        }
        data.columns = data.columns.astype(str)
        self._model = self._MODEL(field_types=field_types, **self._MODEL_KWARGS.copy())
        self._model.fit(data)

    def sample(self, num_rows):
        return self._model.sample(num_rows).values.astype('float32')


class GaussianCopulaCategorical(SDVTabular):
    _MODEL = tabular.GaussianCopula
    _MODEL_KWARGS = {
        'categorical_transformer': 'categorical'
    }


class GaussianCopulaCategoricalFuzzy(SDVTabular):
    _MODEL = tabular.GaussianCopula
    _MODEL_KWARGS = {
        'categorical_transformer': 'categorical_fuzzy'
    }


class GaussianCopulaOneHot(SDVTabular):
    _MODEL = tabular.GaussianCopula
    _MODEL_KWARGS = {
        'categorical_transformer': 'one_hot_encoding'
    }


class CTGAN(SDVTabular):
    _MODEL = tabular.CTGAN
    _MODEL_KWARGS = {}

    def fit(self, data, categorical_columns, ordinal_columns):
        self._MODEL_KWARGS = {'cuda': select_device()}
        super().fit(data, categorical_columns, ordinal_columns)


class CopulaGAN(CTGAN):
    _MODEL = tabular.CopulaGAN
    _MODEL_KWARGS = {}
