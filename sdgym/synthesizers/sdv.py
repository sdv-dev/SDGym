import logging

import sdv

from sdgym.synthesizers.base import Baseline, SingleTableBaseline
from sdgym.synthesizers.utils import select_device

LOGGER = logging.getLogger(__name__)


class SDV(Baseline):

    def fit_sample(self, data, metadata):
        LOGGER.info('Fitting SDV')
        model = sdv.SDV()
        model.fit(metadata, data)

        LOGGER.info('Sampling SDV')
        return model.sample_all()


class SDVTabular(SingleTableBaseline):

    _MODEL = None
    _MODEL_KWARGS = None

    def _fit_sample(self, data, metadata):
        LOGGER.info('Fitting %s', self.__class__.__name__)
        model_kwargs = self._MODEL_KWARGS.copy() if self._MODEL_KWARGS else {}
        model = self._MODEL(table_metadata=metadata, **model_kwargs)
        model.fit(data)

        LOGGER.info('Sampling %s', self.__class__.__name__)
        return model.sample()


class GaussianCopulaCategorical(SDVTabular):
    _MODEL = sdv.tabular.GaussianCopula
    _MODEL_KWARGS = {
        'categorical_transformer': 'categorical'
    }


class GaussianCopulaCategoricalFuzzy(SDVTabular):
    _MODEL = sdv.tabular.GaussianCopula
    _MODEL_KWARGS = {
        'categorical_transformer': 'categorical_fuzzy'
    }


class GaussianCopulaOneHot(SDVTabular):
    _MODEL = sdv.tabular.GaussianCopula
    _MODEL_KWARGS = {
        'categorical_transformer': 'one_hot_encoding'
    }


class CTGAN(SDVTabular):
    _MODEL = sdv.tabular.CTGAN

    def fit(self, data, metadata):
        self._MODEL_KWARGS = {'cuda': select_device()}
        super().fit(data, metadata)


class CopulaGAN(CTGAN):
    _MODEL = sdv.tabular.CopulaGAN


class SDVRelational(Baseline):

    def fit_sample(self, data, metadata):
        LOGGER.info('Fitting %s', self.__class__.__name__)
        model_kwargs = self._MODEL_KWARGS.copy() if self._MODEL_KWARGS else {}
        model = self._MODEL(metadata=metadata, **model_kwargs)
        model.fit(data)

        LOGGER.info('Sampling %s', self.__class__.__name__)
        return model.sample()


class HMA1(SDVRelational):

    _MODEL = sdv.relational.HMA1
