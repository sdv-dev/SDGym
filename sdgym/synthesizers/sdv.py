import logging

import sdv
import sdv.timeseries

from sdgym.synthesizers.base import Baseline, SingleTableBaseline
from sdgym.synthesizers.utils import select_device

LOGGER = logging.getLogger(__name__)


class SDV(Baseline):

    MODALITIES = ('single-table', 'multi-table')

    def fit_sample(self, data, metadata):
        LOGGER.info('Fitting SDV')
        model = sdv.SDV()
        model.fit(metadata, data)

        LOGGER.info('Sampling SDV')
        return model.sample_all()


class SDVTabular(SingleTableBaseline):

    MODALITIES = ('single-table', )
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


class CUDATabular(SDVTabular):

    def _fit_sample(self, data, metadata):
        LOGGER.info('Fitting %s', self.__class__.__name__)
        model_kwargs = self._MODEL_KWARGS.copy() if self._MODEL_KWARGS else {}
        model_kwargs.setdefault('cuda', select_device())
        model = self._MODEL(table_metadata=metadata, **model_kwargs)
        model.fit(data)

        LOGGER.info('Sampling %s', self.__class__.__name__)
        return model.sample()


class CTGAN(CUDATabular):

    _MODEL = sdv.tabular.CTGAN


class TVAE(CUDATabular):

    _MODEL = sdv.tabular.TVAE


class CopulaGAN(CUDATabular):

    _MODEL = sdv.tabular.CopulaGAN


class SDVRelational(Baseline):

    MODALITIES = ('single-table', 'multi-table')
    _MODEL = None
    _MODEL_KWARGS = None

    def fit_sample(self, data, metadata):
        LOGGER.info('Fitting %s', self.__class__.__name__)
        model_kwargs = self._MODEL_KWARGS.copy() if self._MODEL_KWARGS else {}
        model = self._MODEL(metadata=metadata, **model_kwargs)
        model.fit(data)

        LOGGER.info('Sampling %s', self.__class__.__name__)
        return model.sample()


class HMA1(SDVRelational):

    _MODEL = sdv.relational.HMA1


class SDVTimeseries(SingleTableBaseline):

    MODALITIES = ('timeseries', )
    _MODEL = None
    _MODEL_KWARGS = None

    def _fit_sample(self, data, metadata):
        LOGGER.info('Fitting %s', self.__class__.__name__)
        model_kwargs = self._MODEL_KWARGS.copy() if self._MODEL_KWARGS else {}
        model = self._MODEL(table_metadata=metadata, **model_kwargs)
        model.fit(data)

        LOGGER.info('Sampling %s', self.__class__.__name__)
        return model.sample()


class PAR(SDVTimeseries):

    def _fit_sample(self, data, metadata):
        LOGGER.info('Fitting %s', self.__class__.__name__)
        model = sdv.timeseries.PAR(table_metadata=metadata, epochs=1024, verbose=False)
        model.device = select_device()
        model.fit(data)

        LOGGER.info('Sampling %s', self.__class__.__name__)
        return model.sample()
