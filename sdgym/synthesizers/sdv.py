import abc
import logging

import sdv
import sdv.timeseries

from sdgym.synthesizers.base import BaselineSynthesizer, SingleTableBaselineSynthesizer
from sdgym.synthesizers.utils import select_device

LOGGER = logging.getLogger(__name__)


class SDVSynthesizer(BaselineSynthesizer, abc.ABC):

    MODALITIES = ('single-table', 'multi-table')

    def fit_sample(self, data, metadata):
        LOGGER.info('Fitting SDV')
        model = sdv.SDV()
        model.fit(metadata, data)

        LOGGER.info('Sampling SDV')
        return model.sample_all()


class SDVTabularSynthesizer(SingleTableBaselineSynthesizer, abc.ABC):

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


class GaussianCopulaSynthesizer(SDVTabularSynthesizer):

    _MODEL = sdv.tabular.GaussianCopula


class CUDATabularSynthesizer(SDVTabularSynthesizer, abc.ABC):

    def _fit_sample(self, data, metadata):
        model_kwargs = self._MODEL_KWARGS.copy() if self._MODEL_KWARGS else {}
        model_kwargs.setdefault('cuda', select_device())
        LOGGER.info('Fitting %s with kwargs %s', self.__class__.__name__, model_kwargs)
        model = self._MODEL(table_metadata=metadata, **model_kwargs)
        model.fit(data)

        LOGGER.info('Sampling %s', self.__class__.__name__)
        return model.sample()


class CTGANSynthesizer(CUDATabularSynthesizer):

    _MODEL = sdv.tabular.CTGAN


class TVAESynthesizer(CUDATabularSynthesizer):

    _MODEL = sdv.tabular.TVAE


class CopulaGANSynthesizer(CUDATabularSynthesizer):

    _MODEL = sdv.tabular.CopulaGAN


class SDVRelationalSynthesizer(BaselineSynthesizer, abc.ABC):

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


class HMASynthesizer(SDVRelationalSynthesizer):

    _MODEL = sdv.relational.HMA1


class SDVTimeseriesSynthesizer(SingleTableBaselineSynthesizer, abc.ABC):

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


class PARSynthesizer(SDVTimeseriesSynthesizer):

    def _fit_sample(self, data, metadata):
        LOGGER.info('Fitting %s', self.__class__.__name__)
        model = sdv.timeseries.PAR(table_metadata=metadata, epochs=1024, verbose=False)
        model.device = select_device()
        model.fit(data)

        LOGGER.info('Sampling %s', self.__class__.__name__)
        return model.sample()
