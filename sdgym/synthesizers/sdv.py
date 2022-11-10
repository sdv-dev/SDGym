import abc
import logging

import sdv
import sdv.timeseries

from sdgym.synthesizers.base import BaselineSynthesizer, SingleTableBaselineSynthesizer
from sdgym.synthesizers.utils import select_device

LOGGER = logging.getLogger(__name__)


class FastMLPreset(SingleTableBaselineSynthesizer):

    MODALITIES = ('single-table', )
    _MODEL = None
    _MODEL_KWARGS = None

    def _get_trained_synthesizer(self, data, metadata):
        model_kwargs = self._MODEL_KWARGS.copy() if self._MODEL_KWARGS else {}
        model = sdv.lite.TabularPreset(name='FAST_ML', metadata=metadata, **model_kwargs)
        model.fit(data)

        return model

    def _sample_from_synthesizer(self, synthesizer, n_samples):
        return synthesizer.sample(n_samples)


class SDVTabularSynthesizer(SingleTableBaselineSynthesizer, abc.ABC):

    MODALITIES = ('single-table', )
    _MODEL = None
    _MODEL_KWARGS = None

    def _get_trained_synthesizer(self, data, metadata):
        LOGGER.info('Fitting %s', self.__class__.__name__)
        model_kwargs = self._MODEL_KWARGS.copy() if self._MODEL_KWARGS else {}
        model = self._MODEL(table_metadata=metadata, **model_kwargs)
        model.fit(data)
        return model

    def _sample_from_synthesizer(self, synthesizer, n_samples):
        LOGGER.info('Sampling %s', self.__class__.__name__)
        return synthesizer.sample(n_samples)


class GaussianCopulaSynthesizer(SDVTabularSynthesizer):

    _MODEL = sdv.tabular.GaussianCopula


class CUDATabularSynthesizer(SDVTabularSynthesizer, abc.ABC):

    def _get_trained_synthesizer(self, data, metadata):
        model_kwargs = self._MODEL_KWARGS.copy() if self._MODEL_KWARGS else {}
        model_kwargs.setdefault('cuda', select_device())
        LOGGER.info('Fitting %s with kwargs %s', self.__class__.__name__, model_kwargs)
        model = self._MODEL(table_metadata=metadata, **model_kwargs)
        model.fit(data)
        return model

    def _sample_from_synthesizer(self, synthesizer, n_samples):
        LOGGER.info('Sampling %s', self.__class__.__name__)
        return synthesizer.sample(n_samples)


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

    def _get_trained_synthesizer(self, data, metadata):
        LOGGER.info('Fitting %s', self.__class__.__name__)
        model_kwargs = self._MODEL_KWARGS.copy() if self._MODEL_KWARGS else {}
        model = self._MODEL(metadata=metadata, **model_kwargs)
        model.fit(data)
        return model

    def _sample_from_synthesizer(self, synthesizer, n_samples):
        LOGGER.info('Sampling %s', self.__class__.__name__)
        return synthesizer.sample()


class HMASynthesizer(SDVRelationalSynthesizer):

    _MODEL = sdv.relational.HMA1


class SDVTimeseriesSynthesizer(SingleTableBaselineSynthesizer, abc.ABC):

    MODALITIES = ('timeseries', )
    _MODEL = None
    _MODEL_KWARGS = None

    def _get_trained_synthesizer(self, data, metadata):
        LOGGER.info('Fitting %s', self.__class__.__name__)
        model_kwargs = self._MODEL_KWARGS.copy() if self._MODEL_KWARGS else {}
        model = self._MODEL(table_metadata=metadata, **model_kwargs)
        model.fit(data)
        return model

    def _sample_from_synthesizer(self, synthesizer, n_samples):
        LOGGER.info('Sampling %s', self.__class__.__name__)
        return synthesizer.sample()


class PARSynthesizer(SDVTimeseriesSynthesizer):

    def _get_trained_synthesizer(self, data, metadata):
        LOGGER.info('Fitting %s', self.__class__.__name__)
        model = sdv.timeseries.PAR(table_metadata=metadata, epochs=1024, verbose=False)
        model.device = select_device()
        model.fit(data)
        return model

    def _sample_from_synthesizer(self, synthesizer, n_samples):
        LOGGER.info('Sampling %s', self.__class__.__name__)
        return synthesizer.sample()
