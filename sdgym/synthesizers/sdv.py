import abc
import logging

import sdv
import sdv.sequential
from sdv.metadata.single_table import SingleTableMetadata

from sdgym.synthesizers.base import BaselineSynthesizer, BaselineSynthesizer
from sdgym.utils import select_device

LOGGER = logging.getLogger(__name__)


class FastMLPreset(BaselineSynthesizer):

    _MODEL = None
    _MODEL_KWARGS = None

    def _get_trained_synthesizer(self, data, metadata):
        metadata = SingleTableMetadata().load_from_dict(metadata)
        model = sdv.lite.SingleTablePreset(name='FAST_ML', metadata=metadata)
        model.fit(data)

        return model

    def _sample_from_synthesizer(self, synthesizer, n_samples):
        return synthesizer.sample(n_samples)


class SDVTabularSynthesizer(BaselineSynthesizer, abc.ABC):

    _MODEL = None
    _MODEL_KWARGS = None

    def _get_trained_synthesizer(self, data, metadata):
        metadata = SingleTableMetadata().load_from_dict(metadata)
        LOGGER.info('Fitting %s', self.__class__.__name__)
        model_kwargs = self._MODEL_KWARGS.copy() if self._MODEL_KWARGS else {}
        model = self._MODEL(metadata=metadata, **model_kwargs)
        model.fit(data)
        return model

    def _sample_from_synthesizer(self, synthesizer, n_samples):
        LOGGER.info('Sampling %s', self.__class__.__name__)
        return synthesizer.sample(n_samples)


class GaussianCopulaSynthesizer(SDVTabularSynthesizer):

    _MODEL = sdv.single_table.GaussianCopulaSynthesizer


class CUDATabularSynthesizer(SDVTabularSynthesizer, abc.ABC):

    def _get_trained_synthesizer(self, data, metadata):
        metadata = SingleTableMetadata().load_from_dict(metadata)
        model_kwargs = self._MODEL_KWARGS.copy() if self._MODEL_KWARGS else {}
        model_kwargs.setdefault('cuda', select_device())
        LOGGER.info('Fitting %s with kwargs %s', self.__class__.__name__, model_kwargs)
        model = self._MODEL(metadata=metadata, **model_kwargs)
        model.fit(data)
        return model

    def _sample_from_synthesizer(self, synthesizer, n_samples):
        LOGGER.info('Sampling %s', self.__class__.__name__)
        return synthesizer.sample(n_samples)


class CTGANSynthesizer(CUDATabularSynthesizer):

    _MODEL = sdv.single_table.CTGANSynthesizer


class TVAESynthesizer(CUDATabularSynthesizer):

    _MODEL = sdv.single_table.TVAESynthesizer


class CopulaGANSynthesizer(CUDATabularSynthesizer):

    _MODEL = sdv.single_table.CopulaGANSynthesizer


class SDVRelationalSynthesizer(BaselineSynthesizer, abc.ABC):

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

    _MODEL = sdv.multi_table.hma.HMASynthesizer


class SDVTimeseriesSynthesizer(BaselineSynthesizer, abc.ABC):

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


class PARSynthesizer(SDVTimeseriesSynthesizer):

    def _get_trained_synthesizer(self, data, metadata):
        LOGGER.info('Fitting %s', self.__class__.__name__)
        model = sdv.sequential.PARSynthesizer(metadata=metadata, epochs=1024, verbose=False)
        model.device = select_device()
        model.fit(data)
        return model

    def _sample_from_synthesizer(self, synthesizer, n_samples):
        LOGGER.info('Sampling %s', self.__class__.__name__)
        return synthesizer.sample()
