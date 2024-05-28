"""SDV synthesizers module."""

import abc
import logging

import sdv

from sdgym.synthesizers.base import BaselineSynthesizer
from sdgym.utils import select_device

LOGGER = logging.getLogger(__name__)


class SDVTabularSynthesizer(BaselineSynthesizer, abc.ABC):
    """Base class for single-table models."""

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
        return synthesizer.sample(n_samples)


class GaussianCopulaSynthesizer(SDVTabularSynthesizer):
    """Model wrapping the ``GaussianCopulaSynthesizer`` model."""

    _MODEL = sdv.single_table.GaussianCopulaSynthesizer


class CUDATabularSynthesizer(SDVTabularSynthesizer, abc.ABC):
    """Base class for CUDA dependent models."""

    def _get_trained_synthesizer(self, data, metadata):
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
    """Model wrapping the ``CTGANSynthesizer`` model."""

    _MODEL = sdv.single_table.CTGANSynthesizer


class TVAESynthesizer(CUDATabularSynthesizer):
    """Model wrapping the ``TVAESynthesizer`` model."""

    _MODEL = sdv.single_table.TVAESynthesizer


class CopulaGANSynthesizer(CUDATabularSynthesizer):
    """Model wrapping the ``CopulaGANSynthesizer`` model."""

    _MODEL = sdv.single_table.CopulaGANSynthesizer


class SDVRelationalSynthesizer(BaselineSynthesizer, abc.ABC):
    """Base class for multi-table models."""

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
    """Model wrapping the ``HMASynthesizer`` model."""

    _MODEL = sdv.multi_table.hma.HMASynthesizer


class SDVTimeseriesSynthesizer(BaselineSynthesizer, abc.ABC):
    """Base class for time-series models."""

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
    """Model wrapping the ``PARSynthesizer`` model."""

    def _get_trained_synthesizer(self, data, metadata):
        LOGGER.info('Fitting %s', self.__class__.__name__)
        model = sdv.sequential.PARSynthesizer(metadata=metadata, epochs=1024, verbose=False)
        model.device = select_device()
        model.fit(data)
        return model

    def _sample_from_synthesizer(self, synthesizer, n_samples):
        LOGGER.info('Sampling %s', self.__class__.__name__)
        return synthesizer.sample()
