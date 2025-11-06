"""SDV synthesizers module."""

import abc
import logging

from sdgym.synthesizers.base import BaselineSynthesizer

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
