"""SDV-based synthesizers for SDGym."""

import abc
import importlib
import inspect
import logging
import pkgutil

import sdv

from sdgym.synthesizers.base import BaselineSynthesizer
from sdgym.utils import select_device

LOGGER = logging.getLogger(__name__)


class SDVSingleTableBaseline(BaselineSynthesizer, abc.ABC):
    """Generic SDGym baseline that wraps an SDV single-table synthesizer."""

    # this will be set on the dynamically created subclasses
    _SDV_CLASS = None
    _MODEL_KWARGS = None  # keep for compatibility with your old code

    def _get_trained_synthesizer(self, data, metadata):
        if self._SDV_CLASS is None:
            raise ValueError(f'{self.__class__.__name__} has no _SDV_CLASS set')

        LOGGER.info('Fitting %s', self.__class__.__name__)

        model_kwargs = self._MODEL_KWARGS.copy() if self._MODEL_KWARGS else {}
        model = self._SDV_CLASS(metadata=metadata, **model_kwargs)
        model.fit(data)
        return model

    def _sample_from_synthesizer(self, synthesizer, n_samples):
        LOGGER.info('Sampling %s', self.__class__.__name__)
        return synthesizer.sample(n_samples)


def _iter_sdv_single_table_classes():
    """Yield (name, cls) for every SDV single-table synthesizer in the environment."""
    try:
        from sdv.single_table.base import BaseSingleTableSynthesizer
    except ImportError:
        return  # SDV not installed

    try:
        st_root = importlib.import_module('sdv.single_table')
    except ImportError:
        return

    for module_info in pkgutil.walk_packages(st_root.__path__, prefix=st_root.__name__ + '.'):
        try:
            module = importlib.import_module(module_info.name)
        except Exception:
            # don't break discovery because one module has an import-time issue
            continue

        for attr_name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, BaseSingleTableSynthesizer)
                and obj is not BaseSingleTableSynthesizer
            ):
                yield attr_name, obj


# Dynamically create one SDGym baseline per SDV single-table synthesizer
for _name, _cls in _iter_sdv_single_table_classes():
    # avoid clobbering something we already defined manually
    if _name in globals():
        continue

    globals()[_name] = type(
        _name,
        (SDVSingleTableBaseline,),
        {
            '__doc__': f'SDGym baseline wrapping sdv.single_table.{_name}',
            '_SDV_CLASS': _cls,
        },
    )


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
