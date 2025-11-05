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


LOGGER = logging.getLogger(__name__)

# --- single place where we deal with SDV being present or not ---
try:
    from sdv.single_table.base import BaseSingleTableSynthesizer
except ImportError:  # SDV not installed or old version
    BaseSingleTableSynthesizer = None
# ---------------------------------------------------------------


class SDVSingleTableBaseline(BaselineSynthesizer, abc.ABC):
    """Generic SDGym baseline that wraps an SDV single-table synthesizer."""

    _SDV_CLASS = None
    _MODEL_KWARGS = None

    def _get_trained_synthesizer(self, data, metadata):
        if self._SDV_CLASS is None:
            raise ValueError(f'{self.__class__.__name__} has no _SDV_CLASS set')

        LOGGER.info('Fitting %s', self.__class__.__name__)
        kwargs = self._MODEL_KWARGS.copy() if self._MODEL_KWARGS else {}
        model = self._SDV_CLASS(metadata=metadata, **kwargs)
        model.fit(data)
        return model

    def _sample_from_synthesizer(self, synthesizer, n_samples):
        LOGGER.info('Sampling %s', self.__class__.__name__)
        return synthesizer.sample(n_samples)


def _iter_sdv_single_table_classes():
    """Yield (name, cls) for every SDV single-table synthesizer we can find."""
    # if SDV isn't available, just stop here
    if BaseSingleTableSynthesizer is None:
        return

    try:
        st_root = importlib.import_module('sdv.single_table')
    except ImportError:
        return

    # 1) look in the root module (where your enterprise/bundle exports live)
    for name, obj in inspect.getmembers(st_root, inspect.isclass):
        if issubclass(obj, BaseSingleTableSynthesizer) and obj is not BaseSingleTableSynthesizer:
            yield name, obj

    # 2) also walk submodules
    for module_info in pkgutil.walk_packages(st_root.__path__, prefix=st_root.__name__ + '.'):
        try:
            module = importlib.import_module(module_info.name)
        except Exception:
            continue

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, BaseSingleTableSynthesizer)
                and obj is not BaseSingleTableSynthesizer
            ):
                yield name, obj


# dynamically create one baseline per SDV synth
for _name, _cls in _iter_sdv_single_table_classes():
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
