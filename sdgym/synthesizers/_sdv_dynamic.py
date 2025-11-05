from __future__ import annotations

import abc
import importlib
import inspect
import logging
import pkgutil

from sdgym.synthesizers.base import BaselineSynthesizer

LOGGER = logging.getLogger(__name__)
try:
    from sdv.single_table.base import BaseSingleTableSynthesizer
except ImportError:
    BaseSingleTableSynthesizer = None

try:
    from sdv.multi_table.base import BaseMultiTableSynthesizer
except ImportError:
    BaseMultiTableSynthesizer = None


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


class SDVMultiTableBaseline(BaselineSynthesizer, abc.ABC):
    """Generic SDGym baseline that wraps an SDV multi-table synthesizer."""

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


def _discover_and_create(
    *,
    root_module_name: str,
    base_sdv_class,
    baseline_wrapper_class,
    target_globals: dict,
):
    """Discover SDV synthesizers and create SDGym baselines for them.

    Args:
        root_module_name (str): e.g. "sdv.single_table" or "sdv.multi_table"
        base_sdv_class (type|None): e.g. BaseSingleTableSynthesizer
        baseline_wrapper_class (type): e.g. SDVSingleTableBaseline
        target_globals (dict): usually globals() of this module
    """
    if base_sdv_class is None:
        return

    try:
        root_mod = importlib.import_module(root_module_name)
    except ImportError:
        return

    def _iter_sdv_classes():
        for name, obj in inspect.getmembers(root_mod, inspect.isclass):
            if issubclass(obj, base_sdv_class) and obj is not base_sdv_class:
                yield name, obj

        if hasattr(root_mod, '__path__'):  # packages have __path__
            for module_info in pkgutil.walk_packages(
                root_mod.__path__, prefix=root_mod.__name__ + '.'
            ):
                try:
                    module = importlib.import_module(module_info.name)
                except Exception:
                    continue

                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, base_sdv_class) and obj is not base_sdv_class:
                        yield name, obj

    for name, sdv_cls in _iter_sdv_classes():
        if name in target_globals:
            continue

        target_globals[name] = type(
            name,
            (baseline_wrapper_class,),
            {
                '__module__': __name__,  # important for pickling
                '__doc__': f'SDGym baseline wrapping {root_module_name}.{name}',
                '_SDV_CLASS': sdv_cls,
            },
        )


# create single-table baselines
_discover_and_create(
    root_module_name='sdv.single_table',
    base_sdv_class=BaseSingleTableSynthesizer,
    baseline_wrapper_class=SDVSingleTableBaseline,
    target_globals=globals(),
)

# create multi-table baselines
_discover_and_create(
    root_module_name='sdv.multi_table',
    base_sdv_class=BaseMultiTableSynthesizer,
    baseline_wrapper_class=SDVMultiTableBaseline,
    target_globals=globals(),
)
