from __future__ import annotations

import importlib
import inspect
import pkgutil

try:
    from sdv.single_table.base import BaseSingleTableSynthesizer
except Exception:
    BaseSingleTableSynthesizer = None

try:
    from sdv.multi_table.base import BaseMultiTableSynthesizer
except Exception:
    BaseMultiTableSynthesizer = None

SYNTHESIZER_TYPE_TO_BASE_CLASS = {
    'single_table': BaseSingleTableSynthesizer,
    'multi_table': BaseMultiTableSynthesizer,
}


def _find_synthesizer_type(name, synthesizer_type):
    """Helper to find synthesizer of a given type."""
    base_class = SYNTHESIZER_TYPE_TO_BASE_CLASS.get(synthesizer_type)
    if base_class is not None:
        try:
            root = importlib.import_module(f'sdv.{synthesizer_type}')
        except ImportError:
            root = None

        if root is not None:
            # root
            if hasattr(root, name):
                obj = getattr(root, name)
                if inspect.isclass(obj) and issubclass(obj, base_class):
                    return obj

            # submodules
            for info in pkgutil.walk_packages(root.__path__, prefix=root.__name__ + '.'):
                try:
                    mod = importlib.import_module(info.name)
                except Exception:
                    continue

                if hasattr(mod, name):
                    obj = getattr(mod, name)
                    if inspect.isclass(obj) and issubclass(obj, base_class):
                        return obj

    return None


def find_sdv_synthesizer(name):
    """Return (sdv_class, type) where type is 'single_table' or 'multi_table'.

    Raises KeyError if not found.
    """
    for synthesizer_type in SYNTHESIZER_TYPE_TO_BASE_CLASS.keys():
        sdv_cls = _find_synthesizer_type(name, synthesizer_type)
        if sdv_cls is not None:
            return sdv_cls, synthesizer_type

    raise KeyError(f'SDV synthesizer {name} not found')
