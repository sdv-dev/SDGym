# sdgym/synthesizers/_sdv_lookup.py
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


def find_sdv_synthesizer(name: str):
    """Return (sdv_class, kind) where kind is 'single' or 'multi'.

    Raises KeyError if not found.
    """
    # 1) single-table
    if BaseSingleTableSynthesizer is not None:
        try:
            st_root = importlib.import_module('sdv.single_table')
        except ImportError:
            st_root = None

        if st_root is not None:
            # root
            if hasattr(st_root, name):
                obj = getattr(st_root, name)
                if inspect.isclass(obj) and issubclass(obj, BaseSingleTableSynthesizer):
                    return obj, 'single_table'

            # submodules
            for info in pkgutil.walk_packages(st_root.__path__, prefix=st_root.__name__ + '.'):
                try:
                    mod = importlib.import_module(info.name)
                except Exception:
                    continue

                if hasattr(mod, name):
                    obj = getattr(mod, name)
                    if inspect.isclass(obj) and issubclass(obj, BaseSingleTableSynthesizer):
                        return obj, 'single_table'

    # 2) multi-table
    if BaseMultiTableSynthesizer is not None:
        try:
            mt_root = importlib.import_module('sdv.multi_table')
        except ImportError:
            mt_root = None

        if mt_root is not None:
            if hasattr(mt_root, name):
                obj = getattr(mt_root, name)
                if inspect.isclass(obj) and issubclass(obj, BaseMultiTableSynthesizer):
                    return obj, 'multi_table'

            for info in pkgutil.walk_packages(mt_root.__path__, prefix=mt_root.__name__ + '.'):
                try:
                    mod = importlib.import_module(info.name)
                except Exception:
                    continue

                if hasattr(mod, name):
                    obj = getattr(mod, name)
                    if inspect.isclass(obj) and issubclass(obj, BaseMultiTableSynthesizer):
                        return obj, 'multi_table'

    raise KeyError(name)
