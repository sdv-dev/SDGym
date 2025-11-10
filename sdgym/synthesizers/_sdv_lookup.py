from __future__ import annotations

import importlib
import inspect

from sdv.multi_table.base import BaseMultiTableSynthesizer
from sdv.single_table.base import BaseSingleTableSynthesizer

SYNTHESIZER_TYPE_TO_BASE_CLASS = {
    'single_table': BaseSingleTableSynthesizer,
    'multi_table': BaseMultiTableSynthesizer,
}


def _find_synthesizer_type(name, synthesizer_type):
    """Helper to find synthesizer of a given type."""
    if synthesizer_type not in SYNTHESIZER_TYPE_TO_BASE_CLASS:
        raise ValueError('`synthesizer_type` must be one `single_table` or `multi_table`.')

    base_class = SYNTHESIZER_TYPE_TO_BASE_CLASS.get(synthesizer_type)
    root = importlib.import_module(f'sdv.{synthesizer_type}')
    if hasattr(root, name):
        obj = getattr(root, name)
        if inspect.isclass(obj) and issubclass(obj, base_class):
            return obj

    return None


def find_sdv_synthesizer(name):
    """Find an SDV synthesizer by name."""
    for synthesizer_type in SYNTHESIZER_TYPE_TO_BASE_CLASS.keys():
        sdv_cls = _find_synthesizer_type(name, synthesizer_type)
        if sdv_cls is not None:
            return sdv_cls, synthesizer_type

    raise KeyError(f"SDV synthesizer '{name}' not found")
