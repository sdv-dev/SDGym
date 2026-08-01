"""Synthesizers module.

This module exposes the main dataset synthesis functions and models for SDGym.
To load all standard external libraries such as SDV automatically, it supports lazy-loaded initializations
saving base CPU boot costs overhead drastically during CI operations or generic library module testing.
"""

from typing import List
import logging

from sdgym.synthesizers.column import ColumnSynthesizer
from sdgym.synthesizers.generate import (
    create_multi_table_synthesizer,
    create_single_table_synthesizer,
    create_synthesizer_variant,
)
from sdgym.synthesizers.identity import DataIdentity
from sdgym.synthesizers.realtabformer import RealTabFormerSynthesizer
from sdgym.synthesizers.uniform import MultiTableUniformSynthesizer, UniformSynthesizer
from sdgym.synthesizers.utils import (
    get_available_multi_table_synthesizers,
    get_available_single_table_synthesizers,
)

LOGGER = logging.getLogger(__name__)

__all__ =[
    'DataIdentity',
    'ColumnSynthesizer',
    'UniformSynthesizer',
    'RealTabFormerSynthesizer',
    'create_single_table_synthesizer',
    'create_multi_table_synthesizer',
    'create_synthesizer_variant',
    'get_available_single_table_synthesizers',
    'get_available_multi_table_synthesizers',
    'MultiTableUniformSynthesizer',
    'register_sdv_synthesizers',
]

_SDV_SYNTHESIZERS_REGISTERED = False


def register_sdv_synthesizers() -> None:
    """Explicit initialization helper establishing dynamically driven generic algorithm wrappers.

    Avoids creating costly model hooks at `__init__` resolution, drastically speeding up root SDGym load sizes.
    Called explicitly to seed system globals dynamically generating base synthetic variations properly via the registry.
    """
    global _SDV_SYNTHESIZERS_REGISTERED
    if _SDV_SYNTHESIZERS_REGISTERED:
        return

    try:
        from sdgym.synthesizers.sdv import _get_all_sdv_synthesizers, create_sdv_synthesizer_class

        for sdv_name in _get_all_sdv_synthesizers():
            create_sdv_synthesizer_class(sdv_name)
        _SDV_SYNTHESIZERS_REGISTERED = True

    except ImportError as e:
        LOGGER.warning(
            f"SDV Optional Dependences missing: Unable to bootstrap variants. ({e})"
        )


# Hook intercept overriding function calls guaranteeing drop-in legacy workflow consistency seamlessly internally
def _patched_get_single_table_synths(*args, **kwargs) -> List:
    register_sdv_synthesizers()
    return get_available_single_table_synthesizers(*args, **kwargs)


def _patched_get_multi_table_synths(*args, **kwargs) -> List:
    register_sdv_synthesizers()
    return get_available_multi_table_synthesizers(*args, **kwargs)
