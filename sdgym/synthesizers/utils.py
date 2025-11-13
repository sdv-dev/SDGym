"""Utility functions for synthesizers in SDGym."""

from sdgym.synthesizers.base import BaselineSynthesizer
from sdgym.synthesizers.sdv import _get_all_sdv_synthesizers, _get_sdv_synthesizers


def _get_sdgym_synthesizers():
    """Get SDGym synthesizers.

    Returns:
        list:
            A list of available SDGym synthesizer names.
    """
    synthesizers = BaselineSynthesizer._get_supported_synthesizers()
    sdv_synthesizer = _get_all_sdv_synthesizers()
    sdgym_synthesizer = [
        synthesizer for synthesizer in synthesizers if synthesizer not in sdv_synthesizer
    ]
    return sorted(sdgym_synthesizer)


def get_available_single_table_synthesizers():
    """List all available single-table synthesizers in SDGym.

    Returns:
        list:
            A sorted list of available single-table synthesizer names.
    """
    sdv_synthesizers = _get_sdv_synthesizers('single_table')
    sdgym_synthesizers = _get_sdgym_synthesizers()
    return sorted(sdv_synthesizers + sdgym_synthesizers)


def get_available_multi_table_synthesizers():
    """List all available multi-table synthesizers in SDGym.

    Returns:
        list:
            A sorted list of available multi-table synthesizer names.
    """
    return sorted(_get_sdv_synthesizers('multi_table'))


def _get_supported_synthesizers():
    """Get SDGym supported synthesizers.

    Returns:
        list:
            A list of available SDGym supported synthesizer names.
    """
    return BaselineSynthesizer._get_supported_synthesizers()
