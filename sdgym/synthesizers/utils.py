"""Utility functions for synthesizers in SDGym."""

from sdgym.synthesizers.sdv import _get_sdv_synthesizers

NON_SDV_SYNTHESIZERS = [
    'UniformSynthesizer',
    'ColumnSynthesizer',
    'DataIdentity',
    'RealTabFormerSynthesizer',
]


def get_available_single_table_synthesizers():
    """List all available single-table synthesizers in SDGym.

    Returns:
        list:
            A sorted list of available single-table synthesizer names.
    """
    sdv_synthesizers = _get_sdv_synthesizers('single_table')
    return sorted(sdv_synthesizers + NON_SDV_SYNTHESIZERS)


def get_available_multi_table_synthesizers():
    """List all available multi-table synthesizers in SDGym.

    Returns:
        list:
            A sorted list of available multi-table synthesizer names.
    """
    return sorted(_get_sdv_synthesizers('multi_table'))
