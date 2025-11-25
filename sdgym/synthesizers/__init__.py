"""Synthesizers module."""

from sdgym.synthesizers.generate import (
    create_synthesizer_variant,
    create_single_table_synthesizer,
    create_multi_table_synthesizer,
)
from sdgym.synthesizers.identity import DataIdentity
from sdgym.synthesizers.column import ColumnSynthesizer
from sdgym.synthesizers.realtabformer import RealTabFormerSynthesizer
from sdgym.synthesizers.uniform import UniformSynthesizer, MultiTableUniformSynthesizer
from sdgym.synthesizers.utils import (
    get_available_single_table_synthesizers,
    get_available_multi_table_synthesizers,
)
from sdgym.synthesizers.sdv import create_sdv_synthesizer_class, _get_all_sdv_synthesizers


__all__ = [
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
]

for sdv_name in _get_all_sdv_synthesizers():
    create_sdv_synthesizer_class(sdv_name)
