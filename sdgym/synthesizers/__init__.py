"""Synthesizers module."""

from sdgym.synthesizers.generate import (
    SYNTHESIZER_MAPPING,
    create_sdv_synthesizer_variant,
    create_single_table_synthesizer,
    create_multi_table_synthesizer,
)
from sdgym.synthesizers.identity import DataIdentity
from sdgym.synthesizers.column import ColumnSynthesizer
from sdgym.synthesizers.realtabformer import RealTabFormerSynthesizer
from sdgym.synthesizers.uniform import UniformSynthesizer
from sdgym.synthesizers import sdv as sdgym_sdv
from sdgym.synthesizers.sdv import (
    SDVSingleTableBaseline,
    SDVMultiTableBaseline,
)

__all__ = [
    'DataIdentity',
    'ColumnSynthesizer',
    'UniformSynthesizer',
    'RealTabFormerSynthesizer',
    'create_single_table_synthesizer',
    'create_multi_table_synthesizer',
    'create_sdv_synthesizer_variant',
    'SYNTHESIZER_MAPPING',
    'SDVSingleTableBaseline',
    'SDVMultiTableBaseline',
]


for name in dir(sdgym_sdv):
    obj = getattr(sdgym_sdv, name)
    if isinstance(obj, type) and (
        issubclass(obj, SDVSingleTableBaseline) or issubclass(obj, SDVMultiTableBaseline)
    ):
        if obj in (SDVSingleTableBaseline, SDVMultiTableBaseline):
            continue

        globals()[name] = obj
        __all__.append(name)

__all__ = tuple(__all__)
