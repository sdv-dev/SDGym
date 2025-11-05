"""Synthesizers module."""

from sdgym.synthesizers.generate import (
    SYNTHESIZER_MAPPING,
    create_multi_table_synthesizer,
    create_sdv_synthesizer_variant,
    create_sequential_synthesizer,
    create_single_table_synthesizer,
)
from sdgym.synthesizers.identity import DataIdentity
from sdgym.synthesizers.column import ColumnSynthesizer
from sdgym.synthesizers.realtabformer import RealTabFormerSynthesizer
from sdgym.synthesizers.uniform import UniformSynthesizer
from sdgym.synthesizers import _sdv_dynamic as sdv_dynamic  # noqa: F401
from sdgym.synthesizers._sdv_dynamic import (
    SDVSingleTableBaseline,
    SDVMultiTableBaseline,
)
from sdgym.synthesizers.sdv import (
    SDVTabularSynthesizer,
    SDVRelationalSynthesizer,
    SDVTimeseriesSynthesizer,
)

__all__ = [
    'DataIdentity',
    'ColumnSynthesizer',
    'UniformSynthesizer',
    'RealTabFormerSynthesizer',
    'create_single_table_synthesizer',
    'create_multi_table_synthesizer',
    'create_sdv_synthesizer_variant',
    'create_sequential_synthesizer',
    'SYNTHESIZER_MAPPING',
    'SDVSingleTableBaseline',
    'SDVMultiTableBaseline',
    'SDVTabularSynthesizer',
    'SDVRelationalSynthesizer',
    'SDVTimeseriesSynthesizer',
]


for name in dir(sdv_dynamic):
    obj = getattr(sdv_dynamic, name)
    if isinstance(obj, type) and (
        issubclass(obj, SDVSingleTableBaseline) or issubclass(obj, SDVMultiTableBaseline)
    ):
        if obj in (SDVSingleTableBaseline, SDVMultiTableBaseline):
            continue

        globals()[name] = obj
        __all__.append(name)

__all__ = tuple(__all__)
