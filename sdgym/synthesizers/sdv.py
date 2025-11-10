"""SDV synthesizers wrappers for SDGym."""

import abc
import logging

from sdgym.synthesizers._sdv_lookup import (
    find_sdv_synthesizer,
)
from sdgym.synthesizers.base import BaselineSynthesizer

LOGGER = logging.getLogger(__name__)


def _get_trained_synthesizer(sdv_class, model_kwargs, data, metadata):
    if sdv_class is None:
        raise ValueError('The synthesizer has no `_SDV_CLASS` set')

    kwargs = dict(model_kwargs or {})
    model = sdv_class(metadata=metadata, **kwargs)
    model.fit(data)
    return model


class SDVSingleTableBaseline(BaselineSynthesizer, abc.ABC):
    """Base class for SDV single-table synthesizers."""

    _SDV_CLASS = None
    _MODEL_KWARGS = None

    def _get_trained_synthesizer(self, data, metadata):
        return _get_trained_synthesizer(self._SDV_CLASS, self._MODEL_KWARGS, data, metadata)

    def _sample_from_synthesizer(self, synthesizer, n_samples):
        return synthesizer.sample(n_samples)


class SDVMultiTableBaseline(BaselineSynthesizer, abc.ABC):
    """Base class for SDV multi-table synthesizers."""

    _SDV_CLASS = None
    _MODEL_KWARGS = None

    def _get_trained_synthesizer(self, data, metadata):
        return _get_trained_synthesizer(self._SDV_CLASS, self._MODEL_KWARGS, data, metadata)

    def _sample_from_synthesizer(self, synthesizer, n_samples):
        return synthesizer.sample()


def _create_wrappers():
    """Method to create SDV synthesizer wrappers dynamically."""
    names_to_try = []
    for module_name in ('sdv.single_table', 'sdv.multi_table'):
        try:
            module = __import__(module_name, fromlist=['*'])
        except Exception:
            continue

        for attr in dir(module):
            if attr[0].isupper():
                names_to_try.append(attr)

    for name in set(names_to_try):
        try:
            sdv_cls, synthesizer_type = find_sdv_synthesizer(name)
        except KeyError:
            continue

        if name in globals():
            continue

        if synthesizer_type == 'single_table':
            cls = type(
                name,
                (SDVSingleTableBaseline,),
                {
                    '__module__': __name__,
                    '__doc__': f'SDGym wrapper for {sdv_cls}',
                    '_SDV_CLASS': sdv_cls,
                },
            )
        else:
            cls = type(
                name,
                (SDVMultiTableBaseline,),
                {
                    '__module__': __name__,
                    '__doc__': f'SDGym wrapper for {sdv_cls}',
                    '_SDV_CLASS': sdv_cls,
                },
            )

        globals()[name] = cls


_create_wrappers()
