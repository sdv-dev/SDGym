import abc
import logging

from sdgym.synthesizers._sdv_lookup import (
    find_sdv_synthesizer,
)
from sdgym.synthesizers.base import BaselineSynthesizer

LOGGER = logging.getLogger(__name__)


class SDVSingleTableBaseline(BaselineSynthesizer, abc.ABC):
    _SDV_CLASS = None
    _MODEL_KWARGS = None

    def _get_trained_synthesizer(self, data, metadata):
        if self._SDV_CLASS is None:
            raise ValueError(f'{self.__class__.__name__} has no _SDV_CLASS set')
        kwargs = dict(self._MODEL_KWARGS or {})
        model = self._SDV_CLASS(metadata=metadata, **kwargs)
        model.fit(data)
        return model

    def _sample_from_synthesizer(self, synthesizer, n_samples):
        return synthesizer.sample(n_samples)


class SDVMultiTableBaseline(BaselineSynthesizer, abc.ABC):
    _SDV_CLASS = None
    _MODEL_KWARGS = None

    def _get_trained_synthesizer(self, data, metadata):
        if self._SDV_CLASS is None:
            raise ValueError(f'{self.__class__.__name__} has no _SDV_CLASS set')
        kwargs = dict(self._MODEL_KWARGS or {})
        model = self._SDV_CLASS(metadata=metadata, **kwargs)
        model.fit(data)
        return model

    def _sample_from_synthesizer(self, synthesizer, n_samples):
        return synthesizer.sample(n_samples)


# discover everything once
def _create_wrappers():
    names_to_try = []

    # just to be thorough, list the attrs we saw during lookup
    # but simplest is to try to wrap common SDV names by iterating over sdv.single_table, etc.
    # for now we can be opportunistic: try to wrap whatever find_sdv_synthesizer accepts
    for module_name in ('sdv.single_table', 'sdv.multi_table'):
        try:
            m = __import__(module_name, fromlist=['*'])
        except Exception:
            continue
        for attr in dir(m):
            if attr[0].isupper():
                names_to_try.append(attr)

    for name in set(names_to_try):
        try:
            sdv_cls, kind = find_sdv_synthesizer(name)
        except KeyError:
            continue

        if name in globals():
            continue

        if kind == 'single_table':
            cls = type(
                name,
                (SDVSingleTableBaseline,),
                {
                    '__module__': __name__,
                    '__doc__': f'SDGym baseline wrapping {sdv_cls}',
                    '_SDV_CLASS': sdv_cls,
                },
            )
        else:
            cls = type(
                name,
                (SDVMultiTableBaseline,),
                {
                    '__module__': __name__,
                    '__doc__': f'SDGym baseline wrapping {sdv_cls}',
                    '_SDV_CLASS': sdv_cls,
                },
            )

        globals()[name] = cls


_create_wrappers()
