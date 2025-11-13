"""SDV synthesizers wrappers for SDGym."""

import logging
import sys
from importlib import import_module

from sdv import multi_table, single_table

from sdgym.synthesizers.base import BaselineSynthesizer

LOGGER = logging.getLogger(__name__)
UNSUPPORTED_SDV_SYNTHESIZERS = ['DayZSynthesizer']
MODALITY_TO_MODULE = {
    'single_table': single_table,
    'multi_table': multi_table,
}


def _validate_modality(modality):
    """Validate that the modality is correct."""
    if modality not in ['single_table', 'multi_table']:
        raise ValueError("`modality` must be one of 'single_table' or 'multi_table'.")


def _get_sdv_synthesizers(modality):
    _validate_modality(modality)
    module = MODALITY_TO_MODULE[modality]
    available_synthesizer = {name for name, cls in module.__dict__.items() if isinstance(cls, type)}
    available_synthesizer = available_synthesizer - set(UNSUPPORTED_SDV_SYNTHESIZERS)
    return sorted(available_synthesizer)


def _get_all_sdv_synthesizers():
    """Get all available SDV synthesizers."""
    synthesizers = set()
    for modality in MODALITY_TO_MODULE.keys():
        synthesizers.update(_get_sdv_synthesizers(modality))

    return sorted(synthesizers)


def _get_trained_synthesizer(self, data, metadata):
    LOGGER.info('Fitting %s', self.__class__.__name__)
    sdv_class = getattr(import_module(f'sdv.{self.modality}'), self.SDV_NAME)
    synthesizer = sdv_class(metadata=metadata, **self._MODEL_KWARGS)
    synthesizer.fit(data)
    return synthesizer


def _sample_from_synthesizer(self, synthesizer, sample_arg):
    LOGGER.info('Sampling %s', self.__class__.__name__)
    if self.modality == 'multi_table':
        return synthesizer.sample(scale=sample_arg)

    return synthesizer.sample(num_rows=sample_arg)


def _retrieve_sdv_class(sdv_name):
    current_module = sys.modules[__name__]
    if hasattr(current_module, sdv_name):
        existing_class = getattr(current_module, sdv_name)
        if isinstance(existing_class, type):
            return existing_class

    return None


def _get_modality(sdv_name):
    """Get the modality of a SDV synthesizer."""
    st_synthesizers = _get_sdv_synthesizers('single_table')
    if sdv_name in st_synthesizers:
        return 'single_table'

    mt_synthesizers = _get_sdv_synthesizers('multi_table')
    if sdv_name in mt_synthesizers:
        return 'multi_table'

    raise ValueError(f"Synthesizer '{sdv_name}' is not a SDV synthesizer.")


def _create_sdv_class(sdv_name):
    """Create a SDV synthesizer class dynamically."""
    current_module = sys.modules[__name__]
    modality = _get_modality(sdv_name)
    synthesizer_class = type(
        sdv_name,
        (BaselineSynthesizer,),
        {
            '__module__': __name__,
            'SDV_NAME': sdv_name,
            'modality': modality,
            '_MODEL_KWARGS': {},
            '_get_trained_synthesizer': _get_trained_synthesizer,
            '_sample_from_synthesizer': _sample_from_synthesizer,
        },
    )
    setattr(current_module, sdv_name, synthesizer_class)

    return synthesizer_class


def create_sdv_synthesizer_class(sdv_name):
    """Factory for dynamically creating or retrieving SDV synthesizer classes."""
    if sdv_name not in _get_all_sdv_synthesizers():
        raise ValueError(f"Synthesizer '{sdv_name}' is not a supported SDV synthesizer.")

    return _retrieve_sdv_class(sdv_name) or _create_sdv_class(sdv_name)
