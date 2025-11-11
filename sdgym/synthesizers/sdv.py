"""SDV synthesizers wrappers for SDGym."""

import abc
import inspect
import logging
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


def _validate_parameters(provided_parameters, parameters):
    """Validate that the provided parameters are valid for the synthesizer."""
    parameter_names = {param.name for param in parameters}
    for param in provided_parameters.keys():
        if param not in parameter_names:
            raise ValueError(f"Parameter '{param}' is not valid for the selected synthesizer.")


def _validate_inputs(sdv_name, modality, parameters):
    """Validate the inputs for the SDV synthesizer wrapper."""
    _validate_modality(modality)
    if not isinstance(sdv_name, str):
        raise ValueError('`sdv_name` must be a string.')

    if parameters is not None and not isinstance(parameters, dict):
        raise ValueError('`parameters` must be a dictionary or None.')

    sdv_class = getattr(import_module(f'sdv.{modality}'), sdv_name, None)
    if sdv_class is None:
        raise ValueError(f"Synthesizer {sdv_name!r} is not a SDV '{modality}' synthesizers.")

    sdv_parameters = list(inspect.signature(sdv_class.__init__).parameters.values())[1:]
    parameters_to_test = {'metadata': None} if not parameters else parameters
    _validate_parameters(parameters_to_test, sdv_parameters)


def _get_sdv_synthesizers(modality):
    _validate_modality(modality)
    module = MODALITY_TO_MODULE[modality]
    available_synthesizer = {name for name, cls in module.__dict__.items() if isinstance(cls, type)}
    available_synthesizer = available_synthesizer - set(UNSUPPORTED_SDV_SYNTHESIZERS)
    return sorted(available_synthesizer)


class BaselineSDVSynthesizer(BaselineSynthesizer, abc.ABC):
    """Base class for SDV synthesizers."""

    def __init__(self, sdv_name, modality, parameters=None):
        _validate_inputs(sdv_name, modality, parameters)
        self.sdv_name = sdv_name
        self.modality = modality
        self.parameters = parameters or {}

    def __repr__(self):
        """Define the string representation of the synthesizer."""
        return f'<{self.__class__.__name__} sdv_name={self.sdv_name!r} modality={self.modality!r}>'

    def _get_trained_synthesizer(self, data, metadata):
        LOGGER.info('Fitting %s', self.sdv_name)
        sdv_class = getattr(import_module(f'sdv.{self.modality}'), self.sdv_name)
        synthesizer = sdv_class(metadata=metadata, **self.parameters)
        synthesizer.fit(data)
        return synthesizer

    def _sample_from_synthesizer(self, synthesizer, sample_arg):
        LOGGER.info('Sampling %s', self.sdv_name)
        if self.modality == 'multi_table':
            return synthesizer.sample(scale=sample_arg)

        return synthesizer.sample(num_rows=sample_arg)
