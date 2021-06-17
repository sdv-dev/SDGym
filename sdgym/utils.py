"""Random utils used by SDGym."""

import importlib
import json
import logging
import os
import sys
import traceback
import types

import humanfriendly
import psutil

from sdgym.errors import SDGymError
from sdgym.synthesizers.base import Baseline
from sdgym.synthesizers.utils import select_device

LOGGER = logging.getLogger(__name__)


def used_memory():
    """Get the memory used by this process nicely formatted."""
    process = psutil.Process(os.getpid())
    return humanfriendly.format_size(process.memory_info().rss)


def import_object(object_name):
    """Import an object from its Fully Qualified Name."""

    if isinstance(object_name, str):
        parent_name, attribute = object_name.rsplit('.', 1)
        try:
            parent = importlib.import_module(parent_name)
        except ImportError:
            grand_parent_name, parent_name = parent_name.rsplit('.', 1)
            grand_parent = importlib.import_module(grand_parent_name)
            parent = getattr(grand_parent, parent_name)

        return getattr(parent, attribute)

    return object_name


def format_exception():
    exception = traceback.format_exc()
    exc_type, exc_value, _ = sys.exc_info()
    error = traceback.format_exception_only(exc_type, exc_value)[0].strip()
    return exception, error


def _get_synthesizer_name(synthesizer):
    """Get the name of the synthesizer function or class.

    If the given synthesizer is a function, return its name.
    If it is a method, return the name of the class to which
    the method belongs.

    Args:
        synthesizer (function or method):
            The synthesizer function or method.

    Returns:
        str:
            Name of the function or the class to which the method belongs.
    """
    if isinstance(synthesizer, types.MethodType):
        synthesizer_name = synthesizer.__self__.__class__.__name__
    else:
        synthesizer_name = synthesizer.__name__

    return synthesizer_name


def _get_synthesizer(synthesizer, name=None):
    if isinstance(synthesizer, dict):
        return synthesizer

    if isinstance(synthesizer, str):
        if synthesizer.endswith('.json'):
            LOGGER.info('Trying to load synthesizer from a JSON file.')
            with open(synthesizer, 'r') as json_file:
                return json.load(json_file)

        baselines = Baseline.get_subclasses()
        if synthesizer in baselines:
            LOGGER.info('Trying to import synthesizer by name.')
            synthesizer = baselines[synthesizer]

        else:
            try:
                LOGGER.info('Trying to load synthesizer from JSON string.')
                return json.loads(synthesizer)

            except Exception:
                try:
                    LOGGER.info('Trying to import synthesizer from fully qualified name.')
                    synthesizer = import_object(synthesizer)

                except Exception:
                    raise SDGymError(f'Unknown synthesizer {synthesizer}') from None

    if not name:
        name = _get_synthesizer_name(synthesizer)

    return {
        'name': name,
        'synthesizer': synthesizer,
        'modalities': getattr(synthesizer, 'MODALITIES', []),
    }


def get_synthesizers(synthesizers):
    """Get the dict of synthesizers from the input value.

    If the input is a synthesizer or an iterable of synthesizers, get their names
    and put them on a dict.

    Args:
        synthesizers (function, class, list, tuple or dict):
            A synthesizer (function or method or class) or an iterable of synthesizers
            or a dict containing synthesizer names as keys and synthesizers as values.

    Returns:
        dict[str, function]:
            dict containing synthesizer names as keys and function as values.

    Raises:
        TypeError:
            if neither a synthesizer or an iterable or a dict is passed.
    """
    if callable(synthesizers):
        return [_get_synthesizer(synthesizers)]

    if isinstance(synthesizers, (list, tuple)):
        return [
            _get_synthesizer(synthesizer)
            for synthesizer in synthesizers
        ]

    if isinstance(synthesizers, dict):
        return [
            _get_synthesizer(synthesizer, name)
            for name, synthesizer in synthesizers.items()
        ]

    raise TypeError('`synthesizers` can only be a function, a class, a list or a dict')


def _get_kwargs(synthesizer_dict, method_name, replace):
    method_kwargs = synthesizer_dict.get(method_name + '_kwargs', {})
    for key, value in method_kwargs.items():
        for replace_keyword, replace_value in replace:
            if isinstance(value, type(replace_keyword)) and value == replace_keyword:
                method_kwargs[key] = replace_value

    return method_kwargs


def build_synthesizer(synthesizer, synthesizer_dict):
    """Build a synthesizer function for a dict specification.

    The dict specification may contain any combination of the following entries:
        * ``modalities``: List of modalities supported by this synthesizer.
        * ``init``: Dict with the keyword arguments to pass to the ``__init__`` method. The
          arguments should include the metadata, real data or device arguments when
          required, encoded as the corresponding keywords.
        * ``fit``: Dict with the keyword arguments to pass to the ``fit`` method. The
          arguments should include the metadata, real data or device arguments when
          required, encoded as the corresponding keywords.
        * ``modalities``: List of modalities supported by this synthesizer.
        * ``metadata``: Keyword that should be replaced by the ``metadata`` object
          when building the ``__init__`` and ``fit`` arguments. Defaults to ``$metadata``.
        * ``real_data``: Keyword that should be replaced by the ``real_data`` object
          when building the ``__init__`` and ``fit`` arguments. Defaults to ``$real_data``.
        * ``device``: Keyword that should be replaced by the CUDA device to use
          when building the ``__init__`` and ``fit`` arguments. Defaults to ``$device``.
        * ``device_attr``: boolean indicating whether the CUDA device must be set
          as an attribute instead of passing it as an argument. Defaults to ``False``.

    Args:
        synthesizer (class):
            The class to be used in the synthesizer function.
        synthesizer_dict (dict):
            Dictionary with the specification of how to use the synthesizer.

    Returns:
        callable:
            The synthesizer function
    """
    def _synthesizer_function(real_data, metadata):
        metadata_keyword = synthesizer_dict.get('metadata', '$metadata')
        real_data_keyword = synthesizer_dict.get('real_data', '$real_data')
        device_keyword = synthesizer_dict.get('device', '$device')
        device_attribute = synthesizer_dict.get('device_attribute')
        device = select_device()

        multi_table = 'multi-table' in synthesizer_dict['modalities']
        if not multi_table:
            table = metadata.get_tables()[0]
            metadata = metadata.get_table_meta(table)
            real_data = real_data[table]

        replace = [
            (metadata_keyword, metadata),
            (real_data_keyword, real_data),
            (device_keyword, device),
        ]

        init_kwargs = _get_kwargs(synthesizer_dict, 'init', replace)
        fit_kwargs = _get_kwargs(synthesizer_dict, 'fit', replace)

        instance = synthesizer(**init_kwargs)
        if device_attribute:
            setattr(instance, device_attribute, device)

        instance.fit(**fit_kwargs)

        sampled = instance.sample()
        if not multi_table:
            sampled = {table: sampled}

        return sampled

    return _synthesizer_function
