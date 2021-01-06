"""Random utils used by SDGym."""

import importlib
import os
import sys
import traceback

import humanfriendly
import psutil


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
    if isinstance(synthesizer, str):
        baselines = Baseline.get_subclasses()
        if synthesizer in baselines:
            synthesizer = baselines[synthesizer]
        else:
            try:
                synthesizer = import_object(synthesizer)
            except Exception:
                raise SDGymError(f'Unknown synthesizer {synthesizer}') from None

    if name:
        synthesizer.name = name
    elif not hasattr(synthesizer, 'name'):
        synthesizer.name = _get_synthesizer_name(synthesizer)

    return synthesizer


def get_synthesizers_dict(synthesizers):
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

