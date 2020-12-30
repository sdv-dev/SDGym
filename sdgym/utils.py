"""Random utils used by SDGym."""

import importlib
import multiprocessing
import os
import sys
import traceback
from datetime import datetime

import humanfriendly
import psutil

from sdgym.errors import SDGymTimeout


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


def timed(function, *args, **kwargs):
    now = datetime.utcnow()
    out = function(*args, **kwargs)
    elapsed = datetime.utcnow() - now
    return out, elapsed


def format_exception():
    exception = traceback.format_exc()
    exc_type, exc_value, _ = sys.exc_info()
    error = traceback.format_exception_only(exc_type, exc_value)[0].strip()
    return exception, error


def _timeout_function(output, function, args):
    output['output'] = function(*args)


def with_timeout(timeout, function, *args):
    with multiprocessing.Manager() as manager:
        output = manager.dict()
        process = multiprocessing.Process(
            target=_timeout_function,
            args=(output, function, args)
        )

        process.start()
        process.join(timeout)
        process.terminate()

        if not output:
            raise SDGymTimeout()

        return dict(output)['output']
