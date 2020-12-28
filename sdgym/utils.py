"""Random utils used by SDGym."""

import importlib
import os

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
