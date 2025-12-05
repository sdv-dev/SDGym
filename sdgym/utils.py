"""Random utils used by SDGym."""

import logging
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone

import humanfriendly
import numpy as np
import pandas as pd
import psutil

from sdgym.errors import SDGymError
from sdgym.synthesizers.base import BaselineSynthesizer
from sdgym.synthesizers.sdv import _get_all_sdv_synthesizers, create_sdv_synthesizer_class

LOGGER = logging.getLogger(__name__)


def used_memory():
    """Get the memory used by this process nicely formatted."""
    process = psutil.Process(os.getpid())
    return humanfriendly.format_size(process.memory_info().rss)


def format_exception():
    """Format exceptions."""
    exception = traceback.format_exc()
    exc_type, exc_value, _ = sys.exc_info()
    error = traceback.format_exception_only(exc_type, exc_value)[0].strip()
    return exception, error


def get_synthesizers(synthesizers):
    """Get the dict of synthesizer name and object for each synthesizer.

    Args:
        synthesizers (list):
            An iterable of synthesizer classes and strings.

    Returns:
        dict[str, function]:
            Dict with the synthesizer name and object.

    Raises:
        TypeError:
            If neither a list is not passed.
    """
    synthesizers = [] if synthesizers is None else synthesizers
    if not isinstance(synthesizers, list):
        raise TypeError('`synthesizers` must be a list.')

    synthesizers_dicts = []
    baselines = BaselineSynthesizer.get_subclasses(include_parents=True)
    for synthesizer in synthesizers:
        if isinstance(synthesizer, str):
            LOGGER.info('Trying to import synthesizer by name.')
            synthesizer_class = baselines.get(synthesizer)
            if synthesizer_class is None:
                if synthesizer in _get_all_sdv_synthesizers():
                    synthesizer_class = create_sdv_synthesizer_class(synthesizer)
                else:
                    raise SDGymError(f'Unknown synthesizer {synthesizer}') from None

            synthesizers_dicts.append({
                'name': synthesizer,
                'synthesizer': synthesizer_class(),
            })
            continue

        if isinstance(synthesizer, type) or hasattr(synthesizer, '__name__'):
            synthesizer_name = getattr(synthesizer, '__name__', 'undefined')
        else:
            synthesizer_name = getattr(type(synthesizer), '__name__', 'undefined')

        synthesizers_dicts.append({
            'name': synthesizer_name,
            'synthesizer': synthesizer,
        })

    return synthesizers_dicts


def get_size_of(obj, obj_ids=None):
    """Get the memory used by a given object in bytes.

    Args:
        obj (object):
            The object to get the size of.
        obj_ids (set):
            The ids of the objects that have already been evaluated.

    Returns:
        int:
            The size in bytes.
    """
    size = 0
    if obj_ids is None:
        obj_ids = set()

    obj_id = id(obj)
    if obj_id in obj_ids:
        return 0

    obj_ids.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size_of(v, obj_ids) for v in obj.values()])
    elif isinstance(obj, pd.DataFrame):
        size += obj.memory_usage(index=True).sum()
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size_of(i, obj_ids) for i in obj])
    else:
        size += sys.getsizeof(obj)

    return size


def get_duplicates(items):
    """Get any duplicate items in the given list.

    Args:
        items (list):
            The list of items to de-duplicate.

    Returns:
        set:
            The duplicate items.
    """
    seen = set()
    return {item for item in items if item in seen or seen.add(item)}


def get_num_gpus():
    """Get number of gpus.

    Returns:
        int:
            Number of gpus to use.
    """
    try:
        command = ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits']
        output = subprocess.run(command, stdout=subprocess.PIPE)
        return len(output.stdout.decode().split())

    except Exception:
        return 0


def select_device():
    """Select gpu if available, otherwise select cpu.

    Returns:
        str:
            The cuda device if available, otherwise ``'cpu'``.
    """
    try:
        command = ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits']
        output = subprocess.run(command, stdout=subprocess.PIPE)
        loads = np.array(output.stdout.decode().split()).astype(float)
        device = loads.argmin()
        return f'cuda:{device}'

    except Exception:
        return 'cpu'


def get_utc_now():
    """Get the current datetime, in UTC timezone.

    Returns:
        datetime:
            The current datetime in UTC.
    """
    return datetime.now(tz=timezone.utc)


def calculate_score_time(start):
    """Calculate the total duration in seconds.

    Args:
        start (datetime):
            The start datetime to use to calculate duration.

    Returns:
        float:
            The duration in total seconds.

    """
    return (get_utc_now() - start).total_seconds()


def convert_metadata_to_sdmetrics(metadata_dict):
    """Convert a sdv metadata dictionary into sdmetrics expected metadata."""
    table_name = next(iter(metadata_dict['tables']))
    return metadata_dict['tables'][table_name]
