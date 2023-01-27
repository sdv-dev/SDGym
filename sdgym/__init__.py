"""SDGym - Synthetic data Gym.

SDGym is a framework to benchmark the performance of synthetic data generators for
tabular data.
"""

__author__ = 'DataCebo, Inc.'
__copyright__ = 'Copyright (c) 2022 DataCebo, Inc.'
__email__ = 'info@sdv.dev'
__license__ = 'BSL-1.1'
__version__ = '0.6.0.dev1'

import logging

from sdgym import benchmark, synthesizers
from sdgym.benchmark import benchmark_single_table
from sdgym.collect import collect_results
from sdgym.datasets import load_dataset
from sdgym.summary import make_summary_spreadsheet

# Clear the logging wrongfully configured by tensorflow/absl
list(map(logging.root.removeHandler, logging.root.handlers))
list(map(logging.root.removeFilter, logging.root.filters))

__all__ = [
    'benchmark',
    'synthesizers',
    'load_dataset',
    'collect_results',
    'make_summary_spreadsheet',
    'benchmark_single_table',
]
