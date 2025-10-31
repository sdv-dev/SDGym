"""SDGym - Synthetic data Gym.

SDGym is a framework to benchmark the performance of synthetic data generators for
tabular data.
"""

__author__ = 'DataCebo, Inc.'
__copyright__ = 'Copyright (c) 2022 DataCebo, Inc.'
__email__ = 'info@sdv.dev'
__license__ = 'BSL-1.1'
__version__ = '0.11.1.dev0'

import logging

from sdgym.benchmark import benchmark_single_table
from sdgym.cli.collect import collect_results
from sdgym.cli.summary import make_summary_spreadsheet
from sdgym.dataset_explorer import DatasetExplorer
from sdgym.datasets import get_available_datasets, load_dataset
from sdgym.synthesizers import create_sdv_synthesizer_variant, create_single_table_synthesizer
from sdgym.result_explorer import ResultsExplorer

# Clear the logging wrongfully configured by tensorflow/absl
list(map(logging.root.removeHandler, logging.root.handlers))
list(map(logging.root.removeFilter, logging.root.filters))

__all__ = [
    'DatasetExplorer',
    'ResultsExplorer',
    'benchmark_single_table',
    'collect_results',
    'create_sdv_synthesizer_variant',
    'create_single_table_synthesizer',
    'get_available_datasets',
    'load_dataset',
    'make_summary_spreadsheet',
]
