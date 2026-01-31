"""SDGym - Synthetic data Gym.

SDGym is a framework to benchmark the performance of synthetic data generators for
tabular data.
"""

__author__ = 'DataCebo, Inc.'
__copyright__ = 'Copyright (c) 2022 DataCebo, Inc.'
__email__ = 'info@sdv.dev'
__license__ = 'BSL-1.1'
__version__ = '0.13.0'

import logging

from sdgym.benchmark import (
    benchmark_multi_table,
    benchmark_single_table,
    benchmark_single_table_aws,
    benchmark_multi_table_aws,
)
from sdgym.cli.collect import collect_results
from sdgym.cli.summary import make_summary_spreadsheet
from sdgym.dataset_explorer import DatasetExplorer
from sdgym.datasets import load_dataset
from sdgym.synthesizers import (
    create_synthesizer_variant,
    create_single_table_synthesizer,
    create_multi_table_synthesizer,
)
from sdgym.result_explorer import ResultsExplorer

# Clear the logging wrongfully configured by tensorflow/absl
list(map(logging.root.removeHandler, logging.root.handlers))
list(map(logging.root.removeFilter, logging.root.filters))

__all__ = [
    'DatasetExplorer',
    'ResultsExplorer',
    'benchmark_multi_table',
    'benchmark_multi_table_aws',
    'benchmark_single_table',
    'benchmark_single_table_aws',
    'collect_results',
    'create_multi_table_synthesizer',
    'create_single_table_synthesizer',
    'create_synthesizer_variant',
    'load_dataset',
    'make_summary_spreadsheet',
]
