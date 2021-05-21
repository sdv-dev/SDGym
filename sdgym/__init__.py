"""SDGym - Synthetic data Gym.

SDGym is a framework to benchmark the performance of synthetic data generators for
tabular data.
"""

__author__ = 'MIT Data To AI Lab'
__copyright__ = 'Copyright (c) 2018, MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com'
__license__ = 'MIT'
__version__ = '0.3.1'

from sdgym import benchmark, results, synthesizers
from sdgym.benchmark import run
from sdgym.collect import collect_results
from sdgym.datasets import load_dataset

__all__ = [
    'benchmark',
    'synthesizers',
    'results',
    'run',
    'load_dataset',
    'collect_results'
]
