"""Main SDGym benchmarking module."""

import logging
import multiprocessing
import os
import types
from datetime import datetime
from pathlib import Path

import compress_pickle
import pandas as pd
import tqdm

from sdgym.datasets import load_dataset, load_tables
from sdgym.errors import SDGymError, SDGymTimeout
from sdgym.metrics import get_metrics
from sdgym.progress import TqdmLogger, progress
from sdgym.synthesizers.base import Baseline
from sdgym.utils import format_exception, used_memory, with_timeout

LOGGER = logging.getLogger(__name__)


DEFAULT_DATASETS = [
    "adult",
    "alarm",
    "asia",
    "census",
    "child",
    "covtype",
    "credit",
    "grid",
    "gridr",
    "insurance",
    "intrusion",
    "mnist12",
    "mnist28",
    "news",
    "ring"
]


def _synthesize(synthesizer, real_data, metadata):
    if isinstance(synthesizer, type) and issubclass(synthesizer, Baseline):
        synthesizer = synthesizer().fit_sample

    now = datetime.utcnow()
    synthetic_data = synthesizer(real_data.copy(), metadata)
    elapsed = datetime.utcnow() - now
    return synthetic_data, elapsed


def _prepare_data(real_data, synthetic_data, metadata):
    modality = metadata.modality
    if modality == 'multi-table':
        metadata = metadata.to_dict()
    else:
        table = metadata.get_tables()[0]
        metadata = metadata.get_table_meta(table)
        real_data = real_data[table]
        synthetic_data = synthetic_data[table]

    return real_data, synthetic_data, metadata


def _compute_scores(metrics, real_data, synthetic_data, metadata):
    metrics = get_metrics(metrics, metadata)
    real_data, synthetic_data, metadata_dict = _prepare_data(real_data, synthetic_data, metadata)

    scores = []
    for metric_name, metric in metrics.items():
        error = None
        score = None
        start = datetime.utcnow()
        try:
            LOGGER.info('Computing %s on dataset %s', metric_name, metadata.name)
            score = metric.compute(real_data, synthetic_data, metadata_dict)
        except Exception:
            LOGGER.exception('Metric %s failed on dataset %s. Skipping.',
                             metric_name, metadata.name)
            _, error = format_exception()

        scores.append({
            'metric': metric_name,
            'score': score,
            'error': error,
            'metric_time': (datetime.utcnow() - start).total_seconds()
        })

    return pd.DataFrame(scores)


def __score_synthesizer_on_dataset(synthesizer, metadata, metrics, iteration):
    LOGGER.info('Evaluating %s on dataset %s; iteration %s; %s',
                synthesizer.name, metadata.name, iteration, used_memory())
    real_data = load_tables(metadata)

    LOGGER.info('Running %s on dataset %s; iteration %s; %s',
                synthesizer.name, metadata.name, iteration, used_memory())

    synthetic_data, model_time = _synthesize(synthesizer, real_data.copy(), metadata)

    LOGGER.info('Scoring %s on dataset %s; iteration %s; %s',
                synthesizer.name, metadata.name, iteration, used_memory())
    scores = _compute_scores(metrics, real_data, synthetic_data, metadata)

    return scores, metadata, model_time


def _score_synthesizer_on_dataset(args):
    synthesizer, metadata, metrics, iteration, cache_dir, timeout = args
    synthetic_data = None
    exception = None
    scores = None
    model_time = None
    try:
        if timeout:
            scores, metadata, model_time = with_timeout(
                timeout, __score_synthesizer_on_dataset, synthesizer, metadata, metrics, iteration)
        else:
            scores, metadata, model_time = __score_synthesizer_on_dataset(
                synthesizer, metadata, metrics, iteration)
    except SDGymTimeout:
        LOGGER.error('Timeout running %s on dataset %s; iteration %s',
                     synthesizer.name, metadata.name, iteration)
        scores = pd.DataFrame({
            'error': ['Timeout']
        })
    except Exception:
        LOGGER.exception('Error running %s on dataset %s; iteration %s',
                         synthesizer.name, metadata.name, iteration)
        exception, error = format_exception()
        scores = pd.DataFrame({
            'error': [error]
        })

    finally:
        LOGGER.info('Finished %s on dataset %s; iteration %s; %s',
                    synthesizer.name, metadata.name, iteration, used_memory())

    scores['synthesizer'] = synthesizer.name
    scores['dataset'] = metadata.name
    scores['iteration'] = iteration
    if metadata is not None:
        scores['modality'] = metadata.modality

    if model_time is not None:
        scores['model_time'] = model_time.total_seconds()

    if cache_dir:
        base_path = str(cache_dir / f'{synthesizer.name}_{metadata.name}_{iteration}')
        if scores is not None:
            scores.to_csv(base_path + '_scores.csv', index=False)
        if synthetic_data is not None:
            compress_pickle.dump(synthetic_data, base_path + '.data.gz')
        if exception is not None:
            with open(base_path + '_error.txt', 'w') as error_file:
                error_file.write(exception)

    return scores


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


def _get_synthesizers(synthesizers):
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
        synthesizers = {_get_synthesizer_name(synthesizers): synthesizers}
    if isinstance(synthesizers, (list, tuple)):
        synthesizers = {
            _get_synthesizer_name(synthesizer): synthesizer
            for synthesizer in synthesizers
        }
    elif not isinstance(synthesizers, dict):
        raise TypeError('`synthesizers` can only be a function, a class, a list or a dict')

    for synthesizer_name, synthesizer in synthesizers.items():
        if not hasattr(synthesizer, 'name'):
            synthesizer.name = synthesizer_name

    return list(synthesizers.values())


def _get_dataset_paths(datasets, datasets_path):
    """Build the full path to datasets and validate their existance."""
    if datasets_path is None:
        return datasets or DEFAULT_DATASETS

    datasets_path = Path(datasets_path)
    if datasets is None:
        datasets = datasets_path.iterdir()

    dataset_paths = []
    for dataset in datasets:
        if isinstance(dataset, str):
            dataset = datasets_path / dataset

        if not dataset.exists():
            raise ValueError(f'Dataset {dataset} not found')

        dataset_paths.append(dataset)

    return dataset_paths


def _run_on_dask(scorer_args, verbose):
    """Run the tasks in parallel using dask."""
    try:
        import dask
    except ImportError as ie:
        ie.msg += (
            '\n\nIt seems like `dask` is not installed.\n'
            'Please install `dask` and `distributed` using:\n'
            '\n    pip install dask distributed'
        )
        raise

    scorer = dask.delayed(_score_synthesizer_on_dataset)
    persisted = dask.persist(*[scorer(args) for args in scorer_args])
    if verbose:
        try:
            progress(persisted)
        except ValueError:
            pass

    return dask.compute(*persisted)


def run(synthesizers, datasets=None, datasets_path=None, bucket=None, metrics=None, iterations=1,
        add_leaderboard=True, leaderboard_path=None, replace_existing=True, workers=1,
        cache_dir=None, output_path=None, show_progress=False, timeout=None):
    """Run the SDGym benchmark and return a leaderboard.

    The ``synthesizers`` object can either be a single synthesizer or, an iterable of
    synthesizers or a dict containing synthesizer names as keys and synthesizers as values.

    If ``add_leaderboard`` is ``True``, append the obtained scores to the leaderboard
    stored in the ``lederboard_path``. By default, the leaderboard used is the one which
    is included in the package, which contains the scores obtained by the SDGym Synthesizers.

    If ``replace_existing`` is ``True`` and any of the given synthesizers already existed
    in the leaderboard, the old rows are dropped.

    Args:
        synthesizers (function, class, list, tuple or dict):
            The synthesizer or synthesizers to evaluate. It can be a single synthesizer
            (function or method or class), or an iterable of synthesizers, or a dict
            containing synthesizer names as keys and synthesizers as values. If the input
            is not a dict, synthesizer names will be extracted from the given object.
        datasets (list[str]):
            Names of the datasets to use for the benchmark. Defaults to all the ones available.
        datasets_path (str):
            Path to where the datasets can be found. If not given, use the default path.
        metrics (list[str]):
            List of metrics to apply.
        bucket (str):
            Name of the bucket from which the datasets must be downloaded if not found locally.
        iterations (int):
            Number of iterations to perform over each dataset and synthesizer. Defaults to 3.
        add_leaderboard (bool):
            Whether to append the obtained scores to the previous leaderboard or not. Defaults
            to ``True``.
        leaderboard_path (str):
            Path to where the leaderboard is stored. Defaults to the leaderboard included
            with the package, which contains the scores obtained by the SDGym synthesizers.
        replace_existing (bool):
            Whether to replace old scores or keep them in the returned leaderboard. Defaults
            to ``True``.
        workers (int or str):
            If ``workers`` is given as an integer value other than 0 or 1, a multiprocessing
            Pool is used to distribute the computation across the indicated number of workers.
            If the string ``dask`` is given, the computation is distributed using ``dask``.
            In this case, setting up the ``dask`` cluster and client is expected to be handled
            outside of this function.
        cache_dir (str):
            If a ``cache_dir`` is given, intermediate results are stored in the indicated directory
            as CSV files as they get computted. This allows inspecting results while the benchmark
            is still running and also recovering results in case the process does not finish
            properly. Defaults to ``None``.
        output_path (str):
            If an ``output_path`` is given, the generated leaderboard will be stored in the
            indicated path as a CSV file. The given path must be a complete path including
            the ``.csv`` filename.
        show_progress (bool):
            Whether to use tqdm to keep track of the progress. Defaults to ``True``.
        timeout (int):
            Maximum number of seconds to wait for each dataset to
            finish the evaluation process. If not passed, wait until
            all the datasets are done.

    Returns:
        pandas.DataFrame or None:
            If not ``output_path`` is given, a table containing one row per synthesizer and
            one column for each dataset and metric is returned. Otherwise, there is no output.
    """
    synthesizers = _get_synthesizers(synthesizers)
    datasets = _get_dataset_paths(datasets, datasets_path)

    if cache_dir:
        cache_dir = Path(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

    scorer_args = list()
    for synthesizer in synthesizers:
        for dataset in datasets:
            metadata = load_dataset(dataset, bucket=bucket)
            modalities = getattr(synthesizer, 'MODALITIES', None)
            if not modalities or metadata.modality in modalities:
                for iteration in range(iterations):
                    args = (
                        synthesizer,
                        metadata,
                        metrics,
                        iteration,
                        cache_dir,
                        timeout,
                    )
                    scorer_args.append(args)

    if workers == 'dask':
        scores = _run_on_dask(scorer_args, show_progress)
    else:
        if workers in (0, 1):
            scores = map(_score_synthesizer_on_dataset, scorer_args)
        else:
            pool = multiprocessing.Pool(workers)
            scores = pool.imap_unordered(_score_synthesizer_on_dataset, scorer_args)

        scores = tqdm.tqdm(scores, total=len(scorer_args), file=TqdmLogger())
        if show_progress:
            scores = tqdm.tqdm(scores, total=len(scorer_args))

    if not scores:
        raise SDGymError("No valid Dataset/Synthesizer combination given")

    return pd.concat(scores)
