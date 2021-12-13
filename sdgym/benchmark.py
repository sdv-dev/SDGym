"""Main SDGym benchmarking module."""

import concurrent
import logging
import multiprocessing
import os
import uuid
from datetime import datetime
from pathlib import Path

import compress_pickle
import numpy as np
import pandas as pd
import tqdm

from sdgym.datasets import get_dataset_paths, load_dataset, load_tables
from sdgym.errors import SDGymError
from sdgym.metrics import get_metrics
from sdgym.progress import TqdmLogger, progress
from sdgym.s3 import is_s3_path, write_csv, write_file
from sdgym.synthesizers.base import Baseline
from sdgym.synthesizers.utils import get_num_gpus
from sdgym.utils import (
    build_synthesizer, format_exception, get_synthesizers, import_object, used_memory)

LOGGER = logging.getLogger(__name__)


def _synthesize(synthesizer_dict, real_data, metadata):
    synthesizer = synthesizer_dict['synthesizer']

    if isinstance(synthesizer, str):
        synthesizer = import_object(synthesizer)

    if isinstance(synthesizer, type):
        if issubclass(synthesizer, Baseline):
            synthesizer = synthesizer().fit_sample
        else:
            synthesizer = build_synthesizer(synthesizer, synthesizer_dict)

    now = datetime.utcnow()
    synthetic_data = synthesizer(real_data.copy(), metadata)
    elapsed = datetime.utcnow() - now
    return synthetic_data, elapsed


def _prepare_metric_args(real_data, synthetic_data, metadata):
    modality = metadata.modality
    if modality == 'multi-table':
        metadata = metadata.to_dict()
    else:
        table = metadata.get_tables()[0]
        metadata = metadata.get_table_meta(table)
        real_data = real_data[table]
        synthetic_data = synthetic_data[table]

    return real_data, synthetic_data, metadata


def _compute_scores(metrics, real_data, synthetic_data, metadata, output):
    metrics = get_metrics(metrics, metadata)
    metric_args = _prepare_metric_args(real_data, synthetic_data, metadata)

    scores = []
    output['scores'] = scores
    for metric_name, metric in metrics.items():
        scores.append({
            'metric': metric_name,
            'error': 'Metric Timeout',
        })
        output['scores'] = scores  # re-inject list to multiprocessing output

        error = None
        score = None
        normalized_score = None
        start = datetime.utcnow()
        try:
            LOGGER.info('Computing %s on dataset %s', metric_name, metadata._metadata['name'])
            score = metric.compute(*metric_args)
            normalized_score = metric.normalize(score)
        except Exception:
            LOGGER.exception('Metric %s failed on dataset %s. Skipping.',
                             metric_name, metadata._metadata['name'])
            _, error = format_exception()

        scores[-1].update({
            'score': score,
            'normalized_score': normalized_score,
            'error': error,
            'metric_time': (datetime.utcnow() - start).total_seconds()
        })
        output['scores'] = scores  # re-inject list to multiprocessing output


def _score(synthesizer, metadata, metrics, iteration, output=None, max_rows=None):
    if output is None:
        output = {}

    name = synthesizer['name']

    output['timeout'] = True  # To be deleted if there is no error
    output['error'] = 'Load Timeout'  # To be deleted if there is no error
    try:
        real_data = load_tables(metadata, max_rows)

        LOGGER.info('Running %s on %s dataset %s; iteration %s; %s',
                    name, metadata.modality, metadata._metadata['name'], iteration, used_memory())

        output['error'] = 'Synthesizer Timeout'  # To be deleted if there is no error
        synthetic_data, model_time = _synthesize(synthesizer, real_data.copy(), metadata)
        output['synthetic_data'] = synthetic_data
        output['model_time'] = model_time.total_seconds()

        LOGGER.info('Scoring %s on %s dataset %s; iteration %s; %s',
                    name, metadata.modality, metadata._metadata['name'], iteration, used_memory())

        del output['error']   # No error so far. _compute_scores tracks its own errors by metric
        _compute_scores(metrics, real_data, synthetic_data, metadata, output)

        output['timeout'] = False  # There was no timeout

    except Exception:
        LOGGER.exception('Error running %s on dataset %s; iteration %s',
                         name, metadata._metadata['name'], iteration)
        exception, error = format_exception()
        output['exception'] = exception
        output['error'] = error
        output['timeout'] = False  # There was no timeout

    finally:
        LOGGER.info('Finished %s on dataset %s; iteration %s; %s',
                    name, metadata._metadata['name'], iteration, used_memory())

    return output


def _score_with_timeout(timeout, synthesizer, metadata, metrics, iteration):
    with multiprocessing.Manager() as manager:
        output = manager.dict()
        process = multiprocessing.Process(
            target=_score,
            args=(synthesizer, metadata, metrics, iteration, output),
        )

        process.start()
        process.join(timeout)
        process.terminate()

        output = dict(output)
        if output['timeout']:
            LOGGER.error('Timeout running %s on dataset %s; iteration %s',
                         synthesizer['name'], metadata._metadata['name'], iteration)

        return output


def _run_job(args):
    # Reset random seed
    np.random.seed()

    synthesizer, metadata, metrics, iteration, cache_dir, \
        timeout, run_id, aws_key, aws_secret, max_rows = args

    name = synthesizer['name']
    dataset_name = metadata._metadata['name']

    LOGGER.info('Evaluating %s on %s dataset %s with timeout %ss; iteration %s; %s',
                name, metadata.modality, dataset_name, timeout, iteration, used_memory())

    if timeout:
        output = _score_with_timeout(timeout, synthesizer, metadata, metrics, iteration)
    else:
        output = _score(synthesizer, metadata, metrics, iteration, max_rows=max_rows)

    scores = output.get('scores')
    if not scores:
        scores = pd.DataFrame({'score': [None]})
    else:
        scores = pd.DataFrame(scores)

    scores.insert(0, 'synthesizer', name)
    scores.insert(1, 'dataset', metadata._metadata['name'])
    scores.insert(2, 'modality', metadata.modality)
    scores.insert(3, 'iteration', iteration)
    scores['model_time'] = output.get('model_time')
    scores['run_id'] = run_id

    if 'error' in output:
        scores['error'] = output['error']

    if cache_dir:
        cache_dir_name = str(cache_dir)
        base_path = f'{cache_dir_name}/{name}_{dataset_name}_{iteration}_{run_id}'
        if scores is not None:
            write_csv(scores, f'{base_path}_scores.csv', aws_key, aws_secret)
        if 'synthetic_data' in output:
            synthetic_data = compress_pickle.dumps(output['synthetic_data'], compression='gzip')
            write_file(synthetic_data, f'{base_path}.data.gz', aws_key, aws_secret)
        if 'exception' in output:
            exception = output['exception'].encode('utf-8')
            write_file(exception, f'{base_path}_error.txt', aws_key, aws_secret)

    return scores


def _run_on_dask(jobs, verbose):
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

    scorer = dask.delayed(_run_job)
    persisted = dask.persist(*[scorer(args) for args in jobs])
    if verbose:
        try:
            progress(persisted)
        except ValueError:
            pass

    return dask.compute(*persisted)


def run(synthesizers=None, datasets=None, datasets_path=None, modalities=None, bucket=None,
        metrics=None, iterations=1, workers=1, cache_dir=None, show_progress=False,
        timeout=None, output_path=None, aws_key=None, aws_secret=None, jobs=None,
        max_rows=None, max_columns=None):
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
        modalities (list[str]):
            Filter datasets by the given modalities. If not given, filter datasets by the
            synthesizer modalities.
        metrics (list[str]):
            List of metrics to apply.
        bucket (str):
            Name of the bucket from which the datasets must be downloaded if not found locally.
        iterations (int):
            Number of iterations to perform over each dataset and synthesizer. Defaults to 1.
        workers (int or str):
            If ``workers`` is given as an integer value other than 0 or 1, a multiprocessing
            Pool is used to distribute the computation across the indicated number of workers.
            If ``workers`` is -1, the number of workers will be automatically determined by
            the number of GPUs (if available) or the number of CPU cores. If the string ``dask``
            is given, the computation is distributed using ``dask``. In this case, setting up the
            ``dask`` cluster and client is expected to be handled outside of this function.
        cache_dir (str):
            If a ``cache_dir`` is given, intermediate results are stored in the indicated directory
            as CSV files as they get computted. This allows inspecting results while the benchmark
            is still running and also recovering results in case the process does not finish
            properly. Defaults to ``None``.
        show_progress (bool):
            Whether to use tqdm to keep track of the progress. Defaults to ``False``.
        timeout (int):
            Maximum number of seconds to wait for each dataset to
            finish the evaluation process. If not passed, wait until
            all the datasets are done.
        output_path (str):
            If an ``output_path`` is given, the generated leaderboard will be stored in the
            indicated path as a CSV file. The given path must be a complete path including
            the ``.csv`` filename.
        aws_key (str):
            If an ``aws_key`` is provided, the given access key id will be used to read
            from the specified bucket.
        aws_secret (str):
            If an ``aws_secret`` is provided, the given secret access key will be used to read
            from the specified bucket.
        jobs (list[tuple]):
            List of jobs to execute, as a sequence of tuple-like objects containing synthesizer,
            dataset and iteration-id specifications. If not passed, the jobs list is build by
            combining the synthesizers and datasets given instead.
        max_rows (int):
            Cap the number of rows to model from each dataset. Rows will be selected in order.
        max_columns (int):
            Cap the number of columns to model from each dataset.
            Columns will be selected in order.

    Returns:
        pandas.DataFrame:
            A table containing one row per synthesizer + dataset + metric + iteration.
    """
    if cache_dir and not is_s3_path(cache_dir):
        cache_dir = Path(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

    run_id = os.getenv('RUN_ID') or str(uuid.uuid4())[:10]

    if workers == -1:
        num_gpus = get_num_gpus()
        if num_gpus > 0:
            workers = num_gpus
        else:
            workers = multiprocessing.cpu_count()

    if jobs is None:
        synthesizers = get_synthesizers(synthesizers)
        datasets = get_dataset_paths(datasets, datasets_path, bucket, aws_key, aws_secret)

        job_tuples = list()
        for dataset in datasets:
            for synthesizer in synthesizers:
                for iteration in range(iterations):
                    job_tuples.append((synthesizer, dataset, iteration))

    else:
        job_tuples = list()
        for synthesizer, dataset, iteration in jobs:
            job_tuples.append((
                get_synthesizers([synthesizer])[0],
                get_dataset_paths([dataset], datasets_path, bucket, aws_key, aws_secret)[0],
                iteration
            ))

    job_args = list()
    for synthesizer, dataset, iteration in job_tuples:
        metadata = load_dataset(dataset, max_columns=max_columns)
        modalities_ = modalities or synthesizer.get('modalities')
        if not modalities_ or metadata.modality in modalities_:
            args = (
                synthesizer,
                metadata,
                metrics,
                iteration,
                cache_dir,
                timeout,
                run_id,
                aws_key,
                aws_secret,
                max_rows,
            )
            job_args.append(args)

    if workers == 'dask':
        scores = _run_on_dask(job_args, show_progress)
    else:
        if workers in (0, 1):
            scores = map(_run_job, job_args)
        else:
            pool = concurrent.futures.ProcessPoolExecutor(workers)
            scores = pool.map(_run_job, job_args)

        scores = tqdm.tqdm(scores, total=len(job_args), file=TqdmLogger())
        if show_progress:
            scores = tqdm.tqdm(scores, total=len(job_args))

    if not scores:
        raise SDGymError("No valid Dataset/Synthesizer combination given")

    scores = pd.concat(scores)

    if output_path:
        write_csv(scores, output_path, aws_key, aws_secret)

    return scores
