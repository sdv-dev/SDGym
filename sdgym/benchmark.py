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
from sdgym.synthesizers import CTGANSynthesizer, FastMLPreset, GaussianCopulaSynthesizer
from sdgym.synthesizers.base import BaselineSynthesizer, SingleTableBaselineSynthesizer
from sdgym.synthesizers.utils import get_num_gpus
from sdgym.utils import (
    build_synthesizer, format_exception, get_synthesizers, import_object, used_memory)

LOGGER = logging.getLogger(__name__)
DEFAULT_SYNTHESIZERS = [GaussianCopulaSynthesizer, FastMLPreset, CTGANSynthesizer]
DEFAULT_DATASETS = [
    'adult',
    'alarm',
    'census',
    'child',
    'expedia_hotel_logs',
    'insurance',
    'intrusion',
    'news',
    'covtype',
]
DEFAULT_METRICS = [('NewRowSynthesis', {'synthetic_sample_size': 1000})]


def _synthesize(synthesizer_dict, real_data, metadata):
    synthesizer = synthesizer_dict['synthesizer']
    get_synthesizer = None
    sample_from_synthesizer = None

    if isinstance(synthesizer, str):
        synthesizer = import_object(synthesizer)

    if isinstance(synthesizer, type):
        if issubclass(synthesizer, BaselineSynthesizer):
            s_obj = synthesizer()
            get_synthesizer = s_obj.get_trained_synthesizer
            sample_from_synthesizer = s_obj.sample_from_synthesizer
        else:
            get_synthesizer, sample_from_synthesizer = build_synthesizer(
                synthesizer, synthesizer_dict)

    if isinstance(synthesizer, tuple):
        get_synthesizer, sample_from_synthesizer = synthesizer

    data = real_data.copy()
    num_samples = None
    modalities = getattr(synthesizer, 'MODALITIES', [])
    is_single_table = (
        isinstance(synthesizer, type)
        and issubclass(synthesizer, SingleTableBaselineSynthesizer)
    ) or (
        len(modalities) == 1
        and 'single-table' in modalities
    )
    if is_single_table:
        table_name = list(real_data.keys())[0]
        metadata = metadata.get_table_meta(table_name)
        data = list(real_data.values())[0]
        num_samples = len(data)

    now = datetime.utcnow()
    synthesizer_obj = get_synthesizer(data, metadata)
    train_now = datetime.utcnow()
    synthetic_data = sample_from_synthesizer(synthesizer_obj, num_samples)
    sample_now = datetime.utcnow()

    if is_single_table:
        synthetic_data = {list(real_data.keys())[0]: synthetic_data}

    return synthetic_data, train_now - now, sample_now - train_now


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
    metrics, metric_kwargs = get_metrics(metrics, metadata)
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
            score = metric.compute(*metric_args, **metric_kwargs.get(metric_name, {}))
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
        synthetic_data, train_time, sample_time = _synthesize(
            synthesizer, real_data.copy(), metadata)
        output['synthetic_data'] = synthetic_data
        output['train_time'] = train_time.total_seconds()
        output['sample_time'] = sample_time.total_seconds()

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
    scores['train_time'] = output.get('train_time')
    scores['sample_time'] = output.get('sample_time')
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


def benchmark_single_table(synthesizers=DEFAULT_SYNTHESIZERS, custom_synthesizers=None,
                           sdv_datasets=DEFAULT_DATASETS, additional_datasets_folder=None,
                           limit_dataset_size=False, evaluate_quality=True,
                           sdmetrics=DEFAULT_METRICS, timeout=None, output_filepath=None,
                           detailed_results_folder=None, show_progress=False,
                           multi_processing_config=None):
    """Run the SDGym benchmark on single-table datasets.

    The ``synthesizers`` object can either be a single synthesizer or, an iterable of
    synthesizers or a dict containing synthesizer names as keys and synthesizers as values.

    Args:
        synthesizers (list[string]):
            The synthesizer(s) to evaluate. Defaults to ``[GaussianCopulaSynthesizer, FASTMLPreset,
            CTGANSynthesizer]``. The available options are:

                - ``GaussianCopulaSynthesizer``
                - ``CTGANSynthesizer``
                - ``CopulaGANSynthesizer``
                - ``TVAESynthesizer``
                - ``FASTMLPreset``
                - any custom created synthesizer or variant

        custom_synthesizers (list[class]):
            A list of custom synthesizer classes to use. These can be completely custom or
            they can be synthesizer variants (the output from ``create_single_table_synthesizer``
            or ``create_sdv_synthesizer_variant``). Defaults to ``None``.
        sdv_datasets (list[str] or ``None``):
            Names of the SDV demo datasets to use for the benchmark. Defaults to
            ``[adult, alarm, census, child, expedia_hotel_logs, insurance, intrusion, news,
            covtype]``. Use ``None`` to disable using any sdv datasets.
        additional_datasets_folder (str or ``None``):
            The path to a folder (local or an S3 bucket). Datasets found in this folder are
            run in addition to the SDV datasets. If ``None``, no additional datasets are used.
        limit_dataset_size (bool):
            Use this flag to limit the size of the datasets for faster evaluation. If ``True``,
            limit the size of every table to 1,000 rows (randomly sampled) and the first 10
            columns.
        evaluate_quality (bool):
            Whether or not to evaluate an overall quality score.
        sdmetrics (list[str]):
            A list of the different SDMetrics to use. If you'd like to input specific parameters
            into the metric, provide a tuple with the metric name followed by a dictionary of
            the parameters.
        timeout (bool or ``None``):
            The maximum number of seconds to wait for synthetic data creation. If ``None``, no
            timeout is enforced.
        output_filepath (str or ``None``):
            A file path for where to write the output as a csv file. If ``None``, no output
            is written.
        detailed_results_folder (str or ``None``):
            The folder for where to store the intermediary results. If ``None``, do not store
            the intermediate results anywhere.
        show_progress (bool):
            Whether to use tqdm to keep track of the progress. Defaults to ``False``.
        multi_processing_config (dict or ``None``):
            The config to use if multi-processing is desired. For example,
            {
             'package_name': 'dask' or 'multiprocessing',
             'num_workers': 4
            }

    Returns:
        pandas.DataFrame:
            A table containing one row per synthesizer + dataset + metric.
    """
    if detailed_results_folder and not is_s3_path(detailed_results_folder):
        detailed_results_folder = Path(detailed_results_folder)
        os.makedirs(detailed_results_folder, exist_ok=True)

    max_rows, max_columns = (1000, 10) if limit_dataset_size else (None, None)

    run_id = os.getenv('RUN_ID') or str(uuid.uuid4())[:10]

    synthesizers = get_synthesizers(synthesizers)
    if custom_synthesizers:
        custom_synthesizers = get_synthesizers(custom_synthesizers)
        synthesizers.extend(custom_synthesizers)

    datasets = get_dataset_paths(sdv_datasets, None, None, None, None)
    if additional_datasets_folder:
        additional_datasets = get_dataset_paths(None, None, additional_datasets_folder, None, None)
        datasets.extend(additional_datasets)

    job_tuples = list()
    for dataset in datasets:
        for synthesizer in synthesizers:
            job_tuples.append((synthesizer, dataset, 1))

    job_args = list()
    for synthesizer, dataset, iteration in job_tuples:
        metadata = load_dataset('single_table', dataset, max_columns=max_columns)
        dataset_modality = metadata.modality
        synthesizer_modalities = synthesizer.get('modalities')
        if (dataset_modality and dataset_modality != 'single-table') or (
            synthesizer_modalities and 'single-table' not in synthesizer_modalities
        ):
            continue

        args = (
            synthesizer,
            metadata,
            sdmetrics,
            iteration,
            detailed_results_folder,
            timeout,
            run_id,
            None,
            None,
            max_rows,
        )
        job_args.append(args)

    workers = 1
    if multi_processing_config:
        if multi_processing_config['package_name'] == 'dask':
            workers = 'dask'
            scores = _run_on_dask(job_args, show_progress)
        else:
            num_gpus = get_num_gpus()
            if num_gpus > 0:
                workers = num_gpus
            else:
                workers = multiprocessing.cpu_count()

    if workers in (0, 1):
        scores = map(_run_job, job_args)
    elif workers != 'dask':
        pool = concurrent.futures.ProcessPoolExecutor(workers)
        scores = pool.map(_run_job, job_args)

    scores = tqdm.tqdm(scores, total=len(job_args), file=TqdmLogger())
    if show_progress:
        scores = tqdm.tqdm(scores, total=len(job_args))

    if not scores:
        raise SDGymError("No valid Dataset/Synthesizer combination given")

    scores = pd.concat(scores)

    if output_filepath:
        write_csv(scores, output_filepath)

    return scores
