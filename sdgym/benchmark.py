"""Main SDGym benchmarking module."""

import concurrent
import logging
import multiprocessing
import os
import pickle
import tracemalloc
import warnings
from datetime import datetime
from pathlib import Path

import compress_pickle
import numpy as np
import pandas as pd
import tqdm
from sdmetrics.reports.multi_table import QualityReport as MultiTableQualityReport
from sdmetrics.reports.single_table import QualityReport as SingleTableQualityReport

from sdgym.datasets import get_dataset_paths, load_dataset
from sdgym.errors import SDGymError
from sdgym.metrics import get_metrics
from sdgym.progress import TqdmLogger, progress
from sdgym.s3 import is_s3_path, write_csv, write_file
from sdgym.synthesizers import CTGANSynthesizer, FastMLPreset, GaussianCopulaSynthesizer
from sdgym.synthesizers.base import BaselineSynthesizer
from sdgym.utils import (
    format_exception, get_duplicates, get_num_gpus, get_size_of, get_synthesizers, used_memory)

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
N_BYTES_IN_MB = 1000 * 1000


def _validate_inputs(output_filepath, detailed_results_folder, synthesizers, custom_synthesizers):
    if output_filepath and os.path.exists(output_filepath):
        raise ValueError(
            f'{output_filepath} already exists. '
            'Please provide a file that does not already exist.'
        )

    if detailed_results_folder and os.path.exists(detailed_results_folder):
        raise ValueError(
            f'{detailed_results_folder} already exists. '
            'Please provide a folder that does not already exist.'
        )

    duplicates = get_duplicates(synthesizers) if synthesizers else {}
    if custom_synthesizers:
        duplicates.update(get_duplicates(custom_synthesizers))
    if len(duplicates) > 0:
        raise ValueError(
            'Synthesizers must be unique. Please remove repeated values in the `synthesizers` '
            'and `custom_synthesizers` parameters.'
        )


def _create_detailed_results_directory(detailed_results_folder):
    if detailed_results_folder and not is_s3_path(detailed_results_folder):
        detailed_results_folder = Path(detailed_results_folder)
        os.makedirs(detailed_results_folder, exist_ok=True)


def _generate_job_args_list(limit_dataset_size, sdv_datasets, additional_datasets_folder,
                            sdmetrics, detailed_results_folder, timeout,
                            compute_quality_score, synthesizers, custom_synthesizers):
    # Get list of synthesizer objects
    synthesizers = [] if synthesizers is None else synthesizers
    custom_synthesizers = [] if custom_synthesizers is None else custom_synthesizers
    synthesizers = get_synthesizers(synthesizers + custom_synthesizers)

    # Get list of dataset paths
    sdv_datasets = [] if sdv_datasets is None else get_dataset_paths(datasets=sdv_datasets)
    additional_datasets = [] if additional_datasets_folder is None else get_dataset_paths(
        bucket=additional_datasets_folder)
    datasets = sdv_datasets + additional_datasets

    job_tuples = []
    for dataset in datasets:
        for synthesizer in synthesizers:
            job_tuples.append((synthesizer, dataset))

    job_args_list = []
    for synthesizer, dataset in job_tuples:
        data, metadata_dict = load_dataset(
            'single_table',
            dataset,
            limit_dataset_size=limit_dataset_size
        )

        args = (
            synthesizer,
            data,
            metadata_dict,
            sdmetrics,
            detailed_results_folder,
            timeout,
            compute_quality_score,
            dataset.name,
            'single_table'
        )
        job_args_list.append(args)

    return job_args_list


def _synthesize(synthesizer_dict, real_data, metadata):
    synthesizer = synthesizer_dict['synthesizer']
    assert issubclass(
        synthesizer, BaselineSynthesizer), '`synthesizer` must be a synthesizer class'

    synthesizer_object = synthesizer()
    get_synthesizer = synthesizer_object.get_trained_synthesizer
    sample_from_synthesizer = synthesizer_object.sample_from_synthesizer
    data = real_data.copy()
    num_samples = len(data)

    tracemalloc.start()
    now = datetime.utcnow()
    synthesizer_obj = get_synthesizer(data, metadata)
    synthesizer_size = len(pickle.dumps(synthesizer_obj)) / N_BYTES_IN_MB
    train_now = datetime.utcnow()
    synthetic_data = sample_from_synthesizer(synthesizer_obj, num_samples)
    sample_now = datetime.utcnow()

    peak_memory = tracemalloc.get_traced_memory()[1] / N_BYTES_IN_MB
    tracemalloc.stop()
    tracemalloc.clear_traces()

    return synthetic_data, train_now - now, sample_now - train_now, synthesizer_size, peak_memory


def _compute_scores(metrics, real_data, synthetic_data, metadata,
                    output, compute_quality_score, modality, dataset_name):
    metrics = metrics or []
    if len(metrics) > 0:
        metrics, metric_kwargs = get_metrics(metrics, modality='single-table')
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
                LOGGER.info('Computing %s on dataset %s', metric_name, dataset_name)
                metric_args = (real_data, synthetic_data, metadata)
                score = metric.compute(*metric_args, **metric_kwargs.get(metric_name, {}))
                normalized_score = metric.normalize(score)
            except Exception:
                LOGGER.exception(
                    'Metric %s failed on dataset %s. Skipping.', metric_name, dataset_name)
                _, error = format_exception()

            scores[-1].update({
                'score': score,
                'normalized_score': normalized_score,
                'error': error,
                'metric_time': (datetime.utcnow() - start).total_seconds()
            })
            output['scores'] = scores  # re-inject list to multiprocessing output

    if compute_quality_score:
        start = datetime.utcnow()
        if modality == 'single_table':
            quality_report = SingleTableQualityReport()
        else:
            quality_report = MultiTableQualityReport()

        quality_report.generate(real_data, synthetic_data, metadata, verbose=False)
        output['quality_score_time'] = (datetime.utcnow() - start).total_seconds()
        output['quality_score'] = quality_report.get_score()


def _score(synthesizer, data, metadata, metrics, output=None,
           compute_quality_score=False, modality=None, dataset_name=None):
    if output is None:
        output = {}

    output['timeout'] = True  # To be deleted if there is no error
    output['error'] = 'Load Timeout'  # To be deleted if there is no error
    try:
        LOGGER.info(
            'Running %s on %s dataset %s; %s',
            synthesizer['name'], modality, dataset_name, used_memory()
        )

        output['dataset_size'] = get_size_of(data) / N_BYTES_IN_MB
        output['error'] = 'Synthesizer Timeout'  # To be deleted if there is no error
        synthetic_data, train_time, sample_time, synthesizer_size, peak_memory = _synthesize(
            synthesizer, data.copy(), metadata)

        output['synthetic_data'] = synthetic_data
        output['train_time'] = train_time.total_seconds()
        output['sample_time'] = sample_time.total_seconds()
        output['synthesizer_size'] = synthesizer_size
        output['peak_memory'] = peak_memory

        LOGGER.info(
            'Scoring %s on %s dataset %s; %s',
            synthesizer['name'], modality, dataset_name, used_memory()
        )

        del output['error']   # No error so far. _compute_scores tracks its own errors by metric
        _compute_scores(
            metrics,
            data,
            synthetic_data,
            metadata,
            output,
            compute_quality_score,
            modality,
            dataset_name
        )

        output['timeout'] = False  # There was no timeout

    except Exception:
        LOGGER.exception('Error running %s on dataset %s;', synthesizer['name'], dataset_name)

        exception, error = format_exception()
        output['exception'] = exception
        output['error'] = error
        output['timeout'] = False  # There was no timeout

    finally:
        LOGGER.info(
            'Finished %s on dataset %s; %s', synthesizer['name'], dataset_name, used_memory())

    return output


def _score_with_timeout(timeout, synthesizer, data, metadata, metrics,
                        compute_quality_score=False, modality=None, dataset_name=None):
    with multiprocessing.Manager() as manager:
        output = manager.dict()
        process = multiprocessing.Process(
            target=_score,
            args=(
                synthesizer,
                data,
                metadata,
                metrics,
                output,
                compute_quality_score,
                modality,
                dataset_name
            ),
        )

        process.start()
        process.join(timeout)
        process.terminate()

        output = dict(output)
        if output.get('timeout'):
            LOGGER.error('Timeout running %s on dataset %s;', synthesizer['name'], dataset_name)

        return output


def _format_output(output, name, dataset_name, compute_quality_score, cache_dir):
    evaluate_time = None
    if 'scores' in output or 'quality_score_time' in output:
        evaluate_time = output.get('quality_score_time', 0)

    for score in output.get('scores', []):
        if score['metric'] == 'NewRowSynthesis':
            evaluate_time += score['metric_time']

    scores = pd.DataFrame({
        'Synthesizer': [name],
        'Dataset': [dataset_name],
        'Dataset_Size_MB': [output.get('dataset_size')],
        'Train_Time': [output.get('train_time')],
        'Peak_Memory_MB': [output.get('peak_memory')],
        'Synthesizer_Size_MB': [output.get('synthesizer_size')],
        'Sample_Time': [output.get('sample_time')],
        'Evaluate_Time': [evaluate_time],
    })

    if compute_quality_score:
        scores.insert(len(scores.columns), 'Quality_Score', output.get('quality_score'))

    for score in output.get('scores', []):
        scores.insert(len(scores.columns), score['metric'], score['normalized_score'])

    if 'error' in output:
        scores['error'] = output['error']

    if cache_dir:
        cache_dir_name = str(cache_dir)
        base_path = f'{cache_dir_name}/{name}_{dataset_name}'
        if scores is not None:
            write_csv(scores, f'{base_path}_scores.csv', None, None)
        if 'synthetic_data' in output:
            synthetic_data = compress_pickle.dumps(output['synthetic_data'], compression='gzip')
            write_file(synthetic_data, f'{base_path}.data.gz', None, None)
        if 'exception' in output:
            exception = output['exception'].encode('utf-8')
            write_file(exception, f'{base_path}_error.txt', None, None)

    return scores


def _run_job(args):
    # Reset random seed
    np.random.seed()

    synthesizer, data, metadata, metrics, cache_dir, \
        timeout, compute_quality_score, dataset_name, modality = args

    name = synthesizer['name']
    LOGGER.info('Evaluating %s on dataset %s with timeout %ss; %s',
                name, dataset_name, timeout, used_memory())

    output = {}
    try:
        if timeout:
            output = _score_with_timeout(
                timeout=timeout,
                synthesizer=synthesizer,
                data=data,
                metadata=metadata,
                metrics=metrics,
                compute_quality_score=compute_quality_score,
                modality=modality,
                dataset_name=dataset_name
            )
        else:
            output = _score(
                synthesizer=synthesizer,
                data=data,
                metadata=metadata,
                metrics=metrics,
                compute_quality_score=compute_quality_score,
                modality=modality,
                dataset_name=dataset_name
            )
    except Exception as error:
        output['exception'] = error

    scores = _format_output(output, name, dataset_name, compute_quality_score, cache_dir)

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


def _run_jobs(multi_processing_config, job_args_list, show_progress):
    workers = 1
    if multi_processing_config:
        if multi_processing_config['package_name'] == 'dask':
            workers = 'dask'
            scores = _run_on_dask(job_args_list, show_progress)
        else:
            num_gpus = get_num_gpus()
            if num_gpus > 0:
                workers = num_gpus
            else:
                workers = multiprocessing.cpu_count()

    if workers in (0, 1):
        scores = map(_run_job, job_args_list)
    elif workers != 'dask':
        pool = concurrent.futures.ProcessPoolExecutor(workers)
        scores = pool.map(_run_job, job_args_list)

    if show_progress:
        scores = tqdm.tqdm(scores, total=len(job_args_list), position=0, leave=True)
    else:
        scores = tqdm.tqdm(
            scores, total=len(job_args_list), file=TqdmLogger(), position=0, leave=True)

    if not scores:
        raise SDGymError('No valid Dataset/Synthesizer combination given.')

    scores = pd.concat(scores, ignore_index=True)

    return scores


def _get_empty_dataframe(compute_quality_score, sdmetrics):
    warnings.warn('No datasets/synthesizers found.')

    scores = pd.DataFrame({
        'Synthesizer': [],
        'Dataset': [],
        'Dataset_Size_MB': [],
        'Train_Time': [],
        'Peak_Memory_MB': [],
        'Synthesizer_Size_MB': [],
        'Sample_Time': [],
        'Evaluate_Time': [],
    })

    if compute_quality_score:
        scores['Quality_Score'] = []
    if sdmetrics:
        for metric in sdmetrics:
            scores[metric[0]] = []

    return scores


def benchmark_single_table(synthesizers=DEFAULT_SYNTHESIZERS, custom_synthesizers=None,
                           sdv_datasets=DEFAULT_DATASETS, additional_datasets_folder=None,
                           limit_dataset_size=False, compute_quality_score=True,
                           sdmetrics=DEFAULT_METRICS, timeout=None, output_filepath=None,
                           detailed_results_folder=None, show_progress=False,
                           multi_processing_config=None):
    """Run the SDGym benchmark on single-table datasets.

    Args:
        synthesizers (list[string]):
            The synthesizer(s) to evaluate. Defaults to ``[GaussianCopulaSynthesizer, FASTMLPreset,
            CTGANSynthesizer]``. The available options are:

                - ``GaussianCopulaSynthesizer``
                - ``CTGANSynthesizer``
                - ``CopulaGANSynthesizer``
                - ``TVAESynthesizer``
                - ``FASTMLPreset``

        custom_synthesizers (list[class] or ``None``):
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
        compute_quality_score (bool):
            Whether or not to evaluate an overall quality score.
        sdmetrics (list[str]):
            A list of the different SDMetrics to use. If you'd like to input specific parameters
            into the metric, provide a tuple with the metric name followed by a dictionary of
            the parameters.
        timeout (int or ``None``):
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
    _validate_inputs(output_filepath, detailed_results_folder, synthesizers, custom_synthesizers)

    _create_detailed_results_directory(detailed_results_folder)

    job_args_list = _generate_job_args_list(
        limit_dataset_size, sdv_datasets, additional_datasets_folder, sdmetrics,
        detailed_results_folder, timeout, compute_quality_score, synthesizers, custom_synthesizers)

    if job_args_list:
        scores = _run_jobs(multi_processing_config, job_args_list, show_progress)

    # If no synthesizers/datasets are passed, return an empty dataframe
    else:
        scores = _get_empty_dataframe(compute_quality_score, sdmetrics)

    if output_filepath:
        write_csv(scores, output_filepath, None, None)

    return scores
