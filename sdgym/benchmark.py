"""Main SDGym benchmarking module."""

import functools
import gzip
import logging
import math
import multiprocessing
import os
import re
import textwrap
import threading
import tracemalloc
import warnings
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from typing import Any, NamedTuple, Optional
from urllib.parse import urlparse

import boto3
import cloudpickle
import numpy as np
import pandas as pd
import tqdm
import yaml
from botocore.config import Config
from sdmetrics.reports.multi_table import (
    DiagnosticReport as MultiTableDiagnosticReport,
)
from sdmetrics.reports.multi_table import (
    QualityReport as MultiTableQualityReport,
)
from sdmetrics.reports.single_table import (
    DiagnosticReport as SingleTableDiagnosticReport,
)
from sdmetrics.reports.single_table import (
    QualityReport as SingleTableQualityReport,
)
from sdmetrics.single_table import DCRBaselineProtection

from sdgym.datasets import _load_dataset_with_client, get_dataset_paths
from sdgym.errors import BenchmarkError, SDGymError
from sdgym.metrics import get_metrics
from sdgym.progress import TqdmLogger
from sdgym.result_writer import LocalResultsWriter, S3ResultsWriter
from sdgym.s3 import (
    S3_PREFIX,
    S3_REGION,
    is_s3_path,
    parse_s3_path,
)
from sdgym.synthesizers import MultiTableUniformSynthesizer, UniformSynthesizer
from sdgym.synthesizers.base import BaselineSynthesizer
from sdgym.utils import (
    calculate_score_time,
    convert_metadata_to_sdmetrics,
    format_exception,
    get_duplicates,
    get_size_of,
    get_synthesizers,
    get_utc_now,
    used_memory,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_SINGLE_TABLE_SYNTHESIZERS = [
    'GaussianCopulaSynthesizer',
    'CTGANSynthesizer',
    'UniformSynthesizer',
]
DEFAULT_MULTI_TABLE_SYNTHESIZERS = ['MultiTableUniformSynthesizer', 'HMASynthesizer']
DEFAULT_SINGLE_TABLE_DATASETS = [
    'adult',
    'alarm',
    'census',
    'child',
    'covtype',
    'expedia_hotel_logs',
    'insurance',
    'intrusion',
    'news',
]
DEFAULT_MULTI_TABLE_DATASETS = [
    'fake_hotels',
    'Biodegradability',
    'Student_loan',
    'restbase',
    'airbnb-simplified',
    'financial',
    'NBA',
]

N_BYTES_IN_MB = 1000 * 1000
FILE_INCREMENT_PATTERN = re.compile(r'\((\d+)\)$')
RESULTS_DATE_PATTERN = re.compile(r'SDGym_results_(\d{2}_\d{2}_\d{4})')
METAINFO_FILE_PATTERN = re.compile(r'metainfo(?:\((\d+)\))?\.yaml$')
SDV_SINGLE_TABLE_SYNTHESIZERS = [
    'GaussianCopulaSynthesizer',
    'CTGANSynthesizer',
    'CopulaGANSynthesizer',
    'TVAESynthesizer',
]
SDV_MULTI_TABLE_SYNTHESIZERS = ['HMASynthesizer']
SDV_SYNTHESIZERS = SDV_SINGLE_TABLE_SYNTHESIZERS + SDV_MULTI_TABLE_SYNTHESIZERS


class JobArgs(NamedTuple):
    """Arguments needed to run a single synthesizer + dataset benchmark job."""

    synthesizer: dict
    data: Any
    metadata: Any
    metrics: Any
    timeout: Optional[int]
    compute_quality_score: bool
    compute_diagnostic_score: bool
    compute_privacy_score: bool
    dataset_name: str
    modality: str
    output_directions: Optional[dict]


def _import_and_validate_synthesizers(synthesizers, custom_synthesizers, modality):
    """Import user-provided synthesizer and validate modality and uniqueness.

    This function takes lists of synthesizer, imports them as synthesizer classes,
    and validates two conditions:
        - Modality match – all synthesizers must match the expected `modality`.
        A `ValueError` is raised if any synthesizer has a different modality
        flag.

        - Uniqueness – duplicate synthesizer across the two input lists
        (`synthesizers` and `custom_synthesizers`) are not allowed. A
        `ValueError` is raised if duplicates are found.

    Args:
        synthesizers (list | None):
            A list of synthesizer strings or classes. May be ``None``, in which case it
            is treated as an empty list.
        custom_synthesizers (list | None):
            A list of custom synthesizer.
        modality (str):
            The required modality that all synthesizers must match.

    Returns:
        list:
            A list of synthesizer classes.

    Raises:
        ValueError:
            If any synthesizer does not match the expected modality.
        ValueError:
            If duplicate synthesizer are found across the provided lists.
    """
    # Get list of synthesizer objects
    synthesizers = synthesizers or []
    custom_synthesizers = custom_synthesizers or []
    resolved_synthesizers = get_synthesizers(synthesizers + custom_synthesizers)
    mismatched = [
        synth['synthesizer']
        for synth in resolved_synthesizers
        if synth['synthesizer']._MODALITY_FLAG != modality
    ]
    if mismatched:
        raise ValueError(
            f"Synthesizers must be of modality '{modality}'. "
            "Found these synthesizers that don't match: "
            f'{", ".join([type(synth).__name__ for synth in mismatched])}'
        )

    # Check duplicate input values
    duplicates = get_duplicates(synthesizers + custom_synthesizers)
    if duplicates:
        raise ValueError(
            'Synthesizers must be unique. Please remove repeated values in the provided '
            'synthesizers.'
        )

    return resolved_synthesizers


def _get_metainfo_increment(top_folder, s3_client=None):
    increments = []
    first_file_message = 'No metainfo file found, starting from increment (0)'
    if s3_client:
        bucket, prefix = parse_s3_path(top_folder)
        try:
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
            contents = response.get('Contents', [])
            for obj in contents:
                file_name = Path(obj['Key']).name
                match = METAINFO_FILE_PATTERN.match(file_name)
                if match:
                    # Extract numeric suffix (e.g. metainfo(3).yaml → 3) or 0 if plain metainfo.yaml
                    increments.append(int(match.group(1)) if match.group(1) else 0)

        except Exception:
            LOGGER.info(first_file_message)
            return 0  # start with (0) if error
    else:
        top_folder = Path(top_folder)
        if not top_folder.exists():
            LOGGER.info(first_file_message)
            return 0
        for file in top_folder.glob('metainfo*.yaml'):
            match = METAINFO_FILE_PATTERN.match(file.name)
            if match:
                increments.append(int(match.group(1)) if match.group(1) else 0)

    return max(increments) + 1 if increments else 0


def _setup_output_destination_aws(
    output_destination,
    synthesizers,
    datasets,
    modality,
    s3_client,
):
    paths = defaultdict(dict)
    s3_path = output_destination[len(S3_PREFIX) :].rstrip('/')
    parts = s3_path.split('/')
    bucket_name = parts[0]
    prefix_parts = parts[1:]
    paths['bucket_name'] = bucket_name
    today = datetime.today().strftime('%m_%d_%Y')

    modality_prefix = '/'.join(prefix_parts + [modality])
    top_folder = f'{modality_prefix}/SDGym_results_{today}'
    increment = _get_metainfo_increment(f's3://{bucket_name}/{top_folder}', s3_client)
    suffix = f'({increment})' if increment >= 1 else ''
    s3_client.put_object(Bucket=bucket_name, Key=top_folder + '/')
    synthetic_data_extension = 'zip' if modality == 'multi_table' else 'csv'
    for dataset in datasets:
        dataset_folder = f'{top_folder}/{dataset}_{today}'
        s3_client.put_object(Bucket=bucket_name, Key=dataset_folder + '/')

        for synth_name in synthesizers:
            final_synth_name = f'{synth_name}{suffix}'
            synth_folder = f'{dataset_folder}/{final_synth_name}'
            s3_client.put_object(Bucket=bucket_name, Key=synth_folder + '/')
            paths[dataset][final_synth_name] = {
                'synthesizer': (f's3://{bucket_name}/{synth_folder}/{final_synth_name}.pkl'),
                'synthetic_data': (
                    f's3://{bucket_name}/{synth_folder}/'
                    f'{final_synth_name}_synthetic_data.{synthetic_data_extension}'
                ),
                'benchmark_result': (
                    f's3://{bucket_name}/{synth_folder}/{final_synth_name}_benchmark_result.csv'
                ),
                'metainfo': (f's3://{bucket_name}/{top_folder}/metainfo{suffix}.yaml'),
                'results': (f's3://{bucket_name}/{top_folder}/results{suffix}.csv'),
            }

    s3_client.put_object(
        Bucket=bucket_name,
        Key=f'{top_folder}/metainfo{suffix}.yaml',
        Body='completed_date: null\n'.encode('utf-8'),
    )
    return paths


def _setup_output_destination(
    output_destination,
    synthesizers,
    datasets,
    modality,
    s3_client=None,
):
    """Set up the output destination for the benchmark results.

    Args:
        output_destination (str or None):
            The path to the output directory where results will be saved.
            If None, no output will be saved.
        synthesizers (list):
            The list of synthesizers to benchmark.
        datasets (list):
            The list of datasets to benchmark.
        modality (str):
            The dataset modality to load (e.g., 'single_table' or 'multi_table').
        s3_client (boto3.session.Session.client or None):
            The s3 client that can be used to read / write to s3. Defaults to ``None``.
    """
    if s3_client:
        return _setup_output_destination_aws(
            output_destination, synthesizers, datasets, modality, s3_client
        )

    if output_destination is None:
        return {}

    output_path = Path(output_destination)
    output_path.mkdir(parents=True, exist_ok=True)
    today = datetime.today().strftime('%m_%d_%Y')
    top_folder = output_path / modality / f'SDGym_results_{today}'
    top_folder.mkdir(parents=True, exist_ok=True)
    increment = _get_metainfo_increment(top_folder)
    suffix = f'({increment})' if increment >= 1 else ''
    paths = defaultdict(dict)
    synthetic_data_extension = 'zip' if modality == 'multi_table' else 'csv'
    for dataset in datasets:
        dataset_folder = top_folder / f'{dataset}_{today}'
        dataset_folder.mkdir(parents=True, exist_ok=True)

        for synth_name in synthesizers:
            final_synth_name = f'{synth_name}{suffix}'
            synth_folder = dataset_folder / final_synth_name
            synth_folder.mkdir(parents=True, exist_ok=True)
            paths[dataset][final_synth_name] = {
                'synthesizer': str(synth_folder / f'{final_synth_name}.pkl'),
                'synthetic_data': str(
                    synth_folder / f'{final_synth_name}_synthetic_data.{synthetic_data_extension}'
                ),
                'benchmark_result': str(synth_folder / f'{final_synth_name}_benchmark_result.csv'),
                'metainfo': str(top_folder / f'metainfo{suffix}.yaml'),
                'results': str(top_folder / f'results{suffix}.csv'),
            }

    return paths


def _generate_job_args_list(
    limit_dataset_size,
    sdv_datasets,
    additional_datasets_folder,
    sdmetrics,
    timeout,
    output_destination,
    compute_quality_score,
    compute_diagnostic_score,
    compute_privacy_score,
    synthesizers,
    s3_client,
    modality,
):
    sdv_datasets = (
        []
        if sdv_datasets is None
        else get_dataset_paths(
            modality=modality,
            datasets=sdv_datasets,
            s3_client=s3_client,
        )
    )
    additional_datasets = (
        []
        if additional_datasets_folder is None
        else get_dataset_paths(
            modality=modality,
            bucket=(
                additional_datasets_folder
                if is_s3_path(additional_datasets_folder)
                else os.path.join(additional_datasets_folder, modality)
            ),
            s3_client=s3_client,
        )
    )
    datasets = sdv_datasets + additional_datasets
    synthesizer_names = [synthesizer['name'] for synthesizer in synthesizers]
    dataset_names = [dataset.name for dataset in datasets]
    paths = _setup_output_destination(
        output_destination, synthesizer_names, dataset_names, modality=modality, s3_client=s3_client
    )
    job_tuples = []
    for dataset in datasets:
        for synthesizer in synthesizers:
            if paths:
                final_name = next(
                    (name for name in paths[dataset.name] if name.startswith(synthesizer['name'])),
                    synthesizer['name'],
                )
            else:
                final_name = synthesizer['name']

            synthesizer['name'] = final_name
            job_tuples.append((synthesizer, dataset))

    job_args_list = []
    for synthesizer, dataset in job_tuples:
        data, metadata_dict = _load_dataset_with_client(
            modality, dataset, limit_dataset_size=limit_dataset_size, s3_client=s3_client
        )
        path = paths.get(dataset.name, {}).get(synthesizer['name'], None)
        job_args_list.append(
            JobArgs(
                synthesizer=synthesizer,
                data=data,
                metadata=metadata_dict,
                metrics=sdmetrics,
                timeout=timeout,
                compute_quality_score=compute_quality_score,
                compute_diagnostic_score=compute_diagnostic_score,
                compute_privacy_score=compute_privacy_score,
                dataset_name=dataset.name,
                modality=modality,
                output_directions=path,
            )
        )

    return job_args_list


def _synthesize(
    synthesizer_dict,
    real_data,
    metadata,
    synthesizer_path=None,
    result_writer=None,
    modality=None,
):
    synthesizer = synthesizer_dict['synthesizer']
    if isinstance(synthesizer, type):
        assert issubclass(synthesizer, BaselineSynthesizer), (
            '`synthesizer` must be a synthesizer class'
        )
        synthesizer = synthesizer()
    else:
        assert issubclass(type(synthesizer), BaselineSynthesizer), (
            '`synthesizer` must be an instance of a synthesizer class.'
        )

    get_synthesizer = synthesizer.get_trained_synthesizer
    sample_from_synthesizer = synthesizer.sample_from_synthesizer
    data = real_data.copy()

    tracemalloc.start()
    fitted_synthesizer = None
    synthetic_data = None
    synthesizer_size = None
    peak_memory = None
    start = get_utc_now()
    train_end = None
    try:
        fitted_synthesizer = get_synthesizer(data, metadata)
        synthesizer_size = len(cloudpickle.dumps(fitted_synthesizer)) / N_BYTES_IN_MB
        train_end = get_utc_now()
        train_time = train_end - start

        if modality == 'multi_table':
            synthetic_data = sample_from_synthesizer(fitted_synthesizer, 1.0)
        else:
            synthetic_data = sample_from_synthesizer(fitted_synthesizer, n_samples=len(data))

        sample_end = get_utc_now()
        sample_time = sample_end - train_end
        peak_memory = tracemalloc.get_traced_memory()[1] / N_BYTES_IN_MB

        if synthesizer_path is not None and result_writer is not None:
            internal_synthesizer = getattr(
                fitted_synthesizer, '_internal_synthesizer', fitted_synthesizer
            )
            result_writer.write_pickle(internal_synthesizer, synthesizer_path['synthesizer'])
            if modality == 'multi_table':
                result_writer.write_zipped_dataframes(
                    synthetic_data, synthesizer_path['synthetic_data']
                )

            else:
                result_writer.write_dataframe(synthetic_data, synthesizer_path['synthetic_data'])

        return synthetic_data, train_time, sample_time, synthesizer_size, peak_memory

    except Exception as e:
        peak_memory = None
        now = get_utc_now()
        if train_end is None:
            train_time = now - start
            sample_time = timedelta(0)
        else:
            train_time = train_end - start
            sample_time = now - train_end

        exception_text, error_text = format_exception()
        raise BenchmarkError(
            original_exc=e,
            train_time=train_time,
            sample_time=sample_time,
            synthesizer_size=synthesizer_size,
            peak_memory=peak_memory,
            exception_text=exception_text,
            error_text=error_text,
        ) from e

    finally:
        tracemalloc.stop()
        tracemalloc.clear_traces()


def _compute_scores(
    metrics,
    real_data,
    synthetic_data,
    metadata,
    output,
    compute_quality_score,
    compute_diagnostic_score,
    compute_privacy_score,
    modality,
    dataset_name,
):
    metrics = metrics or []
    if modality == 'single_table':
        sdmetrics_metadata = convert_metadata_to_sdmetrics(metadata)
    else:
        sdmetrics_metadata = metadata

    if len(metrics) > 0:
        metrics, metric_kwargs = get_metrics(metrics, modality=modality)
        scores = []
        output['scores'] = scores
        for metric_name, metric in metrics.items():
            scores.append({
                'metric': metric_name,
                'error': 'Metric Timeout',
            })
            # re-inject list to multiprocessing output
            output['scores'] = scores

            error = None
            score = None
            normalized_score = None
            start = get_utc_now()
            try:
                LOGGER.info('Computing %s on dataset %s', metric_name, dataset_name)
                metric_args = (real_data, synthetic_data, sdmetrics_metadata)
                score = metric.compute(*metric_args, **metric_kwargs.get(metric_name, {}))
                normalized_score = metric.normalize(score)
            except Exception:
                LOGGER.exception(
                    'Metric %s failed on dataset %s. Skipping.', metric_name, dataset_name
                )
                _, error = format_exception()

            scores[-1].update({
                'score': score,
                'normalized_score': normalized_score,
                'error': error,
                'metric_time': calculate_score_time(start),
            })
            # re-inject list to multiprocessing output
            output['scores'] = scores

    if compute_diagnostic_score:
        start = get_utc_now()
        if modality == 'single_table':
            diagnostic_report = SingleTableDiagnosticReport()
        else:
            diagnostic_report = MultiTableDiagnosticReport()

        diagnostic_report.generate(real_data, synthetic_data, sdmetrics_metadata, verbose=False)
        output['diagnostic_score_time'] = calculate_score_time(start)
        output['diagnostic_score'] = diagnostic_report.get_score()

    if compute_quality_score:
        start = get_utc_now()
        if modality == 'single_table':
            quality_report = SingleTableQualityReport()
        else:
            quality_report = MultiTableQualityReport()

        quality_report.generate(real_data, synthetic_data, sdmetrics_metadata, verbose=False)
        output['quality_score_time'] = calculate_score_time(start)
        output['quality_score'] = quality_report.get_score()

    if compute_privacy_score:
        start = get_utc_now()
        num_rows = len(synthetic_data)
        # parameters were determined by running experiments with all datasets
        # and tweaking num_rows_subsample and num_iterations until
        # absolute difference from real score was close (while keeping runtime low)
        # see SDGym/0397 folder in Drive
        num_rows_subsample = math.floor(num_rows * 0.60)
        num_iterations = 2
        score = DCRBaselineProtection.compute_breakdown(
            real_data=real_data,
            synthetic_data=synthetic_data,
            metadata=sdmetrics_metadata,
            num_rows_subsample=num_rows_subsample,
            num_iterations=num_iterations,
        )
        output['privacy_score_time'] = calculate_score_time(start)
        output['privacy_score'] = score.get('score')


def _score(
    synthesizer,
    data,
    metadata,
    metrics,
    output=None,
    compute_quality_score=False,
    compute_diagnostic_score=False,
    compute_privacy_score=False,
    modality=None,
    dataset_name=None,
    synthesizer_path=None,
    result_writer=None,
):
    if output is None:
        output = {}

    output['timeout'] = True  # To be deleted if there is no error
    output['error'] = 'Load Timeout'  # To be deleted if there is no error
    try:
        LOGGER.info(
            'Running %s on %s dataset %s; %s',
            synthesizer['name'],
            modality,
            dataset_name,
            used_memory(),
        )

        output['dataset_size'] = get_size_of(data) / N_BYTES_IN_MB
        # To be deleted if there is no error
        output['error'] = 'Synthesizer Timeout'

        try:
            synthetic_data, train_time, sample_time, synthesizer_size, peak_memory = _synthesize(
                synthesizer_dict=synthesizer,
                real_data=data.copy(),
                metadata=metadata,
                synthesizer_path=synthesizer_path,
                result_writer=result_writer,
                modality=modality,
            )

            output['synthetic_data'] = synthetic_data
            output['train_time'] = train_time.total_seconds()
            output['sample_time'] = sample_time.total_seconds()
            output['synthesizer_size'] = synthesizer_size
            output['peak_memory'] = peak_memory

            LOGGER.info(
                'Scoring %s on %s dataset %s; %s',
                synthesizer['name'],
                modality,
                dataset_name,
                used_memory(),
            )

            # No error so far. _compute_scores tracks its own errors by metric
            del output['error']
            _compute_scores(
                metrics,
                data,
                synthetic_data,
                metadata,
                output,
                compute_quality_score,
                compute_diagnostic_score,
                compute_privacy_score,
                modality,
                dataset_name,
            )

            output['timeout'] = False  # There was no timeout

        except BenchmarkError as err:
            LOGGER.exception(
                'Synthesis failed for %s on dataset %s;',
                synthesizer['name'],
                dataset_name,
            )

            output['train_time'] = err.train_time.total_seconds() if err.train_time else None
            output['sample_time'] = err.sample_time.total_seconds() if err.sample_time else None
            output['synthesizer_size'] = err.synthesizer_size
            output['peak_memory'] = err.peak_memory

            output['exception'] = err.exception
            output['error'] = err.error
            output['timeout'] = False

    except Exception:
        LOGGER.exception('Error running %s on dataset %s;', synthesizer['name'], dataset_name)
        exception, error = format_exception()
        output['exception'] = exception
        output['error'] = error
        output['timeout'] = False  # There was no timeout

    finally:
        LOGGER.info(
            'Finished %s on dataset %s; %s', synthesizer['name'], dataset_name, used_memory()
        )

    return output


@contextmanager
def multiprocessing_context():
    """Override multiprocessing ForkingPickler to use cloudpickle."""
    original_dump = multiprocessing.reduction.ForkingPickler.dumps
    original_load = multiprocessing.reduction.ForkingPickler.loads
    original_method = multiprocessing.get_start_method()

    multiprocessing.set_start_method('spawn', force=True)
    multiprocessing.reduction.ForkingPickler.dumps = cloudpickle.dumps
    multiprocessing.reduction.ForkingPickler.loads = cloudpickle.loads

    try:
        yield
    finally:
        # Restore original methods
        multiprocessing.set_start_method(original_method, force=True)
        multiprocessing.reduction.ForkingPickler.dumps = original_dump
        multiprocessing.reduction.ForkingPickler.loads = original_load


def _score_with_timeout(
    timeout,
    synthesizer,
    data,
    metadata,
    metrics,
    compute_quality_score=False,
    compute_diagnostic_score=False,
    compute_privacy_score=False,
    modality=None,
    dataset_name=None,
    synthesizer_path=None,
    result_writer=None,
):
    output = {} if isinstance(result_writer, S3ResultsWriter) else None
    args = (
        synthesizer,
        data,
        metadata,
        metrics,
        output,
        compute_quality_score,
        compute_diagnostic_score,
        compute_privacy_score,
        modality,
        dataset_name,
        synthesizer_path,
        result_writer,
    )
    if isinstance(result_writer, S3ResultsWriter):
        thread = threading.Thread(target=_score, args=args, daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            LOGGER.error('Timeout running %s on dataset %s;', synthesizer['name'], dataset_name)
            return {'timeout': True, 'error': 'Synthesizer Timeout'}

        return output

    with multiprocessing_context():
        with multiprocessing.Manager() as manager:
            output = manager.dict()
            args = args[:4] + (output,) + args[5:]  # replace output=None with manager.dict()
            process = multiprocessing.Process(target=_score, args=args)
            process.start()
            process.join(timeout)
            process.terminate()

            output = dict(output)
            if output.get('timeout'):
                LOGGER.error('Timeout running %s on dataset %s;', synthesizer['name'], dataset_name)

            return output


def _format_output(
    output,
    name,
    dataset_name,
    compute_quality_score,
    compute_diagnostic_score,
    compute_privacy_score,
):
    evaluate_time = 0
    if 'quality_score_time' in output:
        evaluate_time += output.get('quality_score_time', 0)
    if 'diagnostic_score_time' in output:
        evaluate_time += output.get('diagnostic_score_time', 0)
    if 'privacy_score_time' in output:
        evaluate_time += output.get('privacy_score_time', 0)

    for score in output.get('scores', []):
        if 'metric_time' in score and not np.isnan(score['metric_time']):
            evaluate_time += score['metric_time']

    if (
        'quality_score_time' not in output
        and 'scores' not in output
        and 'diagnostic_score_time' not in output
        and 'privacy_score_time' not in output
    ):
        evaluate_time = None

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

    if compute_diagnostic_score:
        scores.insert(len(scores.columns), 'Diagnostic_Score', output.get('diagnostic_score'))

    if compute_quality_score:
        scores.insert(len(scores.columns), 'Quality_Score', output.get('quality_score'))

    if compute_privacy_score:
        scores.insert(len(scores.columns), 'Privacy_Score', output.get('privacy_score'))

    for score in output.get('scores', []):
        scores.insert(len(scores.columns), score['metric'], score['normalized_score'])

    if 'error' in output:
        scores['error'] = output['error']

    return scores


def _run_job(job_args, result_writer=None):
    # Reset random seed
    np.random.seed()

    synthesizer = job_args.synthesizer
    data = job_args.data
    metadata = job_args.metadata
    metrics = job_args.metrics
    timeout = job_args.timeout
    compute_quality_score = job_args.compute_quality_score
    compute_diagnostic_score = job_args.compute_diagnostic_score
    compute_privacy_score = job_args.compute_privacy_score
    dataset_name = job_args.dataset_name
    modality = job_args.modality
    synthesizer_path = job_args.output_directions

    name = synthesizer['name']
    LOGGER.info(
        'Evaluating %s on dataset %s with timeout %ss; %s',
        name,
        dataset_name,
        timeout,
        used_memory(),
    )
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
                compute_diagnostic_score=compute_diagnostic_score,
                compute_privacy_score=compute_privacy_score,
                modality=modality,
                dataset_name=dataset_name,
                synthesizer_path=synthesizer_path,
                result_writer=result_writer,
            )
        else:
            output = _score(
                synthesizer=synthesizer,
                data=data,
                metadata=metadata,
                metrics=metrics,
                compute_quality_score=compute_quality_score,
                compute_diagnostic_score=compute_diagnostic_score,
                compute_privacy_score=compute_privacy_score,
                modality=modality,
                dataset_name=dataset_name,
                synthesizer_path=synthesizer_path,
                result_writer=result_writer,
            )
    except Exception as error:
        output['exception'] = error

    scores = _format_output(
        output,
        name,
        dataset_name,
        compute_quality_score,
        compute_diagnostic_score,
        compute_privacy_score,
    )
    if synthesizer_path and result_writer:
        result_writer.write_dataframe(scores, synthesizer_path['benchmark_result'])

    return scores


def _run_jobs(job_args_list, show_progress, result_writer=None):
    scores = map(functools.partial(_run_job, result_writer=result_writer), job_args_list)
    if show_progress:
        scores = tqdm.tqdm(scores, total=len(job_args_list), position=0, leave=True)
    else:
        scores = tqdm.tqdm(
            scores, total=len(job_args_list), file=TqdmLogger(), position=0, leave=True
        )

    if not scores:
        raise SDGymError('No valid Dataset/Synthesizer combination given.')

    scores = pd.concat(scores, ignore_index=True)
    _add_adjusted_scores(scores=scores, timeout=job_args_list[0].timeout)
    output_directions = job_args_list[0].output_directions
    if output_directions and result_writer:
        path = output_directions['results']
        result_writer.write_dataframe(scores, path, append=True)

    return scores


def _get_empty_dataframe(
    compute_diagnostic_score, compute_quality_score, compute_privacy_score, sdmetrics
):
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
        'Adjusted_Total_Time': [],
    })

    if compute_diagnostic_score:
        scores['Diagnostic_Score'] = []
    if compute_quality_score:
        scores['Quality_Score'] = []
        scores['Adjusted_Quality_Score'] = []
    if compute_privacy_score:
        scores['Privacy_Score'] = []
    if sdmetrics:
        for metric in sdmetrics:
            scores[metric[0]] = []

    return scores


def _check_write_permissions(s3_client, bucket_name):
    s3_client = s3_client or boto3.client('s3')
    try:
        s3_client.put_object(Bucket=bucket_name, Key='__test__', Body=b'')
        write_permission = True
    except Exception:
        write_permission = False
    finally:
        # Clean up the test object
        if write_permission:
            s3_client.delete_object(Bucket=bucket_name, Key='__test__')
    return write_permission


def _validate_output_destination(output_destination, aws_keys=None):
    """Validate the output destination parameter."""
    if output_destination is None and aws_keys is None:
        return

    if aws_keys is not None:
        return _validate_aws_inputs(
            output_destination, aws_keys['aws_access_key_id'], aws_keys['aws_secret_access_key']
        )

    if not isinstance(output_destination, str):
        raise ValueError(
            'The `output_destination` parameter must be a string representing the output path.'
        )

    if is_s3_path(output_destination):
        raise ValueError(
            'The `output_destination` parameter cannot be an S3 path. '
            'Please use `benchmark_single_table_aws` instead.'
        )


def _write_metainfo_file(synthesizers, job_args_list, modality, result_writer=None):
    jobs = [[job.dataset_name, job.synthesizer['name']] for job in job_args_list]
    if not job_args_list or not job_args_list[0].output_directions:
        return

    descriptions_path = Path(__file__).parent / 'synthesizer_descriptions.yaml'
    with open(descriptions_path, 'r') as file:
        synthesizer_descriptions = yaml.safe_load(file)

    output_directions = job_args_list[0].output_directions
    path = output_directions['metainfo']
    stem = Path(path).stem
    match = FILE_INCREMENT_PATTERN.search(stem)
    increment = int(match.group(1)) if match else 0
    date_match = RESULTS_DATE_PATTERN.search(path)
    if not date_match:
        raise ValueError(f'Could not extract date from metainfo path: {path}')

    date_str = date_match.group(1)
    metadata = {
        'run_id': f'run_{date_str}_{increment}',
        'modality': modality,
        'starting_date': datetime.today().strftime('%m_%d_%Y %H:%M:%S'),
        'completed_date': None,
        'sdgym_version': version('sdgym'),
        'jobs': jobs,
    }

    for synthesizer in synthesizers:
        if synthesizer['name'] not in SDV_SYNTHESIZERS:
            ext_lib = synthesizer_descriptions.get(synthesizer['name'], {}).get('library')
            if ext_lib:
                library_version = version(ext_lib)
                metadata[f'{ext_lib}_version'] = library_version

        elif 'sdv' not in metadata.keys():
            metadata['sdv_version'] = version('sdv')

    if result_writer:
        result_writer.write_yaml(metadata, path)


def _update_metainfo_file(run_file, result_writer=None):
    completed_date = datetime.today().strftime('%m_%d_%Y %H:%M:%S')
    update = {'completed_date': completed_date}
    if result_writer:
        result_writer.write_yaml(update, run_file, append=True)


def _ensure_uniform_included(synthesizers, modality):
    uniform_class = UniformSynthesizer
    if modality == 'multi_table':
        uniform_class = MultiTableUniformSynthesizer

    uniform_not_included = bool(
        uniform_class not in synthesizers and uniform_class.__name__ not in synthesizers
    )
    if uniform_not_included:
        LOGGER.info(f'Adding {uniform_class.__name__} to the list of synthesizers.')
        synthesizers.append(uniform_class.__name__)


def _fill_adjusted_scores_with_none(scores):
    """Fill adjusted total time and quality score with NaN values."""
    scores['Adjusted_Total_Time'] = None
    if 'Quality_Score' in scores.columns:
        scores['Adjusted_Quality_Score'] = None

    return scores


def _add_adjusted_scores(scores, timeout):
    """Add adjusted total time and quality score based on UniformSynthesizer baseline."""
    timeout = timeout or 0
    uniform_mask = scores['Synthesizer'].str.contains('UniformSynthesizer', na=False)
    if not uniform_mask.any():
        return _fill_adjusted_scores_with_none(scores)

    scores['Adjusted_Total_Time'] = np.nan
    if 'Quality_Score' in scores.columns:
        scores['Adjusted_Quality_Score'] = np.nan

    for dataset in scores['Dataset'].unique():
        dataset_mask = scores['Dataset'] == dataset
        uniform_mask_dataset = dataset_mask & uniform_mask
        if not uniform_mask_dataset.any():
            scores.loc[dataset_mask, 'Adjusted_Total_Time'] = None
            if 'Adjusted_Quality_Score' in scores.columns:
                scores.loc[dataset_mask, 'Adjusted_Quality_Score'] = None
            continue

        uniform_row = scores.loc[uniform_mask_dataset].iloc[0]
        base_fit_time = uniform_row.get('Train_Time')
        base_sample_time = uniform_row.get('Sample_Time')
        base_quality_score = uniform_row.get('Quality_Score', None)
        if pd.isna(base_fit_time) or pd.isna(base_sample_time):
            scores.loc[dataset_mask, 'Adjusted_Total_Time'] = None
            if 'Adjusted_Quality_Score' in scores.columns:
                scores.loc[dataset_mask, 'Adjusted_Quality_Score'] = None
            continue

        fit_times = scores.loc[dataset_mask, 'Train_Time'].fillna(0)
        sample_times = scores.loc[dataset_mask, 'Sample_Time'].fillna(0)
        if 'error' in scores.columns:
            errors = scores.loc[dataset_mask, 'error']
        else:
            errors = pd.Series([None] * dataset_mask.sum(), index=scores.index[dataset_mask])

        timeout_mask = errors == 'Synthesizer Timeout'
        other_error_mask = errors.notna() & ~timeout_mask
        no_error_mask = errors.isna()
        adjusted_times = np.select(
            [timeout_mask, other_error_mask, no_error_mask],
            [
                base_fit_time + timeout + base_sample_time,  # timeout
                base_fit_time + fit_times + sample_times + base_sample_time,  # other error
                base_fit_time + fit_times + sample_times,  # no error
            ],
            default=np.nan,
        )
        scores.loc[dataset_mask, 'Adjusted_Total_Time'] = adjusted_times
        if 'Adjusted_Quality_Score' not in scores.columns:
            continue

        if pd.isna(base_quality_score):
            scores.loc[dataset_mask, 'Adjusted_Quality_Score'] = None
            continue

        has_error = errors.notna()
        original_quality = scores.loc[dataset_mask, 'Quality_Score']
        adjusted_quality = np.where(has_error, base_quality_score, original_quality)
        scores.loc[dataset_mask, 'Adjusted_Quality_Score'] = adjusted_quality

    return scores


def benchmark_single_table(
    synthesizers=DEFAULT_SINGLE_TABLE_SYNTHESIZERS,
    custom_synthesizers=None,
    sdv_datasets=DEFAULT_SINGLE_TABLE_DATASETS,
    additional_datasets_folder=None,
    limit_dataset_size=False,
    compute_quality_score=True,
    compute_diagnostic_score=True,
    compute_privacy_score=True,
    sdmetrics=None,
    timeout=None,
    output_destination=None,
    show_progress=False,
):
    """Run the SDGym benchmark on single-table datasets.

    Args:
        synthesizers (list[string]):
            The synthesizer(s) to evaluate. Defaults to ``[GaussianCopulaSynthesizer,
            CTGANSynthesizer]``. The available options are:

                - ``GaussianCopulaSynthesizer``
                - ``CTGANSynthesizer``
                - ``CopulaGANSynthesizer``
                - ``TVAESynthesizer``
                - ``RealTabFormerSynthesizer``

        custom_synthesizers (list[class] or ``None``):
            A list of custom synthesizer classes to use. These can be completely custom or
            they can be synthesizer variants (the output from ``create_single_table_synthesizer``
            or ``create_synthesizer_variant``). Defaults to ``None``.
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
            Whether or not to evaluate an overall quality score. Defaults to ``True``.
        compute_diagnostic_score (bool):
            Whether or not to evaluate an overall diagnostic score. Defaults to ``True``.
        compute_privacy_score (bool):
            Whether or not to evaluate an overall privacy score. Defaults to ``True``.
        sdmetrics (list[str]):
            A list of the different SDMetrics to use.
            If you'd like to input specific parameters into the metric, provide a tuple with
            the metric name followed by a dictionary of the parameters.
        timeout (int or ``None``):
            The maximum number of seconds to wait for synthetic data creation. If ``None``, no
            timeout is enforced.
        output_destination (str or ``None``):
            The path to the output directory where results will be saved. If ``None``, no
            output is saved. The results are saved with the following structure:
            output_destination/
                run_<id>.yaml
                SDGym_results_<date>/
                    results.csv
                    <dataset_name>_<date>/
                    metainfo.yaml
                    <synthesizer_name>/
                        synthesizer.pkl
                        synthetic_data.csv
        show_progress (bool):
            Whether to use tqdm to keep track of the progress. Defaults to ``False``.

    Returns:
        pandas.DataFrame:
            A table containing one row per synthesizer + dataset + metric.
    """
    _validate_output_destination(output_destination)
    if not synthesizers:
        synthesizers = []

    _ensure_uniform_included(synthesizers, 'single_table')
    synthesizers = _import_and_validate_synthesizers(
        synthesizers,
        custom_synthesizers,
        'single_table',
    )
    result_writer = LocalResultsWriter()
    job_args_list = _generate_job_args_list(
        limit_dataset_size=limit_dataset_size,
        sdv_datasets=sdv_datasets,
        additional_datasets_folder=additional_datasets_folder,
        sdmetrics=sdmetrics,
        timeout=timeout,
        output_destination=output_destination,
        compute_quality_score=compute_quality_score,
        compute_diagnostic_score=compute_diagnostic_score,
        compute_privacy_score=compute_privacy_score,
        synthesizers=synthesizers,
        s3_client=None,
        modality='single_table',
    )

    _write_metainfo_file(
        synthesizers=synthesizers,
        job_args_list=job_args_list,
        modality='single_table',
        result_writer=result_writer,
    )
    if job_args_list:
        scores = _run_jobs(job_args_list, show_progress, result_writer)

    # If no synthesizers/datasets are passed, return an empty dataframe
    else:
        scores = _get_empty_dataframe(
            compute_diagnostic_score=compute_diagnostic_score,
            compute_quality_score=compute_quality_score,
            compute_privacy_score=compute_privacy_score,
            sdmetrics=sdmetrics,
        )

    if output_destination and job_args_list:
        metainfo_filename = job_args_list[0].output_directions['metainfo']
        _update_metainfo_file(metainfo_filename, result_writer)

    return scores


def _validate_aws_inputs(output_destination, aws_access_key_id, aws_secret_access_key):
    """Validate AWS S3 inputs for SDGym benchmark.

    If credentials are not provided, the S3 bucket must be public.
    """
    if not isinstance(output_destination, str):
        raise ValueError(
            'The `output_destination` parameter must be a string representing the S3 URL.'
        )

    if not output_destination.startswith('s3://'):
        raise ValueError("'output_destination' must be an S3 URL starting with 's3://'. ")

    bucket_name, _ = parse_s3_path(output_destination)
    if not bucket_name:
        raise ValueError(f'Invalid S3 URL: {output_destination}')

    config = Config(connect_timeout=30, read_timeout=300)
    if aws_access_key_id and aws_secret_access_key:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=S3_REGION,
            config=config,
        )
    else:
        # No credentials provided — rely on default session
        s3_client = boto3.client('s3', config=config)

    s3_client.head_bucket(Bucket=bucket_name)
    if not _check_write_permissions(s3_client, bucket_name):
        raise PermissionError(
            f'No write permissions for the S3 bucket: {bucket_name}. '
            'Please check your AWS credentials or bucket policies.'
        )

    return s3_client


def _store_job_args_in_s3(output_destination, job_args_list, s3_client):
    """Store the job arguments in S3.

    During a run we temporarily store the job arguments in S3 to be able to
    retrieve them later in the EC2 instance. This is necessary because the
    EC2 instance does not have access to the local file system of the machine
    that initiated the run.
    The pkl file generated will be deleted after the run is completed.
    """
    parsed_url = urlparse(output_destination)
    bucket_name = parsed_url.netloc
    path = parsed_url.path.lstrip('/') if parsed_url.path else ''
    filename = os.path.basename(job_args_list[0].output_directions['metainfo'])
    modality = job_args_list[0].modality
    metainfo = os.path.splitext(filename)[0]
    job_args_key = f'{modality}/job_args_list_{metainfo}.pkl.gz'
    job_args_key = f'{path}{job_args_key}' if path else job_args_key

    serialized_data = cloudpickle.dumps(job_args_list)
    compressed = gzip.compress(serialized_data, compresslevel=1)
    s3_client.put_object(Bucket=bucket_name, Key=job_args_key, Body=compressed)

    return bucket_name, job_args_key


def _get_s3_script_content(
    access_key, secret_key, region_name, bucket_name, job_args_key, synthesizers
):
    return f"""
import boto3
import cloudpickle
import gzip
from sdgym.benchmark import _run_jobs, _write_metainfo_file, _update_metainfo_file
from sdgym.result_writer import S3ResultsWriter

s3_client = boto3.client(
    's3',
    aws_access_key_id='{access_key}',
    aws_secret_access_key='{secret_key}',
    region_name='{region_name}'
)
response = s3_client.get_object(Bucket='{bucket_name}', Key='{job_args_key}')
blob = response['Body'].read()
if blob[:2] == b'\\x1f\\x8b':
    blob = gzip.decompress(blob)

job_args_list = cloudpickle.loads(blob)
modality = job_args_list[0].modality
result_writer = S3ResultsWriter(s3_client=s3_client)
_write_metainfo_file({synthesizers}, job_args_list, modality, result_writer=result_writer)
scores = _run_jobs(job_args_list, False, result_writer=result_writer)
metainfo_filename = job_args_list[0].output_directions['metainfo']
_update_metainfo_file(metainfo_filename, result_writer)
s3_client.delete_object(Bucket='{bucket_name}', Key='{job_args_key}')
"""


def _get_user_data_script(access_key, secret_key, region_name, script_content):
    return textwrap.dedent(f"""\
        #!/bin/bash
        set -e

        # Always terminate the instance when the script exits (success or failure)
        trap '
        INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id);
        echo "======== Terminating EC2 instance: $INSTANCE_ID ==========";
        aws ec2 terminate-instances --instance-ids $INSTANCE_ID;
        ' EXIT

        exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1
        echo "======== Update and Install Dependencies ============"
        sudo apt update -y
        sudo apt install -y python3-pip python3-venv awscli
        echo "======== Configure AWS CLI ============"
        aws configure set aws_access_key_id '{access_key}'
        aws configure set aws_secret_access_key '{secret_key}'
        aws configure set default.region '{region_name}'

        echo "======== Create Virtual Environment ============"
        python3 -m venv ~/env
        source ~/env/bin/activate

        echo "======== Install Dependencies in venv ============"
        pip install --upgrade pip
        pip install sdgym[all]
        pip install s3fs

        echo "======== Write Script ==========="
        cat << 'EOF' > ~/sdgym_script.py
{script_content}
EOF

        echo "======== Run Script ==========="
        python ~/sdgym_script.py
        echo "======== Complete ==========="
        INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
        aws ec2 terminate-instances --instance-ids $INSTANCE_ID
    """).strip()


def _run_on_aws(
    output_destination,
    synthesizers,
    s3_client,
    job_args_list,
    aws_access_key_id,
    aws_secret_access_key,
):
    bucket_name, job_args_key = _store_job_args_in_s3(output_destination, job_args_list, s3_client)
    synthesizer_names = [{'name': synthesizer['name']} for synthesizer in synthesizers]
    script_content = _get_s3_script_content(
        aws_access_key_id,
        aws_secret_access_key,
        S3_REGION,
        bucket_name,
        job_args_key,
        synthesizer_names,
    )

    # Create a session and EC2 client using the provided S3 client's credentials
    session = boto3.session.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=S3_REGION,
    )
    ec2_client = session.client('ec2')
    print(f'This instance is being created in region: {session.region_name}')  # noqa
    user_data_script = _get_user_data_script(
        aws_access_key_id, aws_secret_access_key, S3_REGION, script_content
    )
    response = ec2_client.run_instances(
        ImageId='ami-080e1f13689e07408',
        InstanceType='g4dn.4xlarge',
        MinCount=1,
        MaxCount=1,
        UserData=user_data_script,
        TagSpecifications=[
            {'ResourceType': 'instance', 'Tags': [{'Key': 'Name', 'Value': 'SDGym_Temp'}]}
        ],
        BlockDeviceMappings=[
            {
                'DeviceName': '/dev/sda1',
                'Ebs': {
                    'VolumeSize': 32,
                    'VolumeType': 'gp2',
                },
            }
        ],
    )

    # Wait until the instance is running
    instance_id = response['Instances'][0]['InstanceId']
    waiter = ec2_client.get_waiter('instance_status_ok')
    waiter.wait(InstanceIds=[instance_id])
    print(f'Job kicked off for SDGym on {instance_id}')  # noqa


def benchmark_single_table_aws(
    output_destination,
    aws_access_key_id=None,
    aws_secret_access_key=None,
    synthesizers=DEFAULT_SINGLE_TABLE_SYNTHESIZERS,
    sdv_datasets=DEFAULT_SINGLE_TABLE_DATASETS,
    additional_datasets_folder=None,
    limit_dataset_size=False,
    compute_quality_score=True,
    compute_diagnostic_score=True,
    compute_privacy_score=True,
    sdmetrics=None,
    timeout=None,
):
    """Run the SDGym benchmark on single-table datasets.

    Args:
        output_destination (str):
            An S3 bucket or filepath. The results output folder will be written here.
            Should be structured as:
            s3://{s3_bucket_name}/{path_to_file} or s3://{s3_bucket_name}.
        aws_access_key_id (str): The AWS access key id. Optional
        aws_secret_access_key (str): The AWS secret access key. Optional
        synthesizers (list[string]):
            The synthesizer(s) to evaluate. Defaults to
            ``[GaussianCopulaSynthesizer, CTGANSynthesizer]``. The available options
            are:
                - ``GaussianCopulaSynthesizer``
                - ``CTGANSynthesizer``
                - ``CopulaGANSynthesizer``
                - ``TVAESynthesizer``
                - ``RealTabFormerSynthesizer``
        sdv_datasets (list[str] or ``None``):
            Names of the SDV demo datasets to use for the benchmark. Defaults to
            ``[adult, alarm, census, child, expedia_hotel_logs, insurance, intrusion, news,
            covtype]``. Use ``None`` to disable using any sdv datasets.
        additional_datasets_folder (str or ``None``):
            The path to an S3 bucket. Datasets found in this folder are
            run in addition to the SDV datasets. If ``None``, no additional datasets are used.
        limit_dataset_size (bool):
            Use this flag to limit the size of the datasets for faster evaluation. If ``True``,
            limit the size of every table to 1,000 rows (randomly sampled) and the first 10
            columns.
        compute_quality_score (bool):
            Whether or not to evaluate an overall quality score. Defaults to ``True``.
        compute_diagnostic_score (bool):
            Whether or not to evaluate an overall diagnostic score. Defaults to ``True``.
        compute_privacy_score (bool):
            Whether or not to evaluate an overall privacy score. Defaults to ``True``.
        sdmetrics (list[str]):
            A list of the different SDMetrics to use.
            If you'd like to input specific parameters into the metric, provide a tuple with
            the metric name followed by a dictionary of the parameters.
        timeout (int or ``None``):
            The maximum number of seconds to wait for synthetic data creation. If ``None``, no
            timeout is enforced.

    Returns:
        pandas.DataFrame:
            A table containing one row per synthesizer + dataset + metric.
    """
    s3_client = _validate_output_destination(
        output_destination,
        aws_keys={
            'aws_access_key_id': aws_access_key_id,
            'aws_secret_access_key': aws_secret_access_key,
        },
    )
    if not synthesizers:
        synthesizers = []

    _ensure_uniform_included(synthesizers, 'single_table')
    synthesizers = _import_and_validate_synthesizers(
        synthesizers=synthesizers,
        custom_synthesizers=None,
        modality='single_table',
    )

    job_args_list = _generate_job_args_list(
        limit_dataset_size=limit_dataset_size,
        sdv_datasets=sdv_datasets,
        additional_datasets_folder=additional_datasets_folder,
        sdmetrics=sdmetrics,
        timeout=timeout,
        output_destination=output_destination,
        compute_quality_score=compute_quality_score,
        compute_diagnostic_score=compute_diagnostic_score,
        compute_privacy_score=compute_privacy_score,
        synthesizers=synthesizers,
        s3_client=s3_client,
        modality='single_table',
    )
    if not job_args_list:
        return _get_empty_dataframe(
            compute_diagnostic_score=compute_diagnostic_score,
            compute_quality_score=compute_quality_score,
            compute_privacy_score=compute_privacy_score,
            sdmetrics=sdmetrics,
        )

    _run_on_aws(
        output_destination=output_destination,
        synthesizers=synthesizers,
        s3_client=s3_client,
        job_args_list=job_args_list,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )


def benchmark_multi_table(
    synthesizers=DEFAULT_MULTI_TABLE_SYNTHESIZERS,
    custom_synthesizers=None,
    sdv_datasets=DEFAULT_MULTI_TABLE_DATASETS,
    additional_datasets_folder=None,
    limit_dataset_size=False,
    compute_quality_score=True,
    compute_diagnostic_score=True,
    timeout=None,
    output_destination=None,
    show_progress=False,
):
    """Run the SDGym benchmark on multi-table datasets.

    Args:
        synthesizers (list[string]):
            The synthesizer(s) to evaluate. Defaults to ``HMASynthesizer`` and
            ``MultiTableUniformSynthesizer``.
        custom_synthesizers (list[class] or ``None``):
            A list of custom synthesizer classes to use. These can be completely custom or
            they can be synthesizer variants (the output from ``create_single_table_synthesizer``
            or ``create_synthesizer_variant``). Defaults to ``None``.
        sdv_datasets (list[str] or ``None``):
            Names of the SDV demo datasets to use for the benchmark. Defaults to
            ``[NBA, financial, Student_loan, Biodegradability, fake_hotels, restbase,
            airbnb-simplified]``. Use ``None`` to disable using any sdv datasets.
        additional_datasets_folder (str or ``None``):
            The path to a folder (local or an S3 bucket). Datasets found in this folder are
            run in addition to the SDV datasets. If ``None``, no additional datasets are used.
        limit_dataset_size (bool):
            Use this flag to limit the size of the datasets for faster evaluation. If ``True``,
            limit the size of every table to 1,000 rows (randomly sampled) and the first 10
            columns.
        compute_quality_score (bool):
            Whether or not to evaluate an overall quality score. Defaults to ``True``.
        compute_diagnostic_score (bool):
            Whether or not to evaluate an overall diagnostic score. Defaults to ``True``.
        timeout (int or ``None``):
            The maximum number of seconds to wait for synthetic data creation. If ``None``, no
            timeout is enforced.
        output_destination (str or ``None``):
            The path to the output directory where results will be saved. If ``None``, no
            output is saved. The results are saved with the following structure:
            output_destination/
                run_<id>.yaml
                SDGym_results_<date>/
                    results.csv
                    <dataset_name>_<date>/
                    metainfo.yaml
                    <synthesizer_name>/
                        synthesizer.pkl
                        synthetic_data.csv
        show_progress (bool):
            Whether to use tqdm to keep track of the progress. Defaults to ``False``.

    Returns:
        pandas.DataFrame:
            A table containing one row per synthesizer + dataset.
    """
    _validate_output_destination(output_destination)
    if not synthesizers:
        synthesizers = []

    _ensure_uniform_included(synthesizers, 'multi_table')
    result_writer = LocalResultsWriter()

    synthesizers = _import_and_validate_synthesizers(
        synthesizers,
        custom_synthesizers,
        'multi_table',
    )
    job_args_list = _generate_job_args_list(
        limit_dataset_size=limit_dataset_size,
        sdv_datasets=sdv_datasets,
        additional_datasets_folder=additional_datasets_folder,
        sdmetrics=None,
        timeout=timeout,
        output_destination=output_destination,
        compute_quality_score=compute_quality_score,
        compute_diagnostic_score=compute_diagnostic_score,
        compute_privacy_score=None,
        synthesizers=synthesizers,
        s3_client=None,
        modality='multi_table',
    )

    _write_metainfo_file(
        synthesizers=synthesizers,
        job_args_list=job_args_list,
        modality='multi_table',
        result_writer=result_writer,
    )
    if job_args_list:
        scores = _run_jobs(
            job_args_list=job_args_list,
            show_progress=show_progress,
            result_writer=result_writer,
        )

    # If no synthesizers/datasets are passed, return an empty dataframe
    else:
        scores = _get_empty_dataframe(
            compute_diagnostic_score=compute_diagnostic_score,
            compute_quality_score=compute_quality_score,
            compute_privacy_score=None,
            sdmetrics=None,
        )

    if output_destination and job_args_list:
        metainfo_filename = job_args_list[0][-1]['metainfo']
        _update_metainfo_file(metainfo_filename, result_writer)

    return scores


def benchmark_multi_table_aws(
    output_destination,
    aws_access_key_id=None,
    aws_secret_access_key=None,
    synthesizers=DEFAULT_MULTI_TABLE_SYNTHESIZERS,
    sdv_datasets=DEFAULT_MULTI_TABLE_DATASETS,
    additional_datasets_folder=None,
    limit_dataset_size=False,
    compute_quality_score=True,
    compute_diagnostic_score=True,
    timeout=None,
):
    """Run the SDGym benchmark on multi-table datasets.

    Args:
        output_destination (str):
            An S3 bucket or filepath. The results output folder will be written here.
            Should be structured as:
            s3://{s3_bucket_name}/{path_to_file} or s3://{s3_bucket_name}.
        aws_access_key_id (str): The AWS access key id. Optional
        aws_secret_access_key (str): The AWS secret access key. Optional
        synthesizers (list[string]):
            The synthesizer(s) to evaluate. Defaults to
            ``[HMASynthesizer, MultiTableUniformSynthesizer]``. The available options
            are:
                - ``HMASynthesizer``
                - ``MultiTableUniformSynthesizer``
        sdv_datasets (list[str] or ``None``):
            Names of the SDV demo datasets to use for the benchmark. Defaults to
            ``[adult, alarm, census, child, expedia_hotel_logs, insurance, intrusion, news,
            covtype]``. Use ``None`` to disable using any sdv datasets.
        additional_datasets_folder (str or ``None``):
            The path to an S3 bucket. Datasets found in this folder are
            run in addition to the SDV datasets. If ``None``, no additional datasets are used.
        limit_dataset_size (bool):
            Use this flag to limit the size of the datasets for faster evaluation. If ``True``,
            limit the size of every table to 1,000 rows (randomly sampled) and the first 10
            columns.
        compute_quality_score (bool):
            Whether or not to evaluate an overall quality score. Defaults to ``True``.
        compute_diagnostic_score (bool):
            Whether or not to evaluate an overall diagnostic score. Defaults to ``True``.
        timeout (int or ``None``):
            The maximum number of seconds to wait for synthetic data creation. If ``None``, no
            timeout is enforced.

    Returns:
        pandas.DataFrame:
            A table containing one row per synthesizer + dataset.
    """
    s3_client = _validate_output_destination(
        output_destination,
        aws_keys={
            'aws_access_key_id': aws_access_key_id,
            'aws_secret_access_key': aws_secret_access_key,
        },
    )
    if not synthesizers:
        synthesizers = []

    _ensure_uniform_included(synthesizers, modality='multi_table')
    synthesizers = _import_and_validate_synthesizers(
        synthesizers=synthesizers,
        custom_synthesizers=None,
        modality='multi_table',
    )
    job_args_list = _generate_job_args_list(
        limit_dataset_size=limit_dataset_size,
        sdv_datasets=sdv_datasets,
        additional_datasets_folder=additional_datasets_folder,
        sdmetrics=None,
        timeout=timeout,
        output_destination=output_destination,
        compute_quality_score=compute_quality_score,
        compute_diagnostic_score=compute_diagnostic_score,
        compute_privacy_score=None,
        synthesizers=synthesizers,
        s3_client=s3_client,
        modality='multi_table',
    )
    if not job_args_list:
        return _get_empty_dataframe(
            compute_diagnostic_score=compute_diagnostic_score,
            compute_quality_score=compute_quality_score,
            compute_privacy_score=None,
            sdmetrics=None,
        )

    _run_on_aws(
        output_destination=output_destination,
        synthesizers=synthesizers,
        s3_client=s3_client,
        job_args_list=job_args_list,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
