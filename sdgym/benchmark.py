"""Main SDGym benchmarking module."""

import base64
import concurrent
import logging
import math
import multiprocessing
import os
import pickle
import re
import textwrap
import tracemalloc
import warnings
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import boto3
import cloudpickle
import compress_pickle
import numpy as np
import pandas as pd
import tqdm
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

from sdgym.datasets import get_dataset_paths, load_dataset
from sdgym.errors import SDGymError
from sdgym.metrics import get_metrics
from sdgym.progress import TqdmLogger, progress
from sdgym.result_writer import LocalResultsWriter
from sdgym.s3 import (
    S3_PREFIX,
    is_s3_path,
    parse_s3_path,
    write_csv,
    write_file,
)
from sdgym.synthesizers import CTGANSynthesizer, GaussianCopulaSynthesizer
from sdgym.synthesizers.base import BaselineSynthesizer
from sdgym.utils import (
    calculate_score_time,
    format_exception,
    get_duplicates,
    get_num_gpus,
    get_size_of,
    get_synthesizers,
    get_utc_now,
    used_memory,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_SYNTHESIZERS = [GaussianCopulaSynthesizer, CTGANSynthesizer]
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
N_BYTES_IN_MB = 1000 * 1000
EXTERNAL_SYNTHESIZER_TO_LIBRARY = {
    'RealTabFormerSynthesizer': 'realtabformer',
}
SDV_SINGLE_TABLE_SYNTHESIZERS = [
    'GaussianCopulaSynthesizer',
    'CTGANSynthesizer',
    'CopulaGANSynthesizer',
    'TVAESynthesizer',
]


def _validate_inputs(output_filepath, detailed_results_folder, synthesizers, custom_synthesizers):
    if output_filepath and os.path.exists(output_filepath):
        raise ValueError(
            f'{output_filepath} already exists. Please provide a file that does not already exist.'
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


def _setup_output_destination_aws(output_destination, synthesizers, datasets, s3_client):
    paths = defaultdict(dict)
    s3_path = output_destination[len(S3_PREFIX) :]
    parts = s3_path.split('/')
    bucket_name = parts[0]
    prefix_parts = parts[1:]
    paths['bucket_name'] = bucket_name
    today = datetime.today().strftime('%m_%d_%Y')
    top_folder = '/'.join(prefix_parts + [f'SDGym_results_{today}'])
    s3_client.put_object(Bucket=bucket_name, Key=top_folder + '/')
    for dataset in datasets:
        dataset_folder = f'{top_folder}/{dataset}_{today}'
        s3_client.put_object(Bucket=bucket_name, Key=dataset_folder + '/')
        paths[dataset]['meta'] = f's3://{bucket_name}/{dataset_folder}/meta.yaml'
        for synth_name in synthesizers:
            synth_folder = f'{dataset_folder}/{synth_name}'
            s3_client.put_object(Bucket=bucket_name, Key=synth_folder + '/')
            paths[dataset][synth_name] = {
                'synthesizer': f's3://{bucket_name}/{synth_folder}/{synth_name}_synthesizer.pkl',
                'synthetic_data': f's3://{bucket_name}/{synth_folder}/{synth_name}_synthetic_data.csv',
                'benchmark_result': f's3://{bucket_name}/{synth_folder}/{synth_name}_benchmark_result.csv',
            }

    return paths


def _setup_output_destination(output_destination, synthesizers, datasets, s3_client=None):
    """Set up the output destination for the benchmark results.

    Args:
        output_destination (str or None):
            The path to the output directory where results will be saved.
            If None, no output will be saved.
        synthesizers (list):
            The list of synthesizers to benchmark.
        datasets (list):
            The list of datasets to benchmark.
    """
    if s3_client:
        return _setup_output_destination_aws(output_destination, synthesizers, datasets, s3_client)

    if output_destination is None:
        return {}

    output_path = Path(output_destination)
    output_path.mkdir(parents=True, exist_ok=True)
    today = datetime.today().strftime('%m_%d_%Y')
    top_folder = output_path / f'SDGym_results_{today}'
    top_folder.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf'run_{re.escape(today)}_(\d+)\.yaml$')
    increments = []
    for file in top_folder.glob(f'run_{today}_*.yaml'):
        match = pattern.match(file.name)
        if match:
            increments.append(int(match.group(1)))

    if increments:
        next_increment = max(increments) + 1
    else:
        next_increment = 1
    paths = defaultdict(dict)
    for dataset in datasets:
        dataset_folder = top_folder / f'{dataset}_{today}'
        dataset_folder.mkdir(parents=True, exist_ok=True)

        for synth_name in synthesizers:
            synth_folder = dataset_folder / synth_name
            synth_folder.mkdir(parents=True, exist_ok=True)

            paths[dataset][synth_name] = {
                'synthesizer': str(synth_folder / f'{synth_name}_synthesizer.pkl'),
                'synthetic_data': str(synth_folder / f'{synth_name}_synthetic_data.csv'),
                'benchmark_result': str(synth_folder / f'{synth_name}_benchmark_result.csv'),
                'run_id': str(top_folder / f'run_{today}_{next_increment}.yaml'),
                'results': str(top_folder / f'results_{today}_{next_increment}.csv'),
            }

    return paths


def _generate_job_args_list(
    limit_dataset_size,
    sdv_datasets,
    additional_datasets_folder,
    sdmetrics,
    detailed_results_folder,
    timeout,
    output_destination,
    compute_quality_score,
    compute_diagnostic_score,
    compute_privacy_score,
    synthesizers,
    custom_synthesizers,
    s3_client,
):
    # Get list of synthesizer objects
    synthesizers = [] if synthesizers is None else synthesizers
    custom_synthesizers = [] if custom_synthesizers is None else custom_synthesizers
    synthesizers = get_synthesizers(synthesizers + custom_synthesizers)

    # Get list of dataset paths
    sdv_datasets = [] if sdv_datasets is None else get_dataset_paths(datasets=sdv_datasets)
    additional_datasets = (
        []
        if additional_datasets_folder is None
        else get_dataset_paths(bucket=additional_datasets_folder)
    )
    datasets = sdv_datasets + additional_datasets
    synthesizer_names = [synthesizer['name'] for synthesizer in synthesizers]
    dataset_names = [dataset.name for dataset in datasets]
    paths = _setup_output_destination(
        output_destination, synthesizer_names, dataset_names, s3_client=s3_client
    )
    job_tuples = []
    for dataset in datasets:
        for synthesizer in synthesizers:
            job_tuples.append((synthesizer, dataset))

    job_args_list = []
    for synthesizer, dataset in job_tuples:
        data, metadata_dict = load_dataset(
            'single_table', dataset, limit_dataset_size=limit_dataset_size
        )
        path = paths.get(dataset.name, {}).get(synthesizer['name'], None)
        args = (
            synthesizer,
            data,
            metadata_dict,
            sdmetrics,
            detailed_results_folder,
            timeout,
            compute_quality_score,
            compute_diagnostic_score,
            compute_privacy_score,
            dataset.name,
            'single_table',
            path,
        )
        job_args_list.append(args)

    return job_args_list


def _synthesize(synthesizer_dict, real_data, metadata, synthesizer_path=None, result_writer=None):
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
    num_samples = len(data)

    tracemalloc.start()
    now = get_utc_now()
    synthesizer_obj = get_synthesizer(data, metadata)
    synthesizer_size = len(pickle.dumps(synthesizer_obj)) / N_BYTES_IN_MB
    train_now = get_utc_now()
    synthetic_data = sample_from_synthesizer(synthesizer_obj, num_samples)
    sample_now = get_utc_now()

    peak_memory = tracemalloc.get_traced_memory()[1] / N_BYTES_IN_MB
    tracemalloc.stop()
    tracemalloc.clear_traces()
    if synthesizer_path is not None and result_writer is not None:
        result_writer.write_dataframe(synthetic_data, synthesizer_path['synthetic_data'])
        result_writer.write_pickle(synthesizer_obj, synthesizer_path['synthesizer'])

    return synthetic_data, train_now - now, sample_now - train_now, synthesizer_size, peak_memory


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
    if len(metrics) > 0:
        metrics, metric_kwargs = get_metrics(metrics, modality='single-table')
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
                metric_args = (real_data, synthetic_data, metadata)
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

        diagnostic_report.generate(real_data, synthetic_data, metadata, verbose=False)
        output['diagnostic_score_time'] = calculate_score_time(start)
        output['diagnostic_score'] = diagnostic_report.get_score()

    if compute_quality_score:
        start = get_utc_now()
        if modality == 'single_table':
            quality_report = SingleTableQualityReport()
        else:
            quality_report = MultiTableQualityReport()

        quality_report.generate(real_data, synthetic_data, metadata, verbose=False)
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
            metadata=metadata,
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
        synthetic_data, train_time, sample_time, synthesizer_size, peak_memory = _synthesize(
            synthesizer,
            data.copy(),
            metadata,
            synthesizer_path=synthesizer_path,
            result_writer=result_writer,
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
    with multiprocessing_context():
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
                    compute_diagnostic_score,
                    compute_privacy_score,
                    modality,
                    dataset_name,
                    synthesizer_path,
                    result_writer,
                ),
            )

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
    cache_dir,
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

    (
        synthesizer,
        data,
        metadata,
        metrics,
        cache_dir,
        timeout,
        compute_quality_score,
        compute_diagnostic_score,
        compute_privacy_score,
        dataset_name,
        modality,
        synthesizer_path,
        result_writer,
    ) = args

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
        cache_dir,
    )

    if synthesizer_path and result_writer:
        result_writer.write_dataframe(scores, synthesizer_path['benchmark_result'])

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


def _run_jobs(multi_processing_config, job_args_list, show_progress, result_writer=None):
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

    job_args_list = [job_args + (result_writer,) for job_args in job_args_list]
    if workers in (0, 1):
        scores = map(_run_job, job_args_list)
    elif workers != 'dask':
        pool = concurrent.futures.ProcessPoolExecutor(workers)
        scores = pool.map(_run_job, job_args_list)

    if show_progress:
        scores = tqdm.tqdm(scores, total=len(job_args_list), position=0, leave=True)
    else:
        scores = tqdm.tqdm(
            scores, total=len(job_args_list), file=TqdmLogger(), position=0, leave=True
        )

    if not scores:
        raise SDGymError('No valid Dataset/Synthesizer combination given.')

    scores = pd.concat(scores, ignore_index=True)
    output_directions = job_args_list[0][-2]
    result_writer = job_args_list[0][-1]
    if output_directions and result_writer:
        path = output_directions['synthesizer']
        if path.startswith(S3_PREFIX):
            parsed = urlparse(path)
            s3_key = PurePosixPath(parsed.path).parents[2] / 'results.csv'
            result_file = f's3://{parsed.netloc}{s3_key}'
        else:
            root_path = Path(path).parents[2]
            result_file = root_path / 'results.csv'

        result_writer.write_dataframe(scores, result_file, append=True)

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
    })

    if compute_diagnostic_score:
        scores['Diagnostic_Score'] = []
    if compute_quality_score:
        scores['Quality_Score'] = []
    if compute_privacy_score:
        scores['Privacy_Score'] = []
    if sdmetrics:
        for metric in sdmetrics:
            scores[metric[0]] = []

    return scores


def _directory_exists(bucket_name, s3_file_path):
    # Find the last occurrence of '/' in the file path
    last_slash_index = s3_file_path.rfind('/')
    directory_prefix = s3_file_path[: last_slash_index + 1]
    s3_client = boto3.client('s3')
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=directory_prefix, Delimiter='/')
    return 'Contents' in response or 'CommonPrefixes' in response


def _check_write_permissions(s3_client, bucket_name):
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


def _create_sdgym_script(params, output_filepath):
    # Confirm the path works
    if not is_s3_path(output_filepath):
        raise ValueError("""Invalid S3 path format.
                         Expected 's3://<bucket_name>/<path_to_file>'.""")
    bucket_name, key_prefix = parse_s3_path(output_filepath)
    if not _directory_exists(bucket_name, key_prefix):
        raise ValueError(f'Directories in {key_prefix} do not exist')
    if not _check_write_permissions(bucket_name):
        raise ValueError('No write permissions allowed for the bucket.')

    # Add quotes to parameter strings
    if params['additional_datasets_folder']:
        params['additional_datasets_folder'] = "'" + params['additional_datasets_folder'] + "'"
    if params['detailed_results_folder']:
        params['detailed_results_folder'] = "'" + params['detailed_results_folder'] + "'"
    if params['output_filepath']:
        params['output_filepath'] = "'" + params['output_filepath'] + "'"

    # Generate the output script to run on the e2 instance
    synthesizer_string = 'synthesizers=['
    for synthesizer in params['synthesizers']:
        if isinstance(synthesizer, str):
            synthesizer_string += synthesizer + ', '
        else:
            synthesizer_string += synthesizer.__name__ + ', '
    if params['synthesizers']:
        synthesizer_string = synthesizer_string[:-2]
    synthesizer_string += ']'
    # The indentation of the string is important for the python script
    script_content = f"""import boto3
from io import StringIO
import sdgym
from sdgym.synthesizers.sdv import (CopulaGANSynthesizer, CTGANSynthesizer,
    GaussianCopulaSynthesizer, HMASynthesizer, PARSynthesizer, SDVRelationalSynthesizer,
    SDVTabularSynthesizer, TVAESynthesizer)
from sdgym.synthesizers import RealTabFormerSynthesizer

results = sdgym.benchmark_single_table(
    {synthesizer_string}, custom_synthesizers={params['custom_synthesizers']},
    sdv_datasets={params['sdv_datasets']}, output_filepath={params['output_filepath']},
    additional_datasets_folder={params['additional_datasets_folder']},
    limit_dataset_size={params['limit_dataset_size']},
    compute_quality_score={params['compute_quality_score']},
    compute_diagnostic_score={params['compute_diagnostic_score']},
    compute_privacy_score={params['compute_privacy_score']},
    sdmetrics={params['sdmetrics']},
    timeout={params['timeout']},
    detailed_results_folder={params['detailed_results_folder']},
    multi_processing_config={params['multi_processing_config']}
)
"""

    return script_content


def _create_instance_on_ec2(script_content):
    ec2_client = boto3.client('ec2')
    session = boto3.session.Session()
    credentials = session.get_credentials()
    print(f'This instance is being created in region: {session.region_name}')  # noqa
    escaped_script = script_content.strip().replace('"', '\\"')

    # User data script to install the library
    user_data_script = f"""#!/bin/bash
    sudo apt update -y
    sudo apt install -y python3-pip python3-venv awscli
    echo "======== Create Virtual Environment ============"
    python3 -m venv ~/env
    source ~/env/bin/activate
    echo "======== Install Dependencies in venv ============"
    pip install --upgrade pip
    pip install sdgym[all]
    pip install anyio
    echo "======== Configure AWS CLI ============"
    aws configure set aws_access_key_id {credentials.access_key}
    aws configure set aws_secret_access_key {credentials.secret_key}
    aws configure set region {session.region_name}
    echo "======== Write Script ==========="
    printf '%s\\n' "{escaped_script}" > ~/sdgym_script.py
    echo "======== Run Script ==========="
    python ~/sdgym_script.py

    echo "======== Complete ==========="
    INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
    aws ec2 terminate-instances --instance-ids $INSTANCE_ID
    """

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
                    'VolumeSize': 32,  # Specify the desired size in GB
                    'VolumeType': 'gp2',  # Change the volume type as needed
                },
            }
        ],
    )

    # Wait until the instance is running before terminating
    instance_id = response['Instances'][0]['InstanceId']
    waiter = ec2_client.get_waiter('instance_status_ok')
    waiter.wait(InstanceIds=[instance_id])
    print(f'Job kicked off for SDGym on {instance_id}')  # noqa


def _handle_deprecated_parameters(
    output_filepath, detailed_results_folder, multi_processing_config, run_on_ec2
):
    """Handle deprecated parameters and issue warnings."""
    parameters_to_deprecate = {
        'output_filepath': output_filepath,
        'detailed_results_folder': detailed_results_folder,
        'multi_processing_config': multi_processing_config,
        'run_on_ec2': run_on_ec2,
    }
    parameters = []
    for name, value in parameters_to_deprecate.items():
        if value is not None and value:
            parameters.append(name)

    if parameters:
        parameters = "', '".join(sorted(parameters))
        message = (
            f"Parameters '{parameters}' are deprecated in the 'benchmark_single_table' "
            'function and will be removed in October 2025. '
            "For saving results, please use the 'output_destination' parameter. For running SDGym"
            " remotely on AWS please use the 'benchmark_single_table_aws' method."
        )
        warnings.warn(message, FutureWarning)


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


def _write_run_id_file(synthesizers, job_args_list, result_writer=None):
    jobs = [[job[-3], job[0]['name']] for job in job_args_list]
    output_directions = job_args_list[0][-1]
    path = output_directions['run_id']
    run_id = Path(path).stem
    metadata = {
        'run_id': run_id,
        'starting_date': datetime.today().strftime('%m_%d_%Y %H:%M:%S'),
        'completed_date': None,
        'sdgym_version': version('sdgym'),
        'jobs': jobs,
    }
    for synthesizer in synthesizers:
        if synthesizer not in SDV_SINGLE_TABLE_SYNTHESIZERS:
            ext_lib = EXTERNAL_SYNTHESIZER_TO_LIBRARY[synthesizer]
            library_version = version(ext_lib)
            metadata[f'{ext_lib}_version'] = library_version
        elif 'sdv' not in metadata.keys():
            metadata['sdv_version'] = version('sdv')

    if result_writer:
        result_writer.write_yaml(metadata, f'{output_destination}/run_{run_id}.yaml')



def _update_run_id_file(run_file, result_writer=None):
    completed_date = datetime.today().strftime('%m_%d_%Y %H:%M:%S')
    update = {'completed_date': completed_date}
    if result_writer:
        result_writer.write_yaml(update, run_file, append=True)


def benchmark_single_table(
    synthesizers=DEFAULT_SYNTHESIZERS,
    custom_synthesizers=None,
    sdv_datasets=DEFAULT_DATASETS,
    additional_datasets_folder=None,
    limit_dataset_size=False,
    compute_quality_score=True,
    compute_diagnostic_score=True,
    compute_privacy_score=True,
    sdmetrics=None,
    timeout=None,
    output_destination=None,
    output_filepath=None,
    detailed_results_folder=None,
    show_progress=False,
    multi_processing_config=None,
    run_on_ec2=False,
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
                    meta.yaml
                    <synthesizer_name>/
                        synthesizer.pkl
                        synthetic_data.csv
        output_filepath (str or ``None``):
            A file path for where to write the output as a csv file. If ``None``, no output
            is written. If run_on_ec2 flag output_filepath needs to be defined and
            the filepath should be structured as: s3://{s3_bucket_name}/{path_to_file}
            Please make sure the path exists and permissions are given.
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
        run_on_ec2 (bool):
            The flag is used to run the benchmark on an EC2 instance that will be created
            by a scriptusing the authentication of the current user. The EC2 instance
            uses the LATEST released version of sdgym. Local changes or changes NOT
            in the released version will NOT be used in the ec2 instance.

    Returns:
        pandas.DataFrame:
            A table containing one row per synthesizer + dataset + metric.
    """
    _handle_deprecated_parameters(
        output_filepath, detailed_results_folder, multi_processing_config, run_on_ec2
    )
    _validate_output_destination(output_destination)
    result_writer = LocalResultsWriter(output_destination) if output_destination else None
    if run_on_ec2:
        print("This will create an instance for the current AWS user's account.")  # noqa
        if output_filepath is not None:
            script_content = _create_sdgym_script(dict(locals()), output_filepath)
            _create_instance_on_ec2(script_content)
        else:
            raise ValueError('In order to run on EC2, please provide an S3 folder output.')
        return None

    _validate_inputs(output_filepath, detailed_results_folder, synthesizers, custom_synthesizers)
    _create_detailed_results_directory(detailed_results_folder)
    job_args_list = _generate_job_args_list(
        limit_dataset_size,
        sdv_datasets,
        additional_datasets_folder,
        sdmetrics,
        detailed_results_folder,
        timeout,
        output_destination,
        compute_quality_score,
        compute_diagnostic_score,
        compute_privacy_score,
        synthesizers,
        custom_synthesizers,
        s3_client=None,
    )

    _write_run_id_file(synthesizers, job_args_list, result_writer)
    if job_args_list:
        scores = _run_jobs(multi_processing_config, job_args_list, show_progress, result_writer)

    # If no synthesizers/datasets are passed, return an empty dataframe
    else:
        scores = _get_empty_dataframe(
            compute_diagnostic_score=compute_diagnostic_score,
            compute_quality_score=compute_quality_score,
            compute_privacy_score=compute_privacy_score,
            sdmetrics=sdmetrics,
        )

    if output_filepath:
        write_csv(scores, output_filepath, None, None)

    if output_destination:
        run_id_filename = job_args_list[0][-1]['run_id']
        _update_run_id_file(run_id_filename, result_writer)

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

    parsed_url = urlparse(output_destination)
    bucket_name = parsed_url.netloc
    if not bucket_name:
        raise ValueError(f'Invalid S3 URL: {output_destination}')

    if aws_access_key_id and aws_secret_access_key:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
    else:
        # No credentials provided — rely on default session
        s3_client = boto3.client('s3')

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
    job_args_key = f'job_args_list_{str(uuid.uuid4())}.pkl'
    if path:
        job_args_key = f'{path}/{job_args_key}' if path else job_args_key

    serialized_data = pickle.dumps(job_args_list)
    encoded_data = base64.b64encode(serialized_data).decode('utf-8')
    s3_client.put_object(Bucket=bucket_name, Key=job_args_key, Body=encoded_data)

    return bucket_name, job_args_key


def _get_s3_script_content(
    access_key, secret_key, region_name, bucket_name, job_args_key, output_destination, synthesizers
):
    return f"""
import boto3
import pickle
import base64
import pandas as pd
import sdgym
from sdgym.synthesizers.sdv import (
    CopulaGANSynthesizer, CTGANSynthesizer,
    GaussianCopulaSynthesizer, HMASynthesizer, PARSynthesizer,
    SDVRelationalSynthesizer, SDVTabularSynthesizer, TVAESynthesizer
)
from sdgym.synthesizers import RealTabFormerSynthesizer
from sdgym.benchmark import _run_jobs, _write_run_id_file, _update_run_id_file
from io import StringIO
from sdgym.result_writer import S3ResultsWriter

s3_client = boto3.client(
    's3',
    aws_access_key_id='{access_key}',
    aws_secret_access_key='{secret_key}',
    region_name='{region_name}'
)
response = s3_client.get_object(Bucket='{bucket_name}', Key='{job_args_key}')
encoded_data = response['Body'].read().decode('utf-8')
serialized_data = base64.b64decode(encoded_data.encode('utf-8'))
job_args_list = pickle.loads(serialized_data)
result_writer = S3ResultsWriter(s3_client=s3_client)
run_id = _write_run_id_file('{output_destination}', {synthesizers}, job_args_list, result_writer)
scores = _run_jobs(None, job_args_list, False, result_writer=result_writer)
_update_run_id_file('{output_destination}', run_id, result_writer)
s3_client.delete_object(Bucket='{bucket_name}', Key='{job_args_key}')
"""


def _get_user_data_script(access_key, secret_key, region_name, script_content):
    return textwrap.dedent(f"""\
        #!/bin/bash
        set -e
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
        pip install "sdgym[all] @ git+https://github.com/sdv-dev/SDGym.git@issue-414-benchmark_single_table_aws#egg=sdgym"
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


def _run_on_aws(output_destination, synthesizers, s3_client, job_args_list):
    bucket_name, job_args_key = _store_job_args_in_s3(output_destination, job_args_list, s3_client)
    credentials = s3_client._request_signer._credentials
    access_key = credentials.access_key
    secret_key = credentials.secret_key
    region_name = s3_client.meta.region_name
    script_content = _get_s3_script_content(
        access_key,
        secret_key,
        region_name,
        bucket_name,
        job_args_key,
        output_destination,
        synthesizers,
    )

    # Create a session and EC2 client using the provided S3 client's credentials
    session = boto3.session.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region_name,
    )
    ec2_client = session.client('ec2')
    print(f'This instance is being created in region: {session.region_name}')  # noqa
    user_data_script = _get_user_data_script(access_key, secret_key, region_name, script_content)
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
    synthesizers=DEFAULT_SYNTHESIZERS,
    sdv_datasets=DEFAULT_DATASETS,
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
        detailed_results_folder=None,
        custom_synthesizers=None,
        s3_client=s3_client,
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
    )
