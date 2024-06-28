"""SDGym CLI module."""

import argparse
import gc
import json
import logging
import sys
import warnings

import humanfriendly
import pandas as pd
import tabulate
import tqdm

import sdgym
from sdgym.synthesizers.base import BaselineSynthesizer
from sdgym.utils import get_synthesizers


def _env_setup(logfile, verbosity):
    gc.enable()
    warnings.simplefilter('ignore')

    FORMAT = '%(asctime)s - %(process)d - %(levelname)s - %(name)s - %(module)s - %(message)s'
    level = (3 - verbosity) * 10
    logging.basicConfig(filename=logfile, level=level, format=FORMAT)
    logging.getLogger('sdgym').setLevel(level)
    logging.getLogger('sdmetrics').setLevel(level)
    logging.getLogger().setLevel(logging.WARN)


def _print_table(data, sort=None, reverse=False, format=None):
    if sort:
        sort_fields = sort.split(',')
        for field in sort_fields:
            data = data.sort_values(field, ascending=not reverse)

    if format:
        for field, formatter in format.items():
            data[field] = data[field].apply(formatter)

    if 'error' in data:
        error = data['error']
        if pd.isna(error).all():
            del data['error']
        else:
            long_error = error.str.len() > 30
            data.loc[long_error, 'error'] = error[long_error].str[:30] + '...'

    print(tabulate.tabulate(data, tablefmt='github', headers=data.columns, showindex=False))  # noqa: T201


def _run(args):
    _env_setup(args.logfile, args.verbose)

    if args.distributed:
        try:
            from dask.distributed import Client, LocalCluster
        except ImportError as ie:
            ie.msg += (
                '\n\nIt seems like `dask` is not installed.\n'
                'Please install `dask` and `distributed` using:\n'
                '\n    pip install dask distributed'
            )
            raise

        processes = args.workers > 1
        client = Client(
            LocalCluster(
                processes=processes,
                n_workers=args.workers,
                threads_per_worker=args.threads,
            ),
        )
        client.register_worker_callbacks(lambda: _env_setup(args.logfile, args.verbose))

    if args.jobs:
        args.jobs = json.loads(args.jobs)

    scores = sdgym.benchmark_single_table(
        synthesizers=args.synthesizers,
        sdv_datasets=args.datasets,
        sdmetrics=args.metrics,
        timeout=args.timeout,
        show_progress=args.progress,
        output_filepath=args.output_path,
    )

    if args.groupby:
        scores = scores.groupby(args.groupby).mean().reset_index()

    if scores is not None:
        _print_table(scores)


def _download_datasets(args):
    _env_setup(args.logfile, args.verbose)
    datasets = args.datasets
    if not datasets:
        datasets = sdgym.datasets.get_available_datasets(
            args.bucket, args.aws_key, args.aws_secret
        )['name']

    for dataset in tqdm.tqdm(datasets):
        sdgym.datasets.load_dataset(
            dataset, args.datasets_path, args.bucket, args.aws_key, args.aws_secret
        )


def _list_downloaded(args):
    datasets = sdgym.cli.utils.get_downloaded_datasets(args.datasets_path)
    _print_table(datasets, args.sort, args.reverse, {'size': humanfriendly.format_size})
    print(f'Found {len(datasets)} downloaded datasets')  # noqa: T201


def _list_available(args):
    datasets = sdgym.datasets.get_available_datasets(args.bucket, args.aws_key, args.aws_secret)
    _print_table(datasets, args.sort, args.reverse, {'size': humanfriendly.format_size})


def _list_synthesizers(args):
    synthesizers = BaselineSynthesizer.get_baselines()
    _print_table(pd.DataFrame(get_synthesizers(list(synthesizers))))


def _collect(args):
    sdgym.cli.collect.collect_results(
        args.input_path, args.output_file, args.aws_key, args.aws_secret
    )


def _summary(args):
    sdgym.cli.summary.make_summary_spreadsheet(
        args.input_path,
        output_path=args.output_file,
        aws_key=args.aws_key,
        aws_secret=args.aws_secret,
    )


def _get_parser():
    parser = argparse.ArgumentParser(description='SDGym Command Line Interface')
    parser.set_defaults(action=None)
    action = parser.add_subparsers(title='action')
    action.required = True

    # run
    run = action.add_parser('run', help='Run the SDGym Benchmark.')
    run.set_defaults(action=_run)

    run.add_argument(
        '-s',
        '--synthesizers',
        nargs='*',
        type=str,
        required=False,
        help='List of synthesizers to benchmark.',
    )
    run.add_argument(
        '-d',
        '--datasets',
        nargs='*',
        type=str,
        required=False,
        help='List of datasets to benchmark.',
    )
    run.add_argument(
        '-c',
        '--cache-dir',
        type=str,
        required=False,
        help='Directory where the intermediate results will be stored.',
    )
    run.add_argument(
        '-o',
        '--output-path',
        type=str,
        required=False,
        help='Path to the CSV file where the report will be dumped',
    )
    run.add_argument('-m', '--metrics', nargs='+', help='Metrics to apply. Accepts multiple names.')
    run.add_argument('-b', '--bucket', help='Bucket from which to download the datasets.')
    run.add_argument('-dp' '--datasets-path', help='Path where datasets can be found.')
    run.add_argument(
        '-dm' '--modalities', nargs='+', help='Data Modalities to run. Accepts multiple names.'
    )
    run.add_argument('-i', '--iterations', type=int, default=1, help='Number of iterations.')
    run.add_argument(
        '-D', '--distributed', action='store_true', help='Distribute computation using dask.'
    )
    run.add_argument(
        '-W',
        '--workers',
        type=int,
        default=1,
        help='Number of workers to use when distributing locally.',
    )
    run.add_argument(
        '-T', '--threads', type=int, help='Number of threads to use when distributing locally.'
    )
    run.add_argument('-l', '--logfile', type=str, help='Name of the log file.')
    run.add_argument(
        '-v',
        '--verbose',
        action='count',
        default=0,
        help='Be verbose. Repeat for increased verbosity.',
    )
    run.add_argument(
        '-p', '--progress', action='store_true', help='Print a progress bar using tqdm.'
    )
    run.add_argument(
        'run_on_ec2',
        action='store_true',
        help='Run job on created ec2 instance with environment aws variables',
    )
    run.add_argument('-t', '--timeout', type=int, help='Maximum seconds to run for each dataset.')
    run.add_argument(
        '-g', '--groupby', nargs='+', help='Group scores leaderboard by the given fields.'
    )
    run.add_argument(
        '-ak' '--aws-key',
        type=str,
        required=False,
        help='Aws access key ID to use when reading datasets.',
    )
    run.add_argument(
        '-as' '--aws-secret',
        type=str,
        required=False,
        help='Aws secret access key to use when reading datasets.',
    )
    run.add_argument(
        '-j', '--jobs', type=str, required=False, help='Serialized list of jobs to run.'
    )
    run.add_argument(
        '-mr' '--max-rows', type=int, help='Cap the number of rows to model from each dataset.'
    )
    run.add_argument(
        '-mc' '--max-columns',
        type=int,
        help='Cap the number of columns to model from each dataset.',
    )

    # download-datasets
    download = action.add_parser('download-datasets', help='Download datasets.')
    download.set_defaults(action=_download_datasets)
    download.add_argument('-b', '--bucket', help='Bucket from which to download the datasets.')
    download.add_argument(
        '-d', '--datasets', nargs='+', help='Datasets/s to be downloaded. Accepts multiple names.'
    )
    download.add_argument(
        '-dp', '--datasets-path', help='Optional path to download the datasets to.'
    )
    download.add_argument(
        '-v',
        '--verbose',
        action='count',
        default=0,
        help='Be verbose. Repeat for increased verbosity.',
    )
    download.add_argument('-l', '--logfile', type=str, help='Name of the log file.')
    download.add_argument(
        '-ak',
        '--aws-key',
        type=str,
        required=False,
        help='Aws access key ID to use when reading datasets.',
    )
    download.add_argument(
        '-as',
        '--aws-secret',
        type=str,
        required=False,
        help='Aws secret access key to use when reading datasets.',
    )

    # list-downloaded-datasets
    list_downloaded = action.add_parser('list-downloaded', help='List downloaded datasets.')
    list_downloaded.set_defaults(action=_list_downloaded)
    list_downloaded.add_argument(
        '-s', '--sort', default='name', help='Value to sort by (name|size). Defaults to `name`.'
    )
    list_downloaded.add_argument('-r', '--reverse', action='store_true', help='Reverse the order.')
    list_downloaded.add_argument(
        '-dp', '--datasets-path', help='Path where the datasets can be found.'
    )

    # list-available-datasets
    list_available = action.add_parser(
        'list-available', help='List datasets available for download.'
    )
    list_available.set_defaults(action=_list_available)
    list_available.add_argument(
        '-s',
        '--sort',
        default='name',
        help='Value to sort by (name|size|modality). Defaults to `name`.',
    )
    list_available.add_argument('-r', '--reverse', action='store_true', help='Reverse the order.')
    list_available.add_argument(
        '-b', '--bucket', help='Bucket from which to download the datasets.'
    )
    list_available.add_argument(
        '-ak',
        '--aws-key',
        type=str,
        required=False,
        help='Aws access key ID to use when reading datasets.',
    )
    list_available.add_argument(
        '-as',
        '--aws-secret',
        type=str,
        required=False,
        help='Aws secret access key to use when reading datasets.',
    )

    # list-synthesizers
    list_available = action.add_parser(
        'list-synthesizers', help='List synthesizers available for use.'
    )
    list_available.set_defaults(action=_list_synthesizers)

    # collect
    collect = action.add_parser('collect', help='Collect sdgym results.')
    collect.set_defaults(action=_collect)
    collect.add_argument(
        '-i',
        '--input-path',
        type=str,
        required=True,
        help='Path within which to look for sdgym results.',
    )
    collect.add_argument(
        '-o', '--output-file', type=str, help='Output file containing the collected results.'
    )
    collect.add_argument(
        '-ak',
        '--aws-key',
        type=str,
        required=False,
        help='Aws access key ID to use when reading datasets.',
    )
    collect.add_argument(
        '-as',
        '--aws-secret',
        type=str,
        required=False,
        help='Aws secret access key to use when reading datasets.',
    )

    # summary
    summary = action.add_parser('summary', help='Create summary file for sdgym results.')
    summary.set_defaults(action=_summary)
    summary.add_argument(
        '-i', '--input-path', type=str, required=True, help='Path to sdgym results file.'
    )
    summary.add_argument(
        '-o',
        '--output-file',
        type=str,
        required=False,
        help='Output file containing summary xlsx doc.',
    )
    summary.add_argument(
        '-ak',
        '--aws-key',
        type=str,
        required=False,
        help='Aws access key ID to use when reading datasets.',
    )
    summary.add_argument(
        '-as',
        '--aws-secret',
        type=str,
        required=False,
        help='Aws secret access key to use when reading datasets.',
    )

    return parser


def main():
    """Run CLI."""
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.max_rows', 1000)

    parser = _get_parser()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    try:
        args.action(args)
    except sdgym.errors.SDGymError as error:
        print(f'ERROR: {error}')  # noqa: T201


if __name__ == '__main__':
    main()
