"""SDGym CLI module."""

import argparse
import gc
import logging
import sys
import warnings

import humanfriendly
import pandas as pd
import tabulate
import tqdm

import sdgym


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
        if pd.isnull(error).all():
            del data['error']
        else:
            long_error = error.str.len() > 30
            data.loc[long_error, 'error'] = error[long_error].str[:30] + '...'

    print(tabulate.tabulate(
        data,
        tablefmt='github',
        headers=data.columns,
        showindex=False
    ))


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

        workers = 'dask'
    else:
        workers = args.workers

    scores = sdgym.run(
        synthesizers=args.synthesizers,
        datasets=args.datasets,
        datasets_path=args.datasets_path,
        modalities=args.modalities,
        metrics=args.metrics,
        iterations=args.iterations,
        cache_dir=args.cache_dir,
        workers=workers,
        show_progress=args.progress,
        timeout=args.timeout,
        output_path=args.output_path,
    )

    if args.groupby:
        scores = scores.groupby(args.groupby).mean().reset_index()

    if scores is not None:
        _print_table(scores)


def _download_datasets(args):
    _env_setup(args.logfile, args.verbose)
    datasets = args.datasets
    if not datasets:
        datasets = sdgym.datasets.get_available_datasets(args.bucket)['name']

    for dataset in tqdm.tqdm(datasets):
        sdgym.datasets.load_dataset(dataset, args.datasets_path, args.bucket)


def _list_downloaded(args):
    datasets = sdgym.datasets.get_downloaded_datasets(args.datasets_path)
    _print_table(datasets, args.sort, args.reverse, {'size': humanfriendly.format_size})
    print(f'Found {len(datasets)} downloaded datasets')


def _list_available(args):
    datasets = sdgym.datasets.get_available_datasets(args.bucket)
    _print_table(datasets, args.sort, args.reverse, {'size': humanfriendly.format_size})


def _get_parser():
    parser = argparse.ArgumentParser(description='SDGym Command Line Interface')
    parser.set_defaults(action=None)
    action = parser.add_subparsers(title='action')
    action.required = True

    # run
    run = action.add_parser('run', help='Run the SDGym Benchmark.')
    run.set_defaults(action=_run)

    run.add_argument('-c', '--cache-dir', type=str, required=False,
                     help='Directory where the intermediate results will be stored.')
    run.add_argument('-o', '--output-path', type=str, required=False,
                     help='Path to the CSV file where the report will be dumped')
    run.add_argument('-s', '--synthesizers', nargs='+',
                     help='Synthesizer/s to be benchmarked. Accepts multiple names.')
    run.add_argument('-m', '--metrics', nargs='+',
                     help='Metrics to apply. Accepts multiple names.')
    run.add_argument('-d', '--datasets', nargs='+',
                     help='Datasets/s to be used. Accepts multiple names.')
    run.add_argument('-dp', '--datasets-path',
                     help='Path where datasets can be found.')
    run.add_argument('-dm', '--modalities', nargs='+',
                     help='Data Modalities to run. Accepts multiple names.')
    run.add_argument('-i', '--iterations', type=int, default=1,
                     help='Number of iterations.')
    run.add_argument('-D', '--distributed', action='store_true',
                     help='Distribute computation using dask.')
    run.add_argument('-W', '--workers', type=int, default=1,
                     help='Number of workers to use when distributing locally.')
    run.add_argument('-T', '--threads', type=int,
                     help='Number of threads to use when distributing locally.')
    run.add_argument('-l', '--logfile', type=str,
                     help='Name of the log file.')
    run.add_argument('-v', '--verbose', action='count', default=0,
                     help='Be verbose. Repeat for increased verbosity.')
    run.add_argument('-p', '--progress', action='store_true',
                     help='Print a progress bar using tqdm.')
    run.add_argument('-t', '--timeout', type=int,
                     help='Maximum seconds to run for each dataset.')
    run.add_argument('-g', '--groupby', nargs='+',
                     help='Group scores leaderboard by the given fields')

    # download-datasets
    download = action.add_parser('download-datasets', help='Download datasets.')
    download.set_defaults(action=_download_datasets)
    download.add_argument('-b', '--bucket',
                          help='Bucket from which to download the datasets.')
    download.add_argument('-d', '--datasets', nargs='+',
                          help='Datasets/s to be downloaded. Accepts multiple names.')
    download.add_argument('-dp', '--datasets-path',
                          help='Optional path to download the datasets to.')
    download.add_argument('-v', '--verbose', action='count', default=0,
                          help='Be verbose. Repeat for increased verbosity.')
    download.add_argument('-l', '--logfile', type=str,
                          help='Name of the log file.')

    # list-available-datasets
    list_downloaded = action.add_parser('list-downloaded', help='List downloaded datasets.')
    list_downloaded.set_defaults(action=_list_downloaded)
    list_downloaded.add_argument('-s', '--sort', default='name',
                                 help='Value to sort by (name|size). Defaults to `name`.')
    list_downloaded.add_argument('-r', '--reverse', action='store_true',
                                 help='Reverse the order.')
    list_downloaded.add_argument('-dp', '--datasets-path',
                                 help='Path where the datasets can be found.')

    # list-available-datasets
    list_available = action.add_parser('list-available',
                                       help='List datasets available for download.')
    list_available.add_argument('-s', '--sort', default='name',
                                help='Value to sort by (name|size|modality). Defaults to `name`.')
    list_available.add_argument('-r', '--reverse', action='store_true',
                                help='Reverse the order.')
    list_available.add_argument('-b', '--bucket',
                                help='Bucket from which to download the datasets.')
    list_available.set_defaults(action=_list_available)

    return parser


def main():
    pd.set_option('max_columns', 1000)
    pd.set_option('max_rows', 1000)

    parser = _get_parser()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    try:
        args.action(args)
    except sdgym.errors.SDGymError as error:
        print(f'ERROR: {error}')


if __name__ == '__main__':
    main()
