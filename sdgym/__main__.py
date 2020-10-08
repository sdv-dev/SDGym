import argparse
import gc
import logging
import sys
import warnings

import pandas as pd
import tabulate

import sdgym


def _env_setup(logfile, verbose):
    gc.enable()
    warnings.simplefilter('ignore')

    FORMAT = '%(asctime)s - %(process)d - %(levelname)s - %(name)s - %(module)s - %(message)s'
    logging.basicConfig(filename=logfile, level=logging.INFO, format=FORMAT)
    logging.getLogger().setLevel(logging.WARN)
    if verbose:
        logging.getLogger('sdgym.benchmark').setLevel(logging.INFO)


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

        client = Client(LocalCluster(n_workers=args.workers, threads_per_worker=args.threads))
        client.register_worker_callbacks(lambda: _env_setup(args.logfile, args.verbose))

        workers = 'dask'
    else:
        workers = args.workers

    synthesizers = sdgym.get_all_synthesizers()
    if args.models:
        synthesizers = {model: synthesizers[model] for model in args.models}

    lb = sdgym.run(
        synthesizers=synthesizers,
        datasets=args.datasets,
        iterations=args.iterations,
        output_path=args.output_path,
        cache_dir=args.cache_dir,
        workers=workers
    )
    if lb is not None:
        print(lb)


def _make_leaderboard(args):
    lb = sdgym.results.make_leaderboard(args.input, output_path=args.output)
    if not args.output:
        print(lb)


def _make_summary(args):
    summary = sdgym.results.summarize_results(args.input, args.output)

    for title, section in summary.items():
        print('\n### {}\n'.format(title))
        print(tabulate.tabulate(
            section.reset_index(),
            tablefmt='github',
            headers=['Synthesizer'] + list(section.columns),
            showindex=False
        ))


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
    run.add_argument('-m', '--models', nargs='+',
                     help='Models/s to be benchmarked. Accepts multiple names.')
    run.add_argument('-d', '--datasets', nargs='+',
                     help='Datasets/s to be used. Accepts multiple names.')
    run.add_argument('-i', '--iterations', type=int, default=3,
                     help='Number of iterations.')
    run.add_argument('-D', '--distributed', action='store_true',
                     help='Distribute computation using dask.')
    run.add_argument('-W', '--workers', type=int, default=1,
                     help='Number of workers to use when distributing locally.')
    run.add_argument('-T', '--threads', type=int,
                     help='Number of threads to use when distributing locally.')
    run.add_argument('-l', '--logfile', type=str,
                     help='Name of the log file.')
    run.add_argument('-v', '--verbose', action='store_true',
                     help='Be verbose.')

    # make-leaderboard
    make_leaderboard = action.add_parser('make-leaderboard',
                                         help='Make a leaderboard from cached results.')
    make_leaderboard.set_defaults(action=_make_leaderboard)
    make_leaderboard.add_argument('input', help='Input path with results.')
    make_leaderboard.add_argument('output', help='Output file.')

    make_summary = action.add_parser('make-summary', help='Summarize multiple leaderboards.')
    make_summary.set_defaults(action=_make_summary)
    make_summary.add_argument('input', nargs='+', help='Input path with results.')
    make_summary.add_argument('output', help='Output file.')

    return parser


def main():
    pd.set_option('max_columns', 1000)
    pd.set_option('max_rows', 1000)

    parser = _get_parser()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    args.action(args)


if __name__ == '__main__':
    main()
