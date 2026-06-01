import argparse
from itertools import product

from sdgym._benchmark_launcher.benchmark_config import BenchmarkConfig
from sdgym._benchmark_launcher.benchmark_launcher import BenchmarkLauncher
from sdgym._benchmark_launcher.utils import (
    _deep_merge,
    _load_merged_modality_config,
    _resolve_datasets,
    _resolve_modality_config,
)
from sdgym.run_benchmark.utils import OUTPUT_DESTINATION_AWS


def _parse_args():
    """Parse CLI arguments for launching a benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config-filepath',
        default=None,
        help='Path to a YAML benchmark configuration file.',
    )
    parser.add_argument(
        '--modality',
        choices=['single_table', 'multi_table'],
        default=None,
        help='Benchmark modality to run when not using --config-filepath.',
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=None,
        help=(
            'Datasets to benchmark. Defaults to the datasets used by the internal '
            'benchmark for the given modality.'
        ),
    )
    parser.add_argument(
        '--synthesizers',
        nargs='+',
        default=None,
        help=(
            'Synthesizers to benchmark. Defaults to the synthesizers used by the internal '
            'benchmark for the given modality.'
        ),
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=None,
        help='Timeout in seconds to include in method_params.',
    )
    parser.add_argument(
        '--output-destination',
        default=None,
        help='Destination where benchmark results will be written.',
    )

    return parser.parse_args()


def _validate_args(args):
    """Validate command-line arguments."""
    if args.config_filepath is not None:
        if any(
            value is not None
            for value in (
                args.modality,
                args.datasets,
                args.synthesizers,
                args.output_destination,
                args.timeout,
            )
        ):
            raise ValueError(
                "'--config-filepath' cannot be combined with the other benchmark arguments."
            )

        return

    if args.modality is None:
        raise ValueError("'--modality' is required when '--config-filepath' is not provided.")

    if args.output_destination is None:
        raise ValueError(
            "'--output-destination' is required when '--config-filepath' is not provided."
        )

    if args.output_destination == OUTPUT_DESTINATION_AWS:
        raise ValueError(
            f"'--output-destination' cannot be {OUTPUT_DESTINATION_AWS!r} that is reserved "
            'for internal benchmarks'
        )


def _build_instance_jobs(datasets, synthesizers, output_destination):
    """Build one instance job per dataset and synthesizer pair."""
    return [
        {
            'synthesizers': [synthesizer],
            'datasets': [dataset],
            'output_destination': output_destination,
        }
        for dataset, synthesizer in product(datasets, synthesizers)
    ]


def _get_default_datasets_and_synthesizers(modality):
    """Get the default datasets and synthesizers for a modality config."""
    base_dict = _load_merged_modality_config(modality)
    datasets = []
    synthesizers = []
    for instance_job in base_dict.get('instance_jobs', []):
        datasets.extend(_resolve_datasets(instance_job.get('datasets', [])))
        synthesizers.extend(instance_job.get('synthesizers', []))

    return sorted(set(datasets)), sorted(set(synthesizers))


def build_dict_from_args(args):
    """Build a config override dict from command-line arguments."""
    method_params = {}
    if args.timeout is not None:
        method_params['timeout'] = args.timeout

    datasets = args.datasets
    synthesizers = args.synthesizers
    if all(value is None for value in (datasets, synthesizers)):
        config = _resolve_modality_config(args.modality)
        config['method_params'] = method_params
        for config_instance_job in config.get('instance_jobs', []):
            config_instance_job['output_destination'] = args.output_destination

        return config

    default_datasets, default_synthesizers = _get_default_datasets_and_synthesizers(args.modality)
    datasets = datasets if datasets is not None else default_datasets
    synthesizers = synthesizers if synthesizers is not None else default_synthesizers
    return {
        'method_params': method_params,
        'instance_jobs': _build_instance_jobs(
            datasets=datasets,
            synthesizers=synthesizers,
            output_destination=args.output_destination,
        ),
    }


def build_config_from_args(args):
    """Build a BenchmarkConfig from command-line arguments."""
    base_dict = _resolve_modality_config(args.modality)
    args_dict = build_dict_from_args(args)
    config_dict = _deep_merge(base_dict, args_dict)

    return BenchmarkConfig.load_from_dict(config_dict)


def launch_from_args():
    """Launch a benchmark using command-line arguments.

    This function supports two modes:

    1. If ``--config-filepath`` is provided, the benchmark configuration is
       loaded from that file. In this case, it cannot be combined with the
       other benchmark-related command-line arguments.

    2. If ``--config-filepath`` is not provided, the benchmark configuration is
       built from the remaining command-line arguments. In this case,
       ``--modality`` and ``--output-destination`` are required, while the
       other arguments are optional.

    When building the configuration from command-line arguments:

    - If ``--datasets`` and ``--synthesizers`` are both omitted, the default
      monthly benchmark configuration for the selected modality is used.
    - If ``--datasets`` or ``--synthesizers`` is omitted, the corresponding
      default values from the monthly benchmark configuration are used.
    - One instance job is created for each dataset and synthesizer pair.

    Once the configuration is resolved, the benchmark is launched.
    """
    args = _parse_args()
    _validate_args(args)
    if args.config_filepath is not None:
        config = BenchmarkConfig.load_from_yaml(args.config_filepath)
    else:
        config = build_config_from_args(args)

    BenchmarkLauncher(config).launch()


if __name__ == '__main__':
    launch_from_args()
