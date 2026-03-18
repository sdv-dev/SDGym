import argparse
import warnings

from sdgym._benchmark_launcher.benchmark_config import BenchmarkConfig
from sdgym._benchmark_launcher.benchmark_launcher import BenchmarkLauncher
from sdgym._benchmark_launcher.utils import (
    _deep_merge,
    _load_merged_modality_config,
    _resolve_modality_config,
)
from sdgym.run_benchmark.utils import OUTPUT_DESTINATION_AWS

DEFAULT_NUM_INSTANCES = 1


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
        '--num-instances',
        type=int,
        default=None,
        help='Number of benchmark instances to create. Defaults to 1.',
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
                args.num_instances,
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

    if args.num_instances is not None and args.num_instances < 1:
        raise ValueError("'--num-instances' must be greater than or equal to 1.")

    if args.output_destination == OUTPUT_DESTINATION_AWS:
        raise ValueError(
            f"'--output-destination' cannot be {OUTPUT_DESTINATION_AWS!r} that is reserved "
            'for internal benchmarks'
        )


def _split_list(values):
    """Split a list into two non-empty parts, as evenly as possible."""
    midpoint = len(values) // 2
    return values[:midpoint], values[midpoint:]


def _instance_job_size(instance_job):
    """Return the number of synthesizer and dataset combinations."""
    return len(instance_job['synthesizers']) * len(instance_job['datasets'])


def _split_instance_jobs(instance_job):
    """Split an instance job into two smaller instance jobs.

    Prefer splitting synthesizers. If there is only one synthesizer,
    split datasets instead.
    """
    synthesizers = instance_job['synthesizers']
    datasets = instance_job['datasets']
    if len(synthesizers) > 1:
        left_synthesizers, right_synthesizers = _split_list(synthesizers)
        return [
            {
                'synthesizers': left_synthesizers,
                'datasets': datasets,
                'output_destination': instance_job['output_destination'],
            },
            {
                'synthesizers': right_synthesizers,
                'datasets': datasets,
                'output_destination': instance_job['output_destination'],
            },
        ]

    if len(datasets) > 1:
        left_datasets, right_datasets = _split_list(datasets)
        return [
            {
                'synthesizers': synthesizers,
                'datasets': left_datasets,
                'output_destination': instance_job['output_destination'],
            },
            {
                'synthesizers': synthesizers,
                'datasets': right_datasets,
                'output_destination': instance_job['output_destination'],
            },
        ]

    raise ValueError('Cannot split the instance job any further.')


def _build_instance_jobs(datasets, synthesizers, num_instances, output_destination):
    """Build exactly ``num_instances`` instance jobs."""
    max_jobs = len(synthesizers) * len(datasets)
    if num_instances > max_jobs:
        num_instances = max_jobs
        warnings.warn(
            f'num_instances is too high for the number of synthesizers and datasets. '
            f'Maximum number of instances is {max_jobs}. Setting num_instances to {max_jobs}.'
        )

    instance_jobs = [
        {
            'synthesizers': list(synthesizers),
            'datasets': list(datasets),
            'output_destination': output_destination,
        }
    ]
    while len(instance_jobs) < num_instances:
        split_index = None
        split_size = -1
        for index, instance_job in enumerate(instance_jobs):
            if (_instance_job_size(instance_job) > 1) and (
                _instance_job_size(instance_job) > split_size
            ):
                split_index = index
                split_size = _instance_job_size(instance_job)

        instance_job = instance_jobs.pop(split_index)
        instance_jobs.extend(_split_instance_jobs(instance_job))

    return instance_jobs


def _get_default_datasets_and_synthesizers(modality):
    """Get the default datasets and synthesizers for a modality config."""
    base_dict = _load_merged_modality_config(modality)
    datasets = base_dict.get(f'datasets_{modality}', [])
    synthesizers = []
    for instance_job in base_dict.get('instance_jobs', []):
        synthesizers.extend(instance_job.get('synthesizers', []))

    return sorted(set(datasets)), sorted(set(synthesizers))


def build_dict_from_args(args):
    """Build a config override dict from command-line arguments."""
    method_params = {}
    if args.timeout is not None:
        method_params['timeout'] = args.timeout

    datasets = args.datasets
    synthesizers = args.synthesizers
    num_instances = args.num_instances
    if all(value is None for value in (datasets, synthesizers, num_instances)):
        config = _resolve_modality_config(args.modality)
        config['method_params'] = method_params
        for config_instance_job in config.get('instance_jobs', []):
            config_instance_job['output_destination'] = args.output_destination

        return config

    default_datasets, default_synthesizers = _get_default_datasets_and_synthesizers(args.modality)
    datasets = datasets if datasets is not None else default_datasets
    synthesizers = synthesizers if synthesizers is not None else default_synthesizers
    num_instances = num_instances if num_instances is not None else DEFAULT_NUM_INSTANCES
    return {
        'method_params': method_params,
        'instance_jobs': _build_instance_jobs(
            datasets=datasets,
            synthesizers=synthesizers,
            num_instances=num_instances,
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

    - If ``--datasets``, ``--synthesizers``, and ``--num-instances`` are all
      omitted, the default monthly benchmark configuration for the selected
      modality is used.
    - If ``--num-instances`` is omitted, it defaults to ``1``.
    - If ``--datasets`` or ``--synthesizers`` is omitted, the corresponding
      default values from the monthly benchmark configuration are used.

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
