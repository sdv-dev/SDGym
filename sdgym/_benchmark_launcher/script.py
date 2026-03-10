import argparse
import warnings

from sdgym._benchmark_launcher.benchmark_config import BenchmarkConfig
from sdgym._benchmark_launcher.benchmark_launcher import BenchmarkLauncher
from sdgym._benchmark_launcher.utils import _deep_merge, _resolve_modality_config
from sdgym.run_benchmark.utils import OUTPUT_DESTINATION_AWS


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-filepath', type=str, default=None)
    parser.add_argument(
        '--modality',
        choices=['single_table', 'multi_table'],
        default=None,
    )
    parser.add_argument('--datasets', type=str, default=None)
    parser.add_argument('--synthesizers', type=str, default=None)
    parser.add_argument('--num-instances', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=None)
    parser.add_argument('--output-destination', type=str, default=None)

    return parser.parse_args()


def _parse_csv(value):
    """Parse a comma-separated string into a list."""
    if not value:
        return None

    return [item.strip() for item in value.split(',') if item.strip()]


def _validate_args(args):
    """Validate command-line arguments."""
    if args.config_filepath is not None:
        return

    if args.modality is None:
        raise ValueError("'--modality' is required when '--config-filepath' is not provided.")

    if args.datasets is None:
        raise ValueError("'--datasets' is required in manual mode.")

    if args.synthesizers is None:
        raise ValueError("'--synthesizers' is required in manual mode.")

    if args.output_destination is None:
        raise ValueError("'--output-destination' is required in manual mode.")

    if args.output_destination == OUTPUT_DESTINATION_AWS:
        raise ValueError(
            f"'--output-destination' cannot be {OUTPUT_DESTINATION_AWS!r} in manual mode."
        )

    if args.num_instances < 1:
        raise ValueError("'--num-instances' must be greater than or equal to 1.")


def _split_list(values):
    """Split a list into two non-empty parts, as evenly as possible."""
    midpoint = len(values) // 2
    return values[:midpoint], values[midpoint:]


def _instance_job_size(instance_job):
    """Return the number of atomic jobs in an instance job."""
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
            },
            {
                'synthesizers': right_synthesizers,
                'datasets': datasets,
            },
        ]

    if len(datasets) > 1:
        left_datasets, right_datasets = _split_list(datasets)
        return [
            {
                'synthesizers': synthesizers,
                'datasets': left_datasets,
            },
            {
                'synthesizers': synthesizers,
                'datasets': right_datasets,
            },
        ]

    raise ValueError('Cannot split a 1x1 instance job any further.')


def _build_instance_jobs(datasets, synthesizers, num_instances):
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


def build_dict_from_args(args):
    """Build a config override dict from command-line arguments."""
    config = {'method_params': {}}
    if args.timeout is not None:
        config['method_params']['timeout'] = args.timeout

    config['method_params']['output_destination'] = args.output_destination
    datasets = _parse_csv(args.datasets)
    synthesizers = _parse_csv(args.synthesizers)
    config['instance_jobs'] = _build_instance_jobs(
        datasets=datasets,
        synthesizers=synthesizers,
        num_instances=args.num_instances,
    )

    return config


def build_config_from_args(args):
    """Build a BenchmarkConfig from command-line arguments."""
    base_dict = _resolve_modality_config(args.modality)
    args_dict = build_dict_from_args(args)
    config_dict = _deep_merge(base_dict, args_dict)

    return BenchmarkConfig.load_from_dict(config_dict)


def launch_from_args():
    """Launch the benchmark described by command-line arguments."""
    args = _parse_args()
    _validate_args(args)
    if args.config_filepath is not None:
        config = BenchmarkConfig.load_from_yaml(args.config_filepath)
    else:
        config = build_config_from_args(args)

    BenchmarkLauncher(config).launch()


if __name__ == '__main__':
    launch_from_args()
