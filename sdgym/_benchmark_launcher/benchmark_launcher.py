"""Define the BenchmarkLauncher class, which launches and manages benchmark executions."""

import logging
import warnings

import cloudpickle

from sdgym._benchmark_launcher._instance_manager import GCPInstanceManager
from sdgym._benchmark_launcher.utils import (
    _METHODS,
    _resolve_datasets,
    generate_ids,
    resolve_credentials,
)

LOGGER = logging.getLogger(__name__)


class BenchmarkLauncher:
    """Launch and manage benchmark executions.

    The ``BenchmarkLauncher`` orchestrates the execution of a benchmark defined
    by a ``BenchmarkConfig``. It validates the configuration, resolves the
    required credentials, and dispatches the benchmark jobs to the appropriate
    execution method depending on the configured modality and compute service.

    The `launch()` method starts one instance per `instance_job` on the
    configured compute service (e.g., AWS or GCP) and executes the specified
    benchmark method on each instance with the provided parameters. This enables
    multiple machines to run the benchmark in parallel.

    The launcher also provides utilities to monitor and manage running jobs,
    such as retrieving their status or terminating them.

    Args:
        benchmark_config (BenchmarkConfig):
            The benchmark configuration describing the datasets, synthesizers,
            compute environment, credentials, and parameters required to run
            the benchmark.
    """

    def __init__(self, benchmark_config):
        benchmark_config.validate()
        self.benchmark_config = benchmark_config
        self.modality = benchmark_config.modality
        self.compute_service = benchmark_config.compute.get('service')
        self.method_to_run = _METHODS[(self.modality, self.compute_service)]
        self._benchmark_id = generate_ids([
            'BENCMARK_ID',
            self.modality,
            self.compute_service,
        ])
        self._launch_to_instance_names = {}
        self._instance_name_to_status = {}
        self._instance_manager = self._build_instance_manager()

    def _build_instance_manager(self):
        """Build the instance manager for the configured compute service."""
        if self.compute_service == 'gcp':
            return GCPInstanceManager(self.benchmark_config.credentials_filepath)

        raise NotImplementedError(f'Compute service {self.compute_service!r} is not supported.')

    def _launch(self):
        launch_id = generate_ids(['LAUNCH_ID'])
        self._launch_to_instance_names[launch_id] = []
        credentials = resolve_credentials(self.benchmark_config.credentials_filepath)
        for instance_job in self.benchmark_config.instance_jobs:
            sdv_datasets = _resolve_datasets(instance_job['datasets'])
            instance_name = self.method_to_run(
                output_destination=instance_job['output_destination'],
                synthesizers=instance_job['synthesizers'],
                sdv_datasets=sdv_datasets,
                credentials=credentials,
                compute_config=self.benchmark_config.compute,
                **self.benchmark_config.method_params,
            )
            self._launch_to_instance_names[launch_id].append(instance_name)
            self._instance_name_to_status[instance_name] = 'running'

    def launch(self):
        """Run the BenchmarkConfig: validate it and then execute the specified benchmark method."""
        if not self.benchmark_config._is_validated:
            self.benchmark_config.validate()

        self._launch()

    def _update_instance_statuses(self):
        """Update instance statuses using the instance manager."""
        self._instance_manager.update_instance_statuses(
            self._get_all_instance_names(),
            self._instance_name_to_status,
        )

    def _get_all_instance_names(self):
        """Return all instance names launched."""
        instance_names = []
        for names in self._launch_to_instance_names.values():
            instance_names.extend(names)

        return instance_names

    def _get_active_instance_names(self):
        """Return instance names currently marked as running."""
        return [
            instance_name
            for instance_name, status in self._instance_name_to_status.items()
            if status == 'running'
        ]

    def _validate_instance_names(self, instance_names):
        """Validate instance names."""
        launched_instances = self._get_all_instance_names()
        instances = instance_names if instance_names is not None else launched_instances
        unknown_instances = set(instances) - set(launched_instances)
        if unknown_instances:
            unknown_instances_str = "', '".join(sorted(unknown_instances))
            launched_instances_str = "', '".join(sorted(launched_instances))
            raise ValueError(
                'Some provided instance names were not launched by this '
                f"BenchmarkLauncher. Unknown: '{unknown_instances_str}'. "
                f"Launched instances: '{launched_instances_str}'."
            )

        return instances

    def _validate_inputs_and_get_instances(self, instance_names, verbose):
        """Validate terminate inputs and return the instance names to process."""
        if not isinstance(verbose, bool):
            raise ValueError(f'`verbose` must be a boolean. Found: {verbose!r} ({type(verbose)}).')

        return self._validate_instance_names(instance_names)

    def terminate(self, instance_names=None, verbose=True):
        """Terminate running benchmark instances.

        Args:
            instance_names (list of str, optional):
                List of instance names to terminate.
                If None, terminate all instances launched by this benchmark. Defaults to None.
            verbose (bool):
                Whether to print progress information. Defaults to True.
        """
        instances = self._validate_inputs_and_get_instances(instance_names, verbose)
        self._update_instance_statuses()
        active_instances = set(self._get_active_instance_names())
        instances_to_terminate = [name for name in instances if name in active_instances]
        if not instances_to_terminate:
            if instance_names is None:
                LOGGER.info('There are no running instances to terminate.')
            else:
                warnings.warn('All provided instance names are already terminated.')

            return

        already_terminated = sorted(set(instance_names or instances) - active_instances)
        if already_terminated:
            already_terminated_str = "', '".join(already_terminated)
            LOGGER.info(
                "Some provided instance names are already terminated: '%s'.",
                already_terminated_str,
            )

        deleted_instances = self._instance_manager.terminate_instances(
            instances_to_terminate,
            verbose,
        )
        for instance_name in deleted_instances:
            self._instance_name_to_status[instance_name] = 'stopped'

        if verbose:
            print(f'Terminated {len(deleted_instances)} GCP instance(s).')  # noqa: T201

    def get_status(self, dataset_names=None, synthesizer_names=None, instance_ids=None):
        """Get status of running benchmark instance jobs.

        Indicates whether the instance_id is still running, has completed successfully,
        or has failed.

        Args:
            dataset_names (list of str, optional):
                List of dataset names to get status for.
                If None, gets status for all datasets. Default to None.
            synthesizer_names (list of str, optional):
                List of synthesizer names to get status for.
                If None, gets status for all synthesizers. Default to None.
            instance_ids (list of str, optional):
                List of instance IDs to get status for.
                If None, gets status for all instance jobs. Default to None.

        Returns:
            pd.DataFrame:
                A dataframe with one row per job (synthesizer-dataset combination) and columns:
                - Dataset: The dataset used in the job.
                - Synthesizer: The synthesizer used in the job.
                - 'Instance_ID': The ID of the instance running the job.
                - 'Status': The status of the job, which can be 'Running', 'Completed', or 'Failed'.
        """
        raise NotImplementedError

    def save(self, filepath):
        """Save the benchmark configuration to a file."""
        with open(filepath, 'wb') as output:
            cloudpickle.dump(self, output)

    @classmethod
    def load(cls, filepath):
        """Load a benchmark launcher from a file."""
        with open(filepath, 'rb') as input_file:
            benchmark = cloudpickle.load(input_file)

        if getattr(benchmark, '_benchmark_id', None) is None:
            benchmark._benchmark_id = generate_ids([
                'BENCMARK_ID',
                benchmark.modality,
                benchmark.compute_service,
            ])

        return benchmark
