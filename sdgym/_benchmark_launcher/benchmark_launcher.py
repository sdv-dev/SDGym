"""Define the BenchmarkLauncher class, which launches and manages benchmark executions."""

import logging
import warnings

import cloudpickle
from google.cloud import compute_v1
from google.oauth2 import service_account

from sdgym._benchmark_launcher._validation import _validate_gcp_credentials
from sdgym._benchmark_launcher.utils import (
    _METHODS,
    _resolve_datasets,
    generate_benchmark_ids,
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
        self._benchmark_id = generate_benchmark_ids([
            'BENCMARK_ID',
            self.modality,
            self.compute_service,
        ])
        self._launch_to_instance_names = {}
        self._instance_name_to_status = {}

    def _launch(self):
        launch_id = generate_benchmark_ids(['LAUNCH_ID'])
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

    def _update_gcp_instance_name_to_status(self):
        """Update local instance statuses using the current GCP state."""
        client, project_id = self._get_gcp_client()
        running_instances = self._list_gcp_instances(client, project_id)
        running_instance_names = {instance['name'] for instance in running_instances}
        for instance_name in self._get_all_instance_names():
            if instance_name in running_instance_names:
                self._instance_name_to_status[instance_name] = 'running'
            else:
                self._instance_name_to_status[instance_name] = 'terminated'

    def _update_instance_name_to_status(self):
        if self.compute_service == 'gcp':
            self._update_gcp_instance_name_to_status()
            return

        raise NotImplementedError(
            f'`_update_instance_name_to_status()` is not implemented for {self.compute_service!r}.'
        )

    def _list_gcp_instances(self, client, project_id):
        """List all non-terminated GCP instances."""
        instances = []
        response = client.aggregated_list(project=project_id)
        for _, scoped_list in response:
            scoped_instances = getattr(scoped_list, 'instances', None)
            if not scoped_instances:
                continue

            for instance in scoped_instances:
                if instance.status == 'TERMINATED':
                    continue

                instances.append({
                    'id': str(instance.id),
                    'name': instance.name,
                    'zone': instance.zone.split('/')[-1],
                    'status': instance.status,
                })

        return instances

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

    def _get_gcp_client(self):
        """Build and return the GCP client and project id."""
        credentials = resolve_credentials(self.benchmark_config.credentials_filepath)
        errors = _validate_gcp_credentials(credentials)
        if errors:
            error_message = '\n'.join(errors)
            raise ValueError(f'Invalid GCP credentials:\n{error_message}')

        project_id = credentials['gcp']['project_id']
        gcp_credentials = service_account.Credentials.from_service_account_info(credentials['gcp'])
        client = compute_v1.InstancesClient(credentials=gcp_credentials)

        return client, project_id

    def _terminate_gcp_instances(self, instance_names, verbose):
        """Terminate GCP instances by their names."""
        client, project_id = self._get_gcp_client()
        running_instances = self._list_gcp_instances(client, project_id)
        running_instances_by_name = {instance['name']: instance for instance in running_instances}
        instances_to_delete = [
            running_instances_by_name[name]
            for name in instance_names
            if name in running_instances_by_name
        ]

        not_running = sorted(set(instance_names) - set(running_instances_by_name))
        if not_running:
            not_running_str = "', '".join(not_running)
            LOGGER.info(
                f"Some provided instance names are not currently running: '{not_running_str}'."
            )

        deleted_instances = []
        for instance in instances_to_delete:
            if verbose:
                message = (
                    f"Terminating GCP instance '{instance['name']}' "
                    f'(id={instance["id"]}, zone={instance["zone"]})...'
                )
                print(message)  # noqa: T201

            operation = client.delete(
                project=project_id,
                zone=instance['zone'],
                instance=instance['name'],
            )
            operation.result()
            self._instance_name_to_status[instance['name']] = 'terminated'
            deleted_instances.append(instance['name'])

        return deleted_instances

    def _validate_inputs_and_get_instances(self, instance_names, verbose):
        """Validate terminate inputs and return the instance names to process."""
        if self.compute_service != 'gcp':
            raise NotImplementedError(
                '`terminate()` is only implemented for GCP instances for now.'
            )

        if not isinstance(verbose, bool):
            raise ValueError(f'`verbose` must be a boolean. Found: {verbose!r} ({type(verbose)}).')

        instances = self._validate_instance_names(instance_names)
        return instances

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
        self._update_instance_name_to_status()
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

        deleted_instances = self._terminate_gcp_instances(instances_to_terminate, verbose)
        self._update_instance_name_to_status()
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
            benchmark._benchmark_id = generate_benchmark_ids([
                'BENCMARK_ID',
                benchmark.modality,
                benchmark.compute_service,
            ])

        return benchmark
