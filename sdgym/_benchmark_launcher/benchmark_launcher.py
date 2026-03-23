"""Define the BenchmarkLauncher class, which launches and manages benchmark executions."""

import logging
import warnings

import cloudpickle
from google.cloud import compute_v1
from google.oauth2 import service_account

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
        self.launch_to_instance_ids = {}

    def _launch(self):
        launch_id = generate_benchmark_ids(['LAUNCH_ID'])
        self.launch_to_instance_ids[launch_id] = []
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
            self.launch_to_instance_ids[launch_id].append(instance_name)

    def launch(self):
        """Run the BenchmarkConfig: validate it and then execute the specified benchmark method."""
        if not self.benchmark_config._is_validated:
            self.benchmark_config.validate()

        self._launch()

    def _extract_zone_name(self, zone):
        """Extract the zone name from a GCP zone URL."""
        return zone.split('/')[-1]

    def _list_gcp_instances(self, client, project_id):
        """List all non-terminated GCP instances in the configured project."""
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
                    'zone': self._extract_zone_name(instance.zone),
                    'status': instance.status,
                })

        return instances

    def _get_all_instance_names(self):
        """Return all instance names launched by this BenchmarkLauncher."""
        instance_names = []
        for names in self.launch_to_instance_ids.values():
            instance_names.extend(names)

        return instance_names

    def terminate(self, instances=None, verbose=True):
        """Terminate running benchmark instances.

        Args:
            instances (list of str, optional):
                List of instance names to terminate.
                If None, terminate all instances launched by this benchmark.
            verbose (bool):
                Whether to print progress information. Defaults to True.
        """
        if self.compute_service != 'gcp':
            raise NotImplementedError(
                f'terminate is only implemented for GCP right now. Got: {self.compute_service!r}.'
            )

        launched_instances = self._get_all_instance_names()
        instances = instances if instances is not None else launched_instances

        unknown_instances = set(instances) - set(launched_instances)
        if unknown_instances:
            unknown_instances_str = "', '".join(sorted(unknown_instances))
            launched_instances_str = "', '".join(sorted(launched_instances))
            raise ValueError(
                'Some provided instance names were not launched by this '
                f"BenchmarkLauncher. Unknown: '{unknown_instances_str}'. "
                f"Launched instances: '{launched_instances_str}'."
            )

        credentials = resolve_credentials(self.benchmark_config.credentials_filepath)
        gcp_credentials_info = credentials.get('gcp')
        project_id = gcp_credentials_info.get('project_id')
        gcp_credentials = service_account.Credentials.from_service_account_info(
            gcp_credentials_info
        )

        client = compute_v1.InstancesClient(credentials=gcp_credentials)
        running_instances = self._list_gcp_instances(client, project_id)
        running_instances_by_name = {instance['name']: instance for instance in running_instances}
        instances_to_delete = [
            running_instances_by_name[name]
            for name in instances
            if name in running_instances_by_name
        ]

        not_running = sorted(set(instances) - set(running_instances_by_name))
        if not_running and set(instances) != set(launched_instances):
            not_running_str = "', '".join(not_running)
            warnings.warn(
                f"Some provided instance names are not currently running: '{not_running_str}'."
            )

        deleted_instances = []
        for instance in instances_to_delete:
            if verbose:
                LOGGER.info(
                    f'Deleting GCP instance {instance["name"]!r} '
                    f'(id={instance["id"]}, zone={instance["zone"]})...'
                )

            operation = client.delete(
                project=project_id,
                zone=instance['zone'],
                instance=instance['name'],
            )
            operation.result()
            deleted_instances.append(instance['name'])

        if verbose:
            LOGGER.info(f'Terminated {len(deleted_instances)} GCP instance(s).')

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
