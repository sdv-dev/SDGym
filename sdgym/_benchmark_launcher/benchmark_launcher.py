"""Define the BenchmarkLauncher class, which launches and manages benchmark executions."""

import logging
import warnings

import cloudpickle
import pandas as pd

from sdgym._benchmark_launcher._instance_manager import GCPInstanceManager
from sdgym._benchmark_launcher.utils import (
    _METHODS,
    _resolve_datasets,
    generate_ids,
    resolve_credentials,
)
from sdgym.s3 import _list_s3_bucket_contents, get_s3_client, is_s3_path, parse_s3_path

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
        self._instance_name_to_jobs = {}
        self._instance_manager = self._build_instance_manager()

    def _build_instance_manager(self):
        """Build the instance manager for the configured compute service."""
        if self.compute_service == 'gcp':
            return GCPInstanceManager(self.benchmark_config.credentials_filepath)

        raise NotImplementedError(f'Compute service {self.compute_service!r} is not supported.')

    def _add_synthesizer_suffix(self, synthesizer, suffix):
        """Return the synthesizer name with the instance suffix."""
        if suffix == 0:
            return synthesizer

        return f'{synthesizer}({suffix})'

    def _add_synthesizer_suffix(self, synthesizer, suffix):
        """Return the synthesizer name with the instance suffix."""
        if suffix == 0:
            return synthesizer

        return f'{synthesizer}({suffix})'

    def _launch(self):
        launch_id = generate_ids(['LAUNCH_ID'])
        self._launch_to_instance_names[launch_id] = []
        credentials = resolve_credentials(self.benchmark_config.credentials_filepath)

        for instance_idx, instance_job in enumerate(self.benchmark_config.instance_jobs):
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
            jobs = []
            for dataset in sdv_datasets:
                for synthesizer in instance_job['synthesizers']:
                    jobs.append({
                        'dataset': dataset,
                        'synthesizer': synthesizer,
                        'artifact_synthesizer': self._add_synthesizer_suffix(
                            synthesizer, instance_idx
                        ),
                        'output_destination': instance_job['output_destination'],
                    })

            self._instance_name_to_jobs[instance_name] = jobs

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

    def _validate_compute_service(self):
        """Validate that the compute service is supported."""
        if self.compute_service not in ('gcp',):
            raise NotImplementedError(
                f"Compute service '{self.compute_service}' is not supported. "
                "Supported services: 'gcp'."
            )

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
        self._validate_compute_service()
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

    def get_instance_status(self, instance_names=None):
        """Get the status of benchmark instances.

        Args:
            instance_names (list of str, optional):
                List of instance names to get status for.
                If None, gets status for all launched instances. Defaults to None.

        Returns:
            pd.DataFrame:
                A dataframe with one row per instance and columns:
                - Instance Name: The instance name tracked by the launcher.
                - Status: The status of the instance.
        """
        self._validate_compute_service()
        instances = self._validate_instance_names(instance_names)
        self._update_instance_name_to_status()
        rows = []
        for instance_name in instances:
            internal_status = self._instance_name_to_status.get(instance_name)
            rows.append({'Instance Name': instance_name, 'Status': internal_status.capitalize()})

        return pd.DataFrame(rows)

    def _get_all_output_destinations(self, instance_names=None):
        """Return all unique output destinations for the selected instances."""
        instances = self._validate_instance_names(instance_names)
        output_destinations = []
        for instance_name in instances:
            jobs = self._instance_name_to_jobs.get(instance_name, [])
            for job in jobs:
                output_destination = job['output_destination']
                if output_destination not in output_destinations:
                    output_destinations.append(output_destination)

        return output_destinations

    def _get_s3_existing_keys(self, output_destination):
        """Return the existing S3 keys under the output destination."""
        if not is_s3_path(output_destination):
            raise ValueError(
                f'`output_destination` must be an S3 path. Found: {output_destination!r}.'
            )

        credentials = resolve_credentials(self.benchmark_config.credentials_filepath)
        aws_credentials = credentials.get('aws', {})
        s3_client = get_s3_client(
            aws_access_key_id=aws_credentials.get('aws_access_key_id'),
            aws_secret_access_key=aws_credentials.get('aws_secret_access_key'),
        )
        bucket_name, key_prefix = parse_s3_path(output_destination)
        contents = _list_s3_bucket_contents(s3_client, bucket_name, key_prefix)
        existing_keys = {obj['Key'] for obj in contents}

        return existing_keys, key_prefix

    def _get_job_artifact_status(self, dataset, synthesizer, key_prefix, existing_keys):
        """Get the artifact-based status for a benchmark job."""
        job_prefix = f'{key_prefix.rstrip("/")}/{dataset}/{synthesizer}'
        benchmark_results_key = f'{job_prefix}/{synthesizer}_benchmark_results.csv'
        synthetic_data_key = f'{job_prefix}/{synthesizer}_synthetic_data.csv'
        synthesizer_key = f'{job_prefix}/{synthesizer}.pkl'

        has_benchmark_results = benchmark_results_key in existing_keys
        has_synthetic_data = synthetic_data_key in existing_keys
        has_synthesizer = synthesizer_key in existing_keys

        if has_benchmark_results and has_synthetic_data and has_synthesizer:
            return 'Completed'

        if has_benchmark_results and not has_synthetic_data and not has_synthesizer:
            return 'Failed'

        return 'Queued'

    def get_job_status(self, dataset_names=None, synthesizer_names=None, instance_names=None):
        """Get status of benchmark instance jobs.

        Args:
            dataset_names (list of str, optional):
                List of dataset names to get status for.
                If None, gets status for all datasets. Defaults to None.
            synthesizer_names (list of str, optional):
                List of synthesizer names to get status for.
                If None, gets status for all synthesizers. Defaults to None.
            instance_names (list of str, optional):
                List of instance names to get status for.
                If None, gets status for all instance jobs. Defaults to None.

        Returns:
            pd.DataFrame:
                A dataframe with one row per job and columns:
                - Dataset: The dataset used in the job.
                - Synthesizer: The synthesizer used in the job.
                - Instance_Name: The name of the instance running the job.
                - Status: The status of the job.
        """
        instances = self._validate_inputs_and_get_instances(instance_names, verbose=False)
        self._update_instance_name_to_status()
        active_instances = set(self._get_active_instance_names())
        output_destination_cache = {}
        for output_destination in self._get_all_output_destinations(instances):
            output_destination_cache[output_destination] = self._get_s3_existing_keys(
                output_destination
            )

        rows = []
        for instance_name in instances:
            jobs = self._instance_name_to_jobs.get(instance_name, [])
            instance_rows = []
            for job in jobs:
                dataset = job['dataset']
                synthesizer = job['synthesizer']
                artifact_synthesizer = job['artifact_synthesizer']
                output_destination = job['output_destination']

                if dataset_names is not None and dataset not in dataset_names:
                    continue

                if synthesizer_names is not None and synthesizer not in synthesizer_names:
                    continue

                existing_keys, key_prefix = output_destination_cache[output_destination]
                status = self._get_job_artifact_status(
                    dataset=dataset,
                    synthesizer=artifact_synthesizer,
                    key_prefix=key_prefix,
                    existing_keys=existing_keys,
                )
                instance_rows.append({
                    'Dataset': dataset,
                    'Synthesizer': synthesizer,
                    'Instance_Name': instance_name,
                    'Status': status,
                })

            if instance_name in active_instances:
                queued_indexes = [
                    idx for idx, row in enumerate(instance_rows) if row['Status'] == 'Queued'
                ]
                if queued_indexes:
                    instance_rows[queued_indexes[0]]['Status'] = 'Running'

            rows.extend(instance_rows)

        return pd.DataFrame(rows)

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
