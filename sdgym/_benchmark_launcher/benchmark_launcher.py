"""Define the BenchmarkLauncher class, which launches and manages benchmark executions."""

import logging
import warnings

import cloudpickle
import pandas as pd

from sdgym._benchmark_launcher._instance_manager import GCPInstanceManager
from sdgym._benchmark_launcher.utils import (
    _METHODS,
    _add_dataset_suffix,
    _build_job_artifact_keys,
    _get_top_folder_prefix,
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
        return synthesizer if suffix == 0 else f'{synthesizer}({suffix})'

    def _build_instance_jobs(self, datasets, synthesizers, output_destination, instance_idx):
        """Build the job metadata for a launched instance."""
        artifact_key_prefix = _get_top_folder_prefix(output_destination, self.modality)
        jobs = []
        for dataset in datasets:
            artifact_dataset = _add_dataset_suffix(dataset)
            for synthesizer in synthesizers:
                jobs.append({
                    'dataset': dataset,
                    'synthesizer': synthesizer,
                    'artifact_dataset': artifact_dataset,
                    'artifact_synthesizer': self._add_synthesizer_suffix(synthesizer, instance_idx),
                    'artifact_key_prefix': artifact_key_prefix,
                    'output_destination': output_destination,
                })

        return jobs

    def _launch(self):
        """Launch the configured benchmark jobs."""
        launch_id = generate_ids(['LAUNCH_ID'])
        self._launch_to_instance_names[launch_id] = []
        credentials = resolve_credentials(self.benchmark_config.credentials_filepath)

        for instance_idx, instance_job in enumerate(self.benchmark_config.instance_jobs):
            datasets = _resolve_datasets(instance_job['datasets'])
            synthesizers = instance_job['synthesizers']
            output_destination = instance_job['output_destination']

            instance_name = self.method_to_run(
                output_destination=output_destination,
                synthesizers=synthesizers,
                sdv_datasets=datasets,
                credentials=credentials,
                compute_config=self.benchmark_config.compute,
                **self.benchmark_config.method_params,
            )

            self._launch_to_instance_names[launch_id].append(instance_name)
            self._instance_name_to_status[instance_name] = 'running'
            self._instance_name_to_jobs[instance_name] = self._build_instance_jobs(
                datasets=datasets,
                synthesizers=synthesizers,
                output_destination=output_destination,
                instance_idx=instance_idx,
            )

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
        return [
            instance_name
            for names in self._launch_to_instance_names.values()
            for instance_name in names
        ]

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
        instances = launched_instances if instance_names is None else instance_names
        unknown_instances = sorted(set(instances) - set(launched_instances))
        if unknown_instances:
            unknown_instances_str = "', '".join(unknown_instances)
            launched_instances_str = "', '".join(sorted(launched_instances))
            raise ValueError(
                'Some provided instance names were not launched by this '
                f"BenchmarkLauncher. Unknown: '{unknown_instances_str}'. "
                f"Launched instances: '{launched_instances_str}'."
            )

        return instances

    def _validate_compute_service(self):
        """Validate that the compute service is supported."""
        if self.compute_service != 'gcp':
            raise NotImplementedError(
                f"Compute service '{self.compute_service}' is not supported. "
                "Supported services: 'gcp'."
            )

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

        already_terminated = sorted(set(instances) - active_instances)
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
        self._update_instance_statuses()

        rows = [
            {
                'Instance Name': instance_name,
                'Status': self._instance_name_to_status[instance_name].capitalize(),
            }
            for instance_name in instances
        ]
        return pd.DataFrame(rows)

    def _get_all_output_destinations(self, instance_names=None):
        """Return all unique output destinations for the selected instances."""
        instances = self._validate_instance_names(instance_names)
        output_destinations = []
        for instance_name in instances:
            for job in self._instance_name_to_jobs.get(instance_name, []):
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

        return {obj['Key'] for obj in contents}

    def _get_job_artifact_status(
        self, artifact_dataset, artifact_synthesizer, artifact_key_prefix, existing_keys
    ):
        """Get the artifact-based status for a benchmark job."""
        benchmark_result_key, synthetic_data_key, synthesizer_key = _build_job_artifact_keys(
            artifact_key_prefix=artifact_key_prefix,
            artifact_dataset=artifact_dataset,
            artifact_synthesizer=artifact_synthesizer,
            modality=self.modality,
        )

        has_benchmark_result = benchmark_result_key in existing_keys
        has_synthetic_data = synthetic_data_key in existing_keys
        has_synthesizer = synthesizer_key in existing_keys

        if has_benchmark_result and has_synthetic_data and has_synthesizer:
            return 'Completed'

        if has_benchmark_result and not has_synthetic_data and not has_synthesizer:
            return 'Failed'

        return 'Queued'

    def _get_instance_job_rows(
        self,
        instance_name,
        jobs,
        dataset_names,
        synthesizer_names,
        existing_keys_by_output,
    ):
        """Build the job-status rows for one instance."""
        rows = []
        for job in jobs:
            dataset = job['dataset']
            synthesizer = job['synthesizer']
            if dataset_names is not None and dataset not in dataset_names:
                continue
            if synthesizer_names is not None and synthesizer not in synthesizer_names:
                continue

            artifact_status = self._get_job_artifact_status(
                artifact_dataset=job['artifact_dataset'],
                artifact_synthesizer=job['artifact_synthesizer'],
                artifact_key_prefix=job['artifact_key_prefix'],
                existing_keys=existing_keys_by_output[job['output_destination']],
            )
            rows.append({
                'Dataset': dataset,
                'Synthesizer': synthesizer,
                'Instance_Name': instance_name,
                'Status': artifact_status,
            })

        return rows

    def _finalize_instance_job_rows(self, instance_rows, instance_status):
        """Adjust queued job statuses based on the instance status."""
        if instance_status == 'running':
            queued_indexes = [
                idx for idx, row in enumerate(instance_rows) if row['Status'] == 'Queued'
            ]
            if queued_indexes:
                instance_rows[queued_indexes[0]]['Status'] = 'Running'
            return instance_rows

        for row in instance_rows:
            if row['Status'] == 'Queued':
                row['Status'] = 'Failed'

        return instance_rows

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
        self._validate_compute_service()
        instances = self._validate_instance_names(instance_names)
        self._update_instance_statuses()

        existing_keys_by_output = {
            output_destination: self._get_s3_existing_keys(output_destination)
            for output_destination in self._get_all_output_destinations(instances)
        }

        rows = []
        for instance_name in instances:
            jobs = self._instance_name_to_jobs.get(instance_name, [])
            instance_rows = self._get_instance_job_rows(
                instance_name=instance_name,
                jobs=jobs,
                dataset_names=dataset_names,
                synthesizer_names=synthesizer_names,
                existing_keys_by_output=existing_keys_by_output,
            )
            instance_status = self._instance_name_to_status.get(instance_name)
            rows.extend(self._finalize_instance_job_rows(instance_rows, instance_status))

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
