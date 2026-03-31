"""Define the BenchmarkLauncher class, which launches and manages benchmark executions."""

import io
import logging
import warnings

import cloudpickle
import pandas as pd

from sdgym._benchmark_launcher._instance_manager import GCPInstanceManager
from sdgym._benchmark_launcher._storage_manager import S3StorageManager
from sdgym._benchmark_launcher.utils import (
    _METHODS,
    _add_dataset_suffix,
    _build_job_artifact_keys,
    _build_job_output_destination,
    _get_top_folder_prefix,
    _resolve_datasets,
    generate_ids,
    resolve_compute,
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
        self._validate_compute_service()
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
        self._storage_manager = self._build_storage_manager()

    def _build_storage_manager(self):
        """Build the storage manager."""
        try:
            return S3StorageManager(
                credentials_filepath=self.benchmark_config.credentials_filepath,
                instance_jobs=self.benchmark_config.instance_jobs,
            )

        except ValueError as e:
            raise NotImplementedError(
                f'Failed to initialize storage manager. Only S3 storage is currently supported. '
                f'Error details: {e}'
            ) from e

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
                artifact_synthesizer = self._add_synthesizer_suffix(synthesizer, instance_idx)
                jobs.append({
                    'dataset': dataset,
                    'synthesizer': synthesizer,
                    'artifact_dataset': artifact_dataset,
                    'artifact_synthesizer': artifact_synthesizer,
                    'artifact_key_prefix': artifact_key_prefix,
                    'output_destination': output_destination,
                    'job_output_destination': _build_job_output_destination(
                        output_destination=output_destination,
                        artifact_key_prefix=artifact_key_prefix,
                        artifact_dataset=artifact_dataset,
                        artifact_synthesizer=artifact_synthesizer,
                    ),
                })

        return jobs

    def _launch(self):
        launch_id = generate_ids(['LAUNCH_ID'])
        self._launch_to_instance_names[launch_id] = []
        credentials = resolve_credentials(self.benchmark_config.credentials_filepath)
        compute = resolve_compute(self.benchmark_config.compute)

        for instance_idx, instance_job in enumerate(self.benchmark_config.instance_jobs):
            datasets = _resolve_datasets(instance_job['datasets'])
            synthesizers = instance_job['synthesizers']
            output_destination = instance_job['output_destination']

            instance_name = self.method_to_run(
                output_destination=output_destination,
                synthesizers=synthesizers,
                sdv_datasets=datasets,
                credentials=credentials,
                compute_config=compute,
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
        if self.compute_service != 'gcp':
            raise NotImplementedError(
                f"Compute service '{self.compute_service}' is not supported. "
                "Supported services: 'gcp'."
            )

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
            synthesizer_with_suffix = job['artifact_synthesizer']
            if dataset_names is not None and dataset not in dataset_names:
                continue
            if synthesizer_names is not None and synthesizer not in synthesizer_names:
                continue

            artifact_status = self._get_job_artifact_status(
                artifact_dataset=job['artifact_dataset'],
                artifact_synthesizer=synthesizer_with_suffix,
                artifact_key_prefix=job['artifact_key_prefix'],
                existing_keys=existing_keys_by_output[job['output_destination']],
            )
            rows.append({
                'Dataset': dataset,
                'Synthesizer': synthesizer_with_suffix,
                'Instance_Name': instance_name,
                'Output_Destination': job['job_output_destination'],
                'Status': artifact_status,
            })

        return rows

    def _update_status_running_job(self, instance_rows, instance_status):
        """Determine the running job for the instance."""
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
                - Output_Destination: The output destination for the job artifacts.
                - Status: The status of the job.
        """
        instances = self._validate_instance_names(instance_names)
        self._update_instance_statuses()
        existing_keys_by_output = {
            output_destination: self._storage_manager.get_existing_filenames(output_destination)
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
            rows.extend(self._update_status_running_job(instance_rows, instance_status))

        return pd.DataFrame(rows)

    def _finalize_instance_results(self, instance_name, existing_keys_by_output):
        jobs = self._instance_name_to_jobs.get(instance_name, [])
        if not jobs:
            return pd.DataFrame()

        output_destination = jobs[0]['output_destination']
        existing_keys = existing_keys_by_output[output_destination]
        result_filename = self._instance_name_to_result_filename[instance_name]
        artifact_key_prefix = jobs[0]['artifact_key_prefix']
        aggregate_result_key = f'{artifact_key_prefix}/{result_filename}'
        instance_status = self._instance_name_to_status.get(instance_name)

        # If instance completed and aggregate file already exists, just load it
        if instance_status == 'completed' and aggregate_result_key in existing_keys:
            s3_client, bucket_name = self._get_s3_resources(output_destination)
            response = s3_client.get_object(Bucket=bucket_name, Key=aggregate_result_key)
            result_df = pd.read_csv(io.BytesIO(response['Body'].read()))
            self._update_result_columns(result_df)
            return result_df

        # Otherwise rebuild from individual job benchmark_result.csv files
        s3_client, bucket_name = self._get_s3_resources(output_destination)
        frames = []
        for job in jobs:
            if job['benchmark_result_key'] in existing_keys:
                job_result = self._load_job_result(job, s3_client, bucket_name)
                self._update_result_columns(job_result)
                frames.append(job_result)
            else:
                frames.append(self._build_missing_result_row(job))

        return self._align_result_frames(frames)

    def _finalize(self):
        self._validate_compute_service()
        self._update_instance_statuses()
        output_destinations = self._get_all_output_destinations()
        # Look for instances with status completed and check they have their corresping result(i).csv

        # Then for the remaining one (Stopped or still Running)
        # Check if they have a result(i).csv but they should not
        # Build the result(i).csv by looking at the file like
        # s3://sdgym-benchmark/Benchmarks/single_table/SDGym_results_03_01_2026/intrusion_03_01_2026/BootstrapSynthesizer(4)/BootstrapSynthesizer(4)_benchmark_result.csv
        # If the file exist concat, if not as a row for the job with 'Instance Error' in the Error columns, the dataset and synthesizer name and the other columns as empty
        # Upload the missing result(i).csv files
        # Terminate all the remaining running instances .terminate()
        existing_keys_by_output = {
            output_destination: self._get_s3_existing_keys(output_destination)
            for output_destination in output_destinations
        }
        for instance_name in self._get_all_instance_names():
            instance_results = self._finalize_instance_results(
                instance_name=instance_name,
                existing_keys_by_output=existing_keys_by_output,
            )
            self._upload_instance_results(instance_name, instance_results)
            self._update_instance_metainfo(instance_name)

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
