"""Define the BenchmarkLauncher class, which launches and manages benchmark executions."""

import cloudpickle

from sdgym._benchmark_launcher.utils import (
    _METHODS,
    _resolve_datasets,
    generate_benchmark_id,
    resolve_credentials,
)


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
        self.benchmark_id = generate_benchmark_id(self)

    def _launch(self):
        credentials = resolve_credentials(self.benchmark_config.credentials_filepath)
        for instance_job in self.benchmark_config.instance_jobs:
            sdv_datasets = _resolve_datasets(instance_job['datasets'])
            self.method_to_run(
                synthesizers=instance_job['synthesizers'],
                sdv_datasets=sdv_datasets,
                credentials=credentials,
                compute_config=self.benchmark_config.compute,
                **self.benchmark_config.method_params,
            )

    def launch(self):
        """Run the BenchmarkConfig: validate it and then execute the specified benchmark method."""
        if not self.benchmark_config._is_validated:
            self.benchmark_config.validate()

        self._launch()

    def terminate(self, instance_ids=None):
        """Terminate running benchmark instance jobs.

        Args:
            instance_ids (list of str, optional):
                List of instance IDs to terminate.
                If None, terminates all instance jobs. Default to None.
        """
        raise NotImplementedError

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

        if getattr(benchmark, '_synthesizer_id', None) is None:
            benchmark.benchmark_id = generate_benchmark_id(benchmark)

        return benchmark
