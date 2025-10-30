"""SDGym Results Explorer for accessing and managing benchmark results."""

import os

from sdgym.benchmark import DEFAULT_DATASETS
from sdgym.datasets import load_dataset
from sdgym.result_explorer.result_handler import LocalResultsHandler, S3ResultsHandler
from sdgym.s3 import _get_s3_client, is_s3_path


def _validate_local_path(path):
    """Validates that the local path exists and is a directory."""
    if not os.path.isdir(path):
        raise ValueError(f"The provided path '{path}' is not a valid local directory.")


class ResultsExplorer:
    """Explorer for SDGym benchmark results, supporting both local and S3 storage."""

    def __init__(self, path, aws_access_key_id=None, aws_secret_access_key=None):
        self.path = path
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key

        if is_s3_path(path):
            s3_client = _get_s3_client(path, aws_access_key_id, aws_secret_access_key)
            self._handler = S3ResultsHandler(path, s3_client)
        else:
            _validate_local_path(path)
            self._handler = LocalResultsHandler(path)

    def list(self):
        """List all runs available in the results directory."""
        return self._handler.list()

    def _get_file_path(self, results_folder_name, dataset_name, synthesizer_name, file_type):
        """Validate access to the synthesizer or synthetic data file."""
        end_filename = f'{synthesizer_name}'
        if file_type == 'synthetic_data':
            end_filename += '_synthetic_data.csv'
        elif file_type == 'synthesizer':
            end_filename += '.pkl'

        date = '_'.join(results_folder_name.split('_')[-3:])
        path_parts = [results_folder_name, f'{dataset_name}_{date}', synthesizer_name]

        return self._handler.get_file_path(path_parts, end_filename)

    def load_synthesizer(self, results_folder_name, dataset_name, synthesizer_name):
        """Load the synthesizer for a given dataset and synthesizer."""
        file_path = self._get_file_path(
            results_folder_name, dataset_name, synthesizer_name, 'synthesizer'
        )
        return self._handler.load_synthesizer(file_path)

    def load_synthetic_data(self, results_folder_name, dataset_name, synthesizer_name):
        """Load the synthetic data for a given dataset and synthesizer."""
        file_path = self._get_file_path(
            results_folder_name, dataset_name, synthesizer_name, 'synthetic_data'
        )
        return self._handler.load_synthetic_data(file_path)

    def load_real_data(self, dataset_name):
        """Load the real data for a given dataset."""
        if dataset_name not in DEFAULT_DATASETS:
            raise ValueError(
                f"Dataset '{dataset_name}' is not a SDGym dataset. "
                'Please provide a valid dataset name.'
            )

        data, _ = load_dataset(
            modality='single_table',
            dataset=dataset_name,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
        )
        return data

    def summarize(self, folder_name):
        """Summarize the results in the specified folder.

        Args:
            folder_name (str):
                The name of the results folder to summarize.

        Returns:
            tuple (pd.DataFrame, pd.DataFrame):
                - A summary DataFrame with the number of Wins per synthesizer.
                - A DataFrame with the results of the benchmark for the specified folder.
        """
        return self._handler.summarize(folder_name)

    def all_runs_complete(self, folder_name):
        """Check if all runs in the specified folder are complete."""
        return self._handler.all_runs_complete(folder_name)

    def load_results(self, results_folder_name):
        """Load and aggregate all the results CSV files from the specified results folder.

        Args:
            results_folder_name (str):
                The name of the results folder to load results from.

        Returns:
            pd.DataFrame:
                A DataFrame containing the results of the specified folder.
        """
        return self._handler.load_results(results_folder_name)

    def load_metainfo(self, results_folder_name):
        """Load and aggregate all the metainfo YAML files from the specified results folder.

        Args:
            results_folder_name (str):
                The name of the results folder to load metainfo from.

        Returns:
            dict:
                A dictionary containing the metainfo of the specified folder.
        """
        return self._handler.load_metainfo(results_folder_name)
