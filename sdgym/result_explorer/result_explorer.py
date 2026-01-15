"""SDGym Results Explorer for accessing and managing benchmark results."""

import os

from sdgym.datasets import _load_dataset_with_client
from sdgym.result_explorer.result_handler import (
    SYNTHESIZER_BASELINE,
    LocalResultsHandler,
    S3ResultsHandler,
)
from sdgym.s3 import _get_s3_client, is_s3_path
from sdgym.synthesizers.base import _validate_modality


def _validate_local_path(path):
    """Validates that the local path exists and is a directory."""
    if not os.path.isdir(path):
        raise ValueError(f"The provided path '{path}' is not a valid local directory.")


_BASELINE_BY_MODALITY = {
    'single_table': SYNTHESIZER_BASELINE,
    'multi_table': 'IndependentSynthesizer',
}


def _resolve_effective_path(path, modality):
    """Append the modality folder to the given base path if provided."""
    # Avoid double-appending if already included
    if str(path).rstrip('/').endswith(('/' + modality, modality)):
        return path

    if is_s3_path(path):
        return path.rstrip('/') + '/' + modality

    return os.path.join(path, modality)


class ResultsExplorer:
    """Explorer for SDGym benchmark results, supporting both local and S3 storage."""

    def _create_results_handler(self, original_path, effective_path):
        """Create the appropriate results handler for local or S3 storage."""
        baseline_synthesizer = _BASELINE_BY_MODALITY.get(self.modality, SYNTHESIZER_BASELINE)
        if is_s3_path(original_path) and self.s3_client is None:
            self.s3_client = _get_s3_client(
                original_path, self.aws_access_key_id, self.aws_secret_access_key
            )
            return S3ResultsHandler(
                effective_path, self.s3_client, baseline_synthesizer=baseline_synthesizer
            )

        _validate_local_path(effective_path)
        return LocalResultsHandler(effective_path, baseline_synthesizer=baseline_synthesizer)

    def __init__(
        self, path, modality='single_table', aws_access_key_id=None, aws_secret_access_key=None
    ):
        self.path = path
        _validate_modality(modality)
        self.modality = modality.lower()
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.s3_client = None
        effective_path = _resolve_effective_path(path, self.modality)
        self._handler = self._create_results_handler(path, effective_path)

    def list(self):
        """List all runs available in the results directory."""
        return self._handler.list()

    def _get_file_path(self, results_folder_name, dataset_name, synthesizer_name, file_type):
        """Validate access to the synthesizer or synthetic data file."""
        end_filename = f'{synthesizer_name}'
        if file_type == 'synthetic_data':
            # Multi-table synthetic data is zipped (multiple CSVs), single table is CSV
            if self.modality == 'multi_table':
                end_filename += '_synthetic_data.zip'
            else:
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
        data, _ = _load_dataset_with_client(
            modality=self.modality,
            dataset=dataset_name,
            s3_client=self.s3_client,
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
