"""SDGym Results Explorer for accessing and managing benchmark results."""

import operator
import os
from datetime import datetime

from sdgym.datasets import _load_dataset_with_client
from sdgym.result_explorer.result_handler import (
    RESULTS_FOLDER_PREFIX,
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

    def _get_latest_run(self):
        """Get the folder name of the latest SDGym run."""
        candidates = []
        for name in self._handler.list():
            name = name.rstrip('/')
            if not name.startswith(RESULTS_FOLDER_PREFIX):
                continue

            date_str = name[len(RESULTS_FOLDER_PREFIX) :]
            try:
                date_obj = datetime.strptime(date_str, '%m_%d_%Y')
            except ValueError:
                continue

            candidates.append((date_obj, name))
        if not candidates:
            raise ValueError(
                f'No run folders found. Expected folders like '
                f"'{RESULTS_FOLDER_PREFIX}MM_DD_YYYY' under: {self.path}/{self.modality}"
            )

        return max(candidates, key=operator.itemgetter(0))[1]

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

    def load_synthesizer(self, dataset_name, synthesizer_name, results_folder_name=None):
        """Load the synthesizer for a given dataset and synthesizer.

        Args:
            dataset_name (str):
                The name of the dataset.
            synthesizer_name (str):
                The name of the synthesizer.
            results_folder_name (str, optional):
                The name of the results folder to load from. If not provided,
                the latest run will be used. Defaults to None.
        """
        results_folder_name = results_folder_name or self._get_latest_run()
        file_path = self._get_file_path(
            results_folder_name, dataset_name, synthesizer_name, 'synthesizer'
        )
        return self._handler.load_synthesizer(file_path)

    def load_synthetic_data(self, dataset_name, synthesizer_name, results_folder_name=None):
        """Load the synthetic data for a given dataset and synthesizer.

        Args:
            dataset_name (str):
                The name of the dataset.
            synthesizer_name (str):
                The name of the synthesizer.
            results_folder_name (str, optional):
                The name of the results folder to load from. If not provided,
                the latest run will be used. Defaults to None.
        """
        results_folder_name = results_folder_name or self._get_latest_run()
        file_path = self._get_file_path(
            results_folder_name, dataset_name, synthesizer_name, 'synthetic_data'
        )
        return self._handler.load_synthetic_data(file_path)

    def load_real_data(self, dataset_name):
        """Load the real data for a given dataset.

        Args:
            dataset_name (str):
                The name of the dataset.

        Returns:
            pd.DataFrame:
                A DataFrame containing the real data for the specified dataset.
        """
        data, _ = _load_dataset_with_client(
            modality=self.modality,
            dataset=dataset_name,
            s3_client=self.s3_client,
        )
        return data

    def summarize(self, results_folder_name=None):
        """Summarize the results in the specified folder.

        Args:
            results_folder_name (str, optional):
                The name of the results folder to summarize. If not provided,
                the latest run will be used. Defaults to None.

        Returns:
            tuple (pd.DataFrame, pd.DataFrame):
                - A summary DataFrame with the number of Wins per synthesizer.
                - A DataFrame with the results of the benchmark for the specified folder.
        """
        results_folder_name = results_folder_name or self._get_latest_run()
        return self._handler.summarize(results_folder_name)

    def all_runs_complete(self, results_folder_name=None):
        """Check if all runs in the specified folder are complete."""
        results_folder_name = results_folder_name or self._get_latest_run()
        return self._handler.all_runs_complete(results_folder_name)

    def load_results(self, results_folder_name=None):
        """Load and aggregate all the results CSV files from the specified results folder.

        Args:
            results_folder_name (str, optional):
                The name of the results folder to load results from. If not provided,
                the latest run will be used. Defaults to None.

        Returns:
            pd.DataFrame:
                A DataFrame containing the results of the specified folder.
        """
        results_folder_name = results_folder_name or self._get_latest_run()
        return self._handler.load_results(results_folder_name)

    def load_metainfo(self, results_folder_name=None):
        """Load and aggregate all the metainfo YAML files from the specified results folder.

        Args:
            results_folder_name (str, optional):
                The name of the results folder to load metainfo from. If not provided,
                the latest run will be used. Defaults to None.

        Returns:
            dict:
                A dictionary containing the metainfo of the specified folder.
        """
        results_folder_name = results_folder_name or self._get_latest_run()
        return self._handler.load_metainfo(results_folder_name)
