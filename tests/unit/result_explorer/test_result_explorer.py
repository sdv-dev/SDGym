import os
import re
import shutil
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

from sdgym.result_explorer.result_explorer import ResultsExplorer, _validate_local_path
from sdgym.result_explorer.result_handler import LocalResultsHandler, S3ResultsHandler


def test_validate_local_path(tmp_path):
    """Test the `_validate_local_path` function for a local path."""
    # Setup
    path = tmp_path / 'local_results_folder'
    path.mkdir()
    expected_error = re.escape("The provided path 'invalid_path' is not a valid local directory.")

    # Run
    s3_client = _validate_local_path(str(path))
    with pytest.raises(ValueError, match=expected_error):
        _validate_local_path('invalid_path')

    # Assert
    assert s3_client is None


class TestResultsExplorer:
    @patch('sdgym.result_explorer.result_explorer.is_s3_path')
    @patch('sdgym.result_explorer.result_explorer._validate_local_path')
    def test__init__local(self, mock_validate_local_path, mock_is_s3_path):
        """Test the ``__init__`` for accessing local folder."""
        # Setup
        mock_is_s3_path.return_value = False
        path = 'local_results_folder'

        # Run
        result_explorer = ResultsExplorer(path, modality='single_table')

        # Assert
        expected_path = os.path.join(path, 'single_table')
        mock_validate_local_path.assert_called_once_with(expected_path)
        mock_is_s3_path.assert_called_with(path)
        assert isinstance(result_explorer._handler, LocalResultsHandler)
        assert result_explorer.path == path
        assert result_explorer.aws_access_key_id is None
        assert result_explorer.aws_secret_access_key is None

    @patch('sdgym.result_explorer.result_explorer._get_s3_client')
    @patch('sdgym.result_explorer.result_explorer.is_s3_path')
    def test__init__s3(self, mock_is_s3_path, mock_get_s3_client):
        """Test the ``__init__`` for accessing S3 bucket."""
        # Setup
        path = 's3://my-bucket/results'
        aws_access_key_id = 'my_access_key'
        aws_secret_access_key = 'my_secret_key'
        mock_is_s3_path.return_value = True
        s3_client = Mock()
        mock_get_s3_client.return_value = s3_client

        # Run
        result_explorer = ResultsExplorer(
            path,
            modality='single_table',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

        # Assert
        mock_is_s3_path.assert_called_with(path)
        mock_get_s3_client.assert_called_once_with(path, aws_access_key_id, aws_secret_access_key)
        assert result_explorer.path == path
        assert result_explorer.aws_access_key_id == aws_access_key_id
        assert result_explorer.aws_secret_access_key == aws_secret_access_key
        assert isinstance(result_explorer._handler, S3ResultsHandler)

    def test_list_with_modality_local(self, tmp_path):
        """Test the `list` method respects the modality subfolder (local)."""
        # Setup
        base = tmp_path / 'results'
        (base / 'unscoped_run').mkdir(parents=True)
        (base / 'multi_table' / 'run_mt1').mkdir(parents=True)
        (base / 'multi_table' / 'run_mt2').mkdir(parents=True)

        result_explorer = ResultsExplorer(str(base), modality='multi_table')

        # Run
        runs = result_explorer.list()

        # Assert
        assert set(runs) == {'run_mt1', 'run_mt2'}

    def test_list_local(self, tmp_path):
        """Test the `list` method with a local path"""
        # Setup
        path = tmp_path / 'results' / 'single_table'
        path.mkdir(parents=True)
        (path / 'run1').mkdir()
        (path / 'run2').mkdir()
        result_explorer = ResultsExplorer(str(path), modality='single_table')

        # Run
        runs = result_explorer.list()

        # Assert
        assert set(runs) == {'run1', 'run2'}

    def test_list(self, tmp_path):
        """Test the `list` method with an S3 path"""
        # Setup
        path = tmp_path / 'results' / 'single_table'
        path.mkdir(parents=True)
        (path / 'run1').mkdir()
        (path / 'run2').mkdir()
        result_explorer = ResultsExplorer(str(path), modality='single_table')
        result_explorer._handler = Mock()
        result_explorer._handler.list.return_value = ['run1', 'run2']

        # Run
        runs = result_explorer.list()

        # Assert
        assert runs == ['run1', 'run2']
        result_explorer._handler.list.assert_called_once()

    def test__get_file_path(self):
        """Test `_get_file_path` for local and S3 paths."""
        # Setup
        explorer = Mock()
        explorer._handler = Mock()
        results_folder_name = 'results_folder_07_07_2025'
        dataset_name = 'my_dataset'
        synthesizer_name = 'my_synthesizer'
        type = 'synthesizer'
        expected_filepath = (
            f'{results_folder_name}/{dataset_name}_07_07_2025/{synthesizer_name}/'
            f'{synthesizer_name}.pkl'
        )
        explorer._handler.get_file_path.return_value = expected_filepath

        # Run
        file_path = ResultsExplorer._get_file_path(
            explorer, results_folder_name, dataset_name, synthesizer_name, type
        )

        # Assert
        explorer._handler.get_file_path.assert_called_once_with(
            [results_folder_name, f'{dataset_name}_07_07_2025', synthesizer_name],
            f'{synthesizer_name}.pkl',
        )
        assert file_path == expected_filepath

    def test__get_file_path_multi_table_synthetic_data(self, tmp_path):
        """Test `_get_file_path` returns .zip for multi_table synthetic data."""
        base = tmp_path / 'results'
        multi_table_dir = base / 'multi_table'
        multi_table_dir.mkdir(parents=True, exist_ok=True)
        explorer = ResultsExplorer(str(multi_table_dir), modality='multi_table')
        try:
            explorer._handler = Mock()
            explorer._handler.get_file_path.return_value = 'irrelevant'
            explorer._get_file_path(
                results_folder_name='results_folder_07_07_2025',
                dataset_name='my_dataset',
                synthesizer_name='my_synthesizer',
                file_type='synthetic_data',
            )
            explorer._handler.get_file_path.assert_called_once_with(
                ['results_folder_07_07_2025', 'my_dataset_07_07_2025', 'my_synthesizer'],
                'my_synthesizer_synthetic_data.zip',
            )
        finally:
            shutil.rmtree(multi_table_dir)

    def test_load_synthesizer(self, tmp_path):
        """Test `load_synthesizer` method."""
        # Setup
        path = tmp_path / 'single_table'
        path.mkdir(parents=True)
        explorer = ResultsExplorer(str(path), modality='single_table')
        explorer._handler = Mock()
        explorer._handler.load_synthesizer = Mock(
            return_value=GaussianCopulaSynthesizer(Metadata())
        )
        explorer._get_file_path = Mock(return_value='path/to/synthesizer.pkl')

        # Run
        synthesizer = explorer.load_synthesizer(
            results_folder_name='results_folder_07_07_2025',
            dataset_name='my_dataset',
            synthesizer_name='my_synthesizer',
        )

        # Assert
        explorer._get_file_path.assert_called_once_with(
            'results_folder_07_07_2025', 'my_dataset', 'my_synthesizer', 'synthesizer'
        )
        explorer._handler.load_synthesizer.assert_called_once_with('path/to/synthesizer.pkl')
        assert isinstance(synthesizer, GaussianCopulaSynthesizer)

    def test_load_synthetic_data(self, tmp_path):
        # Setup
        path = tmp_path / 'single_table'
        path.mkdir(parents=True)
        explorer = ResultsExplorer(str(path), modality='single_table')
        explorer._handler = Mock()
        data = pd.DataFrame({'column1': [1, 2], 'column2': [3, 4]})
        explorer._handler.load_synthetic_data = Mock(return_value=data)
        explorer._get_file_path = Mock(return_value='path/to/synthetic_data.csv')

        # Run
        synthetic_data = explorer.load_synthetic_data(
            results_folder_name='results_folder_07_07_2025',
            dataset_name='my_dataset',
            synthesizer_name='my_synthesizer',
        )

        # Assert
        explorer._get_file_path.assert_called_once_with(
            'results_folder_07_07_2025', 'my_dataset', 'my_synthesizer', 'synthetic_data'
        )
        explorer._handler.load_synthetic_data.assert_called_once_with('path/to/synthetic_data.csv')
        pd.testing.assert_frame_equal(synthetic_data, data)

    @patch('sdgym.result_explorer.result_explorer.load_dataset')
    def test_load_real_data(self, mock_load_dataset, tmp_path):
        """Test `load_real_data` method."""
        # Setup
        dataset_name = 'adult'
        expected_data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        mock_load_dataset.return_value = (expected_data, None)
        path = tmp_path / 'single_table'
        path.mkdir(parents=True)
        result_explorer = ResultsExplorer(str(path), modality='single_table')

        # Run
        real_data = result_explorer.load_real_data(dataset_name)

        # Assert
        mock_load_dataset.assert_called_once_with(
            modality='single_table', dataset='adult', s3_client=None
        )
        pd.testing.assert_frame_equal(real_data, expected_data)

    @patch('sdgym.result_explorer.result_explorer.load_dataset')
    def test_load_real_data_multi_table(self, mock_load_dataset, tmp_path):
        """Test `load_real_data` for multi_table modality calls load_dataset correctly."""
        dataset_name = 'synthea'
        expected_data = {'patients': pd.DataFrame({'id': [1]})}
        mock_load_dataset.return_value = (expected_data, None)
        multi_table_dir = tmp_path / 'multi_table'
        multi_table_dir.mkdir(parents=True, exist_ok=True)
        result_explorer = ResultsExplorer(tmp_path, modality='multi_table')

        try:
            # Run
            real_data = result_explorer.load_real_data(dataset_name)

            # Assert
            mock_load_dataset.assert_called_once_with(
                modality='multi_table', dataset='synthea', s3_client=None
            )
            assert real_data == expected_data
        finally:
            shutil.rmtree(multi_table_dir)

    def test_load_real_data_invalid_dataset(self, tmp_path):
        """Test `load_real_data` method with an invalid dataset."""
        # Setup
        dataset_name = 'invalid_dataset'
        path = tmp_path / 'single_table'
        path.mkdir(parents=True)
        result_explorer = ResultsExplorer(str(path), modality='single_table')
        expected_error_message = (
            "Dataset 'invalid_dataset' not found in bucket 's3://sdv-datasets-public' "
            "for modality 'single_table'."
        )

        # Run and Assert
        with pytest.raises(ValueError, match=expected_error_message):
            result_explorer.load_real_data(dataset_name)

    def test_summarize(self, tmp_path):
        """Test the `summarize` method."""
        # Setup
        output_dir = tmp_path / 'benchmark_output' / 'single_table'
        output_dir.mkdir(parents=True)
        (output_dir / 'SDGym_results_07_07_2025').mkdir(parents=True)
        result_explorer = ResultsExplorer(str(output_dir), modality='single_table')
        result_explorer._handler = Mock()
        results = pd.DataFrame({
            'Synthesizer': ['CTGANSynthesizer', 'CopulaGANSynthesizer', 'TVAESynthesizer'],
            '10_11_2024 - # datasets: 9 - sdgym version: 0.9.1': [6, 4, 5],
            '05_10_2024 - # datasets: 9 - sdgym version: 0.8.0': [4, 4, 5],
            '04_05_2024 - # datasets: 9 - sdgym version: 0.7.0': [5, 3, 5],
        })
        results = results.set_index('Synthesizer')
        result_explorer._handler.summarize = Mock(return_value=results)

        # Run
        summary = result_explorer.summarize('SDGym_results_07_07_2025')

        # Assert
        result_explorer._handler.summarize.assert_called_once_with('SDGym_results_07_07_2025')
        pd.testing.assert_frame_equal(summary, results)

    def test_load_results(self, tmp_path):
        """Test the `load_results` method."""
        # Setup
        output_dir = tmp_path / 'benchmark_output' / 'single_table'
        output_dir.mkdir(parents=True)
        (output_dir / 'SDGym_results_07_07_2025').mkdir(parents=True)
        result_explorer = ResultsExplorer(str(output_dir), modality='single_table')
        result_explorer._handler = Mock()
        results = pd.DataFrame({
            'Dataset': ['A', 'B'],
            'Synthesizer': ['Synth1', 'Synth2'],
            'Metric1': [0.9, 0.8],
        })
        result_explorer._handler.load_results = Mock(return_value=results)

        # Run
        loaded_results = result_explorer.load_results('SDGym_results_07_07_2025')

        # Assert
        result_explorer._handler.load_results.assert_called_once_with('SDGym_results_07_07_2025')
        pd.testing.assert_frame_equal(loaded_results, results)

    def test_load_metainfo(self, tmp_path):
        """Test the `load_metainfo` method."""
        # Setup
        output_dir = tmp_path / 'benchmark_output' / 'single_table'
        output_dir.mkdir(parents=True)
        (output_dir / 'SDGym_results_07_07_2025').mkdir(parents=True)
        result_explorer = ResultsExplorer(str(output_dir), modality='single_table')
        result_explorer._handler = Mock()
        metainfo = {'synthesizer_versions': {'Synth1': '1.0.0', 'Synth2': '2.0.0'}}
        result_explorer._handler.load_metainfo = Mock(return_value=metainfo)

        # Run
        loaded_metainfo = result_explorer.load_metainfo('SDGym_results_07_07_2025')

        # Assert
        result_explorer._handler.load_metainfo.assert_called_once_with('SDGym_results_07_07_2025')
        assert loaded_metainfo == metainfo
