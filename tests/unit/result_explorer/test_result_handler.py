import os
import pickle
import re
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

from sdgym.result_explorer.result_handler import (
    LocalResultsHandler,
    ResultsHandler,
    S3ResultsHandler,
)


class TestResultsHandler:
    """Unit tests for the ResultsHandler class."""

    def test__validate_folder_name(self):
        """Test the `_validate_folder_name` method."""
        # Setup
        handler = Mock()
        handler.list = Mock(return_value=['run1', 'run2'])
        expected_error = re.escape("Folder 'run3' does not exist in the results directory.")

        # Run and Assert
        ResultsHandler._validate_folder_name(handler, 'run1')
        with pytest.raises(ValueError, match=expected_error):
            ResultsHandler._validate_folder_name(handler, 'run3')

    def test__compute_wins(self):
        """Test the `_compute_wins` method."""
        # Setup
        data = pd.DataFrame({
            'Dataset': ['A', 'A', 'B', 'B'],
            'Synthesizer': [
                'GaussianCopulaSynthesizer',
                'Synth1',
                'GaussianCopulaSynthesizer',
                'Synth1',
            ],
            'Quality_Score': [0.5, 0.6, 0.7, 0.6],
        })
        handler = LocalResultsHandler(base_path='.')

        # Run
        handler._compute_wins(data)

        # Assert
        expected_wins = [0, 1, 0, 0]
        assert data['Win'].tolist() == expected_wins

    def test__get_summarize_table(self):
        """Test the `_get_summarize_table` method."""
        # Setup
        folder_to_results = {
            'SDGym_results_07_15_2025': pd.DataFrame({
                'Dataset': ['A', 'B', 'C'] * 3,
                'Synthesizer': ['GaussianCopulaSynthesizer'] * 3 + ['Synth1'] * 3 + ['Synth2'] * 3,
                'Win': [0, 0, 0, 1, 1, 0, 0, 1, 0],
            })
        }
        folder_infos = {
            'SDGym_results_07_15_2025': {
                'date': '07_15_2025',
                'sdgym_version': '0.9.0',
                '# datasets': 3,
            }
        }
        handler = Mock()
        handler.baseline_synthesizer = 'GaussianCopulaSynthesizer'

        # Run
        result = ResultsHandler._get_summarize_table(handler, folder_to_results, folder_infos)

        # Assert
        expected_summary = pd.DataFrame({
            'Synthesizer': ['Synth1', 'Synth2'],
            '07_15_2025 - # datasets: 3 - sdgym version: 0.9.0': [2, 1],
        })
        pd.testing.assert_frame_equal(result, expected_summary)

    def test_get_column_name_infos(self):
        """Test the `_get_column_name_infos` method."""
        # Setup
        folder = 'SDGym_results_07_15_2025'
        yaml_content = {
            'starting_date': '07_15_2025 15:56:03',
            'sdgym_version': '0.9.0',
        }
        result = pd.DataFrame({
            'Dataset': ['A', 'B', 'C'],
            'Synthesizer': ['GaussianCopulaSynthesizer'] * 3,
        })
        folder_to_results = {folder: result}
        handler = Mock()
        handler._get_results_files = Mock(return_value=['run_config.yaml'])
        handler._load_yaml_file = Mock(return_value=yaml_content)
        handler.baseline_synthesizer = 'GaussianCopulaSynthesizer'

        # Run
        info = ResultsHandler._get_column_name_infos(handler, folder_to_results)

        # Assert
        assert info == {folder: {'date': '07_15_2025', 'sdgym_version': '0.9.0', '# datasets': 3}}

    def test__process_results(self):
        """Test the `_process_results` method."""
        # Setup
        results = [
            pd.DataFrame({
                'Dataset': ['A', 'A', 'B', 'B', 'C'],
                'Synthesizer': ['Synth1', 'Synth2(1)', 'Synth1', 'Synth2(1)', 'Synth1'],
                'Quality_Score': [0.5, 0.6, 0.7, 0.6, 0.8],
            }),
            pd.DataFrame({
                'Dataset': ['D', 'D', 'D'],
                'Synthesizer': ['Synth1(2)', 'Synth2', 'Synth1(2)'],
                'Quality_Score': [0.7, 0.8, 0.9],
            }),
        ]
        invalid_results = [
            pd.DataFrame({
                'Dataset': ['A', 'A', 'B', 'B', 'C'],
                'Synthesizer': ['Synth1', 'Synth2', 'Synth3', 'Synth2', 'Synth1'],
                'Quality_Score': [0.5, 0.6, 0.7, 0.6, 0.8],
            }),
        ]
        handler = Mock()
        expected_error_message = re.escape(
            'There is no dataset that has been run by all synthesizers. Cannot summarize results.'
        )

        # Run
        processed_results = ResultsHandler._process_results(handler, results)
        with pytest.raises(ValueError, match=expected_error_message):
            ResultsHandler._process_results(handler, invalid_results)

        # Assert
        expected_results = pd.DataFrame({
            'Dataset': ['A', 'A', 'B', 'B', 'D', 'D'],
            'Synthesizer': ['Synth1', 'Synth2'] * 3,
            'Quality_Score': [0.5, 0.6, 0.7, 0.6, 0.7, 0.8],
        })
        pd.testing.assert_frame_equal(processed_results, expected_results)

    def test_summarize(self):
        """Test the `summarize` method."""
        # Setup
        folder_name = 'SDGym_results_07_15_2025'
        handler = Mock()
        handler.list = Mock(return_value=[folder_name])
        handler._get_results_files = Mock(return_value=['results.csv', 'results(1).csv'])
        result_1 = pd.DataFrame({
            'Dataset': ['A', 'B'],
            'Synthesizer': ['Synth1'] * 2,
            'Quality_Score': [0.5, 0.6],
        })
        result_2 = pd.DataFrame({
            'Dataset': ['A', 'B'],
            'Synthesizer': ['Synth2'] * 2,
            'Quality_Score': [0.7, 0.8],
        })
        result_list = [result_1, result_2]
        handler._get_results = Mock(return_value=result_list)
        aggregated_results = pd.concat(result_list, ignore_index=True)
        handler._compute_wins = Mock()
        handler._get_column_name_infos = Mock(
            return_value={
                folder_name: {'date': '07_15_2025', 'sdgym_version': '0.9.0', '# datasets': 1}
            }
        )
        result = pd.DataFrame({
            '07_15_2025 - # datasets: 1 - sdgym version: 0.9.0': [1],
            'Synthesizer': ['Synth1'],
        }).set_index('Synthesizer')
        handler._get_summarize_table = Mock(return_value=result)
        handler._process_results = Mock(return_value=aggregated_results)

        # Run
        summary, benchmark_result = ResultsHandler.summarize(handler, folder_name)

        # Assert
        pd.testing.assert_frame_equal(summary, result)
        pd.testing.assert_frame_equal(benchmark_result, aggregated_results)
        handler.list.assert_called_once()
        handler._get_results_files.assert_called_once_with(
            folder_name, prefix='results', suffix='.csv'
        )
        handler._get_results.assert_called_once_with(folder_name, ['results.csv', 'results(1).csv'])
        handler._process_results.assert_called_once_with(result_list)
        compute_wing_args = handler._compute_wins.call_args[0][0]
        pd.testing.assert_frame_equal(compute_wing_args, aggregated_results)
        _get_column_name_infos_args = handler._get_column_name_infos.call_args[0][0]
        for folder, agg_result in _get_column_name_infos_args.items():
            assert folder == folder_name
            pd.testing.assert_frame_equal(agg_result, aggregated_results)

    def test_load_results(self):
        """Test the `load_results` method."""
        # Setup
        folder_name = 'SDGym_results_07_15_2025'
        handler = Mock()
        handler._validate_folder_name = Mock()
        handler._get_results_files = Mock(return_value=['results.csv', 'results(1).csv'])
        result_1 = pd.DataFrame({
            'Dataset': ['A', 'B'],
            'Synthesizer': ['Synth1'] * 2,
            'Quality_Score': [0.5, 0.6],
        })
        result_2 = pd.DataFrame({
            'Dataset': ['A', 'B'],
            'Synthesizer': ['Synth2'] * 2,
            'Quality_Score': [0.7, 0.8],
        })
        result_list = [result_1, result_2]
        handler._get_results = Mock(return_value=result_list)

        # Run
        results = ResultsHandler.load_results(handler, folder_name)

        # Assert
        handler._validate_folder_name.assert_called_once_with(folder_name)
        expected_results = pd.concat(result_list, ignore_index=True)
        pd.testing.assert_frame_equal(results, expected_results)
        handler._get_results_files.assert_called_once_with(
            folder_name, prefix='results', suffix='.csv'
        )
        handler._get_results.assert_called_once_with(folder_name, ['results.csv', 'results(1).csv'])

    def test_load_metainfo(self):
        """Test the `load_metainfo` method."""
        # Setup
        folder_name = 'SDGym_results_07_15_2025'
        handler = Mock()
        handler._validate_folder_name = Mock()
        handler._get_results_files = Mock(return_value=['metainfo.yaml', 'metainfo(1).yaml'])
        yaml_content_1 = {'run_id': 'run_1', 'sdgym_version': '0.9.0'}
        yaml_content_2 = {'run_id': 'run_2', 'sdgym_version': '0.9.1'}
        handler._load_yaml_file = Mock(side_effect=[yaml_content_1, yaml_content_2])

        # Run
        metainfo = ResultsHandler.load_metainfo(handler, folder_name)

        # Assert
        handler._validate_folder_name.assert_called_once_with(folder_name)
        expected_metainfo = {
            'run_1': {'sdgym_version': '0.9.0'},
            'run_2': {'sdgym_version': '0.9.1'},
        }
        assert metainfo == expected_metainfo
        handler._get_results_files.assert_called_once_with(
            folder_name, prefix='metainfo', suffix='.yaml'
        )
        assert handler._load_yaml_file.call_count == 2


class TestLocalResultsHandler:
    """Unit tests for the LocalResultsHandler class."""

    def test_list(self, tmp_path):
        """Test the `list` method"""
        # Setup
        path = tmp_path / 'results'
        path.mkdir()
        (path / 'run1').mkdir()
        (path / 'run2').mkdir()
        result_handler = LocalResultsHandler(str(path))

        # Run
        runs = result_handler.list()

        # Assert
        assert set(runs) == {'run1', 'run2'}

    def test_load_synthetic_data(self, tmp_path):
        """Test the `load_synthetic_data` method"""
        # Setup
        path = (
            tmp_path
            / 'results'
            / 'run1'
            / 'expedia_hotel_logs_07_07_2025'
            / 'GaussianCopulaSynthesizer'
        )
        path.mkdir(parents=True)
        data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        data.to_csv(path / 'synthetic_data.csv', index=False)
        file_path = str(path / 'synthetic_data.csv')
        result_handler = LocalResultsHandler(str(tmp_path))

        # Run
        synthetic_data = result_handler.load_synthetic_data(file_path)

        # Assert
        pd.testing.assert_frame_equal(synthetic_data, data)

    def test_load_synthesizer(self, tmp_path):
        """Test the `load_synthesizer` method"""
        # Setup
        synthesizer = GaussianCopulaSynthesizer(Metadata())
        folder_name = 'SDGym_results_07_07_2025'
        dataset_name = 'adult'
        synthesizer_name = 'GaussianCopulaSynthesizer'
        (tmp_path / folder_name).mkdir(parents=True, exist_ok=True)
        (tmp_path / folder_name / f'{dataset_name}_07_07_2025').mkdir(parents=True, exist_ok=True)
        (tmp_path / folder_name / f'{dataset_name}_07_07_2025' / synthesizer_name).mkdir(
            parents=True, exist_ok=True
        )
        synthesizer_path = (
            tmp_path
            / folder_name
            / f'{dataset_name}_07_07_2025'
            / synthesizer_name
            / f'{synthesizer_name}.pkl'
        )
        synthesizer.save(synthesizer_path)
        result_handler = LocalResultsHandler(str(tmp_path))

        # Run
        loaded_synthesizer = result_handler.load_synthesizer(str(synthesizer_path))

        # Assert
        assert loaded_synthesizer is not None
        assert isinstance(loaded_synthesizer, GaussianCopulaSynthesizer)

    @patch('os.path.exists')
    @patch('os.path.isfile')
    def test_get_file_path_local(self, mock_isfile, mock_exists):
        """Test `get_file_path` when files exist."""
        # Setup
        handler = Mock()
        handler.base_path = '/local/results'
        path_parts = ['results_folder_07_07_2025', 'my_dataset']
        mock_exists.return_value = True
        mock_isfile.return_value = True

        # Run
        file_path = LocalResultsHandler.get_file_path(handler, path_parts, 'synthesizer.pkl')

        # Assert
        expected_file_path = os.path.join(
            'results_folder_07_07_2025', 'my_dataset', 'synthesizer.pkl'
        )
        assert file_path == expected_file_path
        mock_exists.assert_called_once_with(
            os.path.join(
                handler.base_path,
                'results_folder_07_07_2025',
                'my_dataset',
            )
        )
        mock_isfile.assert_called_once_with(
            os.path.join(
                handler.base_path,
                'results_folder_07_07_2025',
                'my_dataset',
                'synthesizer.pkl',
            )
        )

    @patch('os.path.exists')
    @patch('os.path.isfile')
    def test_get_file_path_local_error(self, mock_isfile, mock_exists):
        """Test `get_file_path` for local path when files do not exist."""
        # Setup
        handler = Mock()
        handler.base_path = '/local/results'
        results_folder_name = 'SDGym_results_07_07_2025'
        dataset_name = 'adult'
        mock_exists.return_value = False
        synthesizer_name = 'GaussianCopulaSynthesizer'
        synthesizer_path = os.path.join(
            handler.base_path, results_folder_name, f'{dataset_name}_07_07_2025', synthesizer_name
        )
        path_parts = [results_folder_name, f'{dataset_name}_07_07_2025', synthesizer_name]
        error_message_1 = re.escape(f'Path does not exist: {synthesizer_path}')
        error_message_2 = re.escape('File does not exist: synthesizer.pkl')

        # Run and Assert
        with pytest.raises(ValueError, match=error_message_1):
            LocalResultsHandler.get_file_path(handler, path_parts, 'synthesizer.pkl')

        mock_exists.return_value = True
        mock_isfile.return_value = False
        with pytest.raises(ValueError, match=error_message_2):
            LocalResultsHandler.get_file_path(handler, path_parts, 'synthesizer.pkl')


class TestS3ResultsHandler:
    """Unit tests for the S3ResultsHandler class."""

    def test__init__(
        self,
    ):
        """Test the `__init__` method."""
        # Setup
        path = 's3://my-bucket/prefix'

        # Run
        result_handler = S3ResultsHandler(path, 's3_client')

        # Assert
        assert result_handler.s3_client == 's3_client'
        assert result_handler.bucket_name == 'my-bucket'
        assert result_handler.prefix == 'prefix/'

    def test_list(self):
        """Test the `list` method."""
        # Setup
        mock_s3_client = Mock()
        mock_s3_client.list_objects_v2.return_value = {
            'CommonPrefixes': [{'Prefix': 'run1/'}, {'Prefix': 'run2/'}]
        }
        result_handler = Mock()
        result_handler.s3_client = mock_s3_client
        result_handler.bucket_name = 'my-bucket'
        result_handler.prefix = 'results/'

        # Run
        runs = S3ResultsHandler.list(result_handler)

        # Assert
        assert set(runs) == {'run1', 'run2'}
        mock_s3_client.list_objects_v2.assert_called_once_with(
            Bucket='my-bucket', Prefix='results/', Delimiter='/'
        )

    def test_load_synthetic_data(self):
        """Test the `load_synthetic_data` method."""
        # Setup
        mock_s3_client = Mock()
        mock_s3_client.get_object.return_value = {'Body': Mock(read=lambda: b'col1,col2\n1,3\n2,4')}
        result_handler = S3ResultsHandler('s3://my-bucket/prefix', mock_s3_client)
        result_handler.s3_client = mock_s3_client

        # Run
        synthetic_data = result_handler.load_synthetic_data('synthetic_data.csv')

        # Assert
        expected_data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        pd.testing.assert_frame_equal(synthetic_data, expected_data)
        mock_s3_client.get_object.assert_called_once_with(
            Bucket='my-bucket', Key='prefix/synthetic_data.csv'
        )

    def test_load_synthesizer(self):
        """Test the `load_synthesizer` method."""
        # Setup
        mock_s3_client = Mock()
        synthesizer = GaussianCopulaSynthesizer(Metadata())
        mock_s3_client.get_object.return_value = {
            'Body': Mock(read=lambda: pickle.dumps(synthesizer))
        }
        result_handler = S3ResultsHandler('s3://my-bucket/prefix', mock_s3_client)
        result_handler.s3_client = mock_s3_client

        # Run
        loaded_synthesizer = result_handler.load_synthesizer('synthesizer.pkl')

        # Assert
        assert isinstance(loaded_synthesizer, GaussianCopulaSynthesizer)
        mock_s3_client.get_object.assert_called_once_with(
            Bucket='my-bucket', Key='prefix/synthesizer.pkl'
        )

    def test_get_file_path_s3(self):
        """Test `get_file_path` for S3 path when folders and file exist."""
        # Setup
        mock_s3_client = Mock()
        handler = S3ResultsHandler('s3://my-bucket/prefix', mock_s3_client)
        path_parts = ['results_folder_07_07_2025', 'my_dataset']
        end_filename = 'synthesizer.pkl'
        file_path = 'results_folder_07_07_2025/my_dataset/synthesizer.pkl'
        mock_s3_client.list_objects_v2.return_value = {'Contents': [{}]}

        # Run
        result = handler.get_file_path(path_parts, end_filename)

        # Assert
        assert result == file_path
        mock_s3_client.list_objects_v2.assert_any_call(
            Bucket='my-bucket', Prefix='prefix/results_folder_07_07_2025/', MaxKeys=1
        )
        mock_s3_client.list_objects_v2.assert_any_call(
            Bucket='my-bucket', Prefix='prefix/results_folder_07_07_2025/my_dataset/', MaxKeys=1
        )
        mock_s3_client.head_object.assert_called_once_with(
            Bucket='my-bucket', Key='prefix/results_folder_07_07_2025/my_dataset/synthesizer.pkl'
        )
