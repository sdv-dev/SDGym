import os
import pickle
import re
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

from sdgym.sdgym_result_explorer.result_handler import LocalResultsHandler, S3ResultsHandler


class TestLocalResultsHandler:
    """Unit tests for the LocalResultsHandler class."""

    def test_list_runs(self, tmp_path):
        """Test the `list_runs` method"""
        # Setup
        path = tmp_path / 'results'
        path.mkdir()
        (path / 'run1').mkdir()
        (path / 'run2').mkdir()
        result_handler = LocalResultsHandler(str(path))

        # Run
        runs = result_handler.list_runs()

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
            / f'{synthesizer_name}_synthesizer.pkl'
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
    def test_validate_access_local(self, mock_isfile, mock_exists):
        """Test `validate_access` when files exist."""
        # Setup
        handler = Mock()
        handler.base_path = '/local/results'
        path_parts = ['results_folder_07_07_2025', 'my_dataset']
        mock_exists.return_value = True
        mock_isfile.return_value = True

        # Run
        file_path = LocalResultsHandler.validate_access(handler, path_parts, 'synthesizer.pkl')

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
    def test_validate_access_local_error(self, mock_isfile, mock_exists):
        """Test `validate_access` for local path when files do not exist."""
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
            LocalResultsHandler.validate_access(handler, path_parts, 'synthesizer.pkl')

        mock_exists.return_value = True
        mock_isfile.return_value = False
        with pytest.raises(ValueError, match=error_message_2):
            LocalResultsHandler.validate_access(handler, path_parts, 'synthesizer.pkl')


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

    def test_list_runs(self):
        """Test the `list_runs` method."""
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
        runs = S3ResultsHandler.list_runs(result_handler)

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

    def test_validate_access_s3(self):
        """Test `validate_access` for S3 path when files exist."""
        # Setup
        mock_s3_client = Mock()
        handler = S3ResultsHandler('s3://my-bucket/prefix', mock_s3_client)
        path_parts = ['results_folder_07_07_2025', 'my_dataset']
        end_filename = 'synthesizer.pkl'
        file_path = 'results_folder_07_07_2025/my_dataset/synthesizer.pkl'

        # Run
        result = handler.validate_access(path_parts, end_filename)

        # Assert
        assert result == file_path
        mock_s3_client.head_object.assert_called_once_with(
            Bucket='my-bucket', Key='prefix/results_folder_07_07_2025/my_dataset/synthesizer.pkl'
        )
