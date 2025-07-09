import re
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

from sdgym.sdgym_result_explorer.result_explorer import SDGymResultsExplorer, _validate_path
from sdgym.sdgym_result_explorer.result_handler import LocalResultsHandler, S3ResultsHandler


def test_validate_path_local(tmp_path):
    """Test the `_validate_path` function for a local path."""
    # Setup
    path = tmp_path / 'local_results_folder'
    path.mkdir()
    expected_error = re.escape(
        "The provided path 'invalid_path' is not a valid directory or S3 bucket."
    )

    # Run
    is_valid = _validate_path(str(path))
    with pytest.raises(ValueError, match=expected_error):
        _validate_path('invalid_path')

    # Assert
    assert is_valid is False


@patch('sdgym.sdgym_result_explorer.result_explorer._validate_bucket_access')
def test_validate_path_s3(mock_validate_bucket_access):
    """Test the `_validate_path` function for an S3 path."""
    # Setup
    path = 's3://my-bucket/results'
    aws_access_key_id = 'my_access_key'
    aws_secret_access_key = 'my_secret_key'

    # Run
    is_valid = _validate_path(path, aws_access_key_id, aws_secret_access_key)

    # Assert
    assert is_valid is True
    mock_validate_bucket_access.assert_called_once_with(
        path, aws_access_key_id, aws_secret_access_key
    )


class TestSDGymResultsExplorer:
    @patch('sdgym.sdgym_result_explorer.result_explorer._validate_path')
    def test__init__local(self, mock_validate_path):
        """Test the ``__init__`` for accessing local folder."""
        # Setup
        mock_validate_path.return_value = False
        path = 'local_results_folder'

        # Run
        result_explorer = SDGymResultsExplorer(path)

        # Assert
        mock_validate_path.assert_called_once_with(path, None, None)
        assert isinstance(result_explorer._handler, LocalResultsHandler)
        assert result_explorer.path == path
        assert result_explorer.aws_access_key_id is None
        assert result_explorer.aws_secret_access_key is None

    @patch('sdgym.sdgym_result_explorer.result_explorer._validate_path')
    def test__init__s3(self, mock_validate_path):
        """Test the ``__init__`` for accessing S3 bucket."""
        # Setup
        path = 's3://my-bucket/results'
        aws_access_key_id = 'my_access_key'
        aws_secret_access_key = 'my_secret_key'
        mock_validate_path.return_value = True

        # Run
        result_explorer = SDGymResultsExplorer(path, aws_access_key_id, aws_secret_access_key)

        # Assert
        assert result_explorer.path == path
        assert result_explorer.aws_access_key_id == aws_access_key_id
        assert result_explorer.aws_secret_access_key == aws_secret_access_key
        assert isinstance(result_explorer._handler, S3ResultsHandler)

    def test_list_local(self, tmp_path):
        """Test the `list` method with a local path"""
        # Setup
        path = tmp_path / 'results'
        path.mkdir()
        (path / 'run1').mkdir()
        (path / 'run2').mkdir()
        result_explorer = SDGymResultsExplorer(str(path))

        # Run
        runs = result_explorer.list()

        # Assert
        assert set(runs) == {'run1', 'run2'}

    def test_list(self, tmp_path):
        """Test the `list` method with an S3 path"""
        # Setup
        path = tmp_path / 'results'
        path.mkdir()
        (path / 'run1').mkdir()
        (path / 'run2').mkdir()
        result_explorer = SDGymResultsExplorer(str(path))
        result_explorer._handler = Mock()
        result_explorer._handler.list_runs.return_value = ['run1', 'run2']

        # Run
        runs = result_explorer.list()

        # Assert
        assert runs == ['run1', 'run2']
        result_explorer._handler.list_runs.assert_called_once()

    def test__validate_access(self):
        """Test `_validate_access` for local and S3 paths."""
        # Setup
        explorer = Mock()
        explorer._handler = Mock()
        results_folder_name = 'results_folder_07_07_2025'
        dataset_name = 'my_dataset'
        synthesizer_name = 'my_synthesizer'
        type = 'synthesizer'
        expected_filepath = (
            f'{results_folder_name}/{dataset_name}_07_07_2025/{synthesizer_name}/'
            f'{synthesizer_name}_synthesizer.pkl'
        )
        explorer._handler.validate_access.return_value = expected_filepath

        # Run
        file_path = SDGymResultsExplorer._validate_access(
            explorer, results_folder_name, dataset_name, synthesizer_name, type
        )

        # Assert
        explorer._handler.validate_access.assert_called_once_with(
            [results_folder_name, f'{dataset_name}_07_07_2025', synthesizer_name],
            f'{synthesizer_name}_synthesizer.pkl',
        )
        assert file_path == expected_filepath

    def test_load_synthesizer(self, tmp_path):
        """Test `load_synthesizer` method."""
        # Setup
        explorer = SDGymResultsExplorer(str(tmp_path))
        explorer._handler = Mock()
        explorer._handler.load_synthesizer = Mock(
            return_value=GaussianCopulaSynthesizer(Metadata())
        )
        explorer._validate_access = Mock(return_value='path/to/synthesizer.pkl')

        # Run
        synthesizer = explorer.load_synthesizer(
            results_folder_name='results_folder_07_07_2025',
            dataset_name='my_dataset',
            synthesizer_name='my_synthesizer',
        )

        # Assert
        explorer._validate_access.assert_called_once_with(
            'results_folder_07_07_2025', 'my_dataset', 'my_synthesizer', 'synthesizer'
        )
        explorer._handler.load_synthesizer.assert_called_once_with('path/to/synthesizer.pkl')
        assert isinstance(synthesizer, GaussianCopulaSynthesizer)

    def test_load_synthetic_data(self, tmp_path):
        # Setup
        explorer = SDGymResultsExplorer(str(tmp_path))
        explorer._handler = Mock()
        data = pd.DataFrame({'column1': [1, 2], 'column2': [3, 4]})
        explorer._handler.load_synthetic_data = Mock(return_value=data)
        explorer._validate_access = Mock(return_value='path/to/synthetic_data.csv')

        # Run
        synthetic_data = explorer.load_synthetic_data(
            results_folder_name='results_folder_07_07_2025',
            dataset_name='my_dataset',
            synthesizer_name='my_synthesizer',
        )

        # Assert
        explorer._validate_access.assert_called_once_with(
            'results_folder_07_07_2025', 'my_dataset', 'my_synthesizer', 'synthetic_data'
        )
        explorer._handler.load_synthetic_data.assert_called_once_with('path/to/synthetic_data.csv')
        pd.testing.assert_frame_equal(synthetic_data, data)

    @patch('sdgym.sdgym_result_explorer.result_explorer.load_dataset')
    @patch('sdgym.sdgym_result_explorer.result_explorer.get_dataset_paths')
    def test_load_real_data(self, mock_get_dataset_paths, mock_load_dataset, tmp_path):
        """Test `load_real_data` method."""
        # Setup
        dataset_name = 'adult'
        mock_get_dataset_paths.return_value = ['path/to/adult/dataset']
        expected_data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        mock_load_dataset.return_value = (expected_data, None)
        result_explorer = SDGymResultsExplorer(tmp_path)

        # Run
        real_data = result_explorer.load_real_data(dataset_name)

        # Assert
        mock_get_dataset_paths.assert_called_once_with(
            datasets=[dataset_name], aws_key=None, aws_secret=None
        )
        mock_load_dataset.assert_called_once_with(
            'single_table', 'path/to/adult/dataset', aws_key=None, aws_secret=None
        )
        pd.testing.assert_frame_equal(real_data, expected_data)

    def test_load_real_data_invalid_dataset(self, tmp_path):
        """Test `load_real_data` method with an invalid dataset."""
        # Setup
        dataset_name = 'invalid_dataset'
        result_explorer = SDGymResultsExplorer(tmp_path)
        expected_error_message = re.escape(
            f"Dataset '{dataset_name}' is not a default dataset. "
            'Please provide a valid dataset name.'
        )

        # Run and Assert
        with pytest.raises(ValueError, match=expected_error_message):
            result_explorer.load_real_data(dataset_name)
