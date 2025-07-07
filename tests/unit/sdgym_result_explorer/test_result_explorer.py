import os
import pickle
import re
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from botocore.exceptions import ClientError
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

from sdgym.sdgym_result_explorer.result_explorer import SDGymResultsExplorer, _validate_path


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
        assert result_explorer._is_s3_path is False
        assert result_explorer.path == path
        assert result_explorer.aws_access_key_id is None
        assert result_explorer.aws_secret_access_key is None

    @patch('sdgym.sdgym_result_explorer.result_explorer.boto3.client')
    @patch('sdgym.sdgym_result_explorer.result_explorer._validate_path')
    def test__init__s3(self, mock_validate_path, mock_boto_client):
        """Test the ``__init__`` for accessing S3 bucket."""
        # Setup
        path = 's3://my-bucket/results'
        aws_access_key_id = 'my_access_key'
        aws_secret_access_key = 'my_secret_key'
        mock_boto_client.return_value = 'test_client'
        mock_validate_path.return_value = True

        # Run
        result_explorer = SDGymResultsExplorer(path, aws_access_key_id, aws_secret_access_key)

        # Assert
        assert result_explorer._is_s3_path is True
        assert result_explorer.path == path
        assert result_explorer.aws_access_key_id == aws_access_key_id
        assert result_explorer.aws_secret_access_key == aws_secret_access_key
        assert result_explorer._bucket_name == 'my-bucket'
        assert result_explorer._prefix == 'results/'
        assert result_explorer._s3_client == 'test_client'
        mock_boto_client.assert_called_once_with(
            's3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key
        )

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

    def test_list_s3(self):
        """Test the `list` method with an S3 path"""
        # Setup
        mock_s3_client = Mock()
        mock_s3_client.list_objects_v2.return_value = {
            'CommonPrefixes': [{'Prefix': 'run1/'}, {'Prefix': 'run2/'}]
        }
        result_explorer = Mock()
        result_explorer._s3_client = mock_s3_client
        result_explorer._bucket_name = 'my-bucket'
        result_explorer._prefix = 'results/'

        # Run
        runs = SDGymResultsExplorer.list(result_explorer)

        # Assert
        assert set(runs) == {'run1', 'run2'}
        mock_s3_client.list_objects_v2.assert_called_once_with(
            Bucket='my-bucket', Prefix='results/', Delimiter='/'
        )

    @patch('os.path.exists')
    @patch('os.path.isfile')
    def test__validate_access_local(self, mock_isfile, mock_exists):
        """Test `_validate_access` for local path when files exist."""
        # Setup
        explorer = Mock()
        explorer._is_s3_path = False
        explorer.path = '/local/results'
        results_folder_name = 'results_folder_07_07_2025'
        dataset_name = 'my_dataset'
        synthesizer_name = 'my_synthesizer'
        type = 'synthesizer'
        mock_exists.return_value = True
        mock_isfile.return_value = True

        # Run
        file_path = SDGymResultsExplorer._validate_access(
            explorer, results_folder_name, dataset_name, synthesizer_name, type
        )

        # Assert
        expected_date = '07_07_2025'
        expected_file_path = (
            f'{results_folder_name}/{dataset_name}_{expected_date}/{synthesizer_name}'
            f'/{synthesizer_name}_synthesizer.pkl'
        )
        assert file_path == expected_file_path
        mock_exists.assert_called_once_with(
            os.path.join(
                explorer.path,
                results_folder_name,
                f'{dataset_name}_{expected_date}',
                synthesizer_name,
            )
        )
        mock_isfile.assert_called_once_with(
            os.path.join(
                explorer.path,
                results_folder_name,
                f'{dataset_name}_{expected_date}',
                synthesizer_name,
                f'{synthesizer_name}_synthesizer.pkl',
            )
        )

    @patch('os.path.exists')
    @patch('os.path.isfile')
    def test__validate_access_local_error(self, mock_isfile, mock_exists):
        """Test `_validate_access` for local path when files do not exist."""
        # Setup
        explorer = Mock()
        explorer._is_s3_path = False
        explorer.path = '/local/results'
        results_folder_name = 'SDGym_results_07_07_2025'
        dataset_name = 'adule'
        synthesizer_name = 'GaussianCopulaSynthesizer'
        type = 'synthesizer'
        mock_exists.return_value = False
        synthesizer_path = os.path.join(
            explorer.path, results_folder_name, f'{dataset_name}_07_07_2025', synthesizer_name
        )
        error_message_1 = re.escape(
            f"Synthesizer '{synthesizer_name}' for dataset '{dataset_name}' in"
            f" run '{results_folder_name}' does not exist."
        )
        error_message_2 = re.escape(
            f"Synthesizer file '{synthesizer_name}_synthesizer.pkl' does"
            f" not exist in '{synthesizer_path}'."
        )

        # Run and Assert
        with pytest.raises(ValueError, match=error_message_1):
            SDGymResultsExplorer._validate_access(
                explorer, results_folder_name, dataset_name, synthesizer_name, type
            )

        mock_exists.return_value = True
        mock_isfile.return_value = False
        with pytest.raises(ValueError, match=error_message_2):
            SDGymResultsExplorer._validate_access(
                explorer, results_folder_name, dataset_name, synthesizer_name, type
            )

    def test__validate_access_s3(self):
        """Test `_validate_access` for S3 path."""
        # Setup
        mock_s3_client = Mock()
        explorer = Mock()
        explorer._is_s3_path = True
        explorer._s3_client = mock_s3_client
        explorer._bucket_name = 'my-bucket'
        explorer._prefix = 'results/'
        results_folder_name = 'results_folder_07_07_2025'
        dataset_name = 'my_dataset'
        synthesizer_name = 'my_synthesizer'
        type = 'synthetic_data'
        mock_s3_client.head_object.return_value = {}

        # Run
        file_path = SDGymResultsExplorer._validate_access(
            explorer, results_folder_name, dataset_name, synthesizer_name, type
        )

        # Assert
        expected_date = '07_07_2025'
        expected_file_path = (
            f'{results_folder_name}/{dataset_name}_{expected_date}/'
            f'{synthesizer_name}/{synthesizer_name}_synthetic_data.csv'
        )
        assert file_path == expected_file_path
        mock_s3_client.head_object.assert_called_once_with(
            Bucket='my-bucket', Key=f'results/{expected_file_path}'
        )

    def test__validate_access_s3_error(self):
        """Test `_validate_access` for S3 path when files do not exist."""
        # Setup
        mock_s3_client = Mock()
        explorer = Mock()
        explorer._is_s3_path = True
        explorer._s3_client = mock_s3_client
        explorer._bucket_name = 'my-bucket'
        explorer._prefix = 'results/'
        results_folder_name = 'SDGym_results_07_07_2025'
        dataset_name = 'adule'
        synthesizer_name = 'GaussianCopulaSynthesizer'
        type = 'synthesizer'
        mock_s3_client.head_object.side_effect = ClientError(
            {'Error': {'Code': '404', 'Message': 'Not Found'}}, 'HeadObject'
        )
        expected_error_message = re.escape(
            f"Synthesizer '{synthesizer_name}' for dataset '{dataset_name}' "
            f"in run '{results_folder_name}' does not exist."
        )

        # Run and Assert
        with pytest.raises(ValueError, match=expected_error_message):
            SDGymResultsExplorer._validate_access(
                explorer, results_folder_name, dataset_name, synthesizer_name, type
            )

    def test_load_synthesizer_local(self, tmp_path):
        """Test `load_synthesizer` for local path."""
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
        result_explorer = SDGymResultsExplorer(str(tmp_path))

        # Run
        loaded_synthesizer = result_explorer.load_synthesizer(
            results_folder_name=folder_name,
            dataset_name=dataset_name,
            synthesizer_name=synthesizer_name,
        )

        # Assert
        assert loaded_synthesizer is not None
        assert isinstance(loaded_synthesizer, GaussianCopulaSynthesizer)

    def test_load_synthesizer_s3(self):
        """Test `load_synthesizer` for S3 path."""
        # Setup
        mock_s3_client = Mock()
        explorer = Mock()
        explorer._is_s3_path = True
        explorer._s3_client = mock_s3_client
        explorer._bucket_name = 'my-bucket'
        explorer._prefix = 'results/'
        results_folder_name = 'SDGym_results_07_07_2025'
        dataset_name = 'adult'
        synthesizer_name = 'GaussianCopulaSynthesizer'
        expected_file_path = (
            f'{results_folder_name}/{dataset_name}_07_07_2025/{synthesizer_name}/'
            f'{synthesizer_name}_synthesizer.pkl'
        )
        explorer._validate_access = Mock()
        explorer._validate_access.return_value = expected_file_path
        mock_s3_client.get_object.return_value = {
            'Body': Mock(
                read=Mock(return_value=pickle.dumps(GaussianCopulaSynthesizer(Metadata())))
            )
        }
        # Run
        loaded_synthesizer = SDGymResultsExplorer.load_synthesizer(
            explorer, results_folder_name, dataset_name, synthesizer_name
        )

        # Assert
        assert loaded_synthesizer is not None
        assert isinstance(loaded_synthesizer, GaussianCopulaSynthesizer)
        mock_s3_client.get_object.assert_called_once_with(
            Bucket='my-bucket', Key=f'results/{expected_file_path}'
        )

    def test_load_synthetic_data_local(self, tmp_path):
        """Test `load_synthetic_data` for local path."""
        # Setup
        folder_name = 'SDGym_results_07_07_2025'
        dataset_name = 'adult'
        synthesizer_name = 'GaussianCopulaSynthesizer'
        (tmp_path / folder_name).mkdir(parents=True, exist_ok=True)
        (tmp_path / folder_name / f'{dataset_name}_07_07_2025').mkdir(parents=True, exist_ok=True)
        (tmp_path / folder_name / f'{dataset_name}_07_07_2025' / synthesizer_name).mkdir(
            parents=True, exist_ok=True
        )
        synthetic_data_path = (
            tmp_path
            / folder_name
            / f'{dataset_name}_07_07_2025'
            / synthesizer_name
            / f'{synthesizer_name}_synthetic_data.csv'
        )
        synthetic_data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        synthetic_data.to_csv(synthetic_data_path, index=False)
        result_explorer = SDGymResultsExplorer(str(tmp_path))

        # Run
        loaded_synthetic_data = result_explorer.load_synthetic_data(
            results_folder_name=folder_name,
            dataset_name=dataset_name,
            synthesizer_name=synthesizer_name,
        )

        # Assert
        assert isinstance(loaded_synthetic_data, pd.DataFrame)
        pd.testing.assert_frame_equal(loaded_synthetic_data, synthetic_data)

    def test_load_synthetic_data_s3(self):
        """Test `load_synthetic_data` for S3 path."""
        # Setup
        mock_s3_client = Mock()
        explorer = Mock()
        explorer._is_s3_path = True
        explorer._s3_client = mock_s3_client
        explorer._bucket_name = 'my-bucket'
        explorer._prefix = 'results/'
        results_folder_name = 'SDGym_results_07_07_2025'
        dataset_name = 'adult'
        synthesizer_name = 'GaussianCopulaSynthesizer'
        expected_file_path = (
            f'{results_folder_name}/{dataset_name}_07_07_2025/{synthesizer_name}/'
            f'{synthesizer_name}_synthetic_data.csv'
        )
        explorer._validate_access = Mock()
        explorer._validate_access.return_value = expected_file_path
        synthetic_data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        mock_s3_client.get_object.return_value = {
            'Body': Mock(read=Mock(return_value=synthetic_data.to_csv(index=False).encode()))
        }

        # Run
        loaded_synthetic_data = SDGymResultsExplorer.load_synthetic_data(
            explorer, results_folder_name, dataset_name, synthesizer_name
        )

        # Assert
        assert isinstance(loaded_synthetic_data, pd.DataFrame)
        pd.testing.assert_frame_equal(loaded_synthetic_data, synthetic_data)
        mock_s3_client.get_object.assert_called_once_with(
            Bucket='my-bucket', Key=f'results/{expected_file_path}'
        )

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
