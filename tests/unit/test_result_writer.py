import cloudpickle
from unittest.mock import Mock, patch

import pandas as pd
import yaml

from sdgym.result_writer import LocalResultsWriter, S3ResultsWriter


class TestLocalResultsWriter:
    def test_write_dataframe(self, tmp_path):
        """Test the `write_dataframe` method."""
        # Setup
        base_path = tmp_path / 'sdgym_results'
        base_path.mkdir(parents=True, exist_ok=True)
        result_writer = LocalResultsWriter()
        file_path = base_path / 'test_data.csv'
        data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

        # Run
        result_writer.write_dataframe(data, file_path)

        # Assert
        expected_data = pd.read_csv(file_path)
        pd.testing.assert_frame_equal(data, expected_data)

    def test_write_dataframe_append(self, tmp_path):
        """Test `write_dataframe` with append mode."""
        # Setup
        base_path = tmp_path / 'sdgym_results'
        base_path.mkdir(parents=True, exist_ok=True)
        result_writer = LocalResultsWriter()
        file_path = base_path / 'test_data.csv'
        data1 = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        data2 = pd.DataFrame({'col1': [5, 6], 'col2': [7, 8]})

        # Run
        result_writer.write_dataframe(data1, file_path)
        result_writer.write_dataframe(data2, file_path, append=True)

        # Assert
        expected_data = pd.read_csv(file_path)
        expected_combined = pd.concat([data1, data2], ignore_index=True)
        pd.testing.assert_frame_equal(expected_combined, expected_data)

    def test_write_pickle(self, tmp_path):
        """Test the `write_pickle` method."""
        # Setup
        base_path = tmp_path / 'sdgym_results'
        base_path.mkdir(parents=True, exist_ok=True)
        result_writer = LocalResultsWriter()
        file_path = base_path / 'test_object.pkl'
        obj = {'key': 'value'}

        # Run
        result_writer.write_pickle(obj, file_path)

        # Assert
        with open(file_path, 'rb') as f:
            loaded_obj = cloudpickle.load(f)

        assert loaded_obj == obj

    def test_write_yaml(self, tmp_path):
        """Test the `write_yaml` method."""
        # Setup
        base_path = tmp_path / 'sdgym_results'
        base_path.mkdir(parents=True, exist_ok=True)
        result_writer = LocalResultsWriter()
        file_path = base_path / 'test_data.yaml'
        data = {'key': 'value'}

        # Run
        result_writer.write_yaml(data, file_path)

        # Assert
        with open(file_path, 'r') as f:
            loaded_data = yaml.safe_load(f)

        assert loaded_data == data

    def test_write_yaml_append(self, tmp_path):
        """Test `write_yaml` with append mode."""
        # Setup
        base_path = tmp_path / 'sdgym_results'
        base_path.mkdir(parents=True, exist_ok=True)
        file_path = base_path / 'test_data.yaml'
        result_writer = LocalResultsWriter()
        data1 = {'key1': 'value1'}
        data2 = {'key2': 'value2'}

        # Run
        result_writer.write_yaml(data1, file_path)
        result_writer.write_yaml(data2, file_path, append=True)

        # Assert
        with open(file_path, 'r') as f:
            loaded_data = yaml.safe_load(f)

        expected_data = {**data1, **data2}
        assert loaded_data == expected_data


class TestS3ResultsWriter:
    def test__init__(self):
        """Test the __init__ method."""
        # Setup
        result_writer = S3ResultsWriter('s3_client')

        # Assert
        assert result_writer.s3_client == 's3_client'

    @patch('sdgym.result_writer.parse_s3_path')
    def test_write_dataframe(self, mockparse_s3_path):
        """Test the `write_dataframe` method."""
        # Setup
        mock_s3_client = Mock()
        mockparse_s3_path.return_value = ('bucket_name', 'key_prefix/test_data.csv')
        result_writer = S3ResultsWriter(mock_s3_client)
        data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

        # Run
        result_writer.write_dataframe(data, 'test_data.csv')

        # Assert
        mockparse_s3_path.assert_called_once_with('test_data.csv')
        mock_s3_client.put_object.assert_called_once_with(
            Body=data.to_csv(index=False).encode(),
            Bucket='bucket_name',
            Key='key_prefix/test_data.csv',
        )

    @patch('sdgym.result_writer.parse_s3_path')
    def test_write_dataframe_append(self, mockparse_s3_path):
        """Test `write_dataframe` with append mode."""
        # Setup
        mock_s3_client = Mock()
        mockparse_s3_path.return_value = ('bucket_name', 'key_prefix/test_data.csv')
        data1 = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        mock_s3_client.get_object.return_value = {
            'Body': Mock(read=lambda: data1.to_csv(index=False).encode())
        }
        result_writer = S3ResultsWriter(mock_s3_client)
        data2 = pd.DataFrame({'col1': [5, 6], 'col2': [7, 8]})

        # Run
        result_writer.write_dataframe(data2, 'test_data.csv', append=True)

        # Assert
        mockparse_s3_path.assert_called_with('test_data.csv')
        mock_s3_client.get_object.assert_called_with(
            Bucket='bucket_name', Key='key_prefix/test_data.csv'
        )
        mock_s3_client.put_object.assert_called_with(
            Body=pd.concat([data1, data2], ignore_index=True).to_csv(index=False).encode(),
            Bucket='bucket_name',
            Key='key_prefix/test_data.csv',
        )

    @patch('sdgym.result_writer.parse_s3_path')
    def test_write_pickle(self, mockparse_s3_path):
        """Test the `write_pickle` method."""
        # Setup
        mock_s3_client = Mock()
        mockparse_s3_path.return_value = ('bucket_name', 'key_prefix/test_object.pkl')
        result_writer = S3ResultsWriter(mock_s3_client)
        obj = {'key': 'value'}

        # Run
        result_writer.write_pickle(obj, 'test_object.pkl')

        # Assert
        mockparse_s3_path.assert_called_once_with('test_object.pkl')
        mock_s3_client.put_object.assert_called_once_with(
            Body=cloudpickle.dumps(obj),
            Bucket='bucket_name',
            Key='key_prefix/test_object.pkl',
        )

    @patch('sdgym.result_writer.parse_s3_path')
    def test_write_yaml(self, mockparse_s3_path):
        """Test the `write_yaml` method."""
        # Setup
        mock_s3_client = Mock()
        mockparse_s3_path.return_value = ('bucket_name', 'key_prefix/test_data.yaml')
        result_writer = S3ResultsWriter(mock_s3_client)
        data = {'key': 'value'}

        # Run
        result_writer.write_yaml(data, 'test_data.yaml')

        # Assert
        mockparse_s3_path.assert_called_once_with('test_data.yaml')
        mock_s3_client.put_object.assert_called_once_with(
            Body=yaml.dump(data).encode(),
            Bucket='bucket_name',
            Key='key_prefix/test_data.yaml',
        )

    @patch('sdgym.result_writer.parse_s3_path')
    def test_write_yaml_append(self, mockparse_s3_path):
        """Test `write_yaml` with append mode."""
        # Setup
        mock_s3_client = Mock()
        mockparse_s3_path.return_value = ('bucket_name', 'key_prefix/test_data.yaml')
        data1 = {'key1': 'value1'}
        data2 = {'key2': 'value2'}
        mock_s3_client.get_object.return_value = {
            'Body': Mock(read=lambda: yaml.dump(data1).encode())
        }
        result_writer = S3ResultsWriter(mock_s3_client)

        # Run
        result_writer.write_yaml(data2, 'test_data.yaml', append=True)

        # Assert
        mockparse_s3_path.assert_called_with('test_data.yaml')
        mock_s3_client.get_object.assert_called_with(
            Bucket='bucket_name', Key='key_prefix/test_data.yaml'
        )
        expected_combined = {**data1, **data2}
        mock_s3_client.put_object.assert_called_with(
            Body=yaml.dump(expected_combined).encode(),
            Bucket='bucket_name',
            Key='key_prefix/test_data.yaml',
        )
