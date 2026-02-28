import zipfile
from unittest.mock import ANY, MagicMock, Mock, call, patch

import cloudpickle
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

    def test_write_zipped_dataframes(self, tmp_path):
        """Test the `write_zipped_dataframes` method."""
        # Setup
        base_path = tmp_path / 'sdgym_results'
        base_path.mkdir(parents=True, exist_ok=True)
        result_writer = LocalResultsWriter()
        file_path = base_path / 'data.zip'

        data = {
            'table1': pd.DataFrame({'a': [1, 2], 'b': [3, 4]}),
            'table2': pd.DataFrame({'x': [5, 6], 'y': [7, 8]}),
        }

        # Run
        result_writer.write_zipped_dataframes(data, file_path)

        # Assert
        assert file_path.exists()

        with zipfile.ZipFile(file_path, 'r') as zf:
            # Check that all tables are present
            names = zf.namelist()
            assert 'table1.csv' in names
            assert 'table2.csv' in names

            # Check each table content matches the original
            for table_name, df in data.items():
                with zf.open(f'{table_name}.csv') as f:
                    loaded_df = pd.read_csv(f)
                    pd.testing.assert_frame_equal(df, loaded_df)

    @patch('sdgym.result_writer.Path.exists')
    @patch('sdgym.result_writer.load_workbook')
    @patch('sdgym.result_writer.pd.ExcelWriter')
    @patch('sdgym.result_writer._set_column_width')
    @patch('sdgym.result_writer.pd.DataFrame.to_excel', autospec=True)
    def test_write_xlsx(
        self,
        mock_to_excel,
        mock_set_column_width,
        mock_excel_writer,
        mock_load_workbook,
        mock_exists,
        tmp_path,
    ):
        """Test the `write_xlsx` method."""
        # Setup
        mock_exists.return_value = False
        file_path = tmp_path / 'test.xlsx'
        df1 = pd.DataFrame({'A': [1, 2]})
        df2 = pd.DataFrame({'B': [3, 4]})
        data = {'SheetA': df1, 'SheetB': df2}
        mock_writer = mock_excel_writer.return_value
        mock_writer.__enter__.return_value = mock_writer
        mock_writer.__exit__.return_value = None

        mock_ws_a = Mock()
        mock_ws_b = Mock()
        mock_wb = MagicMock()

        def get_sheet(name):
            sheets = {'SheetA': mock_ws_a, 'SheetB': mock_ws_b}
            return sheets[name]

        mock_wb.__getitem__.side_effect = get_sheet
        mock_load_workbook.return_value = mock_wb
        result_writer = LocalResultsWriter()

        # Run
        result_writer.write_xlsx(data, file_path)

        # Assert
        _, kwargs = mock_excel_writer.call_args
        assert kwargs['mode'] == 'w'
        assert kwargs['engine'] == 'openpyxl'
        mock_to_excel.assert_has_calls(
            [
                call(ANY, mock_writer, sheet_name='SheetA', index=False),
                call(ANY, mock_writer, sheet_name='SheetB', index=False),
            ],
            any_order=False,
        )
        assert mock_to_excel.call_args_list[0].args[0] is df1
        assert mock_to_excel.call_args_list[1].args[0] is df2
        mock_set_column_width.assert_has_calls([
            call(mock_writer, df1, 'SheetA'),
            call(mock_writer, df2, 'SheetB'),
        ])
        assert mock_wb._sheets.remove.call_count == 2
        assert mock_wb._sheets.insert.call_count == 2
        mock_wb.save.assert_called_once_with(file_path)


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

    @patch('sdgym.result_writer.parse_s3_path')
    @patch('sdgym.result_writer.io.StringIO')
    def write_zipped_dataframes(self, mock_string_io, mockparse_s3_path):
        """Test the `write_zipped_dataframes` method."""
        # Setup
        mock_s3_client = Mock()
        mockparse_s3_path.return_value = ('bucket_name', 'key_prefix/test_data.zip')
        result_writer = S3ResultsWriter(mock_s3_client)
        df1 = pd.DataFrame({'col1': [1, 2]})
        df2 = pd.DataFrame({'colA': ['x', 'y']})
        df1.to_csv = Mock(return_value='csv1')
        df2.to_csv = Mock(return_value='csv2')
        data = {'table1': df1, 'table2': df2}
        mock_string_io.side_effect = ['buffer1', 'buffer2']

        # Run
        result_writer.write_zipped_dataframes(data, 'test_data.zip')

        # Assert
        mockparse_s3_path.assert_called_once_with('test_data.zip')
        mock_s3_client.upload_fileobj.assert_called_once()
        df1.to_csv.assert_called_once_with('buffer1', index=False)
        df2.to_csv.assert_called_once_with('buffer2', index=False)
        args, _ = mock_s3_client.upload_fileobj.call_args
        uploaded_buffer = args[0]
        uploaded_buffer.seek(0)
        with zipfile.ZipFile(uploaded_buffer, 'r') as zf:
            assert set(zf.namelist()) == {'table1.csv', 'table2.csv'}
