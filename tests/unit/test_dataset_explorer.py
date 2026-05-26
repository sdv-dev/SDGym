import io
import json
import re
from collections import defaultdict
from unittest.mock import MagicMock, patch
from zipfile import ZipFile

import pandas as pd
import pytest

from sdgym.dataset_explorer import SUMMARY_OUTPUT_COLUMNS, DatasetExplorer
from sdgym.datasets import SDV_DATASETS_PUBLIC_BUCKET


def _make_zipped_csv(csvs):
    buffer = io.BytesIO()
    with ZipFile(buffer, 'w') as zip_file:
        for file_name, contents in csvs.items():
            zip_file.writestr(file_name, contents)

    return buffer.getvalue()


def _make_s3_client_with_download(data):
    s3_client = MagicMock()

    def download_fileobj(Bucket, Key, Fileobj):
        Fileobj.write(data)

    s3_client.download_fileobj.side_effect = download_fileobj
    return s3_client


class TestDatasetExplorer:
    def test___init__default(self):
        """Test the ``__init__`` method with default parameters."""
        # Run
        explorer = DatasetExplorer()

        # Assert
        assert explorer.s3_url == SDV_DATASETS_PUBLIC_BUCKET
        assert explorer.aws_access_key_id is None
        assert explorer.aws_secret_access_key is None

    @patch('sdgym.dataset_explorer._get_s3_client')
    def test___init__with_parameters(self, mock_s3_client):
        """Test the ``__init__`` method with custom parameters."""
        # Setup
        mock_s3_client.return_value = 's3_client'

        # Run
        explorer = DatasetExplorer(
            s3_url='s3://custom-bucket',
            aws_access_key_id='key123',
            aws_secret_access_key='secret456',
        )

        # Assert
        assert explorer.s3_url == 's3://custom-bucket'
        assert explorer.aws_access_key_id == 'key123'
        assert explorer.aws_secret_access_key == 'secret456'
        assert explorer.s3_client == 's3_client'
        mock_s3_client.assert_called_once_with('s3://custom-bucket', 'key123', 'secret456')

    def test__get_max_schema_branch_factor(self):
        """Test the ``_get_max_schema_branch_factor`` method."""
        # Setup
        relationships = [
            {'parent_table_name': 'A', 'child_table_name': 'B'},
            {'parent_table_name': 'A', 'child_table_name': 'C'},
            {'parent_table_name': 'B', 'child_table_name': 'D'},
        ]

        # Run
        result = DatasetExplorer._get_max_schema_branch_factor(relationships)

        # Assert
        assert result == 2

    def test__get_max_schema_branch_factor_repeated_parent(self):
        """Test the ``_get_max_schema_branch_factor`` method with repeated children."""
        # Setup
        relationships = [
            {'parent_table_name': 'A', 'child_table_name': 'B'},
            {'parent_table_name': 'A', 'child_table_name': 'C'},
            {'parent_table_name': 'A', 'child_table_name': 'C'},
            {'parent_table_name': 'A', 'child_table_name': 'C'},
            {'parent_table_name': 'B', 'child_table_name': 'D'},
        ]

        # Run
        result = DatasetExplorer._get_max_schema_branch_factor(relationships)

        # Assert
        assert result == 4

    def test__get_max_depth(self):
        """Test the ``_get_max_depth`` method with simple parent-child structure."""
        # Setup
        metadata = MagicMock()
        metadata._get_child_map.return_value = defaultdict(set, {'hotels': {'guests'}})
        metadata._get_parent_map.return_value = defaultdict(set, {'guests': {'hotels'}})

        # Run
        result = DatasetExplorer._get_max_depth(metadata)

        # Assert
        assert result == 2

    def test__get_max_depth_detailed_schema(self):
        # Setup
        metadata = MagicMock()
        metadata._get_parent_map.return_value = defaultdict(
            set,
            {
                'account': {'district'},
                'card': {'disp'},
                'client': {'district'},
                'disp': {'account', 'client'},
                'loan': {'account'},
                'order': {'account'},
                'trans': {'account'},
            },
        )
        metadata._get_child_map.return_value = defaultdict(
            set,
            {
                'district': {'account', 'client'},
                'disp': {'card'},
                'client': {'disp'},
                'account': {'disp', 'loan', 'order', 'trans'},
            },
        )

        # Run
        result = DatasetExplorer._get_max_depth(metadata)

        # Assert
        assert result == 4

    def test__summarize_metadata_columns(self):
        """Test the ``_summarize_metadata_columns`` method."""
        # Setup
        metadata = MagicMock()
        metadata.tables = {'table1': MagicMock(), 'table2': MagicMock()}
        metadata.tables['table1'].columns = {
            'id': {'sdtype': 'id'},
            'another_id': {'sdtype': 'id'},
            'category': {'sdtype': 'categorical'},
            'value': {'sdtype': 'numerical'},
        }
        metadata.tables['table2'].columns = {
            'date': {'sdtype': 'datetime'},
            'pii_col': {'sdtype': 'pii'},
            'address': {'sdtype': 'pii'},
        }
        metadata.tables['table1'].primary_key = 'id'
        metadata.tables['table2'].primary_key = 'pii_col'

        # Run
        result = DatasetExplorer._summarize_metadata_columns(metadata)

        # Assert
        assert result['Total_Num_Columns'] == 7
        assert result['Total_Num_Columns_Categorical'] == 1
        assert result['Total_Num_Columns_Numerical'] == 1
        assert result['Total_Num_Columns_Datetime'] == 1
        assert result['Total_Num_Columns_Datetime'] == 1
        assert result['Total_Num_Columns_ID_NonKey'] == 1

    @patch('sdgym.dataset_explorer.Metadata.load_from_dict')
    def test_get_metadata_summary_with_dict(self, mock_load_from_dict):
        """Ensure that the ``get_metadata_summary`` method converts metadata dict to Metadata."""
        # Setup
        mock_metadata = MagicMock()
        mock_metadata.relationships = []
        mock_metadata.tables = {}
        mock_metadata._get_child_map.return_value = {}
        mock_load_from_dict.return_value = mock_metadata

        # Run
        result = DatasetExplorer.get_metadata_summary({})

        # Assert
        mock_load_from_dict.assert_called_once()
        assert isinstance(result, dict)

    def test_get_data_summary_with_multiple_tables(self):
        """Test the ``get_data_summary`` method with multiple tables."""
        # Setup
        data = {'table1': pd.DataFrame({'a': [1, 2, 3]}), 'table2': pd.DataFrame({'b': [4, 5]})}

        # Run
        result = DatasetExplorer.get_data_summary(data)

        # Assert
        assert result['Total_Num_Rows'] == 5
        assert result['Max_Num_Rows_Per_Table'] == 3

    def test__load_metadata_from_s3(self):
        """Test loading only the dataset metadata JSON from S3."""
        # Setup
        explorer = DatasetExplorer()
        s3_client = MagicMock()
        metadata = {'tables': {'table': {'columns': {'col': {'sdtype': 'id'}}}}}
        s3_client.get_object.return_value = {
            'Body': io.BytesIO(json.dumps(metadata).encode('utf-8'))
        }

        # Run
        result = explorer._load_metadata_from_s3(s3_client, 'single_table', 'my_dataset')

        # Assert
        assert result == metadata
        s3_client.get_object.assert_called_once_with(
            Bucket='sdv-datasets-public',
            Key='single_table/my_dataset/metadata.json',
        )

    def test__get_s3_zipped_data_summary_single_table(self):
        """Test counting rows in a single-table zip from S3."""
        # Setup
        explorer = DatasetExplorer()
        zip_data = _make_zipped_csv({'table.csv': 'a,b\n1,2\n3,4\n5,6\n'})
        s3_client = _make_s3_client_with_download(zip_data)

        # Run
        result = explorer._get_s3_zipped_data_summary(s3_client, 'single_table', 'my_dataset')

        # Assert
        assert result == {'Total_Num_Rows': 3, 'Max_Num_Rows_Per_Table': 3}
        s3_client.download_fileobj.assert_called_once()
        assert s3_client.download_fileobj.call_args.args[:2] == (
            'sdv-datasets-public',
            'single_table/my_dataset/data.zip',
        )

    def test__get_s3_zipped_data_summary_multi_table(self):
        """Test counting rows in a multi-table zip from S3."""
        # Setup
        explorer = DatasetExplorer()
        zip_data = _make_zipped_csv({
            'users.csv': 'id,name\n1,a\n2,b\n3,c\n',
            'transactions.csv': 'id,user_id\n1,1\n2,1\n',
        })
        s3_client = _make_s3_client_with_download(zip_data)

        # Run
        result = explorer._get_s3_zipped_data_summary(s3_client, 'multi_table', 'my_dataset')

        # Assert
        assert result == {'Total_Num_Rows': 5, 'Max_Num_Rows_Per_Table': 3}

    def test__get_s3_zipped_data_summary_counts_multiline_csv_records(self):
        """Test quoted multiline values do not inflate row counts."""
        # Setup
        explorer = DatasetExplorer()
        zip_data = _make_zipped_csv({
            'table.csv': 'id,text\n1,"hello\nthere"\n2,plain\n',
        })
        s3_client = _make_s3_client_with_download(zip_data)

        # Run
        result = explorer._get_s3_zipped_data_summary(s3_client, 'single_table', 'my_dataset')

        # Assert
        assert result == {'Total_Num_Rows': 2, 'Max_Num_Rows_Per_Table': 2}

    def test__validate_output_filepath_valid(self, tmp_path):
        """Test the ``_validate_output_filepath`` method with valid CSV path."""
        # Setup
        explorer = DatasetExplorer()

        # Run and Assert
        explorer._validate_output_filepath(tmp_path / 'output.csv')

    def test__validate_output_filepath_invalid(self):
        """Test the ``_validate_output_filepath`` method with invalid file path."""
        # Setup
        explorer = DatasetExplorer()

        # Run and Assert
        expected_msg = "The 'output_filepath' has to be a .csv file, provided: 'output.txt'."
        with pytest.raises(ValueError, match=expected_msg):
            explorer._validate_output_filepath('output.txt')

    def test__validate_output_filepath_existing_file(self, tmp_path):
        """Test the ``_validate_output_filepath`` method when file already exists."""
        # Setup
        explorer = DatasetExplorer()
        existing_file = tmp_path / 'exists.csv'
        existing_file.write_text('already here')

        # Run and Assert
        expected_msg = re.escape(
            f"The file '{existing_file}' already exists. Please provide a new 'output_filepath'."
        )
        with pytest.raises(ValueError, match=expected_msg):
            explorer._validate_output_filepath(str(existing_file))

    def test___init__valid_s3_bucket_urls(self):
        """DatasetExplorer initializes with valid bucket-only S3 URLs."""
        # Run and Assert
        explorer = DatasetExplorer(s3_url='s3://my-bucket')
        assert explorer._bucket_name == 'my-bucket'

        # Run and Assert
        explorer = DatasetExplorer(s3_url='s3://my-bucket/')
        assert explorer._bucket_name == 'my-bucket'

    @pytest.mark.parametrize(
        'value',
        [
            None,
            123,
            12.34,
            '',
            's3://',
            's3:///',
            's3:///path-only',
            's3://my-bucket/path',
            's3://my-bucket/folder/',
            's3://my-bucket?query=1',
            's3://my-bucket#frag',
            'http://my-bucket',
            'https://my-bucket',
            'ftp://my-bucket',
        ],
    )
    def test___init__invalid_s3_bucket_urls(self, value):
        """DatasetExplorer rejects invalid S3 URLs during initialization."""
        err_msg = re.escape(
            f"The provided s3_url parameter ('{value}') is not a valid S3 URL.\n"
            "Please provide a string that starts with 's3://' and refers to a AWS bucket."
        )
        with pytest.raises(ValueError, match=err_msg):
            DatasetExplorer(s3_url=value)

    @patch('sdgym.dataset_explorer.get_s3_client')
    @patch('sdgym.dataset_explorer.DatasetExplorer._get_s3_zipped_data_summary')
    @patch('sdgym.dataset_explorer.DatasetExplorer.get_metadata_summary')
    @patch('sdgym.dataset_explorer.DatasetExplorer._load_metadata_from_s3')
    @patch('sdgym.dataset_explorer._get_available_datasets')
    def test__load_and_summarize_datasets(
        self,
        mock_get_datasets,
        mock_load_metadata,
        mock_get_metadata_summary,
        mock_get_data_summary,
        mock_get_s3_client,
    ):
        """Test the ``_load_and_summarize_datasets`` method."""
        # Setup
        explorer = DatasetExplorer()
        mock_get_s3_client.return_value = 's3_client'
        mock_get_datasets.return_value = pd.DataFrame([
            {'dataset_name': 'test', 'size_MB': 10, 'num_tables': 2}
        ])
        mock_load_metadata.return_value = {'tables': {}, 'relationships': []}
        mock_get_metadata_summary.return_value = {
            'Total_Num_Columns': 1,
            'Total_Num_Columns_Categorical': 0,
            'Total_Num_Columns_Numerical': 1,
            'Total_Num_Columns_Datetime': 0,
            'Total_Num_Columns_PII': 0,
            'Total_Num_Columns_ID_NonKey': 0,
            'Max_Num_Columns_Per_Table': 1,
            'Num_Relationships': 0,
            'Max_Schema_Depth': 1,
            'Max_Schema_Branch': 0,
        }
        mock_get_data_summary.return_value = {
            'Total_Num_Rows': 3,
            'Max_Num_Rows_Per_Table': 3,
        }

        # Run
        result = explorer._load_and_summarize_datasets('single_table')

        # Assert
        mock_get_datasets.assert_called_once_with(
            modality='single_table',
            bucket='sdv-datasets-public',
            s3_client='s3_client',
        )
        mock_load_dataset.assert_called_once_with(
            modality='single_table',
            dataset_name='test',
            bucket='sdv-datasets-public',
            s3_client=None,
        )
        assert isinstance(result, list)
        assert 'Dataset' in result[0]
        assert set(result[0]) == set(SUMMARY_OUTPUT_COLUMNS)

    @patch('sdgym.dataset_explorer.get_s3_client')
    @patch('sdgym.dataset_explorer.DatasetExplorer._get_s3_zipped_data_summary')
    @patch('sdgym.dataset_explorer.DatasetExplorer.get_metadata_summary')
    @patch('sdgym.dataset_explorer.DatasetExplorer._load_metadata_from_s3')
    @patch('sdgym.dataset_explorer._get_available_datasets')
    def test__load_and_summarize_datasets_with_datasets(
        self,
        mock_get_datasets,
        mock_load_metadata,
        mock_get_metadata_summary,
        mock_get_data_summary,
        mock_get_s3_client,
    ):
        """Test that ``_load_and_summarize_datasets`` restricts to the provided dataset list."""
        # Setup
        explorer = DatasetExplorer()
        mock_get_s3_client.return_value = 's3_client'
        mock_get_datasets.return_value = pd.DataFrame([
            {'dataset_name': 'ds1', 'size_MB': 10, 'num_tables': 1},
            {'dataset_name': 'ds2', 'size_MB': 20, 'num_tables': 2},
            {'dataset_name': 'ds3', 'size_MB': 30, 'num_tables': 3},
        ])
        mock_load_metadata.return_value = {'tables': {}, 'relationships': []}
        mock_get_metadata_summary.return_value = {
            'Total_Num_Columns': 1,
            'Total_Num_Columns_Categorical': 0,
            'Total_Num_Columns_Numerical': 1,
            'Total_Num_Columns_Datetime': 0,
            'Total_Num_Columns_PII': 0,
            'Total_Num_Columns_ID_NonKey': 0,
            'Max_Num_Columns_Per_Table': 1,
            'Num_Relationships': 0,
            'Max_Schema_Depth': 1,
            'Max_Schema_Branch': 0,
        }
        mock_get_data_summary.return_value = {
            'Total_Num_Rows': 2,
            'Max_Num_Rows_Per_Table': 2,
        }

        # Run
        results = explorer._load_and_summarize_datasets('single_table', datasets=['ds1', 'ds3'])

        # Assert
        assert mock_load_dataset.call_count == 2
        loaded_names = [call.kwargs['dataset_name'] for call in mock_load_dataset.call_args_list]
        assert set(loaded_names) == {'ds1', 'ds3'}
        assert [result['Dataset'] for result in results] == ['ds1', 'ds3']
        assert len(results) == 2

    @patch('sdgym.dataset_explorer._validate_modality')
    @patch('sdgym.dataset_explorer.DatasetExplorer._load_and_summarize_datasets')
    def test_summarize_datasets_with_output(
        self, mock_load_and_summarize, mock_validate_modality, tmp_path
    ):
        """Test the ``summarize_datasets`` method with output CSV path."""
        # Setup
        explorer = DatasetExplorer()
        mock_load_and_summarize.return_value = [
            {
                'Dataset': 'test',
                'Datasize_Size_MB': 5.0,
                'Num_Tables': 20,
                'Total_Num_Columns': 2503,
                'Total_Num_Columns_Categorical': 10,
                'Total_Num_Columns_Numerical': 20,
                'Total_Num_Columns_Datetime': 30,
                'Total_Num_Columns_PII': 40,
                'Total_Num_Columns_ID_NonKey': 50,
                'Max_Num_Columns_Per_Table': 60,
                'Total_Num_Rows': 1240,
                'Max_Num_Rows_Per_Table': 1240,
                'Num_Relationships': 250,
                'Max_Schema_Depth': 2105,
                'Max_Schema_Branch': 510,
            }
        ]
        output_filepath = tmp_path / 'summary.csv'

        # Run
        df = explorer.summarize_datasets('single_table', output_filepath=str(output_filepath))

        # Assert
        mock_validate_modality.assert_called_once_with('single_table')
        explorer._load_and_summarize_datasets.assert_called_once_with('single_table')
        assert output_filepath.exists()
        assert isinstance(df, pd.DataFrame)
        assert df.columns.to_list() == SUMMARY_OUTPUT_COLUMNS

    @patch('sdgym.dataset_explorer._validate_modality')
    @patch('sdgym.dataset_explorer._get_available_datasets')
    def test_list_datasets_without_output(self, mock_get_available, mock_validate_modality):
        """Test that `list_datasets` returns the expected dataframe."""
        # Setup
        explorer = DatasetExplorer()
        expected_df = pd.DataFrame([
            {'dataset_name': 'ds1', 'size_MB': 12.5, 'num_tables': 1},
            {'dataset_name': 'ds2', 'size_MB': 3.0, 'num_tables': 2},
        ])
        mock_get_available.return_value = expected_df

        # Run
        result = explorer.list_datasets('single_table')

        # Assert
        mock_validate_modality.assert_called_once_with('single_table')
        mock_get_available.assert_called_once_with(
            modality='single_table',
            bucket='sdv-datasets-public',
            s3_client=None,
        )
        pd.testing.assert_frame_equal(result, expected_df)

    @patch('sdgym.dataset_explorer._validate_modality')
    @patch('sdgym.dataset_explorer._get_available_datasets')
    def test_list_datasets_with_output(self, mock_get_available, mock_validate_modality, tmp_path):
        """Test that `list_datasets` writes CSV when output path is provided."""
        # Setup
        explorer = DatasetExplorer()
        expected_df = pd.DataFrame([
            {'dataset_name': 'alpha', 'size_MB': 1.5, 'num_tables': 1},
            {'dataset_name': 'beta', 'size_MB': 2.0, 'num_tables': 3},
        ])
        mock_get_available.return_value = expected_df
        output_filepath = tmp_path / 'datasets_list.csv'

        # Run
        result = explorer.list_datasets('multi_table', output_filepath=str(output_filepath))

        # Assert
        mock_validate_modality.assert_called_once_with('multi_table')
        mock_get_available.assert_called_once_with(
            modality='multi_table',
            bucket='sdv-datasets-public',
            s3_client=None,
        )
        assert output_filepath.exists()
        loaded = pd.read_csv(output_filepath)
        pd.testing.assert_frame_equal(loaded, expected_df)
        pd.testing.assert_frame_equal(result, expected_df)
