from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
from rdt import HyperTransformer
from sdv.metadata import Metadata

from sdgym.synthesizers.uniform import MultiTableUniformSynthesizer, UniformSynthesizer


class TestUniformSynthesizer:
    def test_uniform_synthesizer_sdtypes(self):
        """Ensure that sdtypes uniform are taken from metadata instead of inferred."""
        # Setup
        uniform_synthesizer = UniformSynthesizer()
        metadata = {
            'primary_key': 'guest_email',
            'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1',
            'columns': {
                'guest_email': {'sdtype': 'email', 'pii': True},
                'has_rewards': {'sdtype': 'boolean'},
                'room_type': {'sdtype': 'categorical'},
                'amenities_fee': {'sdtype': 'numerical', 'computer_representation': 'Float'},
                'checkin_date': {'sdtype': 'datetime', 'datetime_format': '%d %b %Y'},
                'checkout_date': {'sdtype': 'datetime', 'datetime_format': '%d %b %Y'},
                'room_rate': {'sdtype': 'numerical', 'computer_representation': 'Float'},
                'billing_address': {'sdtype': 'address', 'pii': True},
                'credit_card_number': {'sdtype': 'credit_card_number', 'pii': True},
            },
        }

        data = {
            'guest_email': {
                0: 'michaelsanders@shaw.net',
                1: 'randy49@brown.biz',
                2: 'webermelissa@neal.com',
                3: 'gsims@terry.com',
                4: 'misty33@smith.biz',
            },
            'has_rewards': {0: False, 1: False, 2: True, 3: False, 4: False},
            'room_type': {0: 'BASIC', 1: 'BASIC', 2: 'DELUXE', 3: 'BASIC', 4: 'BASIC'},
            'amenities_fee': {0: 37.89, 1: 24.37, 2: 0.0, 3: np.nan, 4: 16.45},
            'checkin_date': {
                0: '27 Dec 2020',
                1: '30 Dec 2020',
                2: '17 Sep 2020',
                3: '28 Dec 2020',
                4: '05 Apr 2020',
            },
            'checkout_date': {
                0: '29 Dec 2020',
                1: '02 Jan 2021',
                2: '18 Sep 2020',
                3: '31 Dec 2020',
                4: np.nan,
            },
            'room_rate': {0: 131.23, 1: 114.43, 2: 368.33, 3: 115.61, 4: 122.41},
            'billing_address': {
                0: '49380 Rivers Street\nSpencerville, AK 68265',
                1: '88394 Boyle Meadows\nConleyberg, TN 22063',
                2: '0323 Lisa Station Apt. 208\nPort Thomas, LA 82585',
                3: '77 Massachusetts Ave\nCambridge, MA 02139',
                4: '1234 Corporate Drive\nBoston, MA 02116',
            },
            'credit_card_number': {
                0: 4075084747483975747,
                1: 180072822063468,
                2: 38983476971380,
                3: 4969551998845740,
                4: 3558512986488983,
            },
        }

        real_data = pd.DataFrame(data)

        # Run
        synthesizer = uniform_synthesizer.get_trained_synthesizer(real_data, metadata)

        hyper_transformer_config = synthesizer.hyper_transformer.get_config()
        config_sdtypes = hyper_transformer_config['sdtypes']
        unknown_sdtypes = ['email', 'credit_card_number', 'address']
        for column in metadata['columns']:
            metadata_sdtype = metadata['columns'][column]['sdtype']
            # Only data types that are known are overridden by metadata
            if metadata_sdtype not in unknown_sdtypes:
                assert metadata_sdtype == config_sdtypes[column]
            else:
                assert config_sdtypes[column] == 'pii'


class TestMultiTableUniformSynthesizer:
    @patch('sdgym.synthesizers.uniform.MultiTableBaselineSynthesizer.__init__')
    def test__init__(self, mock_baseline_init):
        """Test the `__init__` method."""
        # Run
        synthesizer = MultiTableUniformSynthesizer()

        # Assert
        mock_baseline_init.assert_called_once()
        assert synthesizer.num_rows_per_table == {}

    @patch('sdgym.synthesizers.uniform.UniformSynthesizer._fit')
    def test__fit_mock(self, mock_uniform_fit):
        """Test the `fit` method with mocking."""
        # Setup
        synthesizer = MultiTableUniformSynthesizer()
        data = {
            'table1': pd.DataFrame({
                'col1': [1, 2, 3],
                'col2': ['A', 'B', 'C'],
            }),
            'table2': pd.DataFrame({
                'col3': [10.0, 20.0, 30.0],
                'col4': [True, False, True],
            }),
        }
        metadata = Mock()
        st_metadatas = [
            {
                'primary_key': 'col1',
                'columns': {
                    'col1': {'sdtype': 'numerical'},
                    'col2': {'sdtype': 'categorical'},
                },
            },
            {
                'primary_key': 'col3',
                'columns': {
                    'col3': {'sdtype': 'numerical'},
                    'col4': {'sdtype': 'boolean'},
                },
            },
        ]
        metadata.get_table_metadata.side_effect = st_metadatas

        # Run
        synthesizer._fit(data, metadata)

        # Assert
        metadata.get_table_metadata.assert_has_calls([
            call('table1'),
            call('table2'),
        ])
        mock_uniform_fit.assert_has_calls([
            call(data['table1'], st_metadatas[0]),
            call(data['table2'], st_metadatas[1]),
        ])
        assert synthesizer.num_rows_per_table == {
            'table1': 3,
            'table2': 3,
        }
        for table_name, table_synthesizer in synthesizer.table_synthesizers.items():
            assert table_name in ('table1', 'table2')
            assert isinstance(table_synthesizer, UniformSynthesizer)

    def test__get_trained_synthesizer(self):
        """Test the `_get_trained_synthesizer` method."""
        # Setup
        synthesizer = MultiTableUniformSynthesizer()
        data = {
            'table1': pd.DataFrame({
                'col1': [1, 2, 3, 4, 5],
                'col2': ['A', 'B', 'C', 'D', 'E'],
            }),
            'table2': pd.DataFrame({
                'col3': [10.0, 20.0, 30.0],
                'col4': [True, False, True],
            }),
        }
        metadata = Metadata.load_from_dict({
            'tables': {
                'table1': {
                    'columns': {
                        'col1': {'sdtype': 'numerical'},
                        'col2': {'sdtype': 'categorical'},
                    },
                    'primary_key': 'col1',
                },
                'table2': {
                    'columns': {
                        'col3': {'sdtype': 'numerical'},
                        'col4': {'sdtype': 'boolean'},
                    },
                    'primary_key': 'col3',
                },
            },
            'relationships': [],
        })

        # Run
        trained_synthesizer = synthesizer._get_trained_synthesizer(data, metadata)

        # Assert
        assert trained_synthesizer.num_rows_per_table == {
            'table1': 5,
            'table2': 3,
        }
        assert set(trained_synthesizer.table_synthesizers.keys()) == {'table1', 'table2'}
        for table_name, table_synthesizer in trained_synthesizer.table_synthesizers.items():
            hyper_transformer = table_synthesizer.hyper_transformer
            transformed = table_synthesizer.transformed_data
            assert isinstance(hyper_transformer, HyperTransformer)
            assert isinstance(transformed, pd.DataFrame)
            assert set(transformed.columns) == set(data[table_name].columns)

    @patch('sdgym.synthesizers.uniform.UniformSynthesizer.sample_from_synthesizer')
    def test_sample_from_synthesizer_mock(self, mock_sample_from_synthesizer):
        """Test the `sample_from_synthesizer` method with mocking."""
        # Setup
        synthesizer = MultiTableUniformSynthesizer()
        trained_synthesizer = MultiTableUniformSynthesizer()
        trained_synthesizer.num_rows_per_table = {
            'table1': 3,
            'table2': 2,
        }
        synthesizer_table1 = Mock()
        synthesizer_table2 = Mock()
        trained_synthesizer.table_synthesizers = {
            'table1': synthesizer_table1,
            'table2': synthesizer_table2,
        }
        mock_sample_from_synthesizer.side_effect = [
            'sampled_data_table1',
            'sampled_data_table2',
        ]
        scale = 2

        # Run
        sampled_data = synthesizer.sample_from_synthesizer(trained_synthesizer, scale=scale)

        # Assert
        assert sampled_data == {
            'table1': 'sampled_data_table1',
            'table2': 'sampled_data_table2',
        }
        mock_sample_from_synthesizer.assert_has_calls([
            call(synthesizer_table1, n_samples=6),
            call(synthesizer_table2, n_samples=4),
        ])

    def test_sample_from_synthesizer(self):
        """Test the `sample_from_synthesizer` method."""
        # Setup
        np.random.seed(0)
        synthesizer = MultiTableUniformSynthesizer()
        data = {
            'table1': pd.DataFrame({
                'col1': [1, 2, 3, 4, 5],
                'col2': ['A', 'B', 'C', 'D', 'E'],
            }),
            'table2': pd.DataFrame({
                'col3': [10, 20, 30],
                'col4': [True, False, True],
            }),
        }
        table_1 = UniformSynthesizer()
        table_1._fit(
            data['table1'],
            Metadata.load_from_dict({
                'columns': {
                    'col1': {'sdtype': 'numerical'},
                    'col2': {'sdtype': 'categorical'},
                },
            }),
        )
        table_2 = UniformSynthesizer()
        table_2._fit(
            data['table2'],
            Metadata.load_from_dict({
                'columns': {
                    'col3': {'sdtype': 'numerical'},
                    'col4': {'sdtype': 'boolean'},
                },
            }),
        )
        trained_synthesizer = MultiTableUniformSynthesizer()

        trained_synthesizer.table_synthesizers = {
            'table1': table_1,
            'table2': table_2,
        }
        trained_synthesizer.num_rows_per_table = {
            'table1': 5,
            'table2': 3,
        }
        scale = 2
        expected_data = {
            'table1': pd.DataFrame({
                'col1': [5, 1, 4, 4, 4, 2, 4, 3, 5, 1],
                'col2': ['A', 'E', 'C', 'B', 'A', 'B', 'B', 'A', 'B', 'E'],
            }),
            'table2': pd.DataFrame({
                'col3': [29, 26, 29, 15, 25, 25],
                'col4': [True, True, False, True, False, False],
            }),
        }

        # Run
        sampled_data = synthesizer.sample_from_synthesizer(trained_synthesizer, scale=scale)

        # Assert
        for table_name, table_data in sampled_data.items():
            pd.testing.assert_frame_equal(
                table_data,
                expected_data[table_name],
            )
