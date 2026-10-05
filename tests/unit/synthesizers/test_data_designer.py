"""Tests for the Data Designer integration."""

import sys
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from sdv.metadata import Metadata

from sdgym.synthesizers.data_designer import (
    PII_PLACEHOLDER_PREFIX,
    DataDesignerSynthesizer,
    _attach_missing_proportion,
    _import_data_designer,
    create_data_designer_config,
    get_missing_value_proportions,
)

dd = pytest.importorskip('data_designer.config')


@pytest.fixture
def metadata():
    return Metadata.load_from_dict({
        'tables': {
            'hotels': {
                'primary_key': 'guest_id',
                'columns': {
                    'guest_id': {'sdtype': 'id', 'regex_format': 'G[0-9]{4}'},
                    'guest_email': {'sdtype': 'email', 'pii': True},
                    'has_rewards': {'sdtype': 'boolean'},
                    'room_type': {'sdtype': 'categorical'},
                    'num_guests': {'sdtype': 'numerical'},
                    'amenities_fee': {'sdtype': 'numerical'},
                    'checkin_date': {'sdtype': 'datetime', 'datetime_format': '%d %b %Y'},
                    'checkout_ts': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d %H:%M:%S'},
                    'review': {'sdtype': 'text'},
                    'notes': {'sdtype': 'unknown', 'pii': True},
                },
            }
        }
    })


@pytest.fixture
def data():
    return pd.DataFrame({
        'guest_id': ['G0001', 'G0002', 'G0003', 'G0004', 'G0005'],
        'guest_email': ['a@x.com', 'b@x.com', 'c@x.com', 'd@x.com', 'e@x.com'],
        'has_rewards': [False, False, True, False, None],
        'room_type': ['BASIC', 'BASIC', 'DELUXE', 'BASIC', 'SUITE'],
        'num_guests': [1, 2, 2, 4, np.nan],
        'amenities_fee': [37.89, 24.37, 0.0, np.nan, 16.45],
        'checkin_date': ['27 Dec 2020', '30 Dec 2020', '17 Sep 2020', '28 Dec 2020', '05 Apr 2020'],
        'checkout_ts': [
            '2020-12-29 10:30:00',
            '2021-01-02 11:00:00',
            '2020-09-18 09:15:00',
            '2020-12-31 12:00:00',
            None,
        ],
        'review': ['Great stay.', 'Room was small.', None, 'Loved the pool!', 'Noisy.'],
        'notes': ['x', 'y', 'x', 'x', 'z'],
    })


@pytest.fixture
def v2_metadata():
    """Metadata V2 with range hints that disagree with the data on purpose."""
    return Metadata.load_from_dict({
        'tables': {
            'guests': {
                'columns': {
                    'room_type': {
                        'sdtype': 'categorical',
                        'range_is_nullable': False,
                        'range_values': ['BASIC', 'DELUXE', 'SUITE'],
                    },
                    'amenities_fee': {
                        'sdtype': 'numerical',
                        'range_is_nullable': True,
                        'range_min': 0.0,
                        'range_max': 46.64,
                        'decimal_places': 2,
                    },
                    'num_guests': {
                        'sdtype': 'numerical',
                        'range_min': 1,
                        'range_max': 6,
                        'decimal_places': 0,
                    },
                    'checkin_date': {
                        'sdtype': 'datetime',
                        'datetime_format': '%d %b %Y',
                        'range_is_nullable': False,
                        'range_min': '03 Jan 2020',
                        'range_max': '05 Jan 2021',
                    },
                    'has_rewards': {'sdtype': 'boolean', 'range_is_nullable': False},
                }
            }
        },
        'METADATA_SPEC_VERSION': 'V2',
    })


@pytest.fixture
def v2_data():
    return pd.DataFrame({
        'room_type': ['BASIC', 'BASIC', 'DELUXE', 'BASIC'],
        'amenities_fee': [10.5, np.nan, 20.25, np.nan],
        'num_guests': [2.5, 3.0, 2.0, 4.0],
        'checkin_date': ['10 Feb 2020', '11 Mar 2020', '12 Apr 2020', '13 May 2020'],
        'has_rewards': [True, False, None, False],
    })


def _get_column(builder, name):
    return builder.get_column_config(name)


def test_import_error_when_sdk_missing():
    """Test a helpful ``ImportError`` is raised when the package is not installed."""
    # Setup
    missing = {'data_designer': None, 'data_designer.config': None}

    # Run and Assert
    with patch.dict(sys.modules, missing):
        with pytest.raises(ImportError, match=r"pip install sdgym\['data_designer'\]"):
            _import_data_designer()


@patch('sdgym.synthesizers.data_designer.sys')
def test_import_error_on_unsupported_python(sys_mock, data, metadata):
    """Test an error is raised when running on python older than 3.10."""
    # Setup
    sys_mock.version_info = (3, 9, 18)

    # Run and Assert
    with pytest.raises(ImportError, match='DataDesigner only supports python >= 3.10'):
        _import_data_designer()

    with pytest.raises(ImportError, match='DataDesigner only supports python >= 3.10'):
        create_data_designer_config(data, metadata)


@patch('sdgym.synthesizers.data_designer.sys')
def test_import_succeeds_on_supported_python(sys_mock):
    """Test the minimum supported python version is accepted."""
    # Setup
    sys_mock.version_info = (3, 10, 0)

    # Run
    module = _import_data_designer()

    # Assert
    assert module is dd


def test_returns_builder_with_one_config_per_column(data, metadata):
    """Test every metadata column gets a config and the builder validates."""
    # Run
    builder = create_data_designer_config(data, metadata)

    # Assert
    assert isinstance(builder, dd.DataDesignerConfigBuilder)
    names = [config.name for config in builder.get_column_configs()]
    assert names == list(metadata.tables['hotels'].columns)
    assert builder.build().columns


def test_categorical_maps_to_weighted_category_sampler(data, metadata):
    """Test categorical columns use a ``category`` sampler with observed frequencies."""
    # Run
    column = _get_column(create_data_designer_config(data, metadata), 'room_type')

    # Assert
    assert column.sampler_type == dd.SamplerType.CATEGORY
    assert column.params.values == ['BASIC', 'DELUXE', 'SUITE']  # noqa: PD011
    np.testing.assert_allclose(column.params.weights, [0.6, 0.2, 0.2])


def test_categorical_values_are_python_scalars():
    """Test numpy scalars in categorical columns are converted to python values."""
    # Setup
    data = pd.DataFrame({'room_type': pd.Series([1, 2, 2], dtype='int64')})
    metadata = Metadata.load_from_dict({
        'tables': {'t': {'columns': {'room_type': {'sdtype': 'categorical'}}}}
    })

    # Run
    column = _get_column(create_data_designer_config(data, metadata), 'room_type')

    # Assert
    assert column.params.values == [2, 1]  # noqa: PD011
    assert all(type(value) is int for value in column.params.values)  # noqa: PD011


def test_all_null_categorical_uses_placeholder():
    """Test a categorical column without observed values falls back to a placeholder."""
    # Setup
    data = pd.DataFrame({'empty': [None, None]})
    metadata = Metadata.load_from_dict({
        'tables': {'t': {'columns': {'empty': {'sdtype': 'categorical'}}}}
    })

    # Run
    column = _get_column(create_data_designer_config(data, metadata), 'empty')

    # Assert
    assert column.sampler_type == dd.SamplerType.UUID
    assert column.params.prefix == PII_PLACEHOLDER_PREFIX
    assert column.params.short_form is True


def test_other_sdtypes_fall_back_to_category_sampler(data, metadata):
    """Test pii and unknown sdtypes are sampled from the observed values."""
    # Run
    builder = create_data_designer_config(data, metadata)

    # Assert
    email = _get_column(builder, 'guest_email')
    assert email.sampler_type == dd.SamplerType.CATEGORY
    assert set(email.params.values) == set(data['guest_email'])  # noqa: PD011

    notes = _get_column(builder, 'notes')
    assert notes.sampler_type == dd.SamplerType.CATEGORY
    assert notes.params.values == ['x', 'y', 'z']  # noqa: PD011
    np.testing.assert_allclose(notes.params.weights, [0.6, 0.2, 0.2])


def test_boolean_maps_to_bernoulli(data, metadata):
    """Test boolean columns use a ``bernoulli`` sampler with the observed True rate."""
    # Run
    column = _get_column(create_data_designer_config(data, metadata), 'has_rewards')

    # Assert
    assert column.sampler_type == dd.SamplerType.BERNOULLI
    assert column.params.p == 0.25


def test_boolean_without_values_uses_even_odds():
    """Test an all null boolean column is sampled with probability 0.5."""
    # Setup
    data = pd.DataFrame({'flag': [None, None]})
    metadata = Metadata.load_from_dict({
        'tables': {'t': {'columns': {'flag': {'sdtype': 'boolean'}}}}
    })

    # Run
    column = _get_column(create_data_designer_config(data, metadata), 'flag')

    # Assert
    assert column.params.p == 0.5


def test_integer_numerical_maps_to_uniform_int(data, metadata):
    """Test whole valued columns use a ``uniform`` sampler converted to int."""
    # Run
    column = _get_column(create_data_designer_config(data, metadata), 'num_guests')

    # Assert
    assert column.sampler_type == dd.SamplerType.UNIFORM
    assert (column.params.low, column.params.high) == (1.0, 4.0)
    assert column.params.decimal_places == 0
    assert column.convert_to == 'int'


def test_float_numerical_keeps_observed_decimal_places(data, metadata):
    """Test float columns use a ``uniform`` sampler with the observed precision."""
    # Run
    column = _get_column(create_data_designer_config(data, metadata), 'amenities_fee')

    # Assert
    assert column.sampler_type == dd.SamplerType.UNIFORM
    assert (column.params.low, column.params.high) == (0.0, 37.89)
    assert column.params.decimal_places == 2
    assert column.convert_to is None


def test_numerical_without_values_samples_zeros():
    """Test an all null numerical column is sampled as zeros."""
    # Setup
    data = pd.DataFrame({'amount': [np.nan, np.nan]})
    metadata = Metadata.load_from_dict({
        'tables': {'t': {'columns': {'amount': {'sdtype': 'numerical'}}}}
    })

    # Run
    column = _get_column(create_data_designer_config(data, metadata), 'amount')

    # Assert
    assert (column.params.low, column.params.high) == (0.0, 0.0)


def test_datetime_range_uses_datetime_format(data, metadata):
    """Test the observed datetime range is expressed in the column ``datetime_format``."""
    # Run
    builder = create_data_designer_config(data, metadata)

    # Assert
    checkin = _get_column(builder, 'checkin_date')
    assert checkin.sampler_type == dd.SamplerType.DATETIME
    assert checkin.params.start == '05 Apr 2020'
    assert checkin.params.end == '30 Dec 2020'

    checkout = _get_column(builder, 'checkout_ts')
    assert checkout.params.start == '2020-09-18 09:15:00'
    assert checkout.params.end == '2021-01-02 11:00:00'


def test_datetime_without_format_uses_default_format():
    """Test datetime columns without a ``datetime_format`` use an ISO like format."""
    # Setup
    data = pd.DataFrame({'when': pd.to_datetime(['2020-01-01 00:00:00', '2020-03-01 10:30:00'])})
    metadata = Metadata.load_from_dict({
        'tables': {'t': {'columns': {'when': {'sdtype': 'datetime'}}}}
    })

    # Run
    column = _get_column(create_data_designer_config(data, metadata), 'when')

    # Assert
    assert column.params.start == '2020-01-01 00:00:00'
    assert column.params.end == '2020-03-01 10:30:00'


def test_all_null_datetime_uses_placeholder():
    """Test a datetime column without parseable values falls back to a placeholder."""
    # Setup
    data = pd.DataFrame({'when': [None, None]})
    metadata = Metadata.load_from_dict({
        'tables': {
            't': {'columns': {'when': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'}}}
        }
    })

    # Run
    column = _get_column(create_data_designer_config(data, metadata), 'when')

    # Assert
    assert column.sampler_type == dd.SamplerType.UUID
    assert column.params.prefix == PII_PLACEHOLDER_PREFIX


def test_id_maps_to_uuid(data, metadata):
    """Test id columns use a full ``uuid`` sampler."""
    # Run
    column = _get_column(create_data_designer_config(data, metadata), 'guest_id')

    # Assert
    assert column.sampler_type == dd.SamplerType.UUID
    assert column.params.prefix is None
    assert column.params.short_form is False


def test_text_maps_to_llm_text_conditioned_on_row(data, metadata):
    """Test text columns are LLM generated with examples and references to the row."""
    # Run
    column = _get_column(create_data_designer_config(data, metadata), 'review')

    # Assert
    assert isinstance(column, dd.LLMTextColumnConfig)
    assert column.model_alias == 'text'
    assert "'Great stay.', 'Room was small.', 'Loved the pool!'" in column.prompt
    for reference in ('room_type', 'num_guests', 'has_rewards', 'checkin_date', 'checkout_ts'):
        assert f'{{{{ {reference} }}}}' in column.prompt

    for excluded in ('guest_id', 'guest_email', 'notes', 'review'):
        assert f'{{ {excluded} }}' not in column.prompt


def test_default_model_config_uses_nvidia_provider(data, metadata):
    """Test a default NVIDIA model config is registered for the model alias."""
    # Run
    builder = create_data_designer_config(data, metadata, model_alias='my-alias')

    # Assert
    (model_config,) = builder.model_configs
    assert model_config.alias == 'my-alias'
    assert model_config.provider == 'nvidia'
    assert _get_column(builder, 'review').model_alias == 'my-alias'


def test_custom_model_configs_are_used(data, metadata):
    """Test explicitly passed model configs replace the default."""
    # Setup
    model_configs = [
        dd.ModelConfig(
            alias='text',
            model='gpt-4.1',
            provider='openai',
            inference_parameters=dd.ChatCompletionInferenceParams(),
        )
    ]

    # Run
    builder = create_data_designer_config(data, metadata, model_configs=model_configs)

    # Assert
    assert builder.model_configs == model_configs


def test_missing_data_column_raises(data, metadata):
    """Test an error is raised when the data lacks a metadata column."""
    # Run and Assert
    with pytest.raises(ValueError, match=r"columns are missing from the data: \['notes'\]"):
        create_data_designer_config(data.drop(columns=['notes']), metadata)


def test_multi_table_metadata_raises(data):
    """Test an error is raised for multi table metadata without a table name."""
    # Setup
    metadata = Metadata.load_from_dict({
        'tables': {
            'a': {'columns': {'x': {'sdtype': 'numerical'}}},
            'b': {'columns': {'y': {'sdtype': 'numerical'}}},
        }
    })

    # Run and Assert
    with pytest.raises(ValueError, match='Metadata has 2 tables, please provide a `table_name`'):
        create_data_designer_config(data, metadata)


def test_table_name_selects_table_from_multi_table_metadata():
    """Test ``table_name`` picks one table out of a multi table Metadata V2."""
    # Setup
    data = pd.DataFrame({'x': [1.0, 2.0]})
    metadata = Metadata.load_from_dict({
        'tables': {
            'a': {'columns': {'x': {'sdtype': 'numerical'}}},
            'b': {'columns': {'y': {'sdtype': 'numerical'}}},
        }
    })

    # Run
    builder = create_data_designer_config(data, metadata, table_name='a')

    # Assert
    assert [config.name for config in builder.get_column_configs()] == ['x']
    with pytest.raises(ValueError, match="Table 'nope' is not present in the metadata"):
        create_data_designer_config(data, metadata, table_name='nope')


@pytest.mark.parametrize('bad_metadata', [['not', 'metadata'], {'tables': {}}])
def test_wrong_metadata_type_raises(data, bad_metadata):
    """Test an error is raised for anything that is not a ``Metadata`` object."""
    # Run and Assert
    with pytest.raises(TypeError, match='Expected sdv.Metadata, got'):
        create_data_designer_config(data, bad_metadata)


def test_range_values_define_categories_with_zero_weight_for_unseen(v2_data, v2_metadata):
    """Test ``range_values`` are used as the categories, weighted by the observed counts."""
    # Run
    column = _get_column(create_data_designer_config(v2_data, v2_metadata), 'room_type')

    # Assert
    assert column.params.values == ['BASIC', 'DELUXE', 'SUITE']  # noqa: PD011
    np.testing.assert_allclose(column.params.weights, [0.75, 0.25, 0.0])


def test_range_values_without_observations_sample_uniformly(v2_metadata):
    """Test ``range_values`` fall back to uniform weights when the data has no values."""
    # Setup
    data = pd.DataFrame({
        'room_type': [None, None],
        'amenities_fee': [1.0, 2.0],
        'num_guests': [1, 2],
        'checkin_date': ['10 Feb 2020', '11 Mar 2020'],
        'has_rewards': [True, False],
    })

    # Run
    column = _get_column(create_data_designer_config(data, v2_metadata), 'room_type')

    # Assert
    assert column.params.values == ['BASIC', 'DELUXE', 'SUITE']  # noqa: PD011
    assert column.params.weights is None


def test_range_min_max_and_decimal_places_override_data(v2_data, v2_metadata):
    """Test numerical ranges and precision come from the metadata when present."""
    # Run
    builder = create_data_designer_config(v2_data, v2_metadata)

    # Assert
    fee = _get_column(builder, 'amenities_fee')
    assert (fee.params.low, fee.params.high, fee.params.decimal_places) == (0.0, 46.64, 2)
    assert fee.convert_to is None

    guests = _get_column(builder, 'num_guests')
    assert (guests.params.low, guests.params.high, guests.params.decimal_places) == (1.0, 6.0, 0)
    assert guests.convert_to == 'int'


def test_datetime_range_min_max_override_data(v2_data, v2_metadata):
    """Test datetime ranges come from the string ``range_min`` and ``range_max``."""
    # Run
    column = _get_column(create_data_designer_config(v2_data, v2_metadata), 'checkin_date')

    # Assert
    assert column.params.start == '03 Jan 2020'
    assert column.params.end == '05 Jan 2021'


def test_get_missing_value_proportions_honors_range_is_nullable(v2_data, v2_metadata):
    """Test missing proportions come from the data unless the metadata says not nullable."""
    # Run
    proportions = get_missing_value_proportions(v2_data, v2_metadata)

    # Assert
    assert proportions == {
        'room_type': 0.0,
        'amenities_fee': 0.5,
        'num_guests': 0.0,
        'checkin_date': 0.0,
        'has_rewards': 0.0,
    }


def test_get_missing_value_proportions_table_name(v2_data, v2_metadata):
    """Test the proportions can be requested for a named table."""
    # Run
    proportions = get_missing_value_proportions(v2_data, v2_metadata, table_name='guests')

    # Assert
    assert proportions['amenities_fee'] == 0.5


def test_attach_missing_proportion_sets_supported_attribute():
    """Test the proportion is attached when the config exposes a missing value field."""
    # Setup
    config = Mock(spec=['name', 'missing_proportion'])
    config.missing_proportion = None

    # Run
    _attach_missing_proportion(config, 0.3)

    # Assert
    assert config.missing_proportion == 0.3


def test_attach_missing_proportion_is_noop_on_real_configs(v2_data, v2_metadata):
    """Test nothing is attached when the config has no missing value field."""
    # Setup
    column = _get_column(create_data_designer_config(v2_data, v2_metadata), 'amenities_fee')

    # Run
    _attach_missing_proportion(column, 0.5)

    # Assert
    assert not any(
        hasattr(column, attribute)
        for attribute in ('missing_values_proportion', 'missing_proportion', 'null_proportion')
    )


class TestDataDesignerSynthesizer:
    """Unit tests for the ``DataDesignerSynthesizer``."""

    def test_modality(self):
        """Test the synthesizer is a single table synthesizer."""
        assert DataDesignerSynthesizer._MODALITY_FLAG == 'single_table'

    @patch('data_designer.interface.DataDesigner')
    @patch('sdgym.synthesizers.data_designer.create_data_designer_config')
    def test__fit(self, create_config_mock, data_designer_mock, data, metadata):
        """Test ``_fit`` stores the config builder and a ``DataDesigner`` instance."""
        # Setup
        synthesizer = DataDesignerSynthesizer()

        # Run
        synthesizer._fit(data, metadata)

        # Assert
        create_config_mock.assert_called_once_with(data, metadata)
        data_designer_mock.assert_called_once_with()
        assert synthesizer._config_builder is create_config_mock.return_value
        assert synthesizer._internal_synthesizer is data_designer_mock.return_value

    @patch('sdgym.synthesizers.data_designer.create_data_designer_config')
    def test__fit_raises_when_interface_is_missing(self, create_config_mock, data, metadata):
        """Test a helpful error is raised when ``data_designer.interface`` is unavailable."""
        # Setup
        synthesizer = DataDesignerSynthesizer()

        # Run and Assert
        with patch.dict(sys.modules, {'data_designer.interface': None}):
            with pytest.raises(ValueError, match=r"pip install sdgym\['data_designer'\]"):
                synthesizer._fit(data, metadata)

    def test__sample_from_synthesizer(self):
        """Test sampling creates ``n_sample`` records and loads them as a dataset."""
        # Setup
        trained = Mock()
        expected = pd.DataFrame({'a': [1, 2]})
        trained._internal_synthesizer.create.return_value.load_dataset.return_value = expected

        # Run
        sampled = DataDesignerSynthesizer()._sample_from_synthesizer(trained, 2)

        # Assert
        trained._internal_synthesizer.create.assert_called_once_with(
            trained._config_builder, num_records=2
        )
        assert sampled is expected
