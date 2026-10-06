"""Tests for the Data Designer integration."""

import re
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import cloudpickle
import numpy as np
import pandas as pd
import pytest
from sdv.metadata import Metadata

from sdgym.synthesizers import (
    DataDesignerSynthesizer,
    get_available_multi_table_synthesizers,
    get_available_single_table_synthesizers,
)
from sdgym.synthesizers.data_designer import (
    DEFAULT_MODEL_ID,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    PII_PLACEHOLDER_PREFIX,
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
        'guest_email': ['a@xyz.com', 'b@xyz.com', 'c@xyz.com', 'd@xyz.com', 'e@xyz.com'],
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
        'review': ['Great stay.', 'Room was small', None, 'Loved the amenities!', 'Noisy.'],
        'notes': ['x', 'y', 'x', 'x', 'z'],
    })


@pytest.fixture
def data_metadata_with_range():
    metadata = Metadata.load_from_dict({
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
        }
    })

    data = pd.DataFrame({
        'room_type': ['BASIC', 'BASIC', 'DELUXE', 'BASIC'],
        'amenities_fee': [10.5, np.nan, 20.25, np.nan],
        'num_guests': [2.5, 3.0, 2.0, 4.0],
        'checkin_date': ['10 Feb 2020', '11 Mar 2020', '12 Apr 2020', '13 May 2020'],
        'has_rewards': [True, False, None, False],
    })

    return data, metadata


def _get_column(builder, name):
    return builder.get_column_config(name)


def test__import_error_when_package_is_missing():
    """Test a helpful error is raised when the package is not installed."""
    # Setup
    missing = {'data_designer': None, 'data_designer.config': None}
    expected_message = (
        "To use 'DataDesignerSynthesizer' you have to install the extra "
        "dependencies by running pip install sdgym['data_designer']"
    )

    # Run and Assert
    with patch.dict(sys.modules, missing):
        with pytest.raises(ImportError, match=re.escape(expected_message)):
            _import_data_designer()


@patch('sdgym.synthesizers.data_designer.sys')
def test__import_error_on_unsupported_python(sys_mock, data, metadata):
    """Test an error is raised when running on python older than 3.10."""
    # Setup
    sys_mock.version_info = (3, 9, 18)
    expected_message = 'DataDesignerSynthesizer only supports python >= 3.10'

    # Run and Assert
    with pytest.raises(ImportError, match=expected_message):
        _import_data_designer()


@patch('sdgym.synthesizers.data_designer.sys')
def test__import_succeeds_on_supported_python(sys_mock):
    """Test the minimum supported python version is accepted."""
    # Setup
    sys_mock.version_info = (3, 10, 0)

    # Run
    module = _import_data_designer()

    # Assert
    assert module is dd


def test_create_data_designer_config_returns_builder_with_one_config_per_column(data, metadata):
    """Test every metadata column gets a config and the builder validates."""
    # Run
    builder, _ = create_data_designer_config(data, metadata)
    names = [config.name for config in builder.get_column_configs()]

    # Assert
    assert isinstance(builder, dd.DataDesignerConfigBuilder)
    assert names == list(metadata.tables['hotels'].columns)


def test_create_data_designer_config_maps_categorical_to_weighted_category_sampler(data, metadata):
    """Test categorical columns use a ``category`` sampler with observed frequencies."""
    # Run
    column = _get_column(create_data_designer_config(data, metadata)[0], 'room_type')

    # Assert
    assert column.sampler_type == dd.SamplerType.CATEGORY
    assert column.params.values == ['BASIC', 'DELUXE', 'SUITE']  # noqa: PD011
    np.testing.assert_allclose(column.params.weights, [0.6, 0.2, 0.2])


def test_create_data_designer_config_categorical_values_are_scalars():
    """Test numpy scalars in categorical columns are converted to python values."""
    # Setup
    data = pd.DataFrame({'room_type': pd.Series([1, 2, 2], dtype='int64')})
    metadata = Metadata.load_from_dict({
        'tables': {'table': {'columns': {'room_type': {'sdtype': 'categorical'}}}}
    })

    # Run
    column = _get_column(create_data_designer_config(data, metadata)[0], 'room_type')

    # Assert
    assert column.params.values == [2, 1]  # noqa: PD011


def test_create_data_designer_config_categorical_nulls_are_one_category():
    """Test every kind of missing value is counted as the single category."""
    # Setup
    data = pd.DataFrame({'cat': ['a', None, 'b', np.nan, pd.NaT, 'a', float('nan'), 'a']})
    metadata = Metadata.load_from_dict({
        'tables': {'table': {'columns': {'cat': {'sdtype': 'categorical'}}}}
    })

    # Run
    column = _get_column(create_data_designer_config(data, metadata)[0], 'cat')

    # Assert
    assert column.sampler_type == dd.SamplerType.CATEGORY
    assert column.params.values == ['__null__', 'a', 'b']  # noqa: PD011
    np.testing.assert_allclose(column.params.weights, [0.5, 0.375, 0.125])


def test_create_data_designer_config_categorical_nulls_added_to_range_values():
    """Test observed nulls extend the metadata ``range_values`` with the null placeholder."""
    # Setup
    data = pd.DataFrame({'cat': ['a', None, 'a', 'c']})
    metadata = Metadata.load_from_dict({
        'tables': {
            'table': {'columns': {'cat': {'sdtype': 'categorical', 'range_values': ['a', 'b']}}}
        }
    })

    # Run
    column = _get_column(create_data_designer_config(data, metadata)[0], 'cat')

    # Assert
    assert column.params.values == ['a', 'b', '__null__']  # noqa: PD011
    np.testing.assert_allclose(column.params.weights, [2 / 3, 0.0, 1 / 3])


def test_create_data_designer_config_range_values_without_nulls_are_unchanged():
    """Test the null placeholder is not added to ``range_values`` when no null was observed."""
    # Setup
    data = pd.DataFrame({'cat': ['a', 'b', 'a']})
    metadata = Metadata.load_from_dict({
        'tables': {
            'table': {'columns': {'cat': {'sdtype': 'categorical', 'range_values': ['a', 'b']}}}
        }
    })

    # Run
    column = _get_column(create_data_designer_config(data, metadata)[0], 'cat')

    # Assert
    assert column.params.values == ['a', 'b']  # noqa: PD011
    np.testing.assert_allclose(column.params.weights, [2 / 3, 1 / 3])


def test_create_data_designer_config_all_null_categorical_uses_single_category():
    """Test a completely null column uses a category sampler with one category."""
    # Setup
    data = pd.DataFrame({'empty': [None, np.nan, pd.NaT]})
    metadata = Metadata.load_from_dict({
        'tables': {'table': {'columns': {'empty': {'sdtype': 'categorical'}}}}
    })

    # Run
    column = _get_column(create_data_designer_config(data, metadata)[0], 'empty')

    # Assert
    assert column.sampler_type == dd.SamplerType.CATEGORY
    assert column.params.values == ['__null__']  # noqa: PD011
    assert column.params.weights == [1.0]


def test_create_data_designer_config_empty_categorical_uses_placeholder():
    """Test a categorical column without any rows falls back to a placeholder."""
    # Setup
    data = pd.DataFrame({'empty': pd.Series([], dtype=object)})
    metadata = Metadata.load_from_dict({
        'tables': {'table': {'columns': {'empty': {'sdtype': 'categorical'}}}}
    })

    # Run
    column = _get_column(create_data_designer_config(data, metadata)[0], 'empty')

    # Assert
    assert column.sampler_type == dd.SamplerType.UUID
    assert column.params.prefix == PII_PLACEHOLDER_PREFIX


def test_create_data_designer_config_other_sdtypes_fall_back_to_category_sampler(data, metadata):
    """Test pii and unknown sdtypes are sampled from the observed values."""
    # Run
    builder, _ = create_data_designer_config(data, metadata)

    # Assert
    email = _get_column(builder, 'guest_email')
    assert email.sampler_type == dd.SamplerType.CATEGORY
    assert set(email.params.values) == set(data['guest_email'])

    notes = _get_column(builder, 'notes')
    assert notes.sampler_type == dd.SamplerType.CATEGORY
    assert notes.params.values == ['x', 'y', 'z']
    np.testing.assert_allclose(notes.params.weights, [0.6, 0.2, 0.2])


def test_create_data_designer_config_boolean_maps_to_bernoulli(data, metadata):
    """Test boolean columns use a ``bernoulli`` sampler with the observed True rate."""
    # Run
    column = _get_column(create_data_designer_config(data, metadata)[0], 'has_rewards')

    # Assert
    assert column.sampler_type == dd.SamplerType.BERNOULLI
    assert column.params.p == 0.25


def test_create_data_designer_config_boolean_without_values_uses_even_odds():
    """Test an all null boolean column is sampled with probability 0.5."""
    # Setup
    data = pd.DataFrame({'flag': [None, None]})
    metadata = Metadata.load_from_dict({
        'tables': {'table': {'columns': {'flag': {'sdtype': 'boolean'}}}}
    })

    # Run
    column = _get_column(create_data_designer_config(data, metadata)[0], 'flag')

    # Assert
    assert column.params.p == 0.5


def test_create_data_designer_config_integer_numerical_maps_to_uniform_int(data, metadata):
    """Test whole valued columns use a ``uniform`` sampler converted to int."""
    # Run
    column = _get_column(create_data_designer_config(data, metadata)[0], 'num_guests')

    # Assert
    assert column.sampler_type == dd.SamplerType.UNIFORM
    assert (column.params.low, column.params.high) == (1.0, 4.0)
    assert column.params.decimal_places == 0
    assert column.convert_to == 'int'


def test_create_data_designer_config_float_numerical_keeps_observed_decimal_places(data, metadata):
    """Test float columns use a ``uniform`` sampler with the observed precision."""
    # Run
    column = _get_column(create_data_designer_config(data, metadata)[0], 'amenities_fee')

    # Assert
    assert column.sampler_type == dd.SamplerType.UNIFORM
    assert (column.params.low, column.params.high) == (0.0, 37.89)
    assert column.params.decimal_places == 2
    assert column.convert_to is None


def test_create_data_designer_config_numerical_without_values_samples_zeros():
    """Test an all null numerical column is sampled as zeros."""
    # Setup
    data = pd.DataFrame({'amount': [np.nan, np.nan]})
    metadata = Metadata.load_from_dict({
        'tables': {'table': {'columns': {'amount': {'sdtype': 'numerical'}}}}
    })

    # Run
    column = _get_column(create_data_designer_config(data, metadata)[0], 'amount')

    # Assert
    assert (column.params.low, column.params.high) == (0.0, 0.0)


def test_create_data_designer_config_datetime_range_uses_datetime_format(data, metadata):
    """Test the observed datetime range is expressed in the column ``datetime_format``."""
    # Run
    builder, _ = create_data_designer_config(data, metadata)

    # Assert
    checkin = _get_column(builder, 'checkin_date')
    assert checkin.sampler_type == dd.SamplerType.DATETIME
    assert checkin.params.start == '05 Apr 2020'
    assert checkin.params.end == '30 Dec 2020'

    checkout = _get_column(builder, 'checkout_ts')
    assert checkout.params.start == '2020-09-18 09:15:00'
    assert checkout.params.end == '2021-01-02 11:00:00'


def test_create_data_designer_config_datetime_without_format_uses_default_format():
    """Test datetime columns without a ``datetime_format` uses default."""
    # Setup
    data = pd.DataFrame({'when': pd.to_datetime(['2020-01-01 00:00:00', '2020-03-01 10:30:00'])})
    metadata = Metadata.load_from_dict({
        'tables': {'table': {'columns': {'when': {'sdtype': 'datetime'}}}}
    })

    # Run
    column = _get_column(create_data_designer_config(data, metadata)[0], 'when')

    # Assert
    assert column.params.start == '2020-01-01 00:00:00'
    assert column.params.end == '2020-03-01 10:30:00'


def test_create_data_designer_config_all_null_datetime_uses_placeholder():
    """Test a datetime column without parseable values falls back to a placeholder."""
    # Setup
    data = pd.DataFrame({'when': [None, None]})
    metadata = Metadata.load_from_dict({
        'tables': {
            'table': {'columns': {'when': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'}}}
        }
    })

    # Run
    column = _get_column(create_data_designer_config(data, metadata)[0], 'when')

    # Assert
    assert column.sampler_type == dd.SamplerType.UUID
    assert column.params.prefix == PII_PLACEHOLDER_PREFIX


def test_create_data_designer_config_maps_id_to_uuid(data, metadata):
    """Test id columns use a full ``uuid`` sampler."""
    # Run
    column = _get_column(create_data_designer_config(data, metadata)[0], 'guest_id')

    # Assert
    assert column.sampler_type == dd.SamplerType.UUID
    assert column.params.prefix is None
    assert column.params.short_form is False


def test_create_data_designer_config_maps_text_to_llm_text_examples_and_context(data, metadata):
    """Test text columns are LLM generated with examples and the context columns."""
    # Run
    column = _get_column(create_data_designer_config(data, metadata)[0], 'review')

    # Assert
    assert isinstance(column, dd.LLMTextColumnConfig)
    assert column.model_alias == 'text'
    assert "Generate a realistic value for the column 'review'" in column.prompt
    assert (
        "Here are a few examples from this column: 'Great stay.', 'Room was small', "
        "'Loved the amenities!'."
    ) in column.prompt
    assert column.prompt.endswith('Respond with the value only, without any explanation.')

    _, _, context = column.prompt.partition('Refer to the values in these columns for context:')
    context_columns = (
        'has_rewards', 'room_type', 'num_guests', 'amenities_fee', 'checkin_date', 'checkout_ts'
    )
    for name in context_columns:
        assert name in context

    for excluded in ('guest_id', 'guest_email', 'notes', 'review'):
        assert excluded not in context


def test_create_data_designer_config_text_without_examples_or_context():
    """Test the prompt omits the examples and context when there are none."""
    # Setup
    data = pd.DataFrame({'review': [None, None]})
    metadata = Metadata.load_from_dict({
        'tables': {'table': {'columns': {'review': {'sdtype': 'text'}}}}
    })

    # Run
    column = _get_column(create_data_designer_config(data, metadata)[0], 'review')

    # Assert
    assert 'examples from this column' not in column.prompt
    assert 'for context' not in column.prompt


def test_create_data_designer_config_default_model_config(data, metadata):
    """Test a default NVIDIA model config is registered for the model alias."""
    # Run
    builder, _ = create_data_designer_config(data, metadata, model_alias='my-alias')

    # Assert
    (model_config,) = builder.model_configs
    assert model_config.alias == 'my-alias'
    assert model_config.model == DEFAULT_MODEL_ID
    assert model_config.provider == 'nvidia'
    assert model_config.inference_parameters.temperature == DEFAULT_TEMPERATURE == 0.85
    assert model_config.inference_parameters.top_p == DEFAULT_TOP_P == 0.95
    assert _get_column(builder, 'review').model_alias == 'my-alias'


def test_create_data_designer_config_custom_temperature_and_top_p(data, metadata):
    """Test ``temperature`` and ``top_p`` override the default inference parameters."""
    # Run
    builder, _ = create_data_designer_config(data, metadata, temperature=0.2, top_p=0.5)

    # Assert
    (model_config,) = builder.model_configs
    assert model_config.inference_parameters.temperature == 0.2
    assert model_config.inference_parameters.top_p == 0.5


def test_create_data_designer_config_zero_temperature(data, metadata):
    """Test a temperature of zero is a valid value and is kept."""
    # Run
    builder, _ = create_data_designer_config(data, metadata, temperature=0)

    # Assert
    (model_config,) = builder.model_configs
    assert model_config.inference_parameters.temperature == 0
    assert model_config.inference_parameters.top_p == DEFAULT_TOP_P


def test_create_data_designer_config_missing_data_column_raises(data, metadata):
    """Test an error is raised when the data lacks a metadata column."""
    # Run and Assert
    with pytest.raises(ValueError, match=r"columns are missing from the data: \['notes'\]"):
        create_data_designer_config(data.drop(columns=['notes']), metadata)


def test_create_data_designer_config_multi_table_metadata_raises(data):
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


def test_create_data_designer_config_table_name_selects_table_from_multi_table_metadata():
    """Test ``table_name`` picks one table out of a multi table dataset."""
    # Setup
    data = pd.DataFrame({'x': [1.0, 2.0]})
    metadata = Metadata.load_from_dict({
        'tables': {
            'a': {'columns': {'x': {'sdtype': 'numerical'}}},
            'b': {'columns': {'y': {'sdtype': 'numerical'}}},
        }
    })

    # Run
    builder, _ = create_data_designer_config(data, metadata, table_name='a')

    # Assert
    assert [config.name for config in builder.get_column_configs()] == ['x']
    with pytest.raises(ValueError, match="Table 'missing' is not present in the metadata"):
        create_data_designer_config(data, metadata, table_name='missing')


@pytest.mark.parametrize('bad_metadata', [['not', 'metadata'], {'tables': {}}])
def test_create_data_designer_config_wrong_metadata_type_raises(data, bad_metadata):
    """Test an error is raised for anything that is not a ``Metadata`` object."""
    # Run and Assert
    with pytest.raises(TypeError, match='Expected sdv.Metadata, got'):
        create_data_designer_config(data, bad_metadata)


def test_create_data_designer_config_range_values_define_categories_with_zero_weight_for_unseen(
        data_metadata_with_range
    ):
    """Test ``range_values`` are used as the categories, weighted by the observed counts."""
    # Setup
    data, metadata = data_metadata_with_range

    # Run
    column = _get_column(create_data_designer_config(data, metadata)[0], 'room_type')

    # Assert
    assert column.params.values == ['BASIC', 'DELUXE', 'SUITE']
    np.testing.assert_allclose(column.params.weights, [0.75, 0.25, 0.0])


def test_create_data_designer_config_range_values_with_only_nulls_observed(
        data_metadata_with_range
    ):
    """Test ``range_values`` are kept and the nulls become the only observed category."""
    # Setup
    _, metadata = data_metadata_with_range
    data = pd.DataFrame({
        'room_type': [None, None],
        'amenities_fee': [1.0, 2.0],
        'num_guests': [1, 2],
        'checkin_date': ['10 Feb 2020', '11 Mar 2020'],
        'has_rewards': [True, False],
    })

    # Run
    column = _get_column(create_data_designer_config(data, metadata)[0], 'room_type')

    # Assert
    assert column.params.values == ['BASIC', 'DELUXE', 'SUITE', '__null__']
    np.testing.assert_allclose(column.params.weights, [0.0, 0.0, 0.0, 1.0])


def test_create_data_designer_config_range_min_max_and_decimal_places_override_data(
        data_metadata_with_range
    ):
    """Test numerical ranges and precision come from the metadata when present."""
    # Setup
    data, metadata = data_metadata_with_range

    # Run
    builder, _ = create_data_designer_config(data, metadata)

    # Assert
    fee = _get_column(builder, 'amenities_fee')
    assert (fee.params.low, fee.params.high, fee.params.decimal_places) == (0.0, 46.64, 2)
    assert fee.convert_to is None

    guests = _get_column(builder, 'num_guests')
    assert (guests.params.low, guests.params.high, guests.params.decimal_places) == (1.0, 6.0, 0)
    assert guests.convert_to == 'int'


def test_create_data_designer_config_datetime_range_min_max_override_data(data_metadata_with_range):
    """Test datetime ranges come from the string ``range_min`` and ``range_max``."""
    # Setup
    data, metadata = data_metadata_with_range

    # Run
    column = _get_column(create_data_designer_config(data, metadata)[0], 'checkin_date')

    # Assert
    assert column.params.start == '03 Jan 2020'
    assert column.params.end == '05 Jan 2021'


def test_get_missing_value_proportions_honors_range_is_nullable(data_metadata_with_range):
    """Test missing proportions come from the data unless the metadata says not nullable."""
    # Setup
    data, metadata = data_metadata_with_range

    # Run
    proportions = get_missing_value_proportions(data, metadata)

    # Assert
    assert proportions == {
        'room_type': 0.0,
        'amenities_fee': 0.5,
        'num_guests': 0.0,
        'checkin_date': 0.0,
        'has_rewards': 0.0,
    }


def test_get_missing_value_proportions_table_name(data_metadata_with_range):
    """Test the proportions can be requested for a named table."""
    # Setup
    data, metadata = data_metadata_with_range

    # Run
    proportions = get_missing_value_proportions(data, metadata, table_name='guests')

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


def test__attach_missing_proportion_is_noop_on_real_configs(data_metadata_with_range):
    """Test nothing is attached when the config has no missing value field."""
    # Setup
    data, metadata = data_metadata_with_range
    column = _get_column(create_data_designer_config(data, metadata)[0], 'amenities_fee')

    # Run
    _attach_missing_proportion(column, 0.5)

    # Assert
    assert not any(
        hasattr(column, attribute)
        for attribute in ('missing_values_proportion', 'missing_proportion', 'null_proportion')
    )