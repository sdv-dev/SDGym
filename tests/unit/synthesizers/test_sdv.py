import inspect
import re
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

from sdgym.synthesizers.sdv import (
    BaselineSDVSynthesizer,
    _get_all_sdv_synthesizers,
    _get_sdv_synthesizers,
    _validate_inputs,
    _validate_modality,
    _validate_parameters,
)


def test__validate_modality():
    """Test the `_validate_modality` method."""
    # Setup
    valid_modalities = ['single_table', 'multi_table']

    # Run and Assert
    for modality in valid_modalities:
        _validate_modality(modality)


def test__validate_modality_invalid():
    """Test the `_validate_modality` method with invalid modality."""
    # Setup
    expected_error = re.escape("`modality` must be one of 'single_table' or 'multi_table'.")

    # Run and Assert
    with pytest.raises(ValueError, match=expected_error):
        _validate_modality('invalid_modality')


def test__validate_parameters():
    """Test the `_validate_parameters` method."""
    # Setup
    valid_parameters = {'enforce_min_max_values': True, 'default_distribution': 'normal'}
    gc_parameters = list(inspect.signature(GaussianCopulaSynthesizer.__init__).parameters.values())[
        1:
    ]

    # Run
    _validate_parameters(valid_parameters, gc_parameters)


def test__validate_parameters_invalid():
    """Test the `_validate_parameters` method with invalid parameter."""
    # Setup
    invalid_parameters = {'invalid_param': 123}
    gc_parameters = list(inspect.signature(GaussianCopulaSynthesizer.__init__).parameters.values())[
        1:
    ]
    expected_error = re.escape(
        "Parameter 'invalid_param' is not valid for the selected synthesizer."
    )

    # Run and Assert
    with pytest.raises(ValueError, match=expected_error):
        _validate_parameters(invalid_parameters, gc_parameters)


@patch('sdgym.synthesizers.sdv._validate_modality')
@patch('sdgym.synthesizers.sdv._validate_parameters')
def test__validate_inputs(mock_validate_parameters, mock_validate_modality):
    """Test the `_validate_inputs` method."""
    # Setup
    sdv_name = 'GaussianCopulaSynthesizer'
    modality = 'single_table'
    parameters = {'enforce_min_max_values': True}
    gc_parameters = list(inspect.signature(GaussianCopulaSynthesizer.__init__).parameters.values())[
        1:
    ]

    # Run
    _validate_inputs(sdv_name, modality, parameters)

    # Assert
    mock_validate_modality.assert_called_once_with(modality)
    mock_validate_parameters.assert_called_once_with(parameters, gc_parameters)


def test__validate_inputs_invalid_parameters():
    """Test the `_validate_inputs` method with invalid parameters."""
    # Setup
    sdv_name = 'GaussianCopulaSynthesizer'
    modality = 'single_table'
    parameters = {'invalid_param': 123}
    expected_error_1 = re.escape('`sdv_name` must be a string.')
    expected_error_2 = re.escape('`parameters` must be a dictionary or None.')
    expected_error_3 = re.escape(
        "Synthesizer 'InvalidSynthesizer' is not a SDV 'single_table' synthesizers."
    )

    # Run and Assert
    with pytest.raises(ValueError, match=expected_error_1):
        _validate_inputs(123, modality, parameters)

    with pytest.raises(ValueError, match=expected_error_2):
        _validate_inputs(sdv_name, modality, 'invalid_parameters')

    with pytest.raises(ValueError, match=expected_error_3):
        _validate_inputs('InvalidSynthesizer', modality, parameters)


def test__get_sdv_synthesizers():
    """Test the `_get_sdv_synthesizers` method."""
    # Setup
    expected_single_table_synthesizers = [
        'CTGANSynthesizer',
        'CopulaGANSynthesizer',
        'GaussianCopulaSynthesizer',
        'TVAESynthesizer',
    ]
    expected_multi_table_synthesizers = ['HMASynthesizer']

    # Run
    single_table_synthesizers = _get_sdv_synthesizers('single_table')
    multi_table_synthesizers = _get_sdv_synthesizers('multi_table')

    # Assert
    assert single_table_synthesizers == expected_single_table_synthesizers
    assert multi_table_synthesizers == expected_multi_table_synthesizers


def test__get_all_sdv_synthesizers():
    """Test the `_get_all_sdv_synthesizers` method."""
    # Setup
    expected_synthesizers = [
        'CTGANSynthesizer',
        'CopulaGANSynthesizer',
        'GaussianCopulaSynthesizer',
        'HMASynthesizer',
        'TVAESynthesizer',
    ]

    # Run
    all_synthesizers = _get_all_sdv_synthesizers()

    # Assert
    assert all_synthesizers == expected_synthesizers


class TestBaselineSDVSynthesizer:
    """Test the `BaselineSDVSynthesizer` class."""

    @patch('sdgym.synthesizers.sdv._validate_inputs')
    def test__init__(self, mock_validate_inputs):
        """Test the `__init__` method."""
        # Setup
        sdv_name = 'GaussianCopulaSynthesizer'
        modality = 'single_table'
        parameters = {'enforce_min_max_values': True}

        # Run
        synthesizer = BaselineSDVSynthesizer(sdv_name, modality, parameters)

        # Assert
        mock_validate_inputs.assert_called_once_with(sdv_name, modality, parameters)
        assert synthesizer.sdv_name == sdv_name
        assert synthesizer.modality == modality
        assert synthesizer.parameters == parameters

    def test___repr__(self):
        """Test the `__repr__` method."""
        # Setup
        synthesizer = BaselineSDVSynthesizer('GaussianCopulaSynthesizer', 'single_table')

        # Run
        repr_str = repr(synthesizer)

        # Assert
        expected_str = (
            "BaselineSDVSynthesizer(sdv_name='GaussianCopulaSynthesizer', modality='single_table')"
        )
        assert repr_str == expected_str

    @patch('sdgym.synthesizers.sdv.LOGGER')
    def test__get_trained_synthesizer(self, mock_logger):
        """Test the `_get_trained_synthesizer` method."""
        # Setup
        data = pd.DataFrame({
            'column1': [1, 2, 3, 4, 5],
            'column2': ['A', 'B', 'C', 'D', 'E'],
        })
        metadata = Metadata().load_from_dict({
            'tables': {
                'table_1': {
                    'columns': {
                        'column1': {'sdtype': 'numerical'},
                        'column2': {'sdtype': 'categorical'},
                    },
                },
            },
        })
        synthesizer = BaselineSDVSynthesizer(
            'GaussianCopulaSynthesizer', 'single_table', {'enforce_min_max_values': False}
        )

        # Run
        valid_model = synthesizer._get_trained_synthesizer(data, metadata)

        # Assert
        mock_logger.info.assert_called_with('Fitting %s', 'GaussianCopulaSynthesizer')
        assert isinstance(valid_model, GaussianCopulaSynthesizer)
        assert valid_model.enforce_min_max_values is False

    @patch('sdgym.synthesizers.sdv.LOGGER')
    def test__sample_from_synthesizer(self, mock_logger):
        """Test the `_sample_from_synthesizer` method."""
        # Setup
        data = pd.DataFrame({
            'column1': [1, 2, 3, 4, 5],
            'column2': ['A', 'B', 'C', 'D', 'E'],
        })
        base_synthesizer = BaselineSDVSynthesizer('GaussianCopulaSynthesizer', 'single_table')
        synthesizer = Mock()
        synthesizer.sample.return_value = data
        n_samples = 3

        # Run
        sampled_data = base_synthesizer._sample_from_synthesizer(synthesizer, n_samples)

        # Assert
        mock_logger.info.assert_called_with('Sampling %s', 'GaussianCopulaSynthesizer')
        pd.testing.assert_frame_equal(sampled_data, data)
        synthesizer.sample.assert_called_once_with(num_rows=n_samples)
