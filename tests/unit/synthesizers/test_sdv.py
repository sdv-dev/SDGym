import re
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

from sdgym.synthesizers.base import BaselineSynthesizer
from sdgym.synthesizers.sdv import (
    _create_sdv_class,
    _fit,
    _get_all_sdv_synthesizers,
    _get_modality,
    _get_sdv_synthesizers,
    _retrieve_sdv_class,
    _sample_from_synthesizer,
    create_sdv_synthesizer_class,
)


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


@patch('sdgym.synthesizers.sdv.LOGGER')
def test__fit(mock_logger):
    """Test the `_fit` method."""
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
    synthesizer = Mock()
    synthesizer.__class__.__name__ = 'GaussianCopulaClass'
    synthesizer._MODEL_KWARGS = {'enforce_min_max_values': False}
    synthesizer._MODALITY_FLAG = 'single_table'
    synthesizer.SDV_NAME = 'GaussianCopulaSynthesizer'

    # Run
    _fit(synthesizer, data, metadata)

    # Assert
    mock_logger.info.assert_called_with('Fitting %s', 'GaussianCopulaClass')
    assert isinstance(synthesizer._internal_synthesizer, GaussianCopulaSynthesizer)
    assert synthesizer._internal_synthesizer.enforce_min_max_values is False
    assert synthesizer._internal_synthesizer._fitted is True


@patch('sdgym.synthesizers.sdv.LOGGER')
def test__sample_from_synthesizer(mock_logger):
    """Test the `_sample_from_synthesizer` method."""
    # Setup
    data = pd.DataFrame({
        'column1': [1, 2, 3, 4, 5],
        'column2': ['A', 'B', 'C', 'D', 'E'],
    })
    base_synthesizer = Mock()
    base_synthesizer.__class__.__name__ = 'GaussianCopulaSynthesizer'
    base_synthesizer._MODALITY_FLAG = 'single_table'
    synthesizer = Mock()
    synthesizer._internal_synthesizer = Mock()
    synthesizer._internal_synthesizer.sample.return_value = data
    n_samples = 3

    # Run
    sampled_data = _sample_from_synthesizer(base_synthesizer, synthesizer, n_samples)

    # Assert
    mock_logger.info.assert_called_with('Sampling %s', 'GaussianCopulaSynthesizer')
    pd.testing.assert_frame_equal(sampled_data, data)
    synthesizer._internal_synthesizer.sample.assert_called_once_with(num_rows=n_samples)


@patch('sdgym.synthesizers.sdv.sys.modules')
def test__retrieve_sdv_class(mock_sys_modules):
    """Test the `_retrieve_sdv_class` method."""
    # Setup
    CTGANSynthesizer = type('CTGANSynthesizer', (), {})
    fake_module = type('FakeMod', (), {'CTGANSynthesizer': CTGANSynthesizer})()
    mock_sys_modules.__getitem__.return_value = fake_module

    # Run
    defined_class = _retrieve_sdv_class('CTGANSynthesizer')
    undefined_class = _retrieve_sdv_class('UndefinedSynthesizer')

    # Assert
    assert defined_class is CTGANSynthesizer
    assert undefined_class is None


def test__get_modality():
    """Test the `_get_modality` method."""
    # Setup
    single_table_sdv = 'GaussianCopulaSynthesizer'
    multi_table_sdv = 'HMASynthesizer'
    invalid_name = 'InvalidSynthesizer'
    expected_error = re.escape(f"Synthesizer '{invalid_name}' is not a SDV synthesizer.")

    # Run
    single_table_modality = _get_modality(single_table_sdv)
    multi_table_modality = _get_modality(multi_table_sdv)
    with pytest.raises(ValueError, match=expected_error):
        _get_modality(invalid_name)

    # Assert
    assert single_table_modality == 'single_table'
    assert multi_table_modality == 'multi_table'


@patch('sdgym.synthesizers.sdv.sys.modules')
@patch('sdgym.synthesizers.sdv._get_modality')
def test__create_sdv_class_mock(mock_get_modality, mock_sys_modules):
    """Test the `_create_sdv_class` method with mocks."""
    # Setup
    sdv_name = 'GaussianCopulaSynthesizer'
    mock_get_modality.return_value = 'single_table'
    fake_module = type('FakeMod', (), {})()
    mock_sys_modules.__getitem__.return_value = fake_module

    # Run
    synt_class = _create_sdv_class(sdv_name)
    instance = synt_class()

    # Assert
    assert synt_class.__name__ == sdv_name
    assert synt_class._MODALITY_FLAG == 'single_table'
    assert synt_class._MODEL_KWARGS == {}
    assert synt_class.SDV_NAME == sdv_name
    assert issubclass(synt_class, BaselineSynthesizer)
    assert getattr(synt_class, '_fit') is _fit
    assert getattr(synt_class, '_sample_from_synthesizer') is _sample_from_synthesizer
    assert getattr(fake_module, sdv_name) is synt_class
    assert instance._fit.__self__ is instance
    assert instance._fit.__func__ is _fit
    assert instance._sample_from_synthesizer.__self__ is instance
    assert instance._sample_from_synthesizer.__func__ is _sample_from_synthesizer
    assert instance.SDV_NAME == sdv_name
    mock_get_modality.assert_called_once_with(sdv_name)


def test__create_sdv_class():
    """Test the `_create_sdv_class` method."""
    # Setup
    sdv_name = 'GaussianCopulaSynthesizer'

    # Run
    synthesizer_class = _create_sdv_class(sdv_name)

    # Assert
    assert synthesizer_class.__name__ == sdv_name
    assert synthesizer_class._MODALITY_FLAG == 'single_table'
    assert synthesizer_class._MODEL_KWARGS == {}
    assert issubclass(synthesizer_class, BaselineSynthesizer)


@patch('sdgym.synthesizers.sdv._create_sdv_class')
@patch('sdgym.synthesizers.sdv._retrieve_sdv_class')
def test_create_sdv_synthesizer_class(
    mock_retrieve_sdv_class,
    mock_create_sdv_class,
):
    """Test the `create_sdv_synthesizer_class` method."""
    # Setup
    mock_retrieve_sdv_class.return_value = None
    mock_create_sdv_class.return_value = 'GaussianCopulaSynthesizerClass'

    # Run
    synthesizer_class = create_sdv_synthesizer_class('GaussianCopulaSynthesizer')

    # Assert
    mock_retrieve_sdv_class.assert_called_once_with('GaussianCopulaSynthesizer')
    mock_create_sdv_class.assert_called_once_with('GaussianCopulaSynthesizer')
    assert synthesizer_class == 'GaussianCopulaSynthesizerClass'


def test_create_sdv_synthesizer_class_invalid():
    """Test the `create_sdv_synthesizer_class` method with invalid synthesizer name."""
    # Setup
    invalid_name = 'InvalidSynthesizer'
    expected_error = re.escape(f"Synthesizer '{invalid_name}' is not a supported SDV synthesizer.")

    # Run and Assert
    with pytest.raises(ValueError, match=expected_error):
        create_sdv_synthesizer_class(invalid_name)
