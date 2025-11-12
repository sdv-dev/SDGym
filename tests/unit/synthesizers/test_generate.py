"""Tests for the generate module."""

import re
from unittest.mock import Mock, patch

import pytest

from sdgym import (
    create_multi_table_synthesizer,
    create_single_table_synthesizer,
    create_synthesizer_variant,
)
from sdgym.synthesizers.generate import _create_synthesizer_class
from sdgym.synthesizers.sdv import BaselineSDVSynthesizer


def test_create_single_table_synthesizer():
    """Test that a single table synthesizer is created."""
    # Run
    out = create_single_table_synthesizer('test_synth', Mock(), Mock())

    # Assert
    assert out.__name__ == 'Custom:test_synth'
    assert hasattr(out, 'get_trained_synthesizer')
    assert hasattr(out, 'sample_from_synthesizer')


def test_create_multi_table_synthesizer():
    """Test that a multi table synthesizer is created."""
    # Run
    out = create_multi_table_synthesizer('test_synth', Mock(), Mock())

    # Assert
    assert out.__name__ == 'Custom:test_synth'
    assert hasattr(out, 'get_trained_synthesizer')
    assert hasattr(out, 'sample_from_synthesizer')


def test_create_sdv_variant_synthesizer():
    """Test that a sdv variant synthesizer is created.

    Expect that if the synthesizer class is a single-table synthesizer, the
    new synthesizer inherits from the SDVSingleTableBaseline base class.
    """
    # Setup
    synthesizer_class = 'GaussianCopulaSynthesizer'
    synthesizer_parameters = {}

    # Run
    out = create_synthesizer_variant('test_synth', synthesizer_class, synthesizer_parameters)

    # Assert
    assert out.__name__ == 'Variant:test_synth'
    assert out.sdv_name == 'GaussianCopulaSynthesizer'
    assert isinstance(out, BaselineSDVSynthesizer)
    assert out.parameters == {}


def test_create_sdv_variant_synthesizer_error():
    """Test an error is raised when the synthesizer is not from SDV."""
    # Setup
    synthesizer_class = 'test'
    synthesizer_parameters = {}
    expected_error = re.escape("Synthesizer 'test' is not a SDV synthesizer.")

    # Run and Assert
    with pytest.raises(ValueError, match=expected_error):
        create_synthesizer_variant('test_synth', synthesizer_class, synthesizer_parameters)


def test_create_sdv_variant_synthesizer_multi_table():
    """Test that a sdv variant synthesizer is created."""
    # Setup
    synthesizer_class = 'HMASynthesizer'
    synthesizer_parameters = {}

    # Run
    out = create_synthesizer_variant('test_synth', synthesizer_class, synthesizer_parameters)

    # Assert
    assert out.__name__ == 'Variant:test_synth'
    assert out.sdv_name == 'HMASynthesizer'
    assert out.parameters == synthesizer_parameters
    assert isinstance(out, BaselineSDVSynthesizer)


def test__create_synthesizer_class():
    """Test the ``_create_synthesizer_class`` method."""
    # Setup
    get_trained_synthesizer_fn = Mock()
    sample_fn = Mock()

    # Run
    out = _create_synthesizer_class(
        'test_synth',
        get_trained_synthesizer_fn,
        sample_fn,
        sample_arg_name='num_samples',
    )

    # Assert
    assert out.__name__ == 'Custom:test_synth'
    assert hasattr(out, 'get_trained_synthesizer')
    assert hasattr(out, 'sample_from_synthesizer')


@patch('sdgym.synthesizers.generate._create_synthesizer_class')
def test_create_single_table_synthesizer_mock(mock_create_class):
    """Test the ``create_single_table_synthesizer`` method."""
    # Setup
    mock_create_class.return_value = 'synthesizer_class'
    get_trained_synthesizer_fn = Mock()
    sample_fn = Mock()

    # Run
    out = create_single_table_synthesizer('test_synth', get_trained_synthesizer_fn, sample_fn)

    # Assert
    mock_create_class.assert_called_once_with(
        'test_synth',
        get_trained_synthesizer_fn,
        sample_fn,
        sample_arg_name='num_samples',
    )
    assert out == 'synthesizer_class'


@patch('sdgym.synthesizers.generate._create_synthesizer_class')
def test_create_multi_table_synthesizer_mock(mock_create_class):
    """Test the ``create_multi_table_synthesizer`` method."""
    # Setup
    mock_create_class.return_value = 'synthesizer_class'
    get_trained_synthesizer_fn = Mock()
    sample_fn = Mock()

    # Run
    out = create_multi_table_synthesizer('test_synth', get_trained_synthesizer_fn, sample_fn)

    # Assert
    mock_create_class.assert_called_once_with(
        'test_synth',
        get_trained_synthesizer_fn,
        sample_fn,
        sample_arg_name='scale',
    )
    assert out == 'synthesizer_class'
