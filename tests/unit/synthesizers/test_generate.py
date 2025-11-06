"""Tests for the generate module."""

import re
from unittest.mock import Mock, patch

import pytest
from sdv.multi_table import HMASynthesizer
from sdv.single_table import GaussianCopulaSynthesizer

from sdgym import create_sdv_synthesizer_variant, create_single_table_synthesizer
from sdgym.synthesizers.generate import (
    create_multi_table_synthesizer,
)
from sdgym.synthesizers.sdv import SDVMultiTableBaseline, SDVSingleTableBaseline


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
    out = create_sdv_synthesizer_variant('test_synth', synthesizer_class, synthesizer_parameters)

    # Assert
    assert out.__name__ == 'Variant:test_synth'
    assert out._SDV_CLASS == GaussianCopulaSynthesizer
    assert issubclass(out, SDVSingleTableBaseline)
    assert out._MODEL_KWARGS == {}


@patch('sdgym.synthesizers.generate._list_available_synthesizers')
def test_create_sdv_variant_synthesizer_error(mock_list):
    """Test an error is raised when the synthesizer is not from SDV."""
    # Setup
    synthesizer_class = 'test'
    synthesizer_parameters = {}
    mock_list.return_value = [
        'GaussianCopulaSynthesizer',
        'CTGANSynthesizer',
        'UniformSynthesizer',
    ]
    expected_error = re.escape(
        "Synthesizer class 'test' is not recognized. Available SDV synthesizers: "
        "'GaussianCopulaSynthesizer', 'CTGANSynthesizer', 'UniformSynthesizer'"
    )

    # Run
    with pytest.raises(ValueError, match=expected_error):
        create_sdv_synthesizer_variant('test_synth', synthesizer_class, synthesizer_parameters)


def test_create_sdv_variant_synthesizer_multi_table():
    """Test that a sdv variant synthesizer is created.

    Expect that if the synthesizer class is a multi-table synthesizer, the
    new synthesizer inherits from the SDVMultiTableBaseline base class.
    """
    # Setup
    synthesizer_class = 'HMASynthesizer'
    synthesizer_parameters = {}

    # Run
    out = create_sdv_synthesizer_variant('test_synth', synthesizer_class, synthesizer_parameters)

    # Assert
    assert out.__name__ == 'Variant:test_synth'
    assert out._SDV_CLASS == HMASynthesizer
    assert out._BASE_SYNTHESIZER_CLASS == HMASynthesizer
    assert out._MODEL_KWARGS == synthesizer_parameters
    assert issubclass(out, SDVMultiTableBaseline)
