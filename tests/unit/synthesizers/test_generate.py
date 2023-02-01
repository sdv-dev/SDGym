from unittest.mock import Mock

import pytest

from sdgym.synthesizers import FastMLPreset, SDVRelationalSynthesizer, SDVTabularSynthesizer
from sdgym.synthesizers.generate import (
    SYNTHESIZER_MAPPING, create_multi_table_synthesizer, create_sdv_synthesizer_variant,
    create_sequential_synthesizer, create_single_table_synthesizer)


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


def test_create_sequential_synthesizer():
    """Test that a sequential synthesizer is created."""
    # Run
    out = create_sequential_synthesizer('test_synth', Mock(), Mock())

    # Assert
    assert out.__name__ == 'Custom:test_synth'
    assert hasattr(out, 'get_trained_synthesizer')
    assert hasattr(out, 'sample_from_synthesizer')


def test_create_sdv_variant_synthesizer():
    """Test that a sdv variant synthesizer is created.

    Expect that if the synthesizer class is a single-table synthesizer, the
    new synthesizer inherits from the SDVTabularSynthesizer base class."""
    # Setup
    synthesizer_class = 'GaussianCopulaSynthesizer'
    synthesizer_parameters = {}

    # Run
    out = create_sdv_synthesizer_variant('test_synth', synthesizer_class, synthesizer_parameters)

    # Assert
    assert out.__name__ == 'Variant:test_synth'
    assert out._MODEL == SYNTHESIZER_MAPPING.get(synthesizer_class)
    assert out._MODEL_KWARGS == synthesizer_parameters
    assert issubclass(out, SDVTabularSynthesizer)


def test_create_sdv_variant_synthesizer_error():
    """Test that a sdv variant synthesizer is created.

    Expect that if the synthesizer class is a single-table synthesizer, the
    new synthesizer inherits from the SDVTabularSynthesizer base class."""
    # Setup
    synthesizer_class = 'test'
    synthesizer_parameters = {}

    # Run
    with pytest.raises(
        ValueError,
        match=r"Synthesizer class test is not recognized. The supported options are "
              "FastMLPreset, GaussianCopulaSynthesizer, CTGANSynthesizer, "
              "CopulaGANSynthesizer, TVAESynthesizer, PARSynthesizer, HMASynthesizer"
    ):
        create_sdv_synthesizer_variant('test_synth', synthesizer_class, synthesizer_parameters)


def test_create_sdv_variant_synthesizer_relational():
    """Test that a sdv variant synthesizer is created.

    Expect that if the synthesizer class is a relational synthesizer, the
    new synthesizer inherits from the SDVRelationalSynthesizer base class."""
    # Setup
    synthesizer_class = 'HMASynthesizer'
    synthesizer_parameters = {}

    # Run
    out = create_sdv_synthesizer_variant('test_synth', synthesizer_class, synthesizer_parameters)

    # Assert
    assert out.__name__ == 'Variant:test_synth'
    assert out._MODEL == SYNTHESIZER_MAPPING.get(synthesizer_class)
    assert out._MODEL_KWARGS == synthesizer_parameters
    assert issubclass(out, SDVRelationalSynthesizer)


def test_create_sdv_variant_synthesizer_preset():
    """Test that a sdv variant synthesizer is created.

    Expect that if the synthesizer class is a preset synthesizer, the
    new synthesizer inherits from the FastMLPreset base class."""
    # Setup
    synthesizer_class = 'FastMLPreset'
    synthesizer_parameters = {}

    # Run
    out = create_sdv_synthesizer_variant('test_synth', synthesizer_class, synthesizer_parameters)

    # Assert
    assert out.__name__ == 'Variant:test_synth'
    assert out._MODEL == SYNTHESIZER_MAPPING.get(synthesizer_class)
    assert out._MODEL_KWARGS == synthesizer_parameters
    assert issubclass(out, FastMLPreset)
