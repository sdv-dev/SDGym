from unittest.mock import patch

from sdgym.synthesizers.utils import _get_sdgym_synthesizers, _get_supported_synthesizers


@patch('sdgym.synthesizers.utils.BaselineSynthesizer.get_subclasses')
def test__get_sdgym_synthesizers(mock_get_subclasses):
    """Test the `_get_sdgym_synthesizers` method."""
    # Setup
    mock_get_subclasses.return_value = {
        'ColumnSynthesizer': None,
        'UniformSynthesizer': None,
        'DataIdentity': None,
        'RealTabFormerSynthesizer': None,
        'CTGANSynthesizer': None,
        'CopulaGANSynthesizer': None,
        'GaussianCopulaSynthesizer': None,
        'HMASynthesizer': None,
        'TVAESynthesizer': None,
    }
    expected_synthesizers = [
        'ColumnSynthesizer',
        'DataIdentity',
        'RealTabFormerSynthesizer',
        'UniformSynthesizer',
    ]

    # Run
    synthesizers = _get_sdgym_synthesizers()

    # Assert
    assert synthesizers == expected_synthesizers


def test__get_supported_synthesizers():
    """Test the `_get_supported_synthesizers` method."""
    # Setup
    expected_synthesizers = [
        'CTGANSynthesizer',
        'ColumnSynthesizer',
        'CopulaGANSynthesizer',
        'DataIdentity',
        'GaussianCopulaSynthesizer',
        'HMASynthesizer',
        'RealTabFormerSynthesizer',
        'TVAESynthesizer',
        'UniformSynthesizer',
    ]

    # Run
    synthesizers = _get_supported_synthesizers()

    # Assert
    assert synthesizers == expected_synthesizers
