from sdgym.synthesizers.utils import _get_supported_synthesizers


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
        'MultiTableUniformSynthesizer',
        'RealTabFormerSynthesizer',
        'TVAESynthesizer',
        'UniformSynthesizer',
    ]

    # Run
    synthesizers = _get_supported_synthesizers()

    # Assert
    assert synthesizers == expected_synthesizers
