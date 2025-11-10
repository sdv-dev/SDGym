from sdgym.synthesizers.generate import _list_available_synthesizers


def test_list_available_synthesizers():
    """Test the `list_available_synthesizers` method"""
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
    synthesizers = _list_available_synthesizers()

    # Assert
    assert synthesizers == expected_synthesizers
