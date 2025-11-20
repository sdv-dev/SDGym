from sdgym.synthesizers import (
    get_available_multi_table_synthesizers,
    get_available_single_table_synthesizers,
)


def test_get_available_single_table_synthesizers():
    """Test the `get_available_single_table_synthesizers` method"""
    # Setup
    expected_synthesizers = [
        'CTGANSynthesizer',
        'ColumnSynthesizer',
        'CopulaGANSynthesizer',
        'DataIdentity',
        'GaussianCopulaSynthesizer',
        'RealTabFormerSynthesizer',
        'TVAESynthesizer',
        'UniformSynthesizer',
    ]

    # Run
    synthesizers = get_available_single_table_synthesizers()

    # Assert
    assert synthesizers == expected_synthesizers


def test_get_available_multi_table_synthesizers():
    """Test the `get_available_multi_table_synthesizers` method"""
    # Setup
    expected_synthesizers = ['HMASynthesizer']

    # Run
    synthesizers = get_available_multi_table_synthesizers()

    # Assert
    assert synthesizers == expected_synthesizers
