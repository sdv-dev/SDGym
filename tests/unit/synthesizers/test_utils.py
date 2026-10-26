import pytest
import pandas as pd
from sdv.metadata import Metadata
from sdgym.synthesizers.utils import _get_supported_synthesizers, _get_single_table_name


def test__get_supported_synthesizers():
    """Test the `_get_supported_synthesizers` method."""
    # Setup
    expected_synthesizers = [
        'CTGANSynthesizer',
        'ClavaDDPMSynthesizer',
        'ColumnSynthesizer',
        'CopulaGANSynthesizer',
        'DataDesignerSynthesizer',
        'DataIdentity',
        'GaussianCopulaSynthesizer',
        'HMASynthesizer',
        'MultiTableUniformSynthesizer',
        'RealTabFormerSynthesizer',
        'TVAESynthesizer',
        'TabDDPMSynthesizer',
        'UniformSynthesizer',
    ]

    # Run
    synthesizers = _get_supported_synthesizers()

    # Assert
    assert synthesizers == expected_synthesizers

def test__get_single_table_name():
    """Test the `_get_single_table_name` method."""
    # Setup
    data = {
        'table1': pd.DataFrame({'col_a': [1, 2, 3], 'col_b': ['a', 'b', 'c']})
    }
    metadata_dict = {
        'tables': {
            'table1': {
                'columns': {
                    'col_a': {'sdtype': 'numerical'},
                    'col_b': {'sdtype': 'categorical'}
                }
            }
        }
    }

    metadata_object = Metadata.load_from_dict(metadata_dict)

    # Run
    result_dict = _get_single_table_name(data, metadata_dict)
    result_object = _get_single_table_name(data, metadata_object)

    # Assert
    assert result_dict == result_object == 'table1'


def test__get_single_table_name_errors():
    """Test the `_get_single_table_name` method.
    
    Expected to error when more than one table is given with
    non Metadata object.
    """
    # Setup
    data = {
        'table1': pd.DataFrame({'col_a': [1, 2, 3], 'col_b': ['a', 'b', 'c']}),
        'table2': pd.DataFrame({'col': ['x', 'y', 'z']})
    }
    metadata_dict = {
        'tables': {
            'table1': {
                'columns': {
                    'col_a': {'sdtype': 'numerical'},
                    'col_b': {'sdtype': 'categorical'}
                }
            }
        }
    }
    metadata_object = Metadata.load_from_dict(metadata_dict)
    expected_msg = "The data contains more than one table. Could not determine 'table name'."

    # Run
    with pytest.raises(ValueError, match=expected_msg):
        _get_single_table_name(data, metadata_dict)
    
    result_object = _get_single_table_name(data, metadata_object)

    # Assert
    assert result_object == 'table1'