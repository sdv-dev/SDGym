import re
from unittest.mock import patch

import pytest
from sdv.single_table.base import BaseSingleTableSynthesizer

from sdgym.synthesizers._sdv_lookup import _find_synthesizer_type, find_sdv_synthesizer


@patch('sdgym.synthesizers._sdv_lookup.importlib.import_module')
def test__find_synthesizer_type(mock_import_module):
    """Test `_find_synthesizer_type` method."""

    # Setup
    class MockSynthesizer(BaseSingleTableSynthesizer):
        pass

    name_valid = 'GaussianCopulaSynthesizer'
    name_invalid = 'InvalidSynthesizer'
    synthesizer_type = 'single_table'
    mock_module = type('MockModule', (), {name_valid: MockSynthesizer})
    mock_import_module.return_value = mock_module
    expected_error = re.escape('`synthesizer_type` must be one `single_table` or `multi_table`.')

    # Run
    with pytest.raises(ValueError, match=expected_error):
        _find_synthesizer_type(name_valid, 'invalid_type')

    output_valid = _find_synthesizer_type(name_valid, synthesizer_type)
    output_invalid = _find_synthesizer_type(name_invalid, synthesizer_type)

    # Assert
    mock_import_module.assert_called_with(f'sdv.{synthesizer_type}')
    assert output_valid == MockSynthesizer
    assert output_invalid is None


@patch('sdgym.synthesizers._sdv_lookup._find_synthesizer_type')
def test_find_sdv_synthesizer(mock_find_synthesizer_type):
    """Test `find_sdv_synthesizer` method."""
    # Setup
    mock_find_synthesizer_type.side_effect = [
        'GaussianCopulaSynthesizer',
        None,
        'HMASynthesizer',
        None,
        None,
    ]
    expected_error = re.escape("SDV synthesizer 'UnknownSynthesizer' not found")

    # Run
    st_synthesizer = find_sdv_synthesizer('GaussianCopulaSynthesizer')
    mt_synthesizer = find_sdv_synthesizer('HMASynthesizer')
    with pytest.raises(KeyError, match=expected_error):
        find_sdv_synthesizer('UnknownSynthesizer')

    # Assert
    assert st_synthesizer == ('GaussianCopulaSynthesizer', 'single_table')
    assert mt_synthesizer == ('HMASynthesizer', 'multi_table')
