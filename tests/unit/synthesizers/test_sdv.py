import types
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from sdv.metadata import Metadata
from sdv.multi_table import HMASynthesizer
from sdv.single_table import GaussianCopulaSynthesizer

import sdgym.synthesizers.sdv as sdv_mod
from sdgym.synthesizers.sdv import SDVMultiTableBaseline, SDVSingleTableBaseline


class TestSDVSingleTableBaseline:
    """Test the `SDVSingleTableBaseline` class."""

    def test__get_trained_synthesizer(self):
        """Test the `_get_trained_synthesizer` method."""
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
        invalid_synthesizer = SDVSingleTableBaseline()
        valid_synthesizer = SDVSingleTableBaseline()
        valid_synthesizer._SDV_CLASS = GaussianCopulaSynthesizer
        valid_synthesizer._MODEL_KWARGS = {'enforce_min_max_values': False}
        expected_error = 'The synthesizer has no `_SDV_CLASS` set'

        # Run
        valid_model = valid_synthesizer._get_trained_synthesizer(data, metadata)
        with pytest.raises(ValueError, match=expected_error):
            invalid_synthesizer._get_trained_synthesizer(data, metadata)

        # Assert
        assert isinstance(valid_model, GaussianCopulaSynthesizer)
        assert valid_model.enforce_min_max_values is False

    def test__sample_from_synthesizer(self):
        """Test the `_sample_from_synthesizer` method."""
        # Setup
        data = pd.DataFrame({
            'column1': [1, 2, 3, 4, 5],
            'column2': ['A', 'B', 'C', 'D', 'E'],
        })
        base_synthesizer = SDVSingleTableBaseline()
        synthesizer = Mock()
        synthesizer.sample.return_value = data
        n_samples = 3

        # Run
        sampled_data = base_synthesizer._sample_from_synthesizer(synthesizer, n_samples)

        # Assert
        pd.testing.assert_frame_equal(sampled_data, data)
        synthesizer.sample.assert_called_once_with(n_samples)


class TestSDVMultiTableBaseline:
    @patch('sdgym.synthesizers.sdv._get_trained_synthesizer')
    def test__get_trained_synthesizer(self, mock_get_trained):
        """Test the `_get_trained_synthesizer` method."""
        # Setup
        data = {
            'table_1': pd.DataFrame({
                'column1': [1, 2, 3, 4, 5],
                'column2': ['A', 'B', 'C', 'D', 'E'],
            })
        }
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
        multi_table_synthesizer = SDVMultiTableBaseline()
        multi_table_synthesizer._SDV_CLASS = HMASynthesizer
        multi_table_synthesizer._MODEL_KWARGS = {'locales': ['es_ES']}

        # Run
        multi_table_synthesizer._get_trained_synthesizer(data, metadata)

        # Assert
        mock_get_trained.assert_called_once_with(
            HMASynthesizer,
            {'locales': ['es_ES']},
            data,
            metadata,
        )

    def test__sample_from_synthesizer(self):
        """Test the `_sample_from_synthesizer` method."""
        # Setup
        data = {
            'table_1': pd.DataFrame({
                'column1': [1, 2, 3, 4, 5],
                'column2': ['A', 'B', 'C', 'D', 'E'],
            })
        }
        base_synthesizer = SDVMultiTableBaseline()
        synthesizer = Mock()
        synthesizer.sample.return_value = data

        # Run
        sampled_data = base_synthesizer._sample_from_synthesizer(synthesizer, scale=1)

        # Assert
        assert sampled_data == data
        synthesizer.sample.assert_called_once_with(1)


@patch('builtins.__import__')
def test__create_wrappers(mock_import):
    """Test the `_create_wrappers` method."""
    # Setup
    fake_single = types.SimpleNamespace(
        GaussianCopulaSynthesizer=object,
        CTGANSynthesizer=object,
        not_upper='ignore_me',
    )
    fake_multi = types.SimpleNamespace(
        HMASynthesizer=object,
        HSASynthesizer=object,
        lower='also_ignore',
    )

    def import_side_effect(name, fromlist=None):
        if name == 'sdv.single_table':
            return fake_single
        if name == 'sdv.multi_table':
            return fake_multi
        raise ImportError

    mock_import.side_effect = import_side_effect
    expected = ['GaussianCopulaSynthesizer', 'CTGANSynthesizer', 'HMASynthesizer']
    before_keys = set(sdv_mod.__dict__.keys())

    try:
        # Run
        sdv_mod._create_wrappers()

        # Assert
        mock_import.assert_any_call('sdv.single_table', fromlist=['*'])
        mock_import.assert_any_call('sdv.multi_table', fromlist=['*'])
        for synth_name in expected:
            assert synth_name in sdv_mod.__dict__
            synth_class = sdv_mod.__dict__[synth_name]
            if synth_name in ('GaussianCopulaSynthesizer', 'CTGANSynthesizer'):
                assert issubclass(synth_class, sdv_mod.SDVSingleTableBaseline)
            else:
                assert issubclass(synth_class, sdv_mod.SDVMultiTableBaseline)

    finally:
        # Clean up: remove whatever this test added
        after_keys = set(sdv_mod.__dict__.keys())
        new_keys = after_keys - before_keys
        for key in new_keys:
            del sdv_mod.__dict__[key]
