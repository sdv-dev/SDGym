"""Tests for the clavaddmp module."""

from unittest.mock import MagicMock, patch

import pandas as pd
from sdv.metadata import Metadata

from sdgym.synthesizers.clavaddpm import ClavaDDPM, ClavaDDPMSynthesizer


class TestClavaDDPMSynthesizer:
    @patch('sdgym.synthesizers.clavaddpm.ClavaDDPM')
    def test__fit_mock(self, mock_clavaddpm):
        """Test `_fit` builds and fits mock ClavaDDPM."""
        # Setup
        synthesizer = ClavaDDPMSynthesizer()
        data = {'table1': pd.DataFrame({'col1': [1, 2, 3]})}
        metadata = MagicMock()

        # Run
        synthesizer._fit(data, metadata)

        # Assert
        mock_clavaddpm.assert_called_once_with(metadata)
        mock_clavaddpm.return_value.fit.assert_called_once_with(data)
        assert synthesizer._internal_synthesizer is mock_clavaddpm.return_value

    @patch('sdgym.synthesizers.clavaddpm.ClavaDDPM')
    def test__fit_passes_model_kwargs(self, mock_clavaddpm):
        """Test `_fit` forwards `_MODEL_KWARGS` to ClavaDDPM."""
        # Setup
        synthesizer = ClavaDDPMSynthesizer()
        synthesizer._MODEL_KWARGS = {'num_clusters': 5, 'num_timesteps': 100}
        data = {'table1': pd.DataFrame({'col1': [1, 2, 3]})}
        metadata = MagicMock()

        # Run
        synthesizer._fit(data, metadata)

        # Assert
        mock_clavaddpm.assert_called_once_with(metadata, num_clusters=5, num_timesteps=100)

    def test__fit(self):
        """Test the `_fit` method."""
        # Setup
        synthesizer = ClavaDDPMSynthesizer()
        synthesizer._MODEL_KWARGS = {'diffusion_iterations': 300, 'num_timesteps': 100}
        data = {
            'table1': pd.DataFrame({
                'col1': [1, 2, 3],
                'col2': ['A', 'B', 'C'],
            }),
            'table2': pd.DataFrame({
                'col3': [10.0, 20.0, 30.0],
                'col4': [True, False, True],
            }),
        }

        metadata = Metadata.load_from_dict({
            'tables': {
                'table1': {
                    'columns': {
                        'col1': {'sdtype': 'numerical'},
                        'col2': {'sdtype': 'categorical'},
                    },
                    'primary_key': 'col1',
                },
                'table2': {
                    'columns': {
                        'col3': {'sdtype': 'numerical'},
                        'col4': {'sdtype': 'boolean'},
                    },
                    'primary_key': 'col3',
                },
            },
            'relationships': [],
        })

        # Run
        synthesizer._fit(data, metadata)

        # Assert
        assert isinstance(synthesizer._internal_synthesizer, ClavaDDPM)
        assert isinstance(synthesizer, ClavaDDPMSynthesizer)

    @patch('sdgym.synthesizers.clavaddpm.ClavaDDPM')
    def test__get_trained_synthesizer(self, mock_clavaddpm):
        """Test `_get_trained_synthesizer` with mock."""
        # Setup
        synthesizer = ClavaDDPMSynthesizer()
        data = {'table1': pd.DataFrame({'col1': [1, 2, 3]})}
        metadata = MagicMock()

        # Run
        trained_synthesizer = synthesizer._get_trained_synthesizer(data, metadata)

        # Assert
        assert isinstance(trained_synthesizer, ClavaDDPMSynthesizer)
        assert trained_synthesizer._internal_synthesizer is mock_clavaddpm.return_value
        mock_clavaddpm.return_value.fit.assert_called_once_with(data)
