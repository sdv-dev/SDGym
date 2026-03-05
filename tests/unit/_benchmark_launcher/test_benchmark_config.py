"""Unit tests for the BenchmarkConfig class."""

from unittest.mock import Mock, patch

import pytest
import yaml

import sdgym._benchmark_launcher.benchmark_config as benchmark_config_module
from sdgym._benchmark_launcher.benchmark_config import BenchmarkConfig
from sdgym.errors import BenchmarkConfigError


class TestBenchmarkConfig:
    def test__init__(self):
        """Test the `__init__` method."""
        # Run
        config = BenchmarkConfig()

        # Assert
        assert config.modality is None
        assert config.method_params is None
        assert config.credentials_config == {}
        assert config.compute == {'service': None}
        assert config.instance_jobs == []
        assert config._is_validated is False

    def test_to_dict(self):
        """Test the `to_dict` method."""
        # Setup
        config = BenchmarkConfig()
        config.modality = 'single_table'
        config.method_params = {'output_destination': 's3://bucket/prefix/'}
        config.credentials_config = {'credential_filepath': 'creds.json'}
        config.compute = {'service': 'gcp', 'region': 'x'}
        config.instance_jobs = [{'synthesizers': ['A'], 'datasets': ['d1']}]

        # Run
        result = config.to_dict()
        result['compute']['region'] = 'changed'
        result['method_params']['output_destination'] = 's3://changed/'

        # Assert
        assert config.compute['region'] == 'x'
        assert config.method_params['output_destination'] == 's3://bucket/prefix/'

    def test__repr__(self):
        """Test the `__repr__` method."""
        # Setup
        config = BenchmarkConfig()
        config.modality = 'single_table'
        config.method_params = {'output_destination': 's3://bucket/prefix/'}
        config.credentials_config = {}
        config.compute = {'service': 'gcp'}
        config.instance_jobs = []

        # Run
        rendered = repr(config)

        # Assert
        assert isinstance(rendered, str)
        assert '"modality"' in rendered
        assert '"single_table"' in rendered

    @patch('sdgym._benchmark_launcher.benchmark_config._validate_structure', return_value=[])
    @patch('sdgym._benchmark_launcher.benchmark_config._validate_method_params', return_value=[])
    @patch(
        'sdgym._benchmark_launcher.benchmark_config._validate_credentials_config', return_value=[]
    )
    @patch('sdgym._benchmark_launcher.benchmark_config._validate_instance_jobs', return_value=[])
    @patch('sdgym._benchmark_launcher.benchmark_config._format_sectioned_errors')
    def test_validate(
        self,
        mock_format_errors,
        mock_validate_jobs,
        mock_validate_creds,
        mock_validate_method_params,
        mock_validate_structure,
    ):
        """Test the `validate` method."""
        # Setup
        config = BenchmarkConfig()
        config.modality = 'single_table'
        config.compute = {'service': 'gcp'}
        config.method_params = {'output_destination': 's3://bucket/prefix/'}
        config.credentials_config = {}
        config.instance_jobs = []

        method_to_run = Mock(name='method_to_run')
        with patch.dict(
            benchmark_config_module._METHODS,
            {('single_table', 'gcp'): method_to_run},
            clear=True,
        ):
            # Run
            config.validate()

        # Assert
        assert config._is_validated is True
        mock_format_errors.assert_not_called()
        mock_validate_structure.assert_called_once_with(config)
        mock_validate_method_params.assert_called_once_with(config.method_params, method_to_run)
        mock_validate_creds.assert_called_once_with(config.credentials_config)
        mock_validate_jobs.assert_called_once_with(config.instance_jobs)

    @patch.dict(
        'sdgym._benchmark_launcher.benchmark_config._METHODS',
        {('single_table', 'gcp'): Mock(name='method_to_run')},
        clear=True,
    )
    @patch(
        'sdgym._benchmark_launcher.benchmark_config._validate_structure',
        return_value=[],
    )
    @patch('sdgym._benchmark_launcher.benchmark_config._validate_method_params', return_value=[])
    @patch(
        'sdgym._benchmark_launcher.benchmark_config._validate_credentials_config', return_value=[]
    )
    @patch(
        'sdgym._benchmark_launcher.benchmark_config._validate_instance_jobs',
        return_value=['bad structure'],
    )
    @patch(
        'sdgym._benchmark_launcher.benchmark_config._format_sectioned_errors',
        return_value='FORMATTED',
    )
    def test_validate_raises_benchmark_config_error(
        self,
        mock_format_errors,
        mock_validate_jobs,
        mock_validate_creds,
        mock_validate_method_params,
        mock_validate_structure,
    ):
        """Test `validate` raises `BenchmarkConfigError` when any section has errors."""
        # Setup
        config = BenchmarkConfig()
        config.modality = 'single_table'
        config.compute = {'service': 'gcp'}
        config.method_params = {'output_destination': 's3://bucket/prefix/'}
        config.credentials_config = {}
        config.instance_jobs = []

        # Run and Assert
        with pytest.raises(BenchmarkConfigError, match='FORMATTED'):
            config.validate()

        # Assert
        mock_validate_structure.assert_called_once_with(config)
        mock_validate_method_params.assert_called_once_with(
            config.method_params, benchmark_config_module._METHODS[('single_table', 'gcp')]
        )
        mock_validate_creds.assert_called_once_with(config.credentials_config)
        mock_validate_jobs.assert_called_once_with(config.instance_jobs)
        assert config._is_validated is False
        mock_format_errors.assert_called_once()

    def test__validate_no_extra_keys(self):
        """Test the `_validate_no_extra_keys` method."""
        # Setup
        config = BenchmarkConfig()
        config_dict = {'modality': 'single_table', 'extra': 1}

        # Run and Assert
        with pytest.raises(ValueError, match='extra keys'):
            config._validate_no_extra_keys(config_dict)

    def test_load_from_dict(self):
        """Test the `load_from_dict` method."""
        # Setup
        config_dict = {
            'modality': 'single_table',
            'method_params': {'output_destination': 's3://bucket/prefix/'},
            'credentials': {'credential_filepath': 'creds.json'},
            'compute': {'service': 'gcp'},
            'instance_jobs': [{'synthesizers': ['A'], 'datasets': ['d1']}],
        }

        # Run
        config = BenchmarkConfig.load_from_dict(config_dict)

        # Assert
        assert config.modality == 'single_table'
        assert config.method_params == {'output_destination': 's3://bucket/prefix/'}
        assert config.credentials_config == {'credential_filepath': 'creds.json'}
        assert config.compute == {'service': 'gcp'}
        assert config.instance_jobs == [{'synthesizers': ['A'], 'datasets': ['d1']}]

    def test_load_from_yaml(self, tmp_path):
        """Test the `load_from_yaml` method."""
        # Setup
        path = tmp_path / 'config.yaml'
        config_dict = {
            'modality': 'single_table',
            'method_params': {'output_destination': 's3://bucket/prefix/'},
            'credentials': {'credential_filepath': 'creds.json'},
            'compute': {'service': 'gcp'},
            'instance_jobs': [{'synthesizers': ['A'], 'datasets': ['d1']}],
        }
        path.write_text(yaml.safe_dump(config_dict))

        # Run
        config = BenchmarkConfig.load_from_yaml(str(path))

        # Assert
        assert config.modality == 'single_table'
        assert config.method_params == {'output_destination': 's3://bucket/prefix/'}
        assert config.credentials_config == {'credential_filepath': 'creds.json'}
        assert config.compute == {'service': 'gcp'}
        assert config.instance_jobs == [{'synthesizers': ['A'], 'datasets': ['d1']}]

    @patch('sdgym._benchmark_launcher.benchmark_config.yaml.dump')
    def test_save_to_yaml(self, mock_yaml_dump, tmp_path):
        """Test the `save_to_yaml` method writes expected dict through `yaml.dump`."""
        # Setup
        config = BenchmarkConfig()
        config.modality = 'single_table'
        config.method_params = {'output_destination': 's3://bucket/prefix/'}
        config.credentials_config = {'credential_filepath': 'creds.json'}
        config.compute = {'service': 'gcp'}
        config.instance_jobs = [{'synthesizers': ['A'], 'datasets': ['d1']}]
        filepath = tmp_path / 'out.yaml'
        expected = {
            'modality': 'single_table',
            'method_params': {'output_destination': 's3://bucket/prefix/'},
            'credentials': {'credential_filepath': 'creds.json'},
            'compute': {'service': 'gcp'},
            'instance_jobs': [{'synthesizers': ['A'], 'datasets': ['d1']}],
        }

        # Run
        config.save_to_yaml(str(filepath))

        # Assert
        assert mock_yaml_dump.call_count == 1
        saved_dict = mock_yaml_dump.call_args[0][0]
        assert saved_dict == expected
        assert filepath.exists()
