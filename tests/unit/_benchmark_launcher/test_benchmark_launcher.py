from unittest.mock import Mock, call, patch

import pytest

from sdgym._benchmark_launcher.benchmark_config import BenchmarkConfig
from sdgym._benchmark_launcher.benchmark_launcher import BenchmarkLauncher
from sdgym._benchmark_launcher.utils import _METHODS


class TestBenchmarkLauncher:
    @patch('sdgym._benchmark_launcher.benchmark_launcher.generate_benchmark_id')
    def test__init__(self, mock_generate_benchmark_id):
        """Test the `__init__` method of BenchmarkLauncher."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        mock_generate_benchmark_id.return_value = 'unique_id'

        # Run
        launcher = BenchmarkLauncher(benchmark_config)

        # Assert
        benchmark_config.validate.assert_called_once()
        mock_generate_benchmark_id.assert_called_once_with(launcher)
        assert launcher.benchmark_config == benchmark_config
        assert launcher.method_to_run == _METHODS[('single_table', 'gcp')]
        assert launcher.benchmark_id == 'unique_id'

    def test_launch_calls_validate_when_not_validated(self):
        """Test `launch` calls `validate` when `_is_validated` is False."""
        # Setup
        config = Mock()
        config.modality = 'single_table'
        config.compute = {'service': 'gcp'}
        config._is_validated = False
        config.validate = Mock()
        launcher = BenchmarkLauncher(config)
        config.validate.reset_mock()  # Reset call count after __init__
        launcher._launch = Mock()

        # Run
        launcher.launch()

        # Assert
        config.validate.assert_called_once()

    def test_launch_already_validated(self):
        """Test `launch` when config already validated."""
        # Setup
        config = BenchmarkConfig.load_from_dict({
            'modality': 'single_table',
            'compute': {'service': 'gcp'},
        })
        config._is_validated = True
        config.validate = Mock()
        launcher = BenchmarkLauncher(config)
        config.validate.reset_mock()  # Reset call count after __init__
        launcher._launch = Mock()

        # Run
        launcher.launch()

        # Assert
        config.validate.assert_not_called()
        launcher._launch.assert_called_once_with()

    @patch(
        'sdgym._benchmark_launcher.benchmark_launcher.resolve_credentials',
        return_value={'aws': {}, 'gcp': {}, 'sdv': {}},
    )
    @patch(
        'sdgym._benchmark_launcher.benchmark_launcher._resolve_datasets',
        side_effect=[['d1'], ['d2']],
    )
    def test_launch_internal_calls_method_for_each_job(
        self, mock_resolve_datasets, mock_resolve_credentials
    ):
        """Test `_launch` calls the underlying benchmark method for each job."""
        # Setup
        config = BenchmarkConfig.load_from_dict({
            'modality': 'single_table',
            'compute': {'service': 'gcp'},
            'method_params': {
                'output_destination': 's3://bucket/prefix/',
                'timeout': 10,
                'compute_quality_score': True,
                'compute_diagnostic_score': True,
                'compute_privacy_score': False,
            },
            'credentials': {'credential_filepath': 'creds.json'},
            'instance_jobs': [
                {'synthesizers': ['Synth1'], 'datasets': ['a']},
                {'synthesizers': ['Synth2'], 'datasets': ['b']},
            ],
        })
        config.validate = Mock()
        launcher = BenchmarkLauncher(config)
        launcher.method_to_run = Mock(name='method_to_run')

        # Run
        launcher._launch()

        # Assert
        mock_resolve_credentials.assert_called_once_with(config.credentials_config)
        assert mock_resolve_datasets.call_args_list == [call(['a']), call(['b'])]
        expected_calls = [
            call(
                synthesizers=['Synth1'],
                sdv_datasets=['d1'],
                credentials={'aws': {}, 'gcp': {}, 'sdv': {}},
                compute_config=config.compute,
                **config.method_params,
            ),
            call(
                synthesizers=['Synth2'],
                sdv_datasets=['d2'],
                credentials={'aws': {}, 'gcp': {}, 'sdv': {}},
                compute_config=config.compute,
                **config.method_params,
            ),
        ]
        launcher.method_to_run.assert_has_calls(expected_calls, any_order=False)
        assert launcher.method_to_run.call_count == 2

    def test_terminate(self):
        """Test the `terminate` method of BenchmarkLauncher."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        launcher = BenchmarkLauncher(benchmark_config)

        # Run and Assert
        with pytest.raises(NotImplementedError):
            launcher.terminate()

    def test_get_status(self):
        """Test the `get_status` method of BenchmarkLauncher."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        launcher = BenchmarkLauncher(benchmark_config)

        # Run and Assert
        with pytest.raises(NotImplementedError):
            launcher.get_status()
