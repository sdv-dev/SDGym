"""Unit tests for the BenchmarkLauncher class."""

import re
from unittest.mock import Mock, call, mock_open, patch

import pytest

from sdgym._benchmark_launcher.benchmark_config import BenchmarkConfig
from sdgym._benchmark_launcher.benchmark_launcher import BenchmarkLauncher
from sdgym._benchmark_launcher.utils import _METHODS


class TestBenchmarkLauncher:
    @patch('sdgym._benchmark_launcher.benchmark_launcher.generate_benchmark_ids')
    def test__init__(self, mock_generate_benchmark_ids):
        """Test the `__init__` method."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        mock_generate_benchmark_ids.return_value = 'unique_id'

        # Run
        launcher = BenchmarkLauncher(benchmark_config)

        # Assert
        benchmark_config.validate.assert_called_once()
        mock_generate_benchmark_ids.assert_called_once_with([
            'BENCMARK_ID',
            'single_table',
            'gcp',
        ])
        assert launcher.benchmark_config == benchmark_config
        assert launcher.method_to_run == _METHODS[('single_table', 'gcp')]
        assert launcher._benchmark_id == 'unique_id'
        assert launcher._launch_to_instance_names == {}
        assert launcher._instance_name_to_status == {}

    def test_launch_calls_validate_when_not_validated(self):
        """Test `launch` calls `validate` when `_is_validated` is False."""
        # Setup
        config = Mock()
        config.modality = 'single_table'
        config.compute = {'service': 'gcp'}
        config._is_validated = False
        config.validate = Mock()
        launcher = BenchmarkLauncher(config)
        config.validate.reset_mock()
        launcher._launch = Mock()

        # Run
        launcher.launch()

        # Assert
        config.validate.assert_called_once()
        launcher._launch.assert_called_once_with()

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
        config.validate.reset_mock()
        launcher._launch = Mock()

        # Run
        launcher.launch()

        # Assert
        config.validate.assert_not_called()
        launcher._launch.assert_called_once_with()

    @patch('sdgym._benchmark_launcher.benchmark_launcher.generate_benchmark_ids')
    @patch(
        'sdgym._benchmark_launcher.benchmark_launcher.resolve_credentials',
        return_value={'aws': {}, 'gcp': {}, 'sdv': {}},
    )
    @patch(
        'sdgym._benchmark_launcher.benchmark_launcher._resolve_datasets',
        side_effect=[['d1'], ['d2']],
    )
    def test_launch_internal_calls_method_for_each_job(
        self, mock_resolve_datasets, mock_resolve_credentials, mock_generate_benchmark_ids
    ):
        """Test `_launch` calls the underlying benchmark method for each job."""
        # Setup
        output_destination = 's3://bucket/prefix/'
        config = BenchmarkConfig.load_from_dict({
            'modality': 'single_table',
            'compute': {'service': 'gcp'},
            'method_params': {
                'timeout': 10,
                'compute_quality_score': True,
                'compute_diagnostic_score': True,
                'compute_privacy_score': False,
            },
            'credentials_filepath': {'credentials_filepath': 'creds.json'},
            'instance_jobs': [
                {
                    'synthesizers': ['Synth1'],
                    'datasets': ['a'],
                    'output_destination': output_destination,
                },
                {
                    'synthesizers': ['Synth2'],
                    'datasets': ['b'],
                    'output_destination': output_destination,
                },
            ],
        })
        config.validate = Mock()
        launcher = BenchmarkLauncher(config)
        launcher.method_to_run = Mock(name='method_to_run')
        launcher.method_to_run.side_effect = ['instance-1', 'instance-2']
        mock_generate_benchmark_ids.return_value = 'LAUNCH_ID_1'

        # Run
        launcher._launch()

        # Assert
        mock_resolve_credentials.assert_called_once_with(config.credentials_filepath)
        assert mock_resolve_datasets.call_args_list == [call(['a']), call(['b'])]
        expected_calls = [
            call(
                output_destination=output_destination,
                synthesizers=['Synth1'],
                sdv_datasets=['d1'],
                credentials={'aws': {}, 'gcp': {}, 'sdv': {}},
                compute_config=config.compute,
                **config.method_params,
            ),
            call(
                output_destination=output_destination,
                synthesizers=['Synth2'],
                sdv_datasets=['d2'],
                credentials={'aws': {}, 'gcp': {}, 'sdv': {}},
                compute_config=config.compute,
                **config.method_params,
            ),
        ]
        launcher.method_to_run.assert_has_calls(expected_calls, any_order=False)
        assert launcher.method_to_run.call_count == 2
        assert launcher._launch_to_instance_names == {'LAUNCH_ID_1': ['instance-1', 'instance-2']}
        assert launcher._instance_name_to_status == {
            'instance-1': 'running',
            'instance-2': 'running',
        }

    def test_update_gcp_instance_name_to_status(self):
        """Test the `_update_gcp_instance_name_to_status` method."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        launcher = BenchmarkLauncher(benchmark_config)
        launcher._instance_name_to_status = {
            'instance-1': 'running',
            'instance-2': 'running',
            'instance-3': 'stopped',
        }
        launcher._get_gcp_client = Mock(return_value=(Mock(), 'test-project'))
        launcher._list_gcp_instances = Mock(
            return_value=[
                {
                    'id': '123',
                    'name': 'instance-1',
                    'zone': 'us-central1-a',
                    'status': 'RUNNING',
                }
            ]
        )
        launcher._get_all_instance_names = Mock(
            return_value=['instance-1', 'instance-2', 'instance-3']
        )

        # Run
        launcher._update_gcp_instance_name_to_status()

        # Assert
        assert launcher._instance_name_to_status == {
            'instance-1': 'running',
            'instance-2': 'completed',
            'instance-3': 'stopped',
        }

    def test_update_instance_name_to_status(self):
        """Test the `_update_instance_name_to_status` method."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        launcher = BenchmarkLauncher(benchmark_config)
        launcher._update_gcp_instance_name_to_status = Mock()

        # Run
        launcher._update_instance_name_to_status()

        # Assert
        launcher._update_gcp_instance_name_to_status.assert_called_once_with()

    def test_list_gcp_instances(self):
        """Test the `_list_gcp_instances` method."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        launcher = BenchmarkLauncher(benchmark_config)

        running_instance = Mock()
        running_instance.id = 123
        running_instance.name = 'instance-1'
        running_instance.zone = (
            'https://www.googleapis.com/compute/v1/projects/test-project/zones/us-central1-a'
        )
        running_instance.status = 'RUNNING'

        terminated_instance = Mock()
        terminated_instance.id = 456
        terminated_instance.name = 'instance-2'
        terminated_instance.zone = (
            'https://www.googleapis.com/compute/v1/projects/test-project/zones/us-central1-b'
        )
        terminated_instance.status = 'TERMINATED'

        empty_scoped_list = Mock()
        empty_scoped_list.instances = None

        scoped_list = Mock()
        scoped_list.instances = [running_instance, terminated_instance]

        client = Mock()
        client.aggregated_list.return_value = [
            ('zones/us-central1-a', scoped_list),
            ('zones/us-central1-b', empty_scoped_list),
        ]

        # Run
        result = launcher._list_gcp_instances(client, 'test-project')

        # Assert
        client.aggregated_list.assert_called_once_with(project='test-project')
        assert result == [
            {
                'id': '123',
                'name': 'instance-1',
                'zone': 'us-central1-a',
                'status': 'RUNNING',
            }
        ]

    def test_get_all_instance_names(self):
        """Test the `_get_all_instance_names` method."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        launcher = BenchmarkLauncher(benchmark_config)
        launcher._launch_to_instance_names = {
            'launch-1': ['instance-1', 'instance-2'],
            'launch-2': ['instance-3'],
        }

        # Run
        result = launcher._get_all_instance_names()

        # Assert
        assert result == ['instance-1', 'instance-2', 'instance-3']

    def test_get_active_instance_names(self):
        """Test the `_get_active_instance_names` method."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        launcher = BenchmarkLauncher(benchmark_config)
        launcher._instance_name_to_status = {
            'instance-1': 'running',
            'instance-2': 'terminated',
            'instance-3': 'running',
        }

        # Run
        result = launcher._get_active_instance_names()

        # Assert
        assert result == ['instance-1', 'instance-3']

    def test_validate_instance_names(self):
        """Test the `_validate_instance_names` method."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        launcher = BenchmarkLauncher(benchmark_config)
        launcher._get_all_instance_names = Mock(return_value=['instance-1', 'instance-2'])

        # Run
        result = launcher._validate_instance_names(['instance-1'])

        # Assert
        assert result == ['instance-1']

    def test_validate_instance_names_raises_error_for_unknown_instances(self):
        """Test `_validate_instance_names` raises an error for unknown instances."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        launcher = BenchmarkLauncher(benchmark_config)
        launcher._get_all_instance_names = Mock(return_value=['instance-1'])
        expected_error = re.escape(
            'Some provided instance names were not launched by this BenchmarkLauncher.'
            " Unknown: 'instance-2'. Launched instances: 'instance-1'."
        )

        # Run and Assert
        with pytest.raises(ValueError, match=expected_error):
            launcher._validate_instance_names(['instance-2'])

    @patch('sdgym._benchmark_launcher.benchmark_launcher.resolve_credentials')
    @patch(
        'sdgym._benchmark_launcher.benchmark_launcher.service_account.Credentials.from_service_account_info'
    )
    @patch('sdgym._benchmark_launcher.benchmark_launcher.compute_v1.InstancesClient')
    @patch('sdgym._benchmark_launcher.benchmark_launcher._validate_gcp_credentials')
    def test_get_gcp_client(
        self,
        mock_validate_gcp_credentials,
        mock_instances_client,
        mock_from_service_account_info,
        mock_resolve_credentials,
    ):
        """Test the `_get_gcp_client` method."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        benchmark_config.credentials_filepath = 'creds.json'
        launcher = BenchmarkLauncher(benchmark_config)

        mock_resolve_credentials.return_value = {
            'gcp': {
                'project_id': 'test-project',
                'client_email': 'test@test.com',
                'token_uri': 'https://oauth2.googleapis.com/token',
                'private_key_id': 'key-id',
                'private_key': '-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----\n',
            }
        }
        mock_validate_gcp_credentials.return_value = []
        mock_credentials = Mock()
        mock_from_service_account_info.return_value = mock_credentials
        mock_client = Mock()
        mock_instances_client.return_value = mock_client

        # Run
        result_client, result_project_id = launcher._get_gcp_client()

        # Assert
        mock_resolve_credentials.assert_called_once_with('creds.json')
        mock_validate_gcp_credentials.assert_called_once_with(mock_resolve_credentials.return_value)
        mock_from_service_account_info.assert_called_once_with(
            mock_resolve_credentials.return_value['gcp']
        )
        mock_instances_client.assert_called_once_with(credentials=mock_credentials)
        assert result_client is mock_client
        assert result_project_id == 'test-project'

    @patch('builtins.print')
    def test__terminate_gcp_instances(self, mock_print):
        """Test the `_terminate_gcp_instances` method."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        launcher = BenchmarkLauncher(benchmark_config)
        launcher._get_gcp_client = Mock(return_value=(Mock(), 'test-project'))
        launcher._list_gcp_instances = Mock(
            return_value=[
                {
                    'id': '123',
                    'name': 'instance-1',
                    'zone': 'us-central1-a',
                    'status': 'RUNNING',
                },
                {
                    'id': '456',
                    'name': 'instance-2',
                    'zone': 'us-central1-b',
                    'status': 'RUNNING',
                },
            ]
        )
        client = launcher._get_gcp_client.return_value[0]
        mock_operation_1 = Mock()
        mock_operation_2 = Mock()
        client.delete.side_effect = [mock_operation_1, mock_operation_2]

        # Run
        deleted_instances = launcher._terminate_gcp_instances(
            instance_names=['instance-1', 'instance-2'],
            verbose=True,
        )

        # Assert
        assert len(deleted_instances) == 2
        launcher._get_gcp_client.assert_called_once_with()
        launcher._list_gcp_instances.assert_called_once_with(client, 'test-project')
        assert client.delete.call_args_list == [
            call(project='test-project', zone='us-central1-a', instance='instance-1'),
            call(project='test-project', zone='us-central1-b', instance='instance-2'),
        ]
        mock_operation_1.result.assert_called_once_with()
        mock_operation_2.result.assert_called_once_with()
        assert deleted_instances == ['instance-1', 'instance-2']
        assert launcher._instance_name_to_status == {
            'instance-1': 'stopped',
            'instance-2': 'stopped',
        }
        mock_print.assert_has_calls([
            call("Terminating GCP instance 'instance-1' (id=123, zone=us-central1-a)..."),
            call("Terminating GCP instance 'instance-2' (id=456, zone=us-central1-b)..."),
        ])

    @patch('sdgym._benchmark_launcher.benchmark_launcher.LOGGER')
    def test__terminate_gcp_instances_for_not_running_instances(self, mock_logger):
        """Test `_terminate_gcp_instances` logs a message for not running instances."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        launcher = BenchmarkLauncher(benchmark_config)
        launcher._get_gcp_client = Mock(return_value=(Mock(), 'test-project'))
        launcher._list_gcp_instances = Mock(
            return_value=[
                {
                    'id': '123',
                    'name': 'instance-1',
                    'zone': 'us-central1-a',
                    'status': 'RUNNING',
                },
            ]
        )

        # Run
        launcher._terminate_gcp_instances(
            instance_names=['instance-1', 'instance-2'],
            verbose=False,
        )

        # Assert
        mock_logger.info.assert_called_once_with(
            "Some provided instance names are not currently running: 'instance-2'."
        )

    def test_validate_inputs_and_get_instances(self):
        """Test the `_validate_inputs_and_get_instances` method."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        launcher = BenchmarkLauncher(benchmark_config)
        launcher._validate_instance_names = Mock(return_value=['instance-1', 'instance-2'])

        # Run
        result = launcher._validate_inputs_and_get_instances(
            instance_names=['instance-1', 'instance-2'],
            verbose=True,
        )

        # Assert
        launcher._validate_instance_names.assert_called_once_with(['instance-1', 'instance-2'])
        assert result == ['instance-1', 'instance-2']

    @pytest.mark.parametrize(
        ('compute_service', 'verbose', 'expected_error'),
        [
            (
                'aws',
                True,
                NotImplementedError('`terminate()` is only implemented for GCP instances for now.'),
            ),
            (
                'gcp',
                1,
                ValueError("`verbose` must be a boolean. Found: 1 (<class 'int'>)."),
            ),
            (
                'gcp',
                'yes',
                ValueError("`verbose` must be a boolean. Found: 'yes' (<class 'str'>)."),
            ),
        ],
    )
    def test_validate_inputs_and_get_instances_invalid_cases(
        self, compute_service, verbose, expected_error
    ):
        """Test `_validate_inputs_and_get_instances` raises errors for invalid inputs."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        launcher = BenchmarkLauncher(benchmark_config)
        launcher.compute_service = compute_service
        launcher._validate_instance_names = Mock(return_value=['instance-1'])

        # Run and Assert
        with pytest.raises(type(expected_error), match=re.escape(str(expected_error))):
            launcher._validate_inputs_and_get_instances(
                instance_names=['instance-1'],
                verbose=verbose,
            )

        launcher._validate_instance_names.assert_not_called()

    @patch('builtins.print')
    def test_terminate_mock(self, mock_print):
        """Test the `terminate` method with a mock."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        launcher = BenchmarkLauncher(benchmark_config)
        launcher._validate_inputs_and_get_instances = Mock(
            return_value=['instance-1', 'instance-2']
        )
        launcher._update_instance_name_to_status = Mock()
        launcher._get_active_instance_names = Mock(return_value=['instance-1', 'instance-2'])
        launcher._terminate_gcp_instances = Mock(return_value=['instance-1', 'instance-2'])

        # Run
        launcher.terminate(instance_names=['instance-1', 'instance-2'], verbose=True)

        # Assert
        launcher._validate_inputs_and_get_instances.assert_called_once_with(
            ['instance-1', 'instance-2'], True
        )
        assert launcher._update_instance_name_to_status.call_count == 2
        launcher._get_active_instance_names.assert_called_once_with()
        launcher._terminate_gcp_instances.assert_called_once_with(
            ['instance-1', 'instance-2'], True
        )
        mock_print.assert_called_once_with('Terminated 2 GCP instance(s).')

    @patch('sdgym._benchmark_launcher.benchmark_launcher.LOGGER')
    def test_terminate_logs_when_no_running_instances(self, mock_logger):
        """Test the `terminate` method logs when there are no running instances to terminate."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        launcher = BenchmarkLauncher(benchmark_config)
        launcher._validate_instance_names = Mock(return_value=['instance-1'])
        launcher._update_instance_name_to_status = Mock()
        launcher._get_active_instance_names = Mock(return_value=[])

        # Run
        launcher.terminate(instance_names=None, verbose=False)

        # Assert
        mock_logger.info.assert_called_once_with('There are no running instances to terminate.')

    def test_terminate_warns_when_all_requested_instances_are_terminated(self):
        """Test `terminate` warns when all requested instances are terminated."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        launcher = BenchmarkLauncher(benchmark_config)
        launcher._validate_instance_names = Mock(return_value=['instance-1'])
        launcher._update_instance_name_to_status = Mock()
        launcher._get_active_instance_names = Mock(return_value=[])
        expected_warning = re.escape('All provided instance names are already terminated.')

        # Run and Assert
        with pytest.warns(UserWarning, match=expected_warning):
            launcher.terminate(instance_names=['instance-1'], verbose=False)

    def test_terminate_not_gcp(self):
        """Test the `terminate` method when not using GCP."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        launcher = BenchmarkLauncher(benchmark_config)
        launcher.compute_service = 'aws'
        expected_error = re.escape('`terminate()` is only implemented for GCP instances for now.')

        # Run and Assert
        with pytest.raises(NotImplementedError, match=expected_error):
            launcher.terminate()

    def test_get_status(self):
        """Test the `get_status` method."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        launcher = BenchmarkLauncher(benchmark_config)

        # Run and Assert
        with pytest.raises(NotImplementedError):
            launcher.get_status()

    @patch('sdgym._benchmark_launcher.benchmark_launcher.cloudpickle.dump')
    @patch('builtins.open', new_callable=mock_open)
    def test_save(self, mock_file, mock_dump):
        """Test the `save` method."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        launcher = BenchmarkLauncher(benchmark_config)

        # Run
        launcher.save('launcher.pkl')

        # Assert
        mock_file.assert_called_once_with('launcher.pkl', 'wb')
        mock_dump.assert_called_once()
        assert mock_dump.call_args[0][0] is launcher

    @patch('sdgym._benchmark_launcher.benchmark_launcher.cloudpickle.load')
    @patch('builtins.open', new_callable=mock_open, read_data=b'test')
    def test_load(self, mock_file, mock_load):
        """Test the `load` method."""
        # Setup
        benchmark = Mock()
        benchmark._benchmark_id = 'existing-id'
        mock_load.return_value = benchmark

        # Run
        result = BenchmarkLauncher.load('launcher.pkl')

        # Assert
        mock_file.assert_called_once_with('launcher.pkl', 'rb')
        mock_load.assert_called_once()
        assert result is benchmark

    @patch('sdgym._benchmark_launcher.benchmark_launcher.generate_benchmark_ids')
    @patch('sdgym._benchmark_launcher.benchmark_launcher.cloudpickle.load')
    @patch('builtins.open', new_callable=mock_open, read_data=b'test')
    def test_load_generates_benchmark_id_if_missing(
        self, mock_file, mock_load, mock_generate_benchmark_ids
    ):
        """Test `load` generates a benchmark id if missing."""
        # Setup
        benchmark = Mock()
        benchmark._benchmark_id = None
        benchmark.modality = 'single_table'
        benchmark.compute_service = 'gcp'
        mock_load.return_value = benchmark
        mock_generate_benchmark_ids.return_value = 'new-id'

        # Run
        result = BenchmarkLauncher.load('launcher.pkl')

        # Assert
        mock_file.assert_called_once_with('launcher.pkl', 'rb')
        mock_load.assert_called_once()
        mock_generate_benchmark_ids.assert_called_once_with([
            'BENCMARK_ID',
            'single_table',
            'gcp',
        ])
        assert result._benchmark_id == 'new-id'
