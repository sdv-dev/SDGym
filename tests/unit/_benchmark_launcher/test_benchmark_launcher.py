"""Unit tests for the BenchmarkLauncher class."""

import re
from unittest.mock import Mock, call, mock_open, patch

import pytest
from botocore.exceptions import ClientError

from sdgym._benchmark_launcher.benchmark_config import BenchmarkConfig
from sdgym._benchmark_launcher.benchmark_launcher import BenchmarkLauncher
from sdgym._benchmark_launcher.utils import _METHODS


class TestBenchmarkLauncher:
    @patch('sdgym._benchmark_launcher.benchmark_launcher.generate_ids')
    @patch('sdgym._benchmark_launcher.benchmark_launcher.GCPInstanceManager')
    def test__init__(self, mock_instance_manager, mock_generate_ids):
        """Test the `__init__` method."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        benchmark_config.credentials_filepath = 'creds.json'
        mock_generate_ids.return_value = 'unique_id'
        instance_manager = Mock()
        mock_instance_manager.return_value = instance_manager

        # Run
        launcher = BenchmarkLauncher(benchmark_config)

        # Assert
        benchmark_config.validate.assert_called_once()
        mock_generate_ids.assert_called_once_with([
            'BENCMARK_ID',
            'single_table',
            'gcp',
        ])
        mock_instance_manager.assert_called_once_with('creds.json')
        assert launcher.benchmark_config == benchmark_config
        assert launcher.method_to_run == _METHODS[('single_table', 'gcp')]
        assert launcher._benchmark_id == 'unique_id'
        assert launcher._launch_to_instance_names == {}
        assert launcher._instance_name_to_status == {}
        assert launcher._instance_manager is instance_manager

    @patch('sdgym._benchmark_launcher.benchmark_launcher.GCPInstanceManager')
    def test_build_instance_manager(self, mock_instance_manager):
        """Test the `_build_instance_manager` method."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        benchmark_config.credentials_filepath = 'creds.json'
        launcher = BenchmarkLauncher(benchmark_config)
        mock_instance_manager.reset_mock()

        # Run
        result = launcher._build_instance_manager()

        # Assert
        mock_instance_manager.assert_called_once_with('creds.json')
        assert result == mock_instance_manager.return_value

    def test_build_instance_manager_not_supported(self):
        """Test `_build_instance_manager` raises an error for unsupported services."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        benchmark_config.credentials_filepath = 'creds.json'
        launcher = BenchmarkLauncher(benchmark_config)
        launcher.compute_service = 'aws'
        expected_error = re.escape("Compute service 'aws' is not supported.")

        # Run and Assert
        with pytest.raises(NotImplementedError, match=expected_error):
            launcher._build_instance_manager()

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

    @patch('sdgym._benchmark_launcher.benchmark_launcher.generate_ids')
    @patch(
        'sdgym._benchmark_launcher.benchmark_launcher.resolve_credentials',
        return_value={'aws': {}, 'gcp': {}, 'sdv': {}},
    )
    @patch(
        'sdgym._benchmark_launcher.benchmark_launcher._resolve_datasets',
        side_effect=[['d1'], ['d2']],
    )
    def test_launch_internal_calls_method_for_each_job(
        self, mock_resolve_datasets, mock_resolve_credentials, mock_generate_ids
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
        mock_generate_ids.return_value = 'LAUNCH_ID_1'

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

    def test_update_instance_statuses(self):
        """Test the `_update_instance_statuses` method."""
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
        launcher._instance_manager = Mock()
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
        launcher._get_all_instance_names = Mock(return_value=['instance-1', 'instance-2'])

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
        launcher._update_instance_statuses()

        # Assert
        launcher._instance_manager.update_instance_statuses.assert_called_once_with(
            ['instance-1', 'instance-2'],
            launcher._instance_name_to_status,
        )

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

    def test_validate_instance_names_uses_all_when_none(self):
        """Test `_validate_instance_names` returns all launched instances when None is passed."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        benchmark_config.credentials_filepath = 'creds.json'
        launcher = BenchmarkLauncher(benchmark_config)
        launcher._get_all_instance_names = Mock(return_value=['instance-1', 'instance-2'])

        # Run
        result = launcher._validate_instance_names(None)

        # Assert
        assert result == ['instance-1', 'instance-2']

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
        ('verbose', 'expected_error'),
        [
            (
                'aws',
                True,
                NotImplementedError(
                    "Compute service 'aws' is not supported. Supported services: 'gcp'."
                ),
            ),
            (
                'gcp',
                1,
                ValueError("`verbose` must be a boolean. Found: 1 (<class 'int'>)."),
            ),
            (
                'yes',
                ValueError("`verbose` must be a boolean. Found: 'yes' (<class 'str'>)."),
            ),
        ],
    )
    def test_validate_inputs_and_get_instances_invalid_cases(self, verbose, expected_error):
        """Test `_validate_inputs_and_get_instances` raises errors for invalid inputs."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        launcher = BenchmarkLauncher(benchmark_config)
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
        launcher._update_instance_statuses = Mock()
        launcher._get_active_instance_names = Mock(return_value=['instance-1', 'instance-2'])
        launcher._instance_manager = Mock()
        launcher._instance_manager.terminate_instances.return_value = ['instance-1', 'instance-2']

        # Run
        launcher.terminate(instance_names=['instance-1', 'instance-2'], verbose=True)

        # Assert
        launcher._validate_inputs_and_get_instances.assert_called_once_with(
            ['instance-1', 'instance-2'], True
        )
        launcher._update_instance_statuses.assert_called_once_with()
        launcher._get_active_instance_names.assert_called_once_with()
        launcher._instance_manager.terminate_instances.assert_called_once_with(
            ['instance-1', 'instance-2'], True
        )
        assert launcher._instance_name_to_status == {
            'instance-1': 'stopped',
            'instance-2': 'stopped',
        }
        mock_print.assert_called_once_with('Terminated 2 GCP instance(s).')

    @patch('sdgym._benchmark_launcher.benchmark_launcher.LOGGER')
    def test_terminate_logs_when_no_running_instances(self, mock_logger):
        """Test the `terminate` method logs when there are no running instances to terminate."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        launcher = BenchmarkLauncher(benchmark_config)
        launcher._validate_inputs_and_get_instances = Mock(return_value=['instance-1'])
        launcher._update_instance_statuses = Mock()
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
        launcher._validate_inputs_and_get_instances = Mock(return_value=['instance-1'])
        launcher._update_instance_statuses = Mock()
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
        expected_error = re.escape(
            "Compute service 'aws' is not supported. Supported services: 'gcp'."
        )

        # Run and Assert
        with pytest.raises(NotImplementedError, match=expected_error):
            launcher.terminate()

    @pytest.mark.parametrize(
        ('compute_service', 'expected_error'),
        [
            (
                'aws',
                "Compute service 'aws' is not supported. Supported services: 'gcp'.",
            ),
            (
                'azure',
                "Compute service 'azure' is not supported. Supported services: 'gcp'.",
            ),
        ],
    )
    def test_validate_compute_service(self, compute_service, expected_error):
        """Test the `_validate_compute_service` method."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        launcher = BenchmarkLauncher(benchmark_config)
        launcher.compute_service = compute_service

        # Run and Assert
        with pytest.raises(NotImplementedError, match=re.escape(expected_error)):
            launcher._validate_compute_service()

    @patch('sdgym._benchmark_launcher.benchmark_launcher.pd')
    def test_get_instance_status(self, mock_pd):
        """Test the `get_instance_status` method."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        launcher = BenchmarkLauncher(benchmark_config)
        launcher._validate_compute_service = Mock()
        launcher._validate_instance_names = Mock(return_value=['instance-1', 'instance-2'])
        launcher._update_instance_name_to_status = Mock()
        launcher._instance_name_to_status = {
            'instance-1': 'running',
            'instance-2': 'completed',
        }

        # Run
        launcher.get_instance_status(instance_names=['instance-1', 'instance-2'])

        # Assert
        launcher._validate_compute_service.assert_called_once_with()
        launcher._validate_instance_names.assert_called_once_with(['instance-1', 'instance-2'])
        launcher._update_instance_name_to_status.assert_called_once_with()
        mock_pd.DataFrame.assert_called_once_with([
            {
                'Instance Name': 'instance-1',
                'Status': 'Running',
            },
            {
                'Instance Name': 'instance-2',
                'Status': 'Completed',
            },
        ])

    def test_s3_object_exists(self):
        """Test the `_s3_object_exists` method."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        launcher = BenchmarkLauncher(benchmark_config)
        s3_client = Mock()

        # Run
        result = launcher._s3_object_exists(s3_client, 'bucket', 'key')

        # Assert
        s3_client.head_object.assert_called_once_with(Bucket='bucket', Key='key')
        assert result is True

    @pytest.mark.parametrize('error_code', ['404', 'NoSuchKey', 'NotFound'])
    def test_s3_object_exists_returns_false(self, error_code):
        """Test `_s3_object_exists` returns False when the object does not exist."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        launcher = BenchmarkLauncher(benchmark_config)
        s3_client = Mock()
        s3_client.head_object.side_effect = ClientError(
            {'Error': {'Code': error_code}},
            'HeadObject',
        )

        # Run
        result = launcher._s3_object_exists(s3_client, 'bucket', 'key')

        # Assert
        assert result is False

    @patch('sdgym._benchmark_launcher.benchmark_launcher.resolve_credentials')
    @patch('sdgym._benchmark_launcher.benchmark_launcher.get_s3_client')
    @pytest.mark.parametrize(
        ('object_exists_side_effect', 'expected_status'),
        [
            ([True, True, True], 'Completed'),
            ([True, False, False], 'Failed'),
            ([False, False, False], 'Queued'),
        ],
    )
    def test_get_job_artifact_status(
        self,
        mock_get_s3_client,
        mock_resolve_credentials,
        object_exists_side_effect,
        expected_status,
    ):
        """Test `_get_job_artifact_status` returns the expected status."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        benchmark_config.credentials_filepath = 'creds.json'
        launcher = BenchmarkLauncher(benchmark_config)
        launcher._s3_object_exists = Mock(side_effect=object_exists_side_effect)

        mock_resolve_credentials.return_value = {
            'aws': {
                'aws_access_key_id': 'AKIA',
                'aws_secret_access_key': 'SECRET',
            }
        }
        mock_get_s3_client.return_value = Mock()

        # Run
        result = launcher._get_job_artifact_status(
            dataset='alarm',
            synthesizer='CTGANSynthesizer',
            output_destination='s3://bucket/prefix',
        )

        # Assert
        assert result == expected_status

    def test_get_job_artifact_status_invalid_output_destination(self):
        """Test `_get_job_artifact_status` validates the output destination."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        launcher = BenchmarkLauncher(benchmark_config)
        expected_error = re.escape("`output_destination` must be an S3 path. Found: 'local/path'.")

        # Run and Assert
        with pytest.raises(ValueError, match=expected_error):
            launcher._get_job_artifact_status(
                dataset='alarm',
                synthesizer='CTGANSynthesizer',
                output_destination='local/path',
            )

    @patch('sdgym._benchmark_launcher.benchmark_launcher.pd')
    def test_get_job_status(self, mock_pd):
        """Test the `get_job_status` method."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        launcher = BenchmarkLauncher(benchmark_config)
        launcher._validate_inputs_and_get_instances = Mock(return_value=['instance-1'])
        launcher._update_instance_name_to_status = Mock()
        launcher._get_active_instance_names = Mock(return_value=['instance-1'])
        launcher._instance_name_to_jobs = {
            'instance-1': [
                {
                    'dataset': 'alarm',
                    'synthesizer': 'CTGAN',
                    'output_destination': 's3://bucket/prefix',
                },
                {
                    'dataset': 'alarm',
                    'synthesizer': 'TVAE',
                    'output_destination': 's3://bucket/prefix',
                },
            ]
        }
        launcher._get_job_artifact_status = Mock(side_effect=['Completed', 'Queued'])

        # Run
        launcher.get_job_status()

        # Assert
        launcher._validate_inputs_and_get_instances.assert_called_once_with(None, verbose=False)
        launcher._update_instance_name_to_status.assert_called_once_with()
        launcher._get_active_instance_names.assert_called_once_with()
        mock_pd.DataFrame.assert_called_once_with([
            {
                'Dataset': 'alarm',
                'Synthesizer': 'CTGAN',
                'Instance_Name': 'instance-1',
                'Status': 'Completed',
            },
            {
                'Dataset': 'alarm',
                'Synthesizer': 'TVAE',
                'Instance_Name': 'instance-1',
                'Status': 'Running',
            },
        ])

    @patch('sdgym._benchmark_launcher.benchmark_launcher.pd')
    def test_get_job_status_filters(self, mock_pd):
        """Test `get_job_status` applies dataset and synthesizer filters."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        launcher = BenchmarkLauncher(benchmark_config)
        launcher._validate_inputs_and_get_instances = Mock(return_value=['instance-1'])
        launcher._update_instance_name_to_status = Mock()
        launcher._get_active_instance_names = Mock(return_value=[])
        launcher._instance_name_to_jobs = {
            'instance-1': [
                {
                    'dataset': 'alarm',
                    'synthesizer': 'CTGAN',
                    'output_destination': 's3://bucket/prefix',
                },
                {
                    'dataset': 'adult',
                    'synthesizer': 'TVAE',
                    'output_destination': 's3://bucket/prefix',
                },
            ]
        }
        launcher._get_job_artifact_status = Mock(return_value='Completed')

        # Run
        launcher.get_job_status(dataset_names=['alarm'], synthesizer_names=['CTGAN'])

        # Assert
        mock_pd.DataFrame.assert_called_once_with([
            {
                'Dataset': 'alarm',
                'Synthesizer': 'CTGAN',
                'Instance_Name': 'instance-1',
                'Status': 'Completed',
            }
        ])

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

    @patch('sdgym._benchmark_launcher.benchmark_launcher.generate_ids')
    @patch('sdgym._benchmark_launcher.benchmark_launcher.cloudpickle.load')
    @patch('builtins.open', new_callable=mock_open, read_data=b'test')
    def test_load_generates_benchmark_id_if_missing(self, mock_file, mock_load, mock_generate_ids):
        """Test `load` generates a benchmark id if missing."""
        # Setup
        benchmark = Mock()
        benchmark._benchmark_id = None
        benchmark.modality = 'single_table'
        benchmark.compute_service = 'gcp'
        mock_load.return_value = benchmark
        mock_generate_ids.return_value = 'new-id'

        # Run
        result = BenchmarkLauncher.load('launcher.pkl')

        # Assert
        mock_file.assert_called_once_with('launcher.pkl', 'rb')
        mock_load.assert_called_once()
        mock_generate_ids.assert_called_once_with([
            'BENCMARK_ID',
            'single_table',
            'gcp',
        ])
        assert result._benchmark_id == 'new-id'
