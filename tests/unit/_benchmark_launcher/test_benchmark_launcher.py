"""Unit tests for the BenchmarkLauncher class."""

import re
from unittest.mock import Mock, call, mock_open, patch

import pytest

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
        assert launcher._instance_name_to_jobs == {}
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

    def test_add_synthesizer_suffix(self):
        """Test the `_add_synthesizer_suffix` method."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        launcher = BenchmarkLauncher(benchmark_config)

        # Run
        result_no_suffix = launcher._add_synthesizer_suffix('CTGAN', 0)
        result_with_suffix = launcher._add_synthesizer_suffix('CTGAN', 2)

        # Assert
        assert result_no_suffix == 'CTGAN'
        assert result_with_suffix == 'CTGAN(2)'

    @patch('sdgym._benchmark_launcher.benchmark_launcher._get_top_folder_prefix')
    @patch('sdgym._benchmark_launcher.benchmark_launcher._add_dataset_suffix')
    @patch('sdgym._benchmark_launcher.benchmark_launcher._build_job_output_destination')
    def test_build_instance_jobs(
        self, mock_build_job_output_destination, mock_add_dataset_suffix, mock_get_top_folder_prefix
    ):
        """Test the `_build_instance_jobs` method."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        launcher = BenchmarkLauncher(benchmark_config)
        mock_build_job_output_destination.side_effect = [
            's3://bucket/prefix/dataset_1/Synth1(1)/',
            's3://bucket/prefix/dataset_2/Synth1(1)/',
        ]
        mock_get_top_folder_prefix.return_value = 'prefix'
        mock_add_dataset_suffix.side_effect = ['dataset_1', 'dataset_2']

        # Run
        result = launcher._build_instance_jobs(
            datasets=['dataset1', 'dataset2'],
            synthesizers=['Synth1'],
            output_destination='s3://bucket/path',
            instance_idx=1,
        )

        # Assert
        mock_get_top_folder_prefix.assert_called_once_with('s3://bucket/path', 'single_table')
        assert mock_add_dataset_suffix.call_args_list == [call('dataset1'), call('dataset2')]
        assert mock_build_job_output_destination.call_args_list == [
            call(
                output_destination='s3://bucket/path',
                artifact_key_prefix='prefix',
                artifact_dataset='dataset_1',
                artifact_synthesizer='Synth1(1)',
            ),
            call(
                output_destination='s3://bucket/path',
                artifact_key_prefix='prefix',
                artifact_dataset='dataset_2',
                artifact_synthesizer='Synth1(1)',
            ),
        ]
        assert result == [
            {
                'dataset': 'dataset1',
                'synthesizer': 'Synth1',
                'artifact_dataset': 'dataset_1',
                'artifact_synthesizer': 'Synth1(1)',
                'artifact_key_prefix': 'prefix',
                'output_destination': 's3://bucket/path',
                'job_output_destination': 's3://bucket/prefix/dataset_1/Synth1(1)/',
            },
            {
                'dataset': 'dataset2',
                'synthesizer': 'Synth1',
                'artifact_dataset': 'dataset_2',
                'artifact_synthesizer': 'Synth1(1)',
                'artifact_key_prefix': 'prefix',
                'output_destination': 's3://bucket/path',
                'job_output_destination': 's3://bucket/prefix/dataset_2/Synth1(1)/',
            },
        ]

    def test_launch_calls_validate_when_not_validated(self):
        """Test `launch` calls `validate` when `_is_validated` is False."""
        # Setup
        config = Mock()
        config.modality = 'single_table'
        config.compute = {'service': 'gcp'}
        config.credentials_filepath = 'creds.json'
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
        config.credentials_filepath = 'creds.json'
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
    @patch('sdgym._benchmark_launcher.benchmark_launcher._get_top_folder_prefix')
    @patch('sdgym._benchmark_launcher.benchmark_launcher._add_dataset_suffix')
    @patch('sdgym._benchmark_launcher.benchmark_launcher._build_job_output_destination')
    def test_launch_internal_calls_method_for_each_job(
        self,
        mock_build_job_output_destination,
        mock_add_dataset_suffix,
        mock_get_top_folder_prefix,
        mock_resolve_datasets,
        mock_resolve_credentials,
        mock_generate_ids,
    ):
        """Test `_launch` calls the underlying benchmark method for each job."""
        # Setup
        output_destination = 's3://bucket/prefix/'
        output_destination_artifact_1 = 's3://bucket/prefix/dataset_1/Synth1(1)/'
        output_destination_artifact_2 = 's3://bucket/prefix/dataset_2/Synth1(1)/'
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
        mock_get_top_folder_prefix.return_value = 'artifact-prefix'
        mock_add_dataset_suffix.side_effect = ['d1_artifact', 'd2_artifact']
        mock_build_job_output_destination.side_effect = [
            output_destination_artifact_1,
            output_destination_artifact_2,
        ]
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
        assert launcher._instance_name_to_jobs == {
            'instance-1': [
                {
                    'dataset': 'd1',
                    'synthesizer': 'Synth1',
                    'artifact_dataset': 'd1_artifact',
                    'artifact_synthesizer': 'Synth1',
                    'artifact_key_prefix': 'artifact-prefix',
                    'output_destination': output_destination,
                    'job_output_destination': output_destination_artifact_1,
                }
            ],
            'instance-2': [
                {
                    'dataset': 'd2',
                    'synthesizer': 'Synth2',
                    'artifact_dataset': 'd2_artifact',
                    'artifact_synthesizer': 'Synth2(1)',
                    'artifact_key_prefix': 'artifact-prefix',
                    'output_destination': output_destination,
                    'job_output_destination': output_destination_artifact_2,
                }
            ],
        }

    def test_update_instance_statuses(self):
        """Test the `_update_instance_statuses` method."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        benchmark_config.credentials_filepath = 'creds.json'
        launcher = BenchmarkLauncher(benchmark_config)
        launcher._instance_manager = Mock()
        launcher._get_all_instance_names = Mock(return_value=['instance-1', 'instance-2'])
        launcher._instance_name_to_status = {
            'instance-1': 'running',
            'instance-2': 'completed',
        }

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
            'instance-2': 'completed',
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

    def test_validate_compute_service(self):
        """Test the `_validate_compute_service` method."""
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
            launcher._validate_compute_service()

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

    def test_validate_inputs_and_get_instances_invalid_verbose(self):
        """Test `_validate_inputs_and_get_instances` raises errors for invalid verbose."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        launcher = BenchmarkLauncher(benchmark_config)
        launcher._validate_compute_service = Mock()
        launcher._validate_instance_names = Mock(return_value=['instance-1'])
        expected_error = re.escape("`verbose` must be a boolean. Found: 'yes' (<class 'str'>).")

        # Run and Assert
        with pytest.raises(ValueError, match=expected_error):
            launcher._validate_inputs_and_get_instances(
                instance_names=['instance-1'],
                verbose='yes',
            )

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
        launcher._update_instance_statuses = Mock()
        launcher._instance_name_to_status = {
            'instance-1': 'running',
            'instance-2': 'completed',
        }

        # Run
        launcher.get_instance_status(instance_names=['instance-1', 'instance-2'])

        # Assert
        launcher._validate_compute_service.assert_called_once_with()
        launcher._validate_instance_names.assert_called_once_with(['instance-1', 'instance-2'])
        launcher._update_instance_statuses.assert_called_once_with()
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

    def test_get_all_output_destinations(self):
        """Test the `_get_all_output_destinations` method."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        launcher = BenchmarkLauncher(benchmark_config)
        launcher._validate_instance_names = Mock(return_value=['instance-1', 'instance-2'])
        launcher._instance_name_to_jobs = {
            'instance-1': [
                {
                    'dataset': 'alarm',
                    'synthesizer': 'CTGAN',
                    'artifact_synthesizer': 'CTGAN',
                    'output_destination': 's3://bucket/prefix-a',
                },
            ],
            'instance-2': [
                {
                    'dataset': 'adult',
                    'synthesizer': 'TVAE',
                    'artifact_synthesizer': 'TVAE(1)',
                    'output_destination': 's3://bucket/prefix-a',
                },
                {
                    'dataset': 'census',
                    'synthesizer': 'CopulaGAN',
                    'artifact_synthesizer': 'CopulaGAN(1)',
                    'output_destination': 's3://bucket/prefix-b',
                },
            ],
        }

        # Run
        result = launcher._get_all_output_destinations()

        # Assert
        assert result == ['s3://bucket/prefix-a', 's3://bucket/prefix-b']

    @patch('sdgym._benchmark_launcher.benchmark_launcher.resolve_credentials')
    @patch('sdgym._benchmark_launcher.benchmark_launcher.get_s3_client')
    @patch('sdgym._benchmark_launcher.benchmark_launcher._list_s3_bucket_contents')
    def test_get_s3_existing_filenames(
        self, mock_list_s3_bucket_contents, mock_get_s3_client, mock_resolve_credentials
    ):
        """Test the `_get_s3_existing_filenames` method."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        benchmark_config.credentials_filepath = 'creds.json'
        launcher = BenchmarkLauncher(benchmark_config)
        mock_resolve_credentials.return_value = {
            'aws': {
                'aws_access_key_id': 'AKIA',
                'aws_secret_access_key': 'SECRET',
            }
        }
        s3_client = Mock()
        mock_get_s3_client.return_value = s3_client
        mock_list_s3_bucket_contents.return_value = [
            {'Key': 'prefix/file1.csv'},
            {'Key': 'prefix/file2.csv'},
        ]

        # Run
        existing_keys = launcher._get_s3_existing_filenames('s3://bucket/prefix')

        # Assert
        mock_get_s3_client.assert_called_once_with(
            aws_access_key_id='AKIA',
            aws_secret_access_key='SECRET',
        )
        mock_list_s3_bucket_contents.assert_called_once_with(s3_client, 'bucket', 'prefix')
        assert existing_keys == {'prefix/file1.csv', 'prefix/file2.csv'}

    @patch('sdgym._benchmark_launcher.benchmark_launcher._build_job_artifact_keys')
    @pytest.mark.parametrize(
        ('existing_keys', 'expected_status'),
        [
            (
                {
                    'prefix/alarm/CTGANSynthesizer/CTGANSynthesizer_benchmark_result.csv',
                    'prefix/alarm/CTGANSynthesizer/CTGANSynthesizer_synthetic_data.csv',
                    'prefix/alarm/CTGANSynthesizer/CTGANSynthesizer.pkl',
                },
                'Completed',
            ),
            (
                {
                    'prefix/alarm/CTGANSynthesizer/CTGANSynthesizer_benchmark_result.csv',
                },
                'Failed',
            ),
            (
                set(),
                'Queued',
            ),
        ],
    )
    def test_get_job_artifact_status(
        self, mock_build_job_artifact_keys, existing_keys, expected_status
    ):
        """Test `_get_job_artifact_status` returns the expected status."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        launcher = BenchmarkLauncher(benchmark_config)
        mock_build_job_artifact_keys.return_value = (
            'prefix/alarm/CTGANSynthesizer/CTGANSynthesizer_benchmark_result.csv',
            'prefix/alarm/CTGANSynthesizer/CTGANSynthesizer_synthetic_data.csv',
            'prefix/alarm/CTGANSynthesizer/CTGANSynthesizer.pkl',
        )

        # Run
        result = launcher._get_job_artifact_status(
            artifact_dataset='alarm_01_01_2026',
            artifact_synthesizer='CTGANSynthesizer',
            artifact_key_prefix='prefix',
            existing_keys=existing_keys,
        )

        # Assert
        mock_build_job_artifact_keys.assert_called_once_with(
            artifact_key_prefix='prefix',
            artifact_dataset='alarm_01_01_2026',
            artifact_synthesizer='CTGANSynthesizer',
            modality='single_table',
        )
        assert result == expected_status

    def test_get_instance_job_rows(self):
        """Test the `_get_instance_job_rows` method."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        launcher = BenchmarkLauncher(benchmark_config)
        launcher._get_job_artifact_status = Mock(side_effect=['Completed', 'Queued'])
        jobs = [
            {
                'dataset': 'alarm',
                'synthesizer': 'CTGAN',
                'artifact_dataset': 'alarm_01_01_2026',
                'artifact_synthesizer': 'CTGAN',
                'artifact_key_prefix': 'artifact-prefix',
                'output_destination': 's3://bucket/prefix',
                'job_output_destination': 's3://bucket/artifact-prefix/alarm_01_01_2026/CTGAN/',
            },
            {
                'dataset': 'adult',
                'synthesizer': 'TVAE',
                'artifact_dataset': 'adult_01_01_2026',
                'artifact_synthesizer': 'TVAE',
                'artifact_key_prefix': 'artifact-prefix',
                'output_destination': 's3://bucket/prefix',
                'job_output_destination': 's3://bucket/artifact-prefix/adult_01_01_2026/TVAE/',
            },
        ]
        existing_keys_by_output = {
            's3://bucket/prefix': {'file1', 'file2'},
        }

        # Run
        result = launcher._get_instance_job_rows(
            instance_name='instance-1',
            jobs=jobs,
            dataset_names=None,
            synthesizer_names=None,
            existing_keys_by_output=existing_keys_by_output,
        )

        # Assert
        assert result == [
            {
                'Dataset': 'alarm',
                'Synthesizer': 'CTGAN',
                'Instance_Name': 'instance-1',
                'Status': 'Completed',
                'Output_Destination': 's3://bucket/artifact-prefix/alarm_01_01_2026/CTGAN/',
            },
            {
                'Dataset': 'adult',
                'Synthesizer': 'TVAE',
                'Instance_Name': 'instance-1',
                'Status': 'Queued',
                'Output_Destination': 's3://bucket/artifact-prefix/adult_01_01_2026/TVAE/',
            },
        ]

    def test_update_status_running_job_running(self):
        """Test `_update_status_running_job` marks the first queued job as running."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        launcher = BenchmarkLauncher(benchmark_config)
        instance_rows = [
            {'Status': 'Completed'},
            {'Status': 'Queued'},
            {'Status': 'Queued'},
        ]

        # Run
        result = launcher._update_status_running_job(instance_rows, 'running')

        # Assert
        assert result == [
            {'Status': 'Completed'},
            {'Status': 'Running'},
            {'Status': 'Queued'},
        ]

    def test_update_status_running_job_not_running(self):
        """Test `_update_status_running_job` marks queued jobs as failed."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        launcher = BenchmarkLauncher(benchmark_config)
        instance_rows = [
            {'Status': 'Completed'},
            {'Status': 'Queued'},
            {'Status': 'Queued'},
        ]

        # Run
        result = launcher._update_status_running_job(instance_rows, 'completed')

        # Assert
        assert result == [
            {'Status': 'Completed'},
            {'Status': 'Failed'},
            {'Status': 'Failed'},
        ]

    @patch('sdgym._benchmark_launcher.benchmark_launcher.pd')
    def test_get_job_status(self, mock_pd):
        """Test the `get_job_status` method."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        launcher = BenchmarkLauncher(benchmark_config)
        launcher._validate_instance_names = Mock(return_value=['instance-1'])
        launcher._update_instance_statuses = Mock()
        launcher._get_all_output_destinations = Mock(return_value=['s3://bucket/prefix'])
        launcher._get_s3_existing_filenames = Mock(return_value={'file1', 'file2'})
        launcher._instance_name_to_status = {'instance-1': 'running'}
        launcher._instance_name_to_jobs = {'instance-1': ['job1']}
        launcher._get_instance_job_rows = Mock(
            return_value=[
                {
                    'Dataset': 'alarm',
                    'Synthesizer': 'CTGAN',
                    'Instance_Name': 'instance-1',
                    'Output_Destination': 's3://bucket/artifact-prefix/alarm_01_01_2026/CTGAN/',
                    'Status': 'Queued',
                }
            ]
        )
        launcher._update_status_running_job = Mock(
            return_value=[
                {
                    'Dataset': 'alarm',
                    'Synthesizer': 'CTGAN',
                    'Instance_Name': 'instance-1',
                    'Output_Destination': 's3://bucket/artifact-prefix/alarm_01_01_2026/CTGAN/',
                    'Status': 'Running',
                }
            ]
        )

        # Run
        launcher.get_job_status()

        # Assert
        launcher._validate_instance_names.assert_called_once_with(None)
        launcher._update_instance_statuses.assert_called_once_with()
        launcher._get_all_output_destinations.assert_called_once_with(['instance-1'])
        launcher._get_s3_existing_filenames.assert_called_once_with('s3://bucket/prefix')
        launcher._get_instance_job_rows.assert_called_once_with(
            instance_name='instance-1',
            jobs=['job1'],
            dataset_names=None,
            synthesizer_names=None,
            existing_keys_by_output={'s3://bucket/prefix': {'file1', 'file2'}},
        )
        launcher._update_status_running_job.assert_called_once_with(
            [
                {
                    'Dataset': 'alarm',
                    'Synthesizer': 'CTGAN',
                    'Instance_Name': 'instance-1',
                    'Output_Destination': 's3://bucket/artifact-prefix/alarm_01_01_2026/CTGAN/',
                    'Status': 'Queued',
                }
            ],
            'running',
        )
        mock_pd.DataFrame.assert_called_once_with([
            {
                'Dataset': 'alarm',
                'Synthesizer': 'CTGAN',
                'Instance_Name': 'instance-1',
                'Status': 'Running',
                'Output_Destination': 's3://bucket/artifact-prefix/alarm_01_01_2026/CTGAN/',
            }
        ])

    def test_get_status(self):
        """Test the `get_status` method."""
        # Setup
        benchmark_config = Mock()
        benchmark_config.modality = 'single_table'
        benchmark_config.compute = {'service': 'gcp'}
        launcher = BenchmarkLauncher(benchmark_config)

        # Run and Assert
        with pytest.raises(AttributeError):
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
