from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest

from sdgym._benchmark.benchmark import (
    _benchmark_compute_gcp,
    _benchmark_multi_table_compute_gcp,
    _benchmark_single_table_compute_gcp,
    _get_user_data_script,
    _gpu_wait_block,
    _make_instance_name,
    _run_on_gcp,
    _terminate_instance,
    _upload_logs,
)
from sdgym.benchmark import (
    DEFAULT_MULTI_TABLE_DATASETS,
    DEFAULT_MULTI_TABLE_SYNTHESIZERS,
    DEFAULT_SINGLE_TABLE_DATASETS,
    DEFAULT_SINGLE_TABLE_SYNTHESIZERS,
)


@pytest.fixture
def base_credentials():
    return {
        'aws': {
            'aws_access_key_id': 'AKIA_TEST',
            'aws_secret_access_key': 'SECRET_TEST',
        },
        'gcp': {
            'gcp_project': 'test-project',
            'gcp_zone': 'us-central1-a',
        },
    }


@patch('sdgym._benchmark.benchmark.uuid.uuid4')
@patch('sdgym._benchmark.benchmark.datetime')
def test_make_instance_name(mock_datetime, mock_uuid):
    """Test `_make_instance_name` generates a stable, readable name."""
    # Setup
    mock_datetime.now.return_value = datetime(2025, 1, 15, tzinfo=timezone.utc)
    mock_uuid.return_value.hex = 'abcdef123456'

    # Run
    result = _make_instance_name('sdgym-run')

    # Assert
    assert result == 'sdgym-run-20250115-abcdef'


def test_terminate_instance_aws():
    """AWS termination script uses EC2 metadata and terminate-instances."""
    script = _terminate_instance('aws')

    assert 'cleanup()' in script
    assert 'latest/meta-data/instance-id' in script
    assert 'aws ec2 terminate-instances' in script
    assert 'compute.googleapis.com' not in script


def test_terminate_instance_gcp():
    """GCP termination script uses metadata server and Compute Engine API."""
    script = _terminate_instance('gcp')

    assert 'cleanup()' in script
    assert 'Metadata-Flavor: Google' in script
    assert 'compute.googleapis.com/compute/v1/projects' in script
    assert 'terminate-instances' not in script


def test_terminate_instance_invalid_service():
    """Invalid compute service raises a clear error."""
    # Run and Assert
    with pytest.raises(ValueError, match='Unsupported compute service'):
        _terminate_instance('azure')


def test_gpu_wait_block_contents():
    """GPU wait block waits for nvidia-smi to become available."""
    # Setup
    block = _gpu_wait_block()

    # Assert
    assert 'Waiting for GPU' in block
    assert 'nvidia-smi' in block
    assert 'sleep' in block
    assert 'for i in' in block or 'while' in block


def test_upload_logs_fn_no_uri():
    """No log URI returns a no-op upload_logs function."""
    # Run
    fn = _upload_logs('')

    # Assert
    assert fn.strip() == 'upload_logs() { :; }'


def test_upload_logs_fn_with_uri():
    """Upload logs function uploads user-data.log to S3."""
    # Setup
    uri = 's3://bucket/prefix/logs/instance-user-data.log'

    # Run
    fn = _upload_logs(uri)

    # Assert
    assert 'upload_logs()' in fn
    assert 'aws s3 cp /var/log/user-data.log' in fn
    assert uri in fn


def test_get_user_data_script_gcp_gpu_wait(base_credentials):
    """Test GCP user-data script includes GPU wait and delete logic."""
    # Setup
    config = {
        'service': 'gcp',
        'swap_gb': 16,
        'gpu_count': 1,
        'gpu_type': 'nvidia-tesla-t4',
        'gpu': True,
        'assert_gpu': True,
        'gpu_wait_seconds': 600,
        'gpu_wait_interval_seconds': 10,
        'install_s3fs': True,
        'sdgym_install': 'sdgym',
        'delete_on_success': True,
        'delete_on_error': True,
        'stop_fallback': True,
        'upload_logs_to_s3': True,
    }

    script = _get_user_data_script(
        credentials=base_credentials,
        script_content="print('hello')",
        config=config,
        instance_name='instance-1',
        output_destination='s3://bucket/output',
    )

    # Assert
    assert '#!/bin/bash' in script
    assert 'Waiting for GPU' in script
    assert 'nvidia-smi' in script
    assert 'nvidia-smi' in script
    assert 'cleanup()' in script
    assert 'compute.googleapis.com' in script
    assert 'Setting up swap (16G)' in script
    assert "print('hello')" in script


def test_get_user_data_script_aws_termination(base_credentials):
    """Test AWS user-data script includes EC2 termination logic."""
    # Setup
    config = {
        'service': 'aws',
        'swap_gb': 32,
        'assert_gpu': False,
        'gpu_wait_seconds': 300,
        'gpu_wait_interval_seconds': 5,
        'install_s3fs': False,
        'sdgym_install': 'sdgym',
        'upload_logs_to_s3': True,
    }

    script = _get_user_data_script(
        credentials=base_credentials,
        script_content="print('aws')",
        config=config,
        instance_name='aws-instance',
        output_destination='s3://bucket/output',
    )

    # Assert
    assert 'aws ec2 terminate-instances' in script
    assert 'INSTANCE_ID=$(curl -sf' in script
    assert 'wait_for_gpu' not in script
    assert "print('aws')" in script


@patch('sdgym._benchmark.benchmark.compute_v1')
@patch('sdgym._benchmark.benchmark._get_user_data_script')
@patch('sdgym._benchmark.benchmark._prepare_script_content')
@patch('sdgym._benchmark.benchmark.service_account.Credentials.from_service_account_info')
@patch('sdgym._benchmark.benchmark._make_instance_name')
def test_run_on_gcp(
    mock_make_instance_name,
    mock_from_service_account,
    mock_prepare_script_content,
    mock_get_user_data_script,
    mock_compute_v1,
):
    """Test `_run_on_gcp` successfully creates a GCP instance."""
    # Setup
    credentials = {
        'aws': {
            'aws_access_key_id': 'AKIA',
            'aws_secret_access_key': 'SECRET',
        },
        'gcp': {
            'gcp_project': 'test-project',
            'gcp_zone': 'us-central1-a',
        },
    }

    resolved_config = {
        'service': 'gcp',
        'name_prefix': 'sdgym-run',
        'machine_type': 'n1-standard-4',
        'source_image': 'image',
        'disk_size_gb': 50,
        'gpu_type': 'nvidia-tesla-t4',
        'gpu_count': 1,
        'install_nvidia_driver': True,
    }
    mock_prepare_script_content.return_value = 'SCRIPT_CONTENT'
    mock_get_user_data_script.return_value = 'STARTUP_SCRIPT'
    mock_make_instance_name.return_value = 'instance-123'
    mock_instances_client = Mock()
    mock_compute_v1.InstancesClient.return_value = mock_instances_client
    mock_operation = Mock()
    mock_operation.error = None
    mock_zone_ops_client = Mock()
    mock_zone_ops_client.wait.return_value = mock_operation
    mock_compute_v1.ZoneOperationsClient.return_value = mock_zone_ops_client
    boot_disk = Mock()
    mock_compute_v1.AttachedDisk.return_value = boot_disk
    gcp_cred = Mock()
    nic = Mock()
    mock_compute_v1.NetworkInterface.return_value = nic
    metadata = Mock()
    mock_compute_v1.Metadata.return_value = metadata
    gpu = Mock()
    mock_compute_v1.AcceleratorConfig.return_value = gpu
    scheduling = Mock()
    mock_compute_v1.Scheduling.return_value = scheduling
    mock_from_service_account.return_value = gcp_cred

    # Run
    result = _run_on_gcp(
        output_destination='s3://bucket/output',
        synthesizers=[],
        s3_client=Mock(),
        job_args_list=[{'job': 1}],
        credentials=credentials,
        compute_config=resolved_config,
    )

    # Assert
    assert result == 'instance-123'
    mock_from_service_account.assert_called_once_with(credentials['gcp'])
    mock_prepare_script_content.assert_called_once()
    mock_get_user_data_script.assert_called_once_with(
        credentials,
        'SCRIPT_CONTENT',
        resolved_config,
        'instance-123',
        's3://bucket/output',
    )
    mock_make_instance_name.assert_called_once_with('sdgym-run')
    mock_compute_v1.InstancesClient.assert_called_once_with(credentials=gcp_cred)
    mock_compute_v1.Instance.assert_called_once_with(
        name='instance-123',
        machine_type='zones/us-central1-a/machineTypes/n1-standard-4',
        disks=[boot_disk],
        network_interfaces=[nic],
        metadata=metadata,
        guest_accelerators=[gpu],
        scheduling=scheduling,
        service_accounts=[
            mock_compute_v1.ServiceAccount(
                email='default',
                scopes=['https://www.googleapis.com/auth/cloud-platform'],
            )
        ],
    )
    mock_instances_client.insert.assert_called_once()
    mock_compute_v1.ZoneOperationsClient.assert_called_once()
    mock_zone_ops_client.wait.assert_called_once()
    mock_compute_v1.Metadata.assert_called_once()
    mock_compute_v1.Instance.assert_called_once()


@patch('sdgym._benchmark.benchmark._run_on_gcp')
@patch('sdgym._benchmark.benchmark._generate_job_args_list')
@patch('sdgym._benchmark.benchmark._import_and_validate_synthesizers')
@patch('sdgym._benchmark.benchmark._ensure_uniform_included')
@patch('sdgym._benchmark.benchmark._validate_output_destination')
@patch('sdgym._benchmark.benchmark.get_credentials')
@patch('sdgym._benchmark.benchmark.resolve_compute_config')
@patch('sdgym._benchmark.benchmark.validate_compute_config')
def test_benchmark_compute_gcp(
    mock_validate_compute_config,
    mock_resolve_compute_config,
    mock_get_credentials,
    mock_validate_output,
    mock_ensure_uniform,
    mock_import_synths,
    mock_generate_jobs,
    mock_run_on_gcp,
):
    """Test `_benchmark_compute_gcp` method."""
    # Setup
    credentials = {
        'aws': {
            'aws_access_key_id': 'AKIA',
            'aws_secret_access_key': 'SECRET',
        }
    }
    mock_get_credentials.return_value = credentials
    s3_client = Mock()
    mock_validate_output.return_value = s3_client
    mock_import_synths.return_value = [{'name': 'Synth'}]
    mock_generate_jobs.return_value = [{'job': 1}]
    config = {'resolved': True, 'service': 'gcp'}
    mock_resolve_compute_config.return_value = config

    # Run
    _benchmark_compute_gcp(
        output_destination='s3://bucket/output',
        credential_filepath='/creds.json',
        compute_config={'foo': 'bar'},
        synthesizers=['Synth'],
        sdv_datasets=['dataset'],
        additional_datasets_folder=None,
        limit_dataset_size=False,
        compute_quality_score=True,
        compute_diagnostic_score=True,
        compute_privacy_score=True,
        sdmetrics=None,
        timeout=3600,
        modality='single_table',
    )

    # Assert
    mock_validate_compute_config.assert_called_once_with(config)
    mock_ensure_uniform.assert_called_once_with(['Synth'], 'single_table')
    mock_import_synths.assert_called_once_with(
        synthesizers=['Synth'], custom_synthesizers=None, modality='single_table'
    )
    mock_generate_jobs.assert_called_once_with(
        limit_dataset_size=False,
        sdv_datasets=['dataset'],
        additional_datasets_folder=None,
        sdmetrics=None,
        timeout=3600,
        output_destination='s3://bucket/output',
        compute_quality_score=True,
        compute_diagnostic_score=True,
        compute_privacy_score=True,
        synthesizers=[{'name': 'Synth'}],
        detailed_results_folder=None,
        s3_client=s3_client,
        modality='single_table',
    )
    mock_run_on_gcp.assert_called_once_with(
        output_destination='s3://bucket/output',
        synthesizers=[{'name': 'Synth'}],
        s3_client=s3_client,
        job_args_list=[{'job': 1}],
        credentials=credentials,
        compute_config=config,
    )


@patch('sdgym._benchmark.benchmark._benchmark_compute_gcp')
def test_benchmark_single_table_compute_gcp(mock_benchmark_compute):
    """Test `_benchmark_single_table_compute_gcp` calls the compute benchmark correctly."""
    # Setup
    synthesizers = ['SynthA', 'SynthB']
    output_destination = 's3://bucket/single_table_output'
    timeout = 7200
    credential_filepath = '/path/to/credentials.json'
    compute_config = 'compute_config_single'
    sdv_datasets = ['single_dataset1', 'single_dataset2']
    additional_datasets_folder = '/path/to/single_additional_datasets'
    limit_dataset_size = 5000
    compute_quality_score = False
    compute_diagnostic_score = True
    sdmetrics = ['SingleMetric1', 'SingleMetric2']

    # Run
    _benchmark_single_table_compute_gcp(
        output_destination=output_destination,
        credential_filepath=credential_filepath,
        compute_config=compute_config,
        synthesizers=synthesizers,
        sdv_datasets=sdv_datasets,
        additional_datasets_folder=additional_datasets_folder,
        limit_dataset_size=limit_dataset_size,
        compute_quality_score=compute_quality_score,
        compute_diagnostic_score=compute_diagnostic_score,
        sdmetrics=sdmetrics,
        timeout=timeout,
    )

    # Assert
    mock_benchmark_compute.assert_called_once_with(
        output_destination=output_destination,
        credential_filepath=credential_filepath,
        compute_config=compute_config,
        synthesizers=synthesizers,
        sdv_datasets=sdv_datasets,
        additional_datasets_folder=additional_datasets_folder,
        limit_dataset_size=limit_dataset_size,
        compute_quality_score=compute_quality_score,
        compute_diagnostic_score=compute_diagnostic_score,
        compute_privacy_score=True,
        sdmetrics=sdmetrics,
        timeout=timeout,
        modality='single_table',
    )


@patch('sdgym._benchmark.benchmark._benchmark_compute_gcp')
def test_benchmark_single_table_compute_gcp_defaults(mock_benchmark_compute):
    """Test `_benchmark_single_table_compute_gcp` with default parameters."""
    # Setup
    output_destination = 's3://bucket/single_table_output'
    credential_filepath = '/path/to/credentials.json'

    # Run
    _benchmark_single_table_compute_gcp(
        output_destination=output_destination,
        credential_filepath=credential_filepath,
    )

    # Assert
    mock_benchmark_compute.assert_called_once_with(
        output_destination=output_destination,
        credential_filepath=credential_filepath,
        compute_config=None,
        synthesizers=DEFAULT_SINGLE_TABLE_SYNTHESIZERS,
        sdv_datasets=DEFAULT_SINGLE_TABLE_DATASETS,
        additional_datasets_folder=None,
        limit_dataset_size=False,
        compute_quality_score=True,
        compute_diagnostic_score=True,
        compute_privacy_score=True,
        sdmetrics=None,
        timeout=None,
        modality='single_table',
    )


@patch('sdgym._benchmark.benchmark._benchmark_compute_gcp')
def test_benchmark_multi_table_compute_gcp(mock_benchmark_compute):
    """Test `_benchmark_multi_table_compute_gcp` calls the compute benchmark correctly."""
    # Setup
    synthesizers = ['Synth1', 'Synth2']
    output_destination = 's3://bucket/output'
    timeout = 3600
    credential_filepath = '/path/to/credentials.json'
    compute_config = 'compute_config'
    sdv_datasets = ['dataset1', 'dataset2']
    additional_datasets_folder = '/path/to/additional_datasets'
    limit_dataset_size = 10000
    compute_quality_score = True
    compute_diagnostic_score = False
    sdmetrics = ['Metric1', 'Metric2']

    # Run
    _benchmark_multi_table_compute_gcp(
        output_destination=output_destination,
        credential_filepath=credential_filepath,
        compute_config=compute_config,
        synthesizers=synthesizers,
        sdv_datasets=sdv_datasets,
        additional_datasets_folder=additional_datasets_folder,
        limit_dataset_size=limit_dataset_size,
        compute_quality_score=compute_quality_score,
        compute_diagnostic_score=compute_diagnostic_score,
        sdmetrics=sdmetrics,
        timeout=timeout,
    )

    # Assert
    mock_benchmark_compute.assert_called_once_with(
        output_destination=output_destination,
        credential_filepath=credential_filepath,
        compute_config=compute_config,
        synthesizers=synthesizers,
        sdv_datasets=sdv_datasets,
        additional_datasets_folder=additional_datasets_folder,
        limit_dataset_size=limit_dataset_size,
        compute_quality_score=compute_quality_score,
        compute_diagnostic_score=compute_diagnostic_score,
        compute_privacy_score=False,
        sdmetrics=sdmetrics,
        timeout=timeout,
        modality='multi_table',
    )


@patch('sdgym._benchmark.benchmark._benchmark_compute_gcp')
def test_benchmark_multi_table_compute_gcp_defaults(mock_benchmark_compute):
    """Test `_benchmark_multi_table_compute_gcp` with default parameters."""
    # Setup
    output_destination = 's3://bucket/output'
    credential_filepath = '/path/to/credentials.json'

    # Run
    _benchmark_multi_table_compute_gcp(
        output_destination=output_destination,
        credential_filepath=credential_filepath,
    )

    # Assert
    mock_benchmark_compute.assert_called_once_with(
        output_destination=output_destination,
        credential_filepath=credential_filepath,
        compute_config=None,
        synthesizers=DEFAULT_MULTI_TABLE_SYNTHESIZERS,
        sdv_datasets=DEFAULT_MULTI_TABLE_DATASETS,
        additional_datasets_folder=None,
        limit_dataset_size=False,
        compute_quality_score=True,
        compute_diagnostic_score=True,
        compute_privacy_score=False,
        sdmetrics=None,
        timeout=None,
        modality='multi_table',
    )
