import textwrap
from urllib.parse import urlparse

from google.cloud import compute_v1
from google.oauth2 import service_account

from sdgym._benchmark.config_utils import (
    _make_instance_name,
    resolve_compute_config,
    validate_compute_config,
)
from sdgym._benchmark.credentials_utils import get_credentials, sdv_install_cmd
from sdgym.benchmark import (
    DEFAULT_MULTI_TABLE_DATASETS,
    DEFAULT_MULTI_TABLE_SYNTHESIZERS,
    DEFAULT_SINGLE_TABLE_DATASETS,
    DEFAULT_SINGLE_TABLE_SYNTHESIZERS,
    S3_REGION,
    _ensure_uniform_included,
    _generate_job_args_list,
    _get_empty_dataframe,
    _get_s3_script_content,
    _import_and_validate_synthesizers,
    _store_job_args_in_s3,
    _validate_output_destination,
)


def _get_logs_s3_uri(output_destination, instance_name):
    """Store logs next to output destination prefix.

    Example:
        output_destination='s3://bucket/prefix'
        -> s3://bucket/prefix/logs/<instance>-user-data.log
    """
    if not output_destination.startswith('s3://'):
        return ''

    parsed = urlparse(output_destination)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip('/').rstrip('/')
    prefix = f'{prefix}/logs' if prefix else 'logs'

    return f's3://{bucket}/{prefix}/{instance_name}-user-data.log'


def _prepare_script_content(
    output_destination,
    synthesizers,
    s3_client,
    job_args_list,
    credentials,
):
    bucket_name, job_args_key = _store_job_args_in_s3(
        output_destination,
        job_args_list,
        s3_client,
    )
    synthesizer_names = [{'name': s['name']} for s in synthesizers]
    return _get_s3_script_content(
        credentials['aws']['aws_access_key_id'],
        credentials['aws']['aws_secret_access_key'],
        S3_REGION,
        bucket_name,
        job_args_key,
        synthesizer_names,
    )


def _terminate_instance(compute_service):
    if compute_service not in ('aws', 'gcp'):
        raise ValueError(f'Unsupported compute service: {compute_service}')

    if compute_service == 'aws':
        return textwrap.dedent(
            """\
            cleanup() {
              log "======== Kernel messages (OOM info) =========="
              dmesg | tail -50 || true
              upload_logs || true
              INSTANCE_ID=$(curl -sf http://169.254.169.254/latest/meta-data/instance-id || true)
              if [ -n "$INSTANCE_ID" ]; then
                log "Terminating EC2 instance: $INSTANCE_ID"
                aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" >/dev/null 2>&1 || true
              fi
            }
            """
        ).strip()

    # GCP
    return textwrap.dedent(
        """\
        cleanup() {
          log "======== Kernel messages (OOM info) =========="
          dmesg | tail -50 || true
          upload_logs || true
          log "Shutting down GCE instance"
          shutdown -h now || true
        }
        """
    ).strip()


def _gpu_wait_block():
    return textwrap.dedent(
        """\
        log "======== Waiting for GPU =========="
        for i in {1..60}; do
          if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null; then
            nvidia-smi
            break
          fi
          sleep 10
        done
        """
    ).strip()


def _upload_logs(log_uri):
    if not log_uri:
        return 'upload_logs() { :; }'

    return textwrap.dedent(
        f"""\
        upload_logs() {{
          log "======== Uploading logs =========="
          aws s3 cp /var/log/user-data.log "{log_uri}" >/dev/null 2>&1 || true
        }}
        """
    ).strip()


def _get_user_data_script(
    credentials,
    script_content,
    config,
    instance_name,
    output_destination,
):
    compute_service = config['service']
    swap_gb = int(config.get('swap_gb', 32))
    gpu = (
        bool(config.get('gpu'))
        or int(config.get('gpu_count', 0)) > 0
        or bool(config.get('gpu_type'))
    )
    upload_logs = bool(config.get('upload_logs', True))

    aws_key = credentials['aws']['aws_access_key_id']
    aws_secret = credentials['aws']['aws_secret_access_key']

    log_uri = _get_logs_s3_uri(output_destination, instance_name) if upload_logs else ''

    sdv_install = sdv_install_cmd(credentials).rstrip()
    sdv_install = textwrap.indent(sdv_install, '        ') if sdv_install else ''
    terminate_fn = _terminate_instance(compute_service)
    upload_logs_fn = _upload_logs(log_uri)
    gpu_block = _gpu_wait_block() if gpu else ''

    return textwrap.dedent(
        f"""\
        #!/bin/bash
        set -e

        LOG_FILE=/var/log/user-data.log
        exec >> "$LOG_FILE" 2>&1

        log() {{
          echo "$@"
        }}

        {upload_logs_fn}
        {terminate_fn}

        # Always cleanup on exit
        trap cleanup EXIT

        log "======== Instance: {instance_name} =========="

        log "======== Configure kernel OOM behavior =========="
        sudo sysctl -w vm.panic_on_oom=1
        sudo sysctl -w kernel.panic=0

        log "======== Update and Install Dependencies =========="
        sudo apt update -y
        sudo apt install -y python3-pip python3-venv awscli git jq

        log "======== Setting up swap ({swap_gb}G) =========="
        sudo fallocate -l {swap_gb}G /swapfile || \
          sudo dd if=/dev/zero of=/swapfile bs=1M count=$(({swap_gb}*1024))
        sudo chmod 600 /swapfile
        sudo mkswap /swapfile
        sudo swapon /swapfile

        log "======== Configure AWS CLI =========="
        aws configure set aws_access_key_id '{aws_key}'
        aws configure set aws_secret_access_key '{aws_secret}'
        aws configure set default.region '{S3_REGION}'

        log "======== Create Virtual Environment =========="
        python3 -m venv ~/env
        source ~/env/bin/activate

        log "======== Install Dependencies =========="
        pip install --upgrade pip
        {sdv_install}
        pip install "sdgym[all]"

        {gpu_block}

        log "======== Write Script =========="
        cat << 'EOF' > ~/sdgym_script.py
{script_content}
EOF

        log "======== Run Script =========="
        python -u ~/sdgym_script.py | tee -a /var/log/sdgym.log

        log "======== Complete =========="
        """
    ).strip()


def _run_on_gcp(
    output_destination, synthesizers, s3_client, job_args_list, credentials, compute_config
):
    """Launch a GCP Compute Engine instance to run a benchmark.

    This method creates and configures a VM using the provided compute settings,
    prepares a startup script with the benchmark configuration, and starts execution
    automatically when the instance boots. It waits for the instance to be created
    and raises an error if provisioning fails.

    Args:
        output_destination (str):
            The S3 URI where results will be stored.
        synthesizers (list of dict):
            The synthesizers to use in the benchmark.
        s3_client (boto3.client):
            The S3 client to use for storing job arguments.
        job_args_list (list of dict):
            The list of job arguments for each dataset.
        credentials (dict):
            The credentials for AWS and GCP.
        compute_config (dict):
            The compute configuration for the GCP instance.
    """
    script_content = _prepare_script_content(
        output_destination,
        synthesizers,
        s3_client,
        job_args_list,
        credentials,
    )

    gcp_zone = credentials['gcp']['gcp_zone']
    gcp_project = credentials['gcp']['gcp_project']
    gcp_creds = service_account.Credentials.from_service_account_info(
        credentials['gcp'],
    )

    instance_name = _make_instance_name(compute_config['name_prefix'])
    print(  # noqa: T201
        f'Launching instance: {instance_name} (service=gcp project={gcp_project} zone={gcp_zone})'
    )
    startup_script = _get_user_data_script(
        credentials,
        script_content,
        compute_config,
        instance_name,
        output_destination,
    )

    machine_type = f'zones/{gcp_zone}/machineTypes/{compute_config["machine_type"]}'
    source_disk_image = compute_config['source_image']
    gpu = compute_v1.AcceleratorConfig(
        accelerator_type=(f'zones/{gcp_zone}/acceleratorTypes/{compute_config["gpu_type"]}'),
        accelerator_count=int(compute_config['gpu_count']),
    )

    boot_disk = compute_v1.AttachedDisk(
        auto_delete=True,
        boot=True,
        initialize_params=compute_v1.AttachedDiskInitializeParams(
            source_image=source_disk_image,
            disk_size_gb=int(compute_config['disk_size_gb']),
        ),
    )

    nic = compute_v1.NetworkInterface()
    nic.network = 'global/networks/default'
    nic.access_configs = [
        compute_v1.AccessConfig(
            name='External NAT',
            type_='ONE_TO_ONE_NAT',
        )
    ]

    items = [compute_v1.Items(key='startup-script', value=startup_script)]
    if compute_config.get('install_nvidia_driver', True):
        items.append(
            compute_v1.Items(key='install-nvidia-driver', value='true'),
        )
    metadata = compute_v1.Metadata(items=items)

    scheduling = compute_v1.Scheduling(
        on_host_maintenance='TERMINATE',
        automatic_restart=False,
    )

    instance = compute_v1.Instance(
        name=instance_name,
        machine_type=machine_type,
        disks=[boot_disk],
        network_interfaces=[nic],
        metadata=metadata,
        guest_accelerators=[gpu],
        scheduling=scheduling,
        service_accounts=[
            compute_v1.ServiceAccount(
                email='default',
                scopes=['https://www.googleapis.com/auth/cloud-platform'],
            )
        ],
    )

    instance_client = compute_v1.InstancesClient(credentials=gcp_creds)
    operation = instance_client.insert(
        project=gcp_project,
        zone=gcp_zone,
        instance_resource=instance,
    )

    op_client = compute_v1.ZoneOperationsClient(credentials=gcp_creds)
    operation = op_client.wait(
        project=gcp_project,
        zone=gcp_zone,
        operation=operation.name,
    )

    if operation.error and operation.error.errors:
        messages = [e.message for e in operation.error.errors if e.message]
        joined = '; '.join(messages) if messages else str(operation.error)
        raise RuntimeError(f'GCP instance creation failed: {joined}')

    print(f'Instance created: {instance_name}')  # noqa: T201
    return instance_name


def _benchmark_compute_gcp(
    output_destination,
    credential_filepath,
    compute_config,
    synthesizers,
    sdv_datasets,
    additional_datasets_folder,
    limit_dataset_size,
    compute_quality_score,
    compute_diagnostic_score,
    compute_privacy_score,
    sdmetrics,
    timeout,
    modality,
):
    """Run the SDGym benchmark on datasets for the given modality."""
    compute_config = resolve_compute_config('gcp', compute_config)
    credentials = get_credentials(credential_filepath)
    validate_compute_config(compute_config)

    s3_client = _validate_output_destination(
        output_destination,
        aws_keys={
            'aws_access_key_id': credentials['aws']['aws_access_key_id'],
            'aws_secret_access_key': credentials['aws']['aws_secret_access_key'],
        },
    )

    if not synthesizers:
        synthesizers = []

    _ensure_uniform_included(synthesizers, modality)
    synthesizers = _import_and_validate_synthesizers(
        synthesizers=synthesizers,
        custom_synthesizers=None,
        modality=modality,
    )

    job_args_list = _generate_job_args_list(
        limit_dataset_size=limit_dataset_size,
        sdv_datasets=sdv_datasets,
        additional_datasets_folder=additional_datasets_folder,
        sdmetrics=sdmetrics,
        timeout=timeout,
        output_destination=output_destination,
        compute_quality_score=compute_quality_score,
        compute_diagnostic_score=compute_diagnostic_score,
        compute_privacy_score=compute_privacy_score,
        synthesizers=synthesizers,
        s3_client=s3_client,
        modality=modality,
    )
    if not job_args_list:
        return _get_empty_dataframe(
            compute_diagnostic_score=compute_diagnostic_score,
            compute_quality_score=compute_quality_score,
            compute_privacy_score=compute_privacy_score,
            sdmetrics=sdmetrics,
        )

    _run_on_gcp(
        output_destination=output_destination,
        synthesizers=synthesizers,
        s3_client=s3_client,
        job_args_list=job_args_list,
        credentials=credentials,
        compute_config=compute_config,
    )


def _benchmark_single_table_compute_gcp(
    output_destination,
    credential_filepath,
    compute_config=None,
    synthesizers=DEFAULT_SINGLE_TABLE_SYNTHESIZERS,
    sdv_datasets=DEFAULT_SINGLE_TABLE_DATASETS,
    additional_datasets_folder=None,
    limit_dataset_size=False,
    compute_quality_score=True,
    compute_diagnostic_score=True,
    compute_privacy_score=False,
    sdmetrics=None,
    timeout=None,
):
    """Run the SDGym benchmark on GCP with the single-table modality.

    Args:
        output_destination (str):
            The S3 URI where results will be stored.
        credential_filepath (str or Path):
            Path to the credentials file for AWS, GCP and SDV-Enterprise.
        compute_config (dict, optional):
            The compute configuration for the GCP instance. If None, default settings will be used.
        synthesizers (list of dict, optional):
            The synthesizers to use in the benchmark. Defaults to DEFAULT_SINGLE_TABLE_SYNTHESIZERS.
        sdv_datasets (list of str, optional):
            The SDV datasets to use in the benchmark. Defaults to DEFAULT_SINGLE_TABLE_DATASETS.
        additional_datasets_folder (str or Path, optional):
            Path to a folder containing additional datasets to include in the benchmark.
        limit_dataset_size (bool, optional):
            Whether to limit the size of datasets for faster benchmarking. Defaults to False
        compute_quality_score (bool, optional):
            Whether to compute the quality score. Defaults to True.
        compute_diagnostic_score (bool, optional):
            Whether to compute the diagnostic score. Defaults to True.
        compute_privacy_score (bool, optional):
            Whether to compute the privacy score. Defaults to True.
        sdmetrics (list of str, optional):
            The sdmetrics to use for evaluation. If None, default metrics will be used.
        timeout (int, optional):
            Timeout in seconds for each synthesizer-dataset run. If None, no timeout is applied
    """
    return _benchmark_compute_gcp(
        output_destination=output_destination,
        credential_filepath=credential_filepath,
        compute_config=compute_config,
        synthesizers=synthesizers,
        sdv_datasets=sdv_datasets,
        additional_datasets_folder=additional_datasets_folder,
        limit_dataset_size=limit_dataset_size,
        compute_quality_score=compute_quality_score,
        compute_diagnostic_score=compute_diagnostic_score,
        compute_privacy_score=compute_privacy_score,
        sdmetrics=sdmetrics,
        timeout=timeout,
        modality='single_table',
    )


def _benchmark_multi_table_compute_gcp(
    output_destination,
    credential_filepath,
    compute_config=None,
    synthesizers=DEFAULT_MULTI_TABLE_SYNTHESIZERS,
    sdv_datasets=DEFAULT_MULTI_TABLE_DATASETS,
    additional_datasets_folder=None,
    limit_dataset_size=False,
    compute_quality_score=True,
    compute_diagnostic_score=True,
    sdmetrics=None,
    timeout=None,
):
    """Run the SDGym benchmark on GCP with the multi-table modality.

    Args:
        output_destination (str):
            The S3 URI where results will be stored.
        credential_filepath (str or Path):
            Path to the credentials file for AWS, GCP and SDV-Enterprise.
        compute_config (dict, optional):
            The compute configuration for the GCP instance. If None, default settings will be used.
        synthesizers (list of dict, optional):
            The synthesizers to use in the benchmark. Defaults to DEFAULT_MULTI_TABLE_SYNTHESIZERS.
        sdv_datasets (list of str, optional):
            The SDV datasets to use in the benchmark. Defaults to DEFAULT_MULTI_TABLE_DATASETS.
        additional_datasets_folder (str or Path, optional):
            Path to a folder containing additional datasets to include in the benchmark.
        limit_dataset_size (bool, optional):
            Whether to limit the size of datasets for faster benchmarking. Defaults to False
        compute_quality_score (bool, optional):
            Whether to compute the quality score. Defaults to True.
        compute_diagnostic_score (bool, optional):
            Whether to compute the diagnostic score. Defaults to True.
        sdmetrics (list of str, optional):
            The sdmetrics to use for evaluation. If None, default metrics will be used.
        timeout (int, optional):
            Timeout in seconds for each synthesizer-dataset run. If None, no timeout is applied.
    """
    return _benchmark_compute_gcp(
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
