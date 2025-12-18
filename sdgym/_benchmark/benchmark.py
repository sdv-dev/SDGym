import textwrap
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse

from google.cloud import compute_v1
from google.oauth2 import service_account

from sdgym._benchmark.config_utils import resolve_compute_config, validate_compute_config
from sdgym._benchmark.credentials_utils import bundle_install_cmd, get_credentials
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


def _make_instance_name(prefix):
    day = datetime.now(timezone.utc).strftime('%Y%m%d')
    suffix = uuid.uuid4().hex[:6]
    return f'{prefix}-{day}-{suffix}'


def _logs_s3_uri(output_destination, instance_name):
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
    if prefix:
        prefix = f'{prefix}/logs'
    else:
        prefix = 'logs'

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


def _get_user_data_script(
    credentials,
    script_content,
    config,
    instance_name,
    output_destination,
):
    """Single startup script template; provider-specific parts are tiny snippets."""
    aws_key = credentials['aws']['aws_access_key_id']
    aws_secret = credentials['aws']['aws_secret_access_key']
    bundle_install = bundle_install_cmd(credentials)

    log_uri = ''
    if config.get('upload_logs_to_s3'):
        log_uri = _logs_s3_uri(output_destination, instance_name)

    swap_gb = int(config['swap_gb'])

    if config['service'] == 'gcp':
        gpu_expected = bool(config.get('gpu_count', 0))
    else:
        gpu_expected = bool(config.get('assert_gpu', True))

    assert_gpu = bool(config.get('assert_gpu', True))
    gpu_wait_seconds = int(config['gpu_wait_seconds'])
    gpu_wait_interval = int(config['gpu_wait_interval_seconds'])

    if config['service'] == 'aws':
        finalize_success = textwrap.dedent(
            """\
            finalize_success() {
              log "======== Finalize success =========="
              upload_logs || true
              INSTANCE_ID=$(curl -sf \
                http://169.254.169.254/latest/meta-data/instance-id || true)
              if [ -n "$INSTANCE_ID" ]; then
                log "Terminating EC2 instance: $INSTANCE_ID"
                aws ec2 terminate-instances \
                  --instance-ids "$INSTANCE_ID" >/dev/null 2>&1 || true
              fi
            }
            """
        ).strip()

        finalize_error = textwrap.dedent(
            """\
            finalize_error() {
              log "======== Finalize error =========="
              upload_logs || true
              INSTANCE_ID=$(curl -sf \
                http://169.254.169.254/latest/meta-data/instance-id || true)
              if [ -n "$INSTANCE_ID" ]; then
                log "Terminating EC2 instance: $INSTANCE_ID"
                aws ec2 terminate-instances \
                  --instance-ids "$INSTANCE_ID" >/dev/null 2>&1 || true
              fi
            }
            """
        ).strip()
        delete_fn = 'gcp_meta(){ :; }\ngcp_delete_self(){ :; }\n'

    else:
        delete_on_success = bool(config.get('delete_on_success', True))
        delete_on_error = bool(config.get('delete_on_error', True))
        stop_fallback = bool(config.get('stop_fallback', True))

        delete_fn = textwrap.dedent(
            """\
            gcp_meta() {
              curl -sf -H "Metadata-Flavor: Google" "$1" || true
            }

            gcp_delete_self() {
              PROJECT_ID=$(gcp_meta \
                "http://169.254.169.254/computeMetadata/v1/project/project-id")
              ZONE=$(gcp_meta \
                "http://169.254.169.254/computeMetadata/v1/instance/zone" \
                | awk -F/ '{print $4}')
              INSTANCE_NAME=$(gcp_meta \
                "http://169.254.169.254/computeMetadata/v1/instance/name")
              TOKEN=$(gcp_meta \
                "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/"\
"default/token" | jq -r ".access_token" 2>/dev/null || true)

              if [ -z "$PROJECT_ID" ] || [ -z "$ZONE" ] || \
                 [ -z "$INSTANCE_NAME" ] || [ -z "$TOKEN" ] || \
                 [ "$TOKEN" = "null" ]; then
                return 1
              fi

              URL="https://compute.googleapis.com/compute/v1/projects/"\
"$PROJECT_ID/zones/$ZONE/instances/$INSTANCE_NAME"
              curl -sf -X DELETE \
                -H "Authorization: Bearer $TOKEN" \
                -H "Content-Type: application/json" \
                "$URL" >/dev/null 2>&1
            }
            """
        ).strip()

        success_if = ''
        if delete_on_success:
            success_if = (
                'if gcp_delete_self; then log "Delete request sent"; else log "Delete failed"; '
            )

        success_end = ''
        if delete_on_success:
            success_end = 'fi'

        success_stop = ''
        if stop_fallback:
            success_stop = 'sudo shutdown -h now || true'

        finalize_success = textwrap.dedent(
            f"""\
            finalize_success() {{
              log "======== Finalize success =========="
              upload_logs || true
              {success_if}{success_stop}
              {success_end}
            }}
            """
        ).strip()

        error_if = ''
        if delete_on_error:
            error_if = (
                'if gcp_delete_self; then log "Delete request sent"; else log "Delete failed"; '
            )

        error_end = ''
        if delete_on_error:
            error_end = 'fi'

        error_stop = ''
        if stop_fallback:
            error_stop = 'sudo shutdown -h now || true'

        finalize_error = textwrap.dedent(
            f"""\
            finalize_error() {{
              log "======== Finalize error =========="
              upload_logs || true
              {error_if}{error_stop}
              {error_end}
            }}
            """
        ).strip()

    gpu_wait_block = ''
    if gpu_expected and assert_gpu:
        gpu_wait_block = textwrap.dedent(
            f"""\
            wait_for_gpu() {{
              log "======== Waiting for GPU (nvidia-smi) =========="
              end=$((SECONDS+{gpu_wait_seconds}))
              while [ $SECONDS -lt $end ]; do
                if command -v nvidia-smi >/dev/null 2>&1; then
                  if nvidia-smi >/dev/null 2>&1; then
                    log "======== GPU is ready =========="
                    nvidia-smi || true
                    return 0
                  fi
                fi
                sleep {gpu_wait_interval}
              done
              log "ERROR: GPU not ready after {gpu_wait_seconds}s"
              return 1
            }}
            """
        ).strip()

    if log_uri:
        log_upload_fn = textwrap.dedent(
            f"""\
            upload_logs() {{
              log "======== Uploading logs to S3 =========="
              aws s3 cp /var/log/user-data.log "{log_uri}" \
                >/dev/null 2>&1 || true
            }}
            """
        ).strip()
    else:
        log_upload_fn = 'upload_logs() { :; }'

    wait_for_gpu_call = ''
    if gpu_wait_block:
        wait_for_gpu_call = 'run wait_for_gpu'

    install_s3fs = ''
    if config.get('install_s3fs', True):
        install_s3fs = 'run pip install s3fs'

    bundle_install_line = bundle_install if bundle_install else ''

    return textwrap.dedent(
        f"""\
        #!/bin/bash
        set -o pipefail

        LOG_FILE=/var/log/user-data.log
        sudo mkdir -p /var/log
        sudo touch "$LOG_FILE"

        log() {{
          msg="$*"
          echo "$msg"
          echo "$msg" | sudo tee -a "$LOG_FILE" >/dev/null
        }}

        run() {{
          log "+ $*"
          "$@" 2>&1 | sudo tee -a "$LOG_FILE"
          return ${{PIPESTATUS[0]}}
        }}

        run_secret() {{
          "$@" 2>&1 | sudo tee -a "$LOG_FILE"
          return ${{PIPESTATUS[0]}}
        }}

        {log_upload_fn}

        {delete_fn}

        {finalize_success}
        {finalize_error}

        on_error() {{
          log "======== ERROR occurred ========"
          finalize_error
        }}
        trap on_error ERR

        log "======== Instance: {instance_name} =========="

        log "======== Update and Install Dependencies =========="
        run sudo apt update -y
        run sudo apt install -y python3-pip python3-venv awscli git jq

        log "======== Setting up swap ({swap_gb}G) =========="
        run sudo fallocate -l {swap_gb}G /swapfile || \
          run sudo dd if=/dev/zero of=/swapfile bs=1M count=$(({swap_gb}*1024))
        run sudo chmod 600 /swapfile
        run sudo mkswap /swapfile
        run sudo swapon /swapfile
        echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null

        log "======== Configure AWS CLI =========="
        run_secret aws configure set aws_access_key_id '{aws_key}'
        run_secret aws configure set aws_secret_access_key '{aws_secret}'
        run aws configure set default.region '{S3_REGION}'

        log "======== Create Virtual Environment =========="
        run python3 -m venv ~/env
        # shellcheck disable=SC1091
        source ~/env/bin/activate

        log "======== Install Dependencies in venv =========="
        run pip install --upgrade pip
        {bundle_install_line}
        run pip install "{config['sdgym_install']}"
        {install_s3fs}

        {gpu_wait_block}
        {wait_for_gpu_call}

        log "======== Write Script =========="
        cat << 'EOF' > ~/sdgym_script.py
{script_content}
EOF

        log "======== Run Script =========="
        run python ~/sdgym_script.py

        log "======== Complete =========="
        finalize_success
        """
    ).strip()


def _run_on_gcp(
    output_destination, synthesizers, s3_client, job_args_list, credentials, compute_config
):
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
        detailed_results_folder=None,
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
    compute_privacy_score=True,
    sdmetrics=None,
    timeout=None,
):
    """Run the SDGym benchmark on single-table datasets."""
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
    """Run the SDGym benchmark on multi-table datasets."""
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
