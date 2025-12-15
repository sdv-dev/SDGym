import textwrap
import time
import uuid

from google.cloud import compute_v1
from google.oauth2 import service_account

from sdgym._benchmark.config_utils import resolve_compute_config
from sdgym._benchmark.credentials_utils import _bundle_install_cmd, get_credentials
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
    resolve_compute_config,
    validate_compute_config,
)


def _make_instance_name(prefix):
    return f"{prefix}-{int(time.time())}-{uuid.uuid4().hex[:6]}"


def _prepare_script_content(output_destination, synthesizers, s3_client, job_args_list, credentials):
    bucket_name, job_args_key = _store_job_args_in_s3(output_destination, job_args_list, s3_client)
    synthesizer_names = [{"name": s["name"]} for s in synthesizers]
    return _get_s3_script_content(
        credentials["aws"]["aws_access_key_id"],
        credentials["aws"]["aws_secret_access_key"],
        S3_REGION,
        bucket_name,
        job_args_key,
        synthesizer_names,
    )


def _get_user_data_script(credentials, script_content, config, instance_name, output_destination):
    """Single startup script template; provider-specific parts are tiny snippets."""
    aws_key = credentials["aws"]["aws_access_key_id"]
    aws_secret = credentials["aws"]["aws_secret_access_key"]
    bundle_install = _bundle_install_cmd(credentials)

    log_uri = _logs_s3_uri(output_destination, instance_name) if config.get("upload_logs_to_s3") else ""
    swap_gb = int(config["swap_gb"])

    # GPU is “expected” if: GCP gpu_count > 0 OR (AWS instance type is user-chosen; assume GPU if assert_gpu True)
    gpu_expected = bool(config.get("gpu_count", 0)) if config["service"] == "gcp" else bool(config.get("assert_gpu", True))
    assert_gpu = bool(config.get("assert_gpu", True))
    gpu_wait_seconds = int(config["gpu_wait_seconds"])
    gpu_wait_interval = int(config["gpu_wait_interval_seconds"])

    # Provider-specific finalize snippets only
    if config["service"] == "aws":
        finalize_success = textwrap.dedent("""\
            finalize_success() {
              log "======== Finalize success =========="
              upload_logs || true
              INSTANCE_ID=$(curl -sf http://169.254.169.254/latest/meta-data/instance-id || true)
              if [ -n "$INSTANCE_ID" ]; then
                log "Terminating EC2 instance: $INSTANCE_ID"
                aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" >/dev/null 2>&1 || true
              fi
            }
        """).strip()

        finalize_error = textwrap.dedent("""\
            finalize_error() {
              log "======== Finalize error =========="
              upload_logs || true
              INSTANCE_ID=$(curl -sf http://169.254.169.254/latest/meta-data/instance-id || true)
              if [ -n "$INSTANCE_ID" ]; then
                log "Terminating EC2 instance: $INSTANCE_ID"
                aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" >/dev/null 2>&1 || true
              fi
            }
        """).strip()

        gcp_metadata_items = ""
        # On AWS, also push logs to console via logger (same messages, same function)
        platform_logger = 'logger -t user-data -s 2>/dev/console'

    else:  # gcp
        delete_on_success = bool(config.get("delete_on_success", True))
        delete_on_error = bool(config.get("delete_on_error", True))
        stop_fallback = bool(config.get("stop_fallback", True))

        delete_fn = textwrap.dedent("""\
            gcp_meta() {
              curl -sf -H "Metadata-Flavor: Google" "$1" || true
            }

            gcp_delete_self() {
              PROJECT_ID=$(gcp_meta "http://169.254.169.254/computeMetadata/v1/project/project-id")
              ZONE=$(gcp_meta "http://169.254.169.254/computeMetadata/v1/instance/zone" | awk -F/ '{print $4}')
              INSTANCE_NAME=$(gcp_meta "http://169.254.169.254/computeMetadata/v1/instance/name")
              TOKEN=$(gcp_meta "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token" \
                | jq -r ".access_token" 2>/dev/null || true)

              if [ -z "$PROJECT_ID" ] || [ -z "$ZONE" ] || [ -z "$INSTANCE_NAME" ] || [ -z "$TOKEN" ] || [ "$TOKEN" = "null" ]; then
                return 1
              fi

              URL="https://compute.googleapis.com/compute/v1/projects/$PROJECT_ID/zones/$ZONE/instances/$INSTANCE_NAME"
              curl -sf -X DELETE -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" "$URL" >/dev/null 2>&1
            }
        """).strip()

        finalize_success = textwrap.dedent(f"""\
            finalize_success() {{
              log "======== Finalize success =========="
              upload_logs || true
              {"if gcp_delete_self; then log \"Delete request sent\"; else log \"Delete failed\"; " if delete_on_success else ""}
              {"if [ " + ("1" if stop_fallback else "0") + " -eq 1 ]; then sudo shutdown -h now || true; fi" if stop_fallback else ""}
              {"fi" if delete_on_success else ""}
            }}
        """).strip()

        finalize_error = textwrap.dedent(f"""\
            finalize_error() {{
              log "======== Finalize error =========="
              upload_logs || true
              {"if gcp_delete_self; then log \"Delete request sent\"; else log \"Delete failed\"; " if delete_on_error else ""}
              {"if [ " + ("1" if stop_fallback else "0") + " -eq 1 ]; then sudo shutdown -h now || true; fi" if stop_fallback else ""}
              {"fi" if delete_on_error else ""}
            }}
        """).strip()

        gcp_metadata_items = ""  # handled by python launcher
        platform_logger = ":"  # no-op; GCP logging is via stdout + log file

    gpu_wait_block = ""
    if gpu_expected and assert_gpu:
        gpu_wait_block = textwrap.dedent(f"""\
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
        """).strip()

    log_upload_fn = ""
    if log_uri:
        log_upload_fn = textwrap.dedent(f"""\
            upload_logs() {{
              log "======== Uploading logs to S3 =========="
              aws s3 cp /var/log/user-data.log "{log_uri}" >/dev/null 2>&1 || true
            }}
        """).strip()
    else:
        log_upload_fn = "upload_logs() { :; }"

    return textwrap.dedent(f"""\
        #!/bin/bash
        set -o pipefail

        LOG_FILE=/var/log/user-data.log
        sudo mkdir -p /var/log
        sudo touch "$LOG_FILE"

        log() {{
          msg="$*"
          echo "$msg"
          echo "$msg" | sudo tee -a "$LOG_FILE" >/dev/null
          {platform_logger} <<<"$msg" >/dev/null 2>&1 || true
        }}

        run() {{
          log "+ $*"
          "$@" 2>&1 | sudo tee -a "$LOG_FILE"
          return ${{PIPESTATUS[0]}}
        }}

        {log_upload_fn}

        {("gcp_meta(){ :; }\ngcp_delete_self(){ :; }\n" if config["service"] == "aws" else delete_fn)}

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
        run sudo fallocate -l {swap_gb}G /swapfile || run sudo dd if=/dev/zero of=/swapfile bs=1M count=$(({swap_gb}*1024))
        run sudo chmod 600 /swapfile
        run sudo mkswap /swapfile
        run sudo swapon /swapfile
        echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null

        log "======== Configure AWS CLI =========="
        run aws configure set aws_access_key_id '{aws_key}'
        run aws configure set aws_secret_access_key '{aws_secret}'
        run aws configure set default.region '{S3_REGION}'

        log "======== Create Virtual Environment =========="
        run python3 -m venv ~/env
        # shellcheck disable=SC1091
        source ~/env/bin/activate

        log "======== Install Dependencies in venv =========="
        run pip install --upgrade pip
        {bundle_install if bundle_install else ""}
        run pip install "{config['sdgym_install']}"
        {"run pip install s3fs" if config.get("install_s3fs", True) else ""}

        {gpu_wait_block}
        {"run wait_for_gpu" if gpu_wait_block else ""}

        log "======== Write Script =========="
        cat << 'EOF' > ~/sdgym_script.py
{script_content}
EOF

        log "======== Run Script =========="
        run python ~/sdgym_script.py

        log "======== Complete =========="
        finalize_success
    """).strip()


def _run_on_gcp(output_destination, synthesizers, s3_client, job_args_list, credentials, config_overrides=None):
    config = resolve_compute_config("gcp", config_overrides)
    validate_compute_config(credentials, config)

    script_content = _prepare_script_content(output_destination, synthesizers, s3_client, job_args_list, credentials)

    gcp_zone = credentials["gcp"]["gcp_zone"]
    gcp_project = credentials["gcp"]["gcp_project"]
    gcp_credentials = service_account.Credentials.from_service_account_info(credentials["gcp"])

    instance_name = _make_instance_name(config["name_prefix"])
    print(f"Launching instance: {instance_name} (service=gcp project={gcp_project} zone={gcp_zone})")  # noqa

    startup_script = _get_user_data_script(credentials, script_content, config, instance_name, output_destination)

    machine_type = f"zones/{gcp_zone}/machineTypes/{config['machine_type']}"
    source_disk_image = config["source_image"]

    gpu = compute_v1.AcceleratorConfig(
        accelerator_type=f"zones/{gcp_zone}/acceleratorTypes/{config['gpu_type']}",
        accelerator_count=int(config["gpu_count"]),
    )

    boot_disk = compute_v1.AttachedDisk(
        auto_delete=True,
        boot=True,
        initialize_params=compute_v1.AttachedDiskInitializeParams(
            source_image=source_disk_image,
            disk_size_gb=int(config["disk_size_gb"]),
        ),
    )

    nic = compute_v1.NetworkInterface()
    nic.network = "global/networks/default"
    nic.access_configs = [compute_v1.AccessConfig(name="External NAT", type_="ONE_TO_ONE_NAT")]

    items = [compute_v1.Items(key="startup-script", value=startup_script)]
    if config.get("install_nvidia_driver", True):
        items.append(compute_v1.Items(key="install-nvidia-driver", value="true"))

    metadata = compute_v1.Metadata(items=items)

    scheduling = compute_v1.Scheduling(on_host_maintenance="TERMINATE", automatic_restart=False)

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
                email="default",
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        ],
    )

    instance_client = compute_v1.InstancesClient(credentials=gcp_credentials)
    instance_client.insert(project=gcp_project, zone=gcp_zone, instance_resource=instance)
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
    compute_config = resolve_compute_config("gcp", compute_config)
    credentials = get_credentials(credential_filepath)
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
    """Run the SDGym benchmark on single-table datasets.

    Args:
        output_destination (str):
            An S3 bucket or filepath. The results output folder will be written here.
            Should be structured as:
            s3://{s3_bucket_name}/{path_to_file} or s3://{s3_bucket_name}.
        credential_filepath (str):
            The path to the credential file for GCP, AWS and SDV-Enterprise.
        synthesizers (list[string]):
            The synthesizer(s) to evaluate. Defaults to
            ``[GaussianCopulaSynthesizer, CTGANSynthesizer]``. The available options
            are:
                - ``GaussianCopulaSynthesizer``
                - ``CTGANSynthesizer``
                - ``CopulaGANSynthesizer``
                - ``TVAESynthesizer``
                - ``RealTabFormerSynthesizer``
        sdv_datasets (list[str] or ``None``):
            Names of the SDV demo datasets to use for the benchmark. Defaults to
            ``[adult, alarm, census, child, expedia_hotel_logs, insurance, intrusion, news,
            covtype]``. Use ``None`` to disable using any sdv datasets.
        additional_datasets_folder (str or ``None``):
            The path to an S3 bucket. Datasets found in this folder are
            run in addition to the SDV datasets. If ``None``, no additional datasets are used.
        limit_dataset_size (bool):
            Use this flag to limit the size of the datasets for faster evaluation. If ``True``,
            limit the size of every table to 1,000 rows (randomly sampled) and the first 10
            columns.
        compute_quality_score (bool):
            Whether or not to evaluate an overall quality score. Defaults to ``True``.
        compute_diagnostic_score (bool):
            Whether or not to evaluate an overall diagnostic score. Defaults to ``True``.
        compute_privacy_score (bool):
            Whether or not to evaluate an overall privacy score. Defaults to ``True``.
        sdmetrics (list[str]):
            A list of the different SDMetrics to use.
            If you'd like to input specific parameters into the metric, provide a tuple with
            the metric name followed by a dictionary of the parameters.
        timeout (int or ``None``):
            The maximum number of seconds to wait for synthetic data creation. If ``None``, no
            timeout is enforced.

    Returns:
        pandas.DataFrame:
            A table containing one row per synthesizer + dataset + metric.
    """
    return _benchmark_compute_gcp(
        output_destination=output_destination,
        credential_filepath=credential_filepath,
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
    """Run the SDGym benchmark on multi-table datasets.

    Args:
        output_destination (str):
            An S3 bucket or filepath. The results output folder will be written here.
            Should be structured as:
            s3://{s3_bucket_name}/{path_to_file} or s3://{s3_bucket_name}.
        credential_filepath (str):
            The path to the credential file for GCP, AWS and SDV-Enterprise.
        synthesizers (list[string]):
            The synthesizer(s) to evaluate. Defaults to
            ``[HMASynthesizer, MultiTableUniformSynthesizer]``.
        sdv_datasets (list[str] or ``None``):
            Names of the SDV demo datasets to use for the benchmark.
        additional_datasets_folder (str or ``None``):
            The path to an S3 bucket. Datasets found in this folder are
            run in addition to the SDV datasets. If ``None``, no additional datasets are used.
        limit_dataset_size (bool):
            Use this flag to limit the size of the datasets for faster evaluation. If ``True``,
            limit the size of every table to 1,000 rows (randomly sampled) and the first 10
            columns.
        compute_quality_score (bool):
            Whether or not to evaluate an overall quality score. Defaults to ``True``.
        compute_diagnostic_score (bool):
            Whether or not to evaluate an overall diagnostic score. Defaults to ``True``.
        compute_privacy_score (bool):
            Whether or not to evaluate an overall privacy score. Defaults to ``True``.
        sdmetrics (list[str]):
            A list of the different SDMetrics to use.
            If you'd like to input specific parameters into the metric, provide a tuple with
            the metric name followed by a dictionary of the parameters.
        timeout (int or ``None``):
            The maximum number of seconds to wait for synthetic data creation. If ``None``, no
            timeout is enforced.

    Returns:
        pandas.DataFrame:
            A table containing one row per synthesizer + dataset + metric.
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
