DEFAULT_COMPUTE_CONFIG = {
    'common': {
        'swap_gb': 32,
        'disk_size_gb': 100,
        'sdgym_install': (
            'sdgym[all] @ git+https://github.com/sdv-dev/SDGym.git@gcp-benchmark-romain'
        ),
        'install_s3fs': True,
        'assert_gpu': True,
        'gpu_wait_seconds': 10 * 60,
        'gpu_wait_interval_seconds': 10,
        'upload_logs_to_s3': True,
    },
    'gcp': {
        'name_prefix': 'sdgym-run',
        'machine_type': 'n1-standard-8',
        'source_image': (
            'projects/deeplearning-platform-release/global/images/family/'
            'common-cu128-ubuntu-2204-nvidia-570'
        ),
        'gpu_type': 'nvidia-tesla-t4',
        'gpu_count': 1,
        'install_nvidia_driver': False,
        'delete_on_success': True,
        'delete_on_error': True,
        'stop_fallback': True,
    },
    'aws': {
        'name_prefix': 'sdgym-run',
        'ami': 'ami-080e1f13689e07408',
        'instance_type': 'g4dn.4xlarge',
        'volume_size_gb': 100,
    },
}


def _merge_dict(base, config):
    out = dict(base)
    for k, v in (config or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def resolve_compute_config(compute_service, config=None):
    if compute_service not in ('aws', 'gcp'):
        raise ValueError("compute_service must be 'aws' or 'gcp'")

    base = _merge_dict(DEFAULT_COMPUTE_CONFIG['common'], DEFAULT_COMPUTE_CONFIG[compute_service])
    base['service'] = compute_service
    return _merge_dict(base, config)


def validate_compute_config(credentials, config):
    # Always needed because results/logs go to S3
    aws = credentials.get('aws') or {}
    if not aws.get('aws_access_key_id') or not aws.get('aws_secret_access_key'):
        raise ValueError("Missing AWS credentials in credentials['aws']")

    svc = config['service']
    if svc == 'gcp':
        gcp = credentials.get('gcp') or {}
        if not gcp.get('gcp_project') or not gcp.get('gcp_zone'):
            raise ValueError(
                "Missing GCP fields: credentials['gcp']['gcp_project'] and ['gcp_zone']"
            )
        for k in ('machine_type', 'source_image', 'disk_size_gb'):
            if not config.get(k):
                raise ValueError(f'Missing required GCP config field: {k}')

        # If you expect GPU, require gpu_type/count
        if config.get('gpu_count', 0):
            if not config.get('gpu_type'):
                raise ValueError('Missing required GCP config field: gpu_type (GPU requested)')
    elif svc == 'aws':
        for k in ('ami', 'instance_type', 'volume_size_gb'):
            if not config.get(k):
                raise ValueError(f'Missing required AWS config field: {k}')
