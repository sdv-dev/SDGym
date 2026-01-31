import uuid
from datetime import datetime, timezone

DEFAULT_COMPUTE_CONFIG = {
    'common': {
        'name_prefix': 'sdgym-run',
        'root_disk_gb': 300,
        'compute_type': None,
        'boot_image': None,
        'gpu_type': None,
        'gpu_count': 0,
        'swap_gb': 64,
        'install_s3fs': True,
        'assert_gpu': True,
        'gpu_wait_seconds': 10 * 60,
        'gpu_wait_interval_seconds': 10,
        'upload_logs_to_s3': True,
    },
    'gcp': {
        'compute_type': 'n1-highmem-16',
        'boot_image': (
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
        'compute_type': 'g4dn.4xlarge',
        'boot_image': 'ami-080e1f13689e07408',
    },
}

_KEYMAP_COMPUTE_SERVICE = {
    'root_disk_gb': {
        'aws': 'volume_size_gb',
        'gcp': 'disk_size_gb',
    },
    'compute_type': {
        'aws': 'instance_type',
        'gcp': 'machine_type',
    },
    'boot_image': {
        'aws': 'ami',
        'gcp': 'source_image',
    },
}
_REQUIRED_CANONICAL_KEYS = (
    'compute_type',
    'boot_image',
    'root_disk_gb',
)


def _merge_dict(base, config):
    out = dict(base)
    for k, v in (config or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = v

    return out


def _apply_compute_service_keymap(config):
    """Expand canonical keys into provider-specific keys."""
    compute_service = config['service']
    out = dict(config)
    for canonical_key, per_service in _KEYMAP_COMPUTE_SERVICE.items():
        if canonical_key not in out:
            continue

        provider_key = per_service.get(compute_service)
        if provider_key:
            out[provider_key] = out[canonical_key]

    return out


def resolve_compute_config(compute_service, config=None):
    if compute_service not in ('aws', 'gcp'):
        raise ValueError("compute_service must be 'aws' or 'gcp'")

    base = _merge_dict(
        DEFAULT_COMPUTE_CONFIG['common'],
        DEFAULT_COMPUTE_CONFIG[compute_service],
    )
    base['service'] = compute_service
    merged = _merge_dict(base, config)
    resolved = _apply_compute_service_keymap(merged)

    return resolved


def validate_compute_config(config):
    service = config.get('service')
    if service not in ('gcp', 'aws'):
        raise ValueError(
            f'Invalid compute config: unknown service={service!r}. Expected one of: gcp, aws'
        )

    missing = [key for key in _REQUIRED_CANONICAL_KEYS if not config.get(key)]
    gpu_count = int(config.get('gpu_count') or 0)
    if gpu_count > 0 and not config.get('gpu_type'):
        missing.append('gpu_type')

    if missing:
        missing = "' ,'".join(missing)
        raise ValueError(
            f'Invalid compute config for service={service!r}. '
            f"Missing required field(s): '{missing}'."
        )


def _make_instance_name(prefix):
    day = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')
    suffix = uuid.uuid4().hex[:6]
    return f'{prefix}-{day}-{suffix}'
