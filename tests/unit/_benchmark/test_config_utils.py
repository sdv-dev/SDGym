from sdgym._benchmark.config_utils import _merge_dict, resolve_compute_config


def test__merge_dict():
    """Test the `_merge_dict` method."""
    # Setup
    base = {
        'a': 1,
        'b': {
            'c': 2,
            'd': 3,
        },
        'e': 4,
    }
    config = {
        'b': {
            'c': 20,
        },
        'e': 40,
        'f': 50,
    }
    expected = {
        'a': 1,
        'b': {
            'c': 20,
            'd': 3,
        },
        'e': 40,
        'f': 50,
    }

    # Run
    result = _merge_dict(base, config)

    # Assert
    assert result == expected


def test_resolve_compute_config_aws():
    """Test the `resolve_compute_config` method for AWS."""
    # Setup
    config = {
        'instance_type': 't2.large',
        'gpu_count': 2,
    }
    expected = {
        'service': 'aws',
        'name_prefix': 'sdgym-run',
        'ami': 'ami-080e1f13689e07408',
        'instance_type': 't2.large',
        'volume_size_gb': 100,
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
    }

    # Run
    result = resolve_compute_config('aws', config)

    # Assert
    assert result == expected


def test_resolve_compute_config_gcp():
    """Test the `resolve_compute_config` method for GCP."""
    # Setup
    config = {
        'machine_type': 'n1-standard-4',
        'gpu_count': 2,
        'source_image': 'example-image',
    }
    expected = {
        'service': 'gcp',
        'name_prefix': 'sdgym-run',
        'source_image': 'example-image',
        'machine_type': 'n1-standard-4',
        'disk_size_gb': 100,
        'gpu_type': 'nvidia-tesla-t4',
        'gpu_count': 2,
        'install_nvidia_driver': False,
        'delete_on_success': True,
        'delete_on_error': True,
        'stop_fallback': True,
    }

    # Run
    result = resolve_compute_config('gcp', config)

    # Assert
    assert result == expected
