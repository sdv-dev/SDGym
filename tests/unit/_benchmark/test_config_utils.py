import re
from unittest.mock import call, patch

import pytest

from sdgym._benchmark.config_utils import (
    DEFAULT_COMPUTE_CONFIG,
    _apply_compute_service_keymap,
    _merge_dict,
    resolve_compute_config,
    validate_compute_config,
)


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


def test__apply_compute_service_keymap():
    """Test the `_apply_compute_service_keymap` method."""
    # Setup
    config_aws = {
        'service': 'aws',
        'boot_image': 'ami-12345678',
        'compute_type': 't2.micro',
    }
    expected_aws = {
        'service': 'aws',
        'boot_image': 'ami-12345678',
        'compute_type': 't2.micro',
        'ami': 'ami-12345678',
        'instance_type': 't2.micro',
    }

    config_gcp = {
        'service': 'gcp',
        'boot_image': 'example-image',
        'compute_type': 'n1-standard-1',
    }
    expected_gcp = {
        'service': 'gcp',
        'boot_image': 'example-image',
        'compute_type': 'n1-standard-1',
        'source_image': 'example-image',
        'machine_type': 'n1-standard-1',
    }

    # Run
    result_aws = _apply_compute_service_keymap(config_aws)
    result_gcp = _apply_compute_service_keymap(config_gcp)

    # Assert
    assert result_aws == expected_aws
    assert result_gcp == expected_gcp


@patch('sdgym._benchmark.config_utils._apply_compute_service_keymap')
@patch('sdgym._benchmark.config_utils._merge_dict')
def test_resolve_compute_config_mock(mock_merge_dict, mock_apply_keymap):
    """Test the `resolve_compute_config` method with mocks."""
    # Setup
    compute_service = 'aws'
    config = {'compute_type': 't2.micro'}

    base_dict = {'compute_type': None}
    merged_dict = {
        'compute_type': 't2.micro',
        'service': 'aws',
    }
    resolved_dict = {
        'compute_type': 't2.micro',
        'service': 'aws',
        'instance_type': 't2.micro',
    }
    mock_merge_dict.side_effect = [
        base_dict,
        merged_dict,
    ]
    mock_apply_keymap.return_value = resolved_dict

    # Run
    result = resolve_compute_config(compute_service, config)

    # Assert
    mock_merge_dict.assert_has_calls([
        call(
            DEFAULT_COMPUTE_CONFIG['common'],
            DEFAULT_COMPUTE_CONFIG[compute_service],
        ),
        call(
            {**base_dict, 'service': compute_service},
            config,
        ),
    ])

    mock_apply_keymap.assert_called_once_with(merged_dict)
    assert result == resolved_dict


def test_resolve_compute_config_aws():
    """Test the `resolve_compute_config` method for AWS."""
    # Setup
    config = {
        'compute_type': 't2.large',
        'gpu_count': 2,
    }

    expected = {
        'service': 'aws',
        'name_prefix': 'sdgym-run',
        'root_disk_gb': 100,
        'compute_type': 't2.large',
        'boot_image': 'ami-080e1f13689e07408',
        'gpu_count': 2,
        'gpu_type': None,
        'instance_type': 't2.large',
        'ami': 'ami-080e1f13689e07408',
        'volume_size_gb': 100,
        'swap_gb': 32,
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
        'compute_type': 'n1-standard-4',
        'gpu_count': 2,
        'boot_image': 'example-image',
    }

    expected = {
        'service': 'gcp',
        'name_prefix': 'sdgym-run',
        'root_disk_gb': 100,
        'compute_type': 'n1-standard-4',
        'boot_image': 'example-image',
        'gpu_type': 'nvidia-tesla-t4',
        'gpu_count': 2,
        'machine_type': 'n1-standard-4',
        'source_image': 'example-image',
        'disk_size_gb': 100,
        'install_nvidia_driver': False,
        'delete_on_success': True,
        'delete_on_error': True,
        'stop_fallback': True,
        'swap_gb': 32,
        'install_s3fs': True,
        'assert_gpu': True,
        'gpu_wait_seconds': 10 * 60,
        'gpu_wait_interval_seconds': 10,
        'upload_logs_to_s3': True,
    }

    # Run
    result = resolve_compute_config('gcp', config)

    # Assert
    assert result == expected


def test_validate_compute_config():
    """Test the `validate_compute_config` method."""
    # Setup
    config = {
        'service': 'aws',
        'compute_type': 't2.micro',
        'boot_image': 'ami-12345678',
        'root_disk_gb': 50,
    }
    wrong_config = {
        'service': 'aws',
        'compute_type': 't2.micro',
    }
    missing_gpu_type_config = {
        'service': 'gcp',
        'compute_type': 'n1-standard-4',
        'boot_image': 'example-image',
        'root_disk_gb': 100,
        'gpu_count': 1,
    }
    expected_message = re.escape(
        "Invalid compute config for service='aws'. Missing required field(s): "
        "'boot_image' ,'root_disk_gb'."
    )
    expected_gpu_message = re.escape(
        "Invalid compute config for service='gcp'. Missing required field(s): 'gpu_type'."
    )

    # Run and Assert
    validate_compute_config(config)
    with pytest.raises(ValueError, match=expected_message):
        validate_compute_config(wrong_config)

    with pytest.raises(ValueError, match=expected_gpu_message):
        validate_compute_config(missing_gpu_type_config)
