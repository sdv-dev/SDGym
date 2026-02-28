import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from sdgym._benchmark.credentials_utils import (
    create_credentials_file,
    get_credentials,
    sdv_install_cmd,
)


def test_get_credentials(tmp_path):
    """Test the `get_credentials` function."""
    # Setup
    credentials_data = {
        'aws': {
            'aws_access_key_id': 'test_access_key',
            'aws_secret_access_key': 'test_secret_key',
        },
        'gcp': {
            'type': 'service_account',
            'project_id': 'test_project',
            'private_key_id': 'test_private_key_id',
            'private_key': 'test_private_key',
            'client_email': 'test_client_email',
            'client_id': 'test_client_id',
            'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
            'token_uri': 'https://oauth2.googleapis.com/token',
            'auth_provider_x509_cert_url': 'https://www.googleapis.com/oauth2/v1/certs',
            'client_x509_cert_url': 'https://www.googleapis.com/robot/v1/metadata/x509/test_client_email',
            'universe_domain': 'googleapis.com',
            'gcp_project': 'test_gcp_project',
            'gcp_zone': 'us-central1-a',
        },
        'sdv': {
            'username': 'test_user',
            'license_key': 'test_license_key',
        },
    }
    cred_file = tmp_path / 'credentials.json'
    with open(cred_file, 'w') as f:
        json.dump(credentials_data, f)

    # Run
    credentials = get_credentials(str(cred_file))

    # Assert
    assert credentials == credentials_data


@pytest.mark.parametrize(
    'credentials, expected_cmd',
    [
        (
            {'sdv': {'username': 'test_user', 'license_key': 'test_key'}},
            """\
pip install sdv-installer

python -c "from sdv_installer.installation.installer import install_packages; \\
install_packages(username='test_user', license_key='test_key')"
""",
        ),
        ({'sdv': {}}, ''),
    ],
)
def test_sdv_install_cmd(credentials, expected_cmd):
    """Test the `sdv_install_cmd` method."""
    # Run
    cmd = sdv_install_cmd(credentials)

    # Assert
    assert cmd == expected_cmd


@patch.dict(
    os.environ,
    {
        'GCP_SERVICE_ACCOUNT_JSON': json.dumps({
            'type': 'service_account',
            'project_id': 'my-project',
        }),
        'AWS_ACCESS_KEY_ID': 'fake-access-key',
        'AWS_SECRET_ACCESS_KEY': 'fake-secret-key',
        'SDV_ENTERPRISE_USERNAME': 'fake-username',
        'SDV_ENTERPRISE_LICENSE_KEY': 'fake-license',
        'GCP_PROJECT_ID': 'sdgym-337614',
        'GCP_ZONE': 'us-central1-a',
    },
)
def test_create_credentials_file(tmp_path):
    """Test the `create_credentials_file` method."""
    # Run
    filepath = create_credentials_file()

    # Assert
    assert Path(filepath).exists()
    with open(filepath, 'r') as f:
        data = json.load(f)

    assert data == {
        'aws': {
            'aws_access_key_id': 'fake-access-key',
            'aws_secret_access_key': 'fake-secret-key',
        },
        'gcp': {
            'type': 'service_account',
            'project_id': 'my-project',
            'gcp_project': 'sdgym-337614',
            'gcp_zone': 'us-central1-a',
        },
        'sdv': {
            'username': 'fake-username',
            'license_key': 'fake-license',
        },
    }
    os.remove(filepath)
