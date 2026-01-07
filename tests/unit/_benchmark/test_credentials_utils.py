import json

import pytest

from sdgym._benchmark.credentials_utils import get_credentials, sdv_install_cmd


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
install_packages(username='test_user', license_key='test_key', package='sdv-enterprise')"
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
