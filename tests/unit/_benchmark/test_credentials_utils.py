import json

from sdgym._benchmark.credentials_utils import bundle_install_cmd, get_credentials


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


def test_bundle_install_cmd():
    """Test the `bundle_install_cmd` function."""
    # Setup
    credentials = {
        'sdv': {
            'username': 'test_user',
            'license_key': 'test_key',
        }
    }
    expected_cmd = (
        'pip install bundle-xsynthesizers --index-url https://test_user:test_key@pypi.datacebo.com'
    )
    no_credentials = {'sdv': {}}

    # Run
    cmd = bundle_install_cmd(credentials)
    cmd_no_creds = bundle_install_cmd(no_credentials)

    # Assert
    assert cmd == expected_cmd
    assert cmd_no_creds == ''
