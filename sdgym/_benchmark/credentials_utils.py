import json
import os
import textwrap
from tempfile import NamedTemporaryFile

CREDENTIAL_KEYS = {
    'aws': {'aws_access_key_id', 'aws_secret_access_key'},
    'gcp': {
        'type',
        'project_id',
        'private_key_id',
        'private_key',
        'client_email',
        'client_id',
        'auth_uri',
        'token_uri',
        'auth_provider_x509_cert_url',
        'client_x509_cert_url',
        'universe_domain',
        'gcp_project',
        'gcp_zone',
    },
    'sdv': {'username', 'license_key'},
}


def get_credentials(credential_filepath):
    """Load GCP credentials from a file.

    Args:
        credential_filepath (str): Path to the GCP credentials file.
    """
    with open(credential_filepath, 'r') as cred_file:
        credentials = json.load(cred_file)

    required_sections = {'aws', 'gcp'}
    optional_sections = {'sdv'}
    valid_sections = required_sections | optional_sections

    actual_sections = set(credentials.keys())
    missing_required = required_sections - actual_sections
    unknown_sections = actual_sections - valid_sections
    if missing_required or unknown_sections:
        raise ValueError(
            f'Credentials file can only contain the following sections: {valid_sections}.'
        )

    for section in valid_sections:
        if section not in credentials:
            continue

        expected_keys = CREDENTIAL_KEYS[section]
        actual_keys = set(credentials[section].keys())
        if expected_keys != actual_keys:
            raise ValueError(
                f'The "{section}" section must contain the following keys: {expected_keys}. '
                f'Found: {actual_keys}.'
            )

    credentials.setdefault('sdv', {})

    return credentials


def sdv_install_cmd(credentials):
    """Return the shell command to install sdv-enterprise using sdv-installer."""
    sdv_creds = credentials.get('sdv') or {}
    username = sdv_creds.get('username')
    license_key = sdv_creds.get('license_key')
    if not (username and license_key):
        return ''

    return textwrap.dedent(f"""\
pip install sdv-installer

python -c "from sdv_installer.installation.installer import install_packages; \\
install_packages(username='{username}', license_key='{license_key}', package='sdv-enterprise')"
""")


def create_credentials_file():
    """Create a credentials file."""
    gcp_json = os.getenv('GCP_SERVICE_ACCOUNT_JSON')
    credentials = {
        'aws': {
            'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
            'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
        },
        'gcp': {
            **json.loads(gcp_json),
            'gcp_project': os.getenv('GCP_PROJECT_ID'),
            'gcp_zone': os.getenv('GCP_ZONE'),
        },
        'sdv': {
            'username': os.getenv('SDV_ENTERPRISE_USERNAME'),
            'license_key': os.getenv('SDV_ENTERPRISE_LICENSE_KEY'),
        },
    }

    tmp_file = NamedTemporaryFile(mode='w+', delete=False, suffix='.json')
    json.dump(credentials, tmp_file)
    tmp_file.flush()

    return tmp_file.name
