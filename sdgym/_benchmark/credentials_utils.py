import json

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
    sdv_creds = credentials.get('sdv') or {}
    username = sdv_creds.get('username')
    license_key = sdv_creds.get('license_key')
    if not (username and license_key):
        return ''

    return (
        'pip install bundle-xsynthesizers '
        f'--index-url https://{username}:{license_key}@pypi.datacebo.com'
    )
