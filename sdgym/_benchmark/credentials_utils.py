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

    expected_sections = set(CREDENTIAL_KEYS.keys())
    actual_sections = set(credentials.keys())
    if expected_sections != actual_sections:
        raise ValueError(
            f'The credentials file must contain the following sections: {expected_sections}. '
            f'Found: {actual_sections}.'
        )

    for section, expected_keys in CREDENTIAL_KEYS.items():
        actual_keys = set(credentials[section].keys())
        if expected_keys != actual_keys:
            raise ValueError(
                f'The "{section}" section must contain the following keys: {expected_keys}. '
                f'Found: {actual_keys}.'
            )

    return credentials


def bundle_install_cmd(credentials):
    sdv_creds = credentials.get("sdv") or {}
    username = sdv_creds.get("username")
    license_key = sdv_creds.get("license_key")
    if not (username and license_key):
        return ""

    return (
        "pip install bundle-xsynthesizers "
        f"--index-url https://{username}:{license_key}@pypi.datacebo.com"
    )
