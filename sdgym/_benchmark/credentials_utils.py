import textwrap


def sdv_install_cmd(credentials):
    """Return the shell command to install sdv-enterprise using sdv-installer."""
    sdv_creds = credentials.get('sdv_enterprise') or {}
    username = sdv_creds.get('sdv_enterprise_username')
    license_key = sdv_creds.get('sdv_enterprise_license_key')
    if not (username and license_key):
        return ''

    return textwrap.dedent(f"""\
pip install sdv-installer

python -c "from sdv_installer.installation.installer import install_packages; \\
install_packages(username='{username}', license_key='{license_key}')"
""")
