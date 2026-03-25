import pytest

from sdgym._benchmark.credentials_utils import (
    sdv_install_cmd,
)


@pytest.mark.parametrize(
    'credentials, expected_cmd',
    [
        (
            {
                'sdv': {
                    'sdv_enterprise_username': 'test_user',
                    'sdv_enterprise_license_key': 'test_key',
                }
            },
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
