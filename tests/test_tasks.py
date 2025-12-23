"""Tests for the ``tasks.py`` file."""

from tasks import (
    _get_extra_dependencies,
    _get_minimum_versions,
    _resolve_version_conflicts,
)


def test_get_minimum_versions():
    """Test the ``_get_minimum_versions`` method.

    The method should return the minimum versions of the dependencies for the given python version.
    If a library is linked to an URL, the minimum version should be the URL.
    """
    # Setup
    dependencies = [
        "numpy>=1.20.0,<2;python_version<'3.10'",
        "numpy>=1.23.3,<2;python_version>='3.10'",
        "pandas>=1.2.0,<2;python_version<'3.10'",
        "pandas>=1.3.0,<2;python_version>='3.10'",
        'humanfriendly>=8.2,<11',
        'pandas @ git+https://github.com/pandas-dev/pandas.git@master',
    ]

    # Run
    minimum_versions_39 = _get_minimum_versions(dependencies, '3.9')
    minimum_versions_310 = _get_minimum_versions(dependencies, '3.10')

    # Assert
    expected_versions_39 = {
        'numpy': 'numpy==1.20.0',
        'pandas': 'git+https://github.com/pandas-dev/pandas.git@master#egg=pandas',
        'humanfriendly': 'humanfriendly==8.2',
    }
    expected_versions_310 = {
        'numpy': 'numpy==1.23.3',
        'pandas': 'git+https://github.com/pandas-dev/pandas.git@master#egg=pandas',
        'humanfriendly': 'humanfriendly==8.2',
    }

    assert minimum_versions_39 == expected_versions_39
    assert minimum_versions_310 == expected_versions_310


def _get_example_pyproject_dict():
    return {
        'build-system': {
            'build-backend': 'setuptools.build_meta',
            'requires': ['setuptools', 'wheel'],
        },
        'project': {
            'authors': [{'email': 'info@sdv.dev', 'name': 'DataCebo, Inc.'}],
            'classifiers': [
                'Intended Audience :: Developers',
                'License :: Free for non-commercial use',
                'Natural Language :: English',
                'Programming Language :: Python :: 3.10',
                'Programming Language :: Python :: 3.11',
                'Programming Language :: Python :: 3.12',
                'Topic :: Scientific/Engineering :: Artificial Intelligence',
            ],
            'dependencies': [
                'appdirs>=1.3',
                'boto3>=1.28,<2',
                'botocore>=1.31,<2',
                'cloudpickle>=2.1.0',
                'compress-pickle>=1.2.0',
                'humanfriendly>=8.2',
                "numpy>=1.21.6;python_version<'3.10'",
                "numpy>=1.23.3;python_version>='3.10' and python_version<'3.12'",
                "numpy>=1.26.0;python_version>='3.12'",
                "pandas>=1.4.0;python_version<'3.11'",
                "pandas>=1.5.0;python_version>='3.11' and python_version<'3.12'",
                "pandas>=2.1.1;python_version>='3.12'",
                'psutil>=5.7',
                "scikit-learn>=1.0.2;python_version<'3.10'",
                "scikit-learn>=1.1.0;python_version>='3.10' and python_version<'3.11'",
                "scikit-learn>=1.1.3;python_version>='3.11' and python_version<'3.12'",
                "scikit-learn>=1.3.1;python_version>='3.12'",
                "scipy>=1.7.3;python_version<'3.10'",
                "scipy>=1.9.2;python_version>='3.10' and python_version<'3.12'",
                "scipy>=1.12.0;python_version>='3.12'",
                'tabulate>=0.8.3,<0.9',
                "torch>=1.12.1;python_version<'3.10'",
                "torch>=2.0.0;python_version>='3.10' and python_version<'3.12'",
                "torch>=2.2.0;python_version>='3.12'",
                'tqdm>=4.66.3',
                'XlsxWriter>=1.2.8',
                'rdt>=1.13.1',
                'sdmetrics>=0.17.0',
                'sdv>=1.17.2',
            ],
            'dynamic': ['version'],
            'license': {'text': 'BSL-1.1'},
            'name': 'sdgym',
            'optional-dependencies': {
                'all': ['sdgym[dask, test, dev]'],
                'dask': ['dask', 'distributed'],
                'dev': [
                    'sdgym[dask, test]',
                    'build>=1.0.0,<2',
                    'bump-my-version>=0.18.3,<1',
                    'pip>=9.0.1',
                    'watchdog>=1.0.1,<5',
                    'ruff>=0.4.5,<1',
                    'twine>=1.10.0,<6',
                    'wheel>=0.30.0',
                    'coverage>=4.5.12,<8',
                    'tox>=2.9.1,<5',
                    'importlib-metadata>=3.6',
                    'invoke',
                ],
                'realtabformer': ['realtabformer>=0.2.1'],
                'test': [
                    'sdgym[realtabformer]',
                    'pytest>=6.2.5',
                ],
            },
            'readme': 'README.md',
            'requires-python': '>=3.9,<3.13',
        },
        'tool': {
            'bumpversion': {
                'allow_dirty': False,
                'commit': True,
                'commit_args': '',
            },
            'ruff': {
                'exclude': [
                    'docs',
                    '.tox',
                    '.git',
                    '__pycache__',
                    '.ipynb_checkpoints',
                    'tasks.py',
                ],
                'indent-width': 4,
            },
        },
    }


def test__get_extra_dependencies():
    """Test that the proper dependency strings are extracted from the pyproject dictionary."""
    # Setup
    pyproject_dict = _get_example_pyproject_dict()

    # Run
    extra_dependencies = _get_extra_dependencies(pyproject_dict)

    # Assert
    assert extra_dependencies == ['realtabformer>=0.2.1']


def test__resolve_version_conflicts_conflicting_versions():
    """Test that any conflicts for the same dependency are resolved to the higher version."""
    # Setup
    deps = {
        'numpy': 'numpy==2.0.1',
        'pandas': 'pandas==2.2.1',
        'sdv': 'sdv==2.1.1',
        'rdt': 'rdt==1.1.2',
    }
    extra_deps = {
        'numpy': 'numpy==2.0.0',
        'pandas': 'pandas==2.3.0',
        'sdv': 'sdv==3.0.0',
        'copulas': 'copulas==0.12.0',
    }

    # Run
    versions = _resolve_version_conflicts(deps, extra_deps)

    # Assert
    assert sorted(versions) == sorted([
        'numpy==2.0.1',
        'pandas==2.3.0',
        'sdv==3.0.0',
        'rdt==1.1.2',
        'copulas==0.12.0',
    ])


def test__resolve_version_conflicts_pointing_to_branch():
    """Test specific branches are always selected over normal version numbers."""
    # Setup
    deps = {
        'numpy': 'git+https://github.com/numpy-dev/numpy.git@master#egg=numpy',
        'pandas': 'pandas==2.2.1',
        'sdv': 'sdv==2.1.1',
        'rdt': 'rdt==1.1.2',
    }
    extra_deps = {
        'numpy': 'numpy==2.0.0',
        'pandas': 'git+https://github.com/pandas-dev/pandas.git@master#egg=pandas',
        'sdv': 'sdv==3.0.0',
        'copulas': 'copulas==0.12.0',
    }

    # Run
    versions = _resolve_version_conflicts(deps, extra_deps)

    # Assert
    assert sorted(versions) == sorted([
        'git+https://github.com/numpy-dev/numpy.git@master#egg=numpy',
        'git+https://github.com/pandas-dev/pandas.git@master#egg=pandas',
        'sdv==3.0.0',
        'rdt==1.1.2',
        'copulas==0.12.0',
    ])
