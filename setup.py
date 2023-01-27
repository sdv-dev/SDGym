#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open('README.md', encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md', encoding='utf-8') as history_file:
    history = history_file.read()

install_requires = [
    'appdirs>=1.3,<2',
    'boto3>=1.15.0,<2',
    'botocore>=1.18,<2',
    'compress-pickle>=1.2.0,<3',
    'humanfriendly>=8.2,<11',
    "numpy>=1.20.0,<2;python_version<'3.10'",
    "numpy>=1.23.3,<2;python_version>='3.10'",
    "pandas>=1.1.3,<2;python_version<'3.10'",
    "pandas>=1.5.0,<2;python_version>='3.10'",
    "pomegranate>=0.14.3,<0.15",
    'psutil>=5.7,<6',
    "scikit-learn>=0.24,<2;python_version<'3.10'",
    "scikit-learn>=1.1.3,<2;python_version>='3.10'",
    "scipy>=1.5.4,<2;python_version<'3.10'",
    "scipy>=1.9.2,<2;python_version>='3.10'",
    'tabulate>=0.8.3,<0.9',
    "torch>=1.8.0,<2;python_version<'3.10'",
    "torch>=1.11.0,<2;python_version>='3.10'",
    'tqdm>=4.15,<5',
    'XlsxWriter>=1.2.8,<4',
    'rdt>=1.3.0,<2.0',
    'sdmetrics>=0.9.0,<1.0',
    'sdv>=0.18.0',
]


dask_requires = [
    'dask',
    'distributed',
]

setup_requires = [
    'pytest-runner>=2.11.1',
]

tests_require = [
    'pytest>=3.4.2',
    'pytest-cov>=2.6.0',
    'jupyter>=1.0.0,<2',
    'rundoc>=0.4.3,<0.5',
]

development_requires = [
    # general
    'bumpversion>=0.5.3,<0.6',
    'pip>=9.0.1',
    'watchdog>=0.8.3,<0.11',

    # style check
    'flake8>=3.7.7,<4',
    'isort>=4.3.4,<5',

    # fix style issues
    'autoflake>=1.1,<2',
    'autopep8>=1.4.3,<2',

    # distribute on PyPI
    'twine>=1.10.0,<4',
    'wheel>=0.30.0',

    # Advanced testing
    'coverage>=4.5.1,<6',
    'tox>=2.9.1,<4',
    'importlib-metadata>=3.6',

    # Invoke
    'invoke',
]

setup(
    author='DataCebo, Inc.',
    author_email='info@sdv.dev',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: Free for non-commercial use',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    description=(
        'Benchmark tabular synthetic data generators using a variety of datasets'
    ),
    entry_points={
        'console_scripts': [
            'sdgym=sdgym.__main__:main'
        ],
    },
    extras_require={
        'all': development_requires + tests_require + dask_requires,
        'dev': development_requires + tests_require + dask_requires,
        'test': tests_require,
        'dask': dask_requires,
    },
    include_package_data=True,
    install_requires=install_requires,
    license='BSL-1.1',
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    keywords='machine learning synthetic data generation benchmark generative models',
    name='sdgym',
    packages=find_packages(include=['sdgym', 'sdgym.*']),
    python_requires='>=3.7,<3.11',
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=tests_require,
    url='https://github.com/sdv-dev/SDGym',
    version='0.6.0.dev1',
    zip_safe=False,
)
