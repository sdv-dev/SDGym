#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open('README.md', encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md', encoding='utf-8') as history_file:
    history = history_file.read()

install_requires = [
    'appdirs>1.1.4,<2',
    'boto3>=1.15.0,<2',
    'compress-pickle>=1.2.0,<2',
    'humanfriendly>=8.2,<9',
    'numpy>=1.15.4,<2',
    'pandas<1.1.5,>=1.1',
    'pomegranate>=0.13.0,<0.13.5',
    'psutil>=5.7,<6',
    'scikit-learn>=0.20,<0.24',
    'tabulate>=0.8.3,<0.9',
    'torch>=1.1.0,<2',
    'tqdm>=4,<5',
    'XlsxWriter>=1.2.8,<1.3',
    'rdt>=0.4.1',
    'sdmetrics>=0.3.0',
    'sdv>=0.9.0',
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

    # docs
    'm2r>=0.2.0,<0.3',
    'Sphinx>=1.7.1,<3',
    'sphinx_rtd_theme>=0.2.4,<0.5',
    'autodocsumm>=0.1.10,<0.2',

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
]

setup(
    author='MIT Data To AI Lab',
    author_email='dailabmit@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    description=(
        'A framework to benchmark the performance of synthetic data generators '
        'for non-temporal tabular data'
    ),
    entry_points={
        'console_scripts': [
            'sdgym=sdgym.__main__:main'
        ],
    },
    extras_require={
        'dev': development_requires + tests_require,
        'test': tests_require,
    },
    include_package_data=True,
    install_requires=install_requires,
    license='MIT license',
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    keywords='machine learning synthetic data generation benchmark generative models',
    name='sdgym',
    packages=find_packages(include=['sdgym', 'sdgym.*']),
    python_requires='>=3.6,<3.9',
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=tests_require,
    url='https://github.com/sdv-dev/SDGym',
    version='0.3.1.dev2',
    zip_safe=False,
)
