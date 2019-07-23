#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

install_requires = [
    'certifi==2019.3.9',
    'cycler==0.10.0',
    'decorator==4.4.0',
    'joblib==0.13.2',
    'kiwisolver==1.1.0',
    'matplotlib==3.1.0',
    'networkx==2.3',
    'numpy==1.16.3',
    'pandas==0.24.2',
    'Pillow==6.0.0',
    'pomegranate==0.11.0',
    'pyparsing==2.4.0',
    'python-dateutil==2.8.0',
    'pytz==2019.1',
    'PyYAML==5.1',
    'scikit-learn==0.21.1',
    'scipy==1.3.0',
    'seaborn==0.9.0',
    'six==1.12.0',
    'sklearn==0.0',
    'torch==1.1.0',
    'torchvision==0.2.2.post3',
    'tqdm==4.32.1',
]

tests_require = [
    'pytest>=3.4.2',
    'pytest-cov>=2.6.0',
]


setup_requires = [
    'pytest-runner>=2.11.1',
]

development_requires = [
    # general
    'bumpversion>=0.5.3',
    'pip>=9.0.1',
    'watchdog>=0.8.3',

    # docs
    'm2r>=0.2.0',
    'Sphinx>=1.7.1',
    'sphinx_rtd_theme>=0.2.4',
    'autodocsumm>=0.1.10',

    # style check
    'flake8>=3.7.7',
    'isort>=4.3.4',

    # fix style issues
    'autoflake>=1.1',
    'autopep8>=1.4.3',

    # distribute on PyPI
    'twine>=1.10.0',
    'wheel>=0.30.0',

    # Advanced testing
    'coverage>=4.5.1',
    'tox>=2.9.1',
]

setup(
    author="MIT Data To AI Lab",
    author_email='dailabmit@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    description=(
        "A framework to benchmark the performance of synthetic data generators "
        "for non-temporal tabular data"),
    extras_require={
        'dev': development_requires + tests_require,
        'tests': tests_require,
    },
    include_package_data=True,
    install_requires=install_requires,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    keywords='machine learning synthetic data benchmark generative models',
    name='sdgym',
    packages=find_packages(include=['sdgym', 'sdgym.*'], exclude=['sdgym.synthesizers.*']),
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=tests_require,
    url='https://github.com/DAI-Lab/SDGym',
    version='0.1.0-dev',
    zip_safe=False,
)
