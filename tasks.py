import inspect
import operator
import os
import shutil
import stat
import sys
from pathlib import Path

import tomli
from invoke import task
from packaging.requirements import Requirement
from packaging.version import Version
COMPARISONS = {'>=': operator.ge, '>': operator.gt, '<': operator.lt, '<=': operator.le}
EGG_STRING = '#egg='

if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec


@task
def check_dependencies(c):
    c.run('python -m pip check')


@task
def unit(c):
    c.run('python -m pytest ./tests/unit --cov=sdgym --cov-report=xml:./unit_cov.xml')


@task
def integration(c):
    c.run('python -m pytest ./tests/integration --cov=sdgym --cov-report=xml:./integration_cov.xml')


@task
def readme(c):
    test_path = Path('tests/readme_test')
    if test_path.exists() and test_path.is_dir():
        shutil.rmtree(test_path)

    cwd = os.getcwd()
    os.makedirs(test_path, exist_ok=True)
    shutil.copy('README.md', test_path / 'README.md')
    os.chdir(test_path)
    c.run('rundoc run --single-session python3 -t python3 README.md')
    os.chdir(cwd)
    shutil.rmtree(test_path)


def _get_minimum_versions(dependencies, python_version):
    min_versions = {}
    for dependency in dependencies:
        if '@' in dependency:
            name, url = dependency.split(' @ ')
            min_versions[name] = f'{url}{EGG_STRING}{name}'
            continue

        req = Requirement(dependency)
        if ';' in dependency:
            marker = req.marker
            if marker and not marker.evaluate({'python_version': python_version}):
                continue  # Skip this dependency if the marker does not apply to the current Python version

        if req.name not in min_versions:
            min_version = next(
                (spec.version for spec in req.specifier if spec.operator in ('>=', '==')), None
            )
            if min_version:
                min_versions[req.name] = f'{req.name}=={min_version}'

        elif '@' not in min_versions[req.name]:
            existing_version = Version(min_versions[req.name].split('==')[1])
            new_version = next(
                (spec.version for spec in req.specifier if spec.operator in ('>=', '==')),
                existing_version,
            )
            if new_version > existing_version:
                min_versions[req.name] = (
                    f'{req.name}=={new_version}'  # Change when a valid newer version is found
                )

    return min_versions


def _get_extra_dependencies(pyproject_data):
    """Get the dependencies for optional synthesizers.

    This function assumes that all external synthesizers we add will have an optional dependency
    section defined and that the section will be listed in the '[test]' optional dependency.

    Args:
        pyproject_data (dict):
            Dictionary representation of our pyproject.toml file.

    Returns:
        list:
            A list of dependency strings (ie. numpy>=x.y.z).
    """
    optional_dependencies = pyproject_data.get('project', {}).get('optional-dependencies', {})
    test_dependencies = optional_dependencies.get('test', [])
    extra_dependencies = []
    start_token = 'sdgym['
    for dep in test_dependencies:
        if dep.startswith(start_token):
            synthesizer = dep[len(start_token): -1]
            extra_dependencies.extend(optional_dependencies.get(synthesizer))

    return extra_dependencies


def _get_version_from_requirement(requirement):
    requirement = requirement.strip()
    equal_index = requirement.find('==')
    version_number = requirement[equal_index + 2:]
    return Version(version_number)


def _resolve_version_conflicts(dependencies, extra_dependencies):
    """Pick the highest version of two minimums.

    Args:
        dependencies (dict):
            A dictionary mapping dependency names to the version.
        extra_dependencies (dict):
            A dictionary mapping the optional dependency names to the version.

    Returns:
        list:
            A list of dependency strings (ie. numpy>=x.y.z).
    """
    all_dependencies = set(dependencies.keys()).union(set(extra_dependencies.keys()))
    selected_versions = []
    for dep in all_dependencies:
        if dep in dependencies and dep in extra_dependencies:
            requirement1 = dependencies.get(dep)
            requirement2 = extra_dependencies.get(dep)
            if EGG_STRING in requirement1:
                selected_versions.append(requirement1)
                continue
            if EGG_STRING in requirement2:
                selected_versions.append(requirement2)
                continue

            version1 = _get_version_from_requirement(requirement1)
            version2 = _get_version_from_requirement(requirement2)
            max_version = requirement1 if version1 > version2 else requirement2
            selected_versions.append(max_version)
        else:
            selected_versions.append(dependencies.get(dep, extra_dependencies.get(dep)))

    return selected_versions


@task
def install_minimum(c):
    with open('pyproject.toml', 'rb') as pyproject_file:
        pyproject_data = tomli.load(pyproject_file)

    dependencies = pyproject_data.get('project', {}).get('dependencies', [])
    extra_synthesizer_dependencies = _get_extra_dependencies(pyproject_data)
    python_version = '.'.join(map(str, sys.version_info[:2]))
    minimum_versions = _get_minimum_versions(dependencies, python_version)
    extra_minimum_versions = _get_minimum_versions(extra_synthesizer_dependencies, python_version)
    minimum_versions = _resolve_version_conflicts(minimum_versions, extra_minimum_versions)
    if minimum_versions:
        install_deps = ' '.join(minimum_versions)
        c.run(f'python -m pip install {install_deps}')


@task
def minimum(c):
    install_minimum(c)
    check_dependencies(c)
    unit(c)
    integration(c)


@task
def lint(c):
    check_dependencies(c)
    c.run('ruff check .')
    c.run('ruff format --check --diff .')


@task
def fix_lint(c):
    check_dependencies(c)
    c.run('ruff check --fix .')
    c.run('ruff format .')


def remove_readonly(func, path, _):
    """Clear the readonly bit and reattempt the removal"""
    os.chmod(path, stat.S_IWRITE)
    func(path)


@task
def rmdir(c, path):
    try:
        shutil.rmtree(path, onerror=remove_readonly)
    except PermissionError:
        pass

@task
def run_sdgym_benchmark(c, modality='single_table'):
    """Run the SDGym benchmark."""
    c.run(f'python sdgym/run_benchmark/run_benchmark.py --modality {modality}')

@task
def upload_benchmark_results(c, modality='single_table'):
    """Upload the benchmark results to S3."""
    c.run(f'python sdgym/run_benchmark/upload_benchmark_results.py --modality {modality}')

@task
def notify_sdgym_benchmark_uploaded(c, folder_name, commit_url=None, modality='single_table'):
    """Notify Slack about the SDGym benchmark upload."""
    from sdgym.run_benchmark.utils import post_benchmark_uploaded_message

    post_benchmark_uploaded_message(folder_name, commit_url, modality)
