import os
import unittest

from sdgym.benchmark import run
from sdgym.synthesizers import IdentitySynthesizer

DEFAULT_DATASETS = [
    "adult",
    "alarm",
    "asia",
    "census",
    "child",
    "covtype",
    "credit",
    "grid",
    "gridr",
    "insurance",
    "intrusion",
    "mnist12",
    "mnist28",
    "news",
    "ring"
]


MANUAL_TESTS = os.environ.get('MANUAL_TESTS', 'false') == 'true'
MESSAGE = 'set environ variable MANUAL_TESTS="true" to execute'


def test_adult():
    run(IdentitySynthesizer, iterations=1, datasets=['adult'])


@unittest.skipUnless(MANUAL_TESTS, MESSAGE)
def test_alarm():
    run(IdentitySynthesizer, iterations=1, datasets=['alarm'])


@unittest.skipUnless(MANUAL_TESTS, MESSAGE)
def test_asia():
    run(IdentitySynthesizer, iterations=1, datasets=['asia'])


@unittest.skipUnless(MANUAL_TESTS, MESSAGE)
def test_census():
    run(IdentitySynthesizer, iterations=1, datasets=['census'])


@unittest.skipUnless(MANUAL_TESTS, MESSAGE)
def test_child():
    run(IdentitySynthesizer, iterations=1, datasets=['child'])


@unittest.skipUnless(MANUAL_TESTS, MESSAGE)
def test_covtype():
    run(IdentitySynthesizer, iterations=1, datasets=['covtype'])


@unittest.skipUnless(MANUAL_TESTS, MESSAGE)
def test_credit():
    run(IdentitySynthesizer, iterations=1, datasets=['credit'])


@unittest.skipUnless(MANUAL_TESTS, MESSAGE)
def test_grid():
    run(IdentitySynthesizer, iterations=1, datasets=['grid'])


@unittest.skipUnless(MANUAL_TESTS, MESSAGE)
def test_gridr():
    run(IdentitySynthesizer, iterations=1, datasets=['gridr'])


@unittest.skipUnless(MANUAL_TESTS, MESSAGE)
def test_insurance():
    run(IdentitySynthesizer, iterations=1, datasets=['insurance'])


@unittest.skipUnless(MANUAL_TESTS, MESSAGE)
def test_intrusion():
    run(IdentitySynthesizer, iterations=1, datasets=['intrusion'])


@unittest.skipUnless(MANUAL_TESTS, MESSAGE)
def test_mnist12():
    run(IdentitySynthesizer, iterations=1, datasets=['mnist12'])


@unittest.skipUnless(MANUAL_TESTS, MESSAGE)
def test_mnist28():
    run(IdentitySynthesizer, iterations=1, datasets=['mnist28'])


@unittest.skipUnless(MANUAL_TESTS, MESSAGE)
def test_news():
    run(IdentitySynthesizer, iterations=1, datasets=['news'])


@unittest.skipUnless(MANUAL_TESTS, MESSAGE)
def test_ring():
    run(IdentitySynthesizer, iterations=1, datasets=['ring'])
