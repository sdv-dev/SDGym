from sdgym.benchmark import benchmark
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


synthesizer = IdentitySynthesizer()


def test_adult():
    benchmark(synthesizer.fit_sample, repeat=1, datasets=['adult'])


def test_alarm():
    benchmark(synthesizer.fit_sample, repeat=1, datasets=['alarm'])


def test_asia():
    benchmark(synthesizer.fit_sample, repeat=1, datasets=['asia'])


def test_census():
    benchmark(synthesizer.fit_sample, repeat=1, datasets=['census'])


def test_child():
    benchmark(synthesizer.fit_sample, repeat=1, datasets=['child'])


def test_covtype():
    benchmark(synthesizer.fit_sample, repeat=1, datasets=['covtype'])


def test_credit():
    benchmark(synthesizer.fit_sample, repeat=1, datasets=['credit'])


def test_grid():
    benchmark(synthesizer.fit_sample, repeat=1, datasets=['grid'])


def test_gridr():
    benchmark(synthesizer.fit_sample, repeat=1, datasets=['gridr'])


def test_insurance():
    benchmark(synthesizer.fit_sample, repeat=1, datasets=['insurance'])


def test_intrusion():
    benchmark(synthesizer.fit_sample, repeat=1, datasets=['intrusion'])


def test_mnist12():
    benchmark(synthesizer.fit_sample, repeat=1, datasets=['mnist12'])


def test_mnist28():
    benchmark(synthesizer.fit_sample, repeat=1, datasets=['mnist28'])


def test_news():
    benchmark(synthesizer.fit_sample, repeat=1, datasets=['news'])


def test_ring():
    benchmark(synthesizer.fit_sample, repeat=1, datasets=['ring'])
