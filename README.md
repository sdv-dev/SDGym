<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“SDGym” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

[![Development Status](https://img.shields.io/badge/Development%20Status-2%20--%20Pre--Alpha-yellow)](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
[![Travis](https://travis-ci.org/sdv-dev/SDGym.svg?branch=master)](https://travis-ci.org/sdv-dev/SDGym)
[![PyPi Shield](https://img.shields.io/pypi/v/sdgym.svg)](https://pypi.python.org/pypi/sdgym)
[![Downloads](https://pepy.tech/badge/sdgym)](https://pepy.tech/project/sdgym)
<!--[![Coverage Status](https://codecov.io/gh/sdv-dev/SDGym/branch/master/graph/badge.svg)](https://codecov.io/gh/sdv-dev/SDGym)-->

# SDGym - Synthetic Data Gym

* License: [MIT](https://github.com/sdv-dev/SDGym/blob/master/LICENSE)
* Development Status: [Pre-Alpha](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
* Homepage: https://github.com/sdv-dev/SDGym
<!--* Documentation: https://sdv-dev.github.io/SDGym/-->

# Overview

Synthetic Data Gym (SDGym) is a framework to benchmark the performance of synthetic data generators
for non-temporal tabular data. SDGym is based on the paper [Modeling Tabular data using Conditional
GAN](https://arxiv.org/abs/1907.00503), and the project is part of the [Data to AI
Laboratory](https://dai.lids.mit.edu/) at MIT.

## What is a Synthetic Data Generator?

A **Synthetic Data Generator** is a Python function (or class method) that takes as input some
data, which call the real data, learns from it, and outputs new synthetic data that has similar
mathematical properties as the real one.

Please refer to the [SYNTHESIZERS.md](SYNTHESIZERS.md) documentation for instructions about how
to implement your own Synthetic Data Generator, as well for references about how to use the
ones included in **SDGym** and the current [LEADERBOARD](SYNTHESIZERS.md#leaderboard)

## Benchmark datasets

**SDGym** evaluates the performance of **Synthetic Data Generators** using datasets
that are in three families:

* Simulated data generated using Gaussian Mixtures
* Simulated data generated using Bayesian Networks
* Real world datasets

Further details about how these datasets were generated can be found in the [Modeling Tabular
data using Conditional GAN](https://arxiv.org/abs/1907.00503) paper and in the [DATASETS.md](
DATASETS.md) document.

# Install

## Requirements

**SDGym** has been developed and tested on [Python 3.5, and 3.6](https://www.python.org/downloads/)

Also, although it is not strictly required, the usage of a [virtualenv](https://virtualenv.pypa.io/en/latest/)
is highly recommended in order to avoid interfering with other software installed in the system
where **SDGym** is run.

## Install with pip

The easiest and recommended way to install **SDGym** is using [pip](https://pip.pypa.io/en/stable/):

```bash
pip install sdgym
```

This will pull and install the latest stable release from [PyPi](https://pypi.org/).

If you want to install it from source or contribute to the project please read the
[Contributing Guide](https://sdv-dev.github.io/SDGym/contributing.html#get-started) for
more details about how to do it.

# Usage

## Benchmark

All you need to do in order to use the SDGym Benchmark, is to import and call the `sdgym.benchmark`
function passing it your synthesizer function:

```python3
from sdgym import benchmark

scores = benchmark(my_synthesizer_function)
```

The output of the `benchmark` function will be a `pd.DataFrame` containing the results obtained
by your synthesizer on each dataset, as well as the results obtained previously by the SDGym
synthesizers:

```
                        adult/accuracy  adult/f1  ...  ring/test_likelihood
IndependentSynthesizer         0.56530  0.134593  ...             -1.958888
UniformSynthesizer             0.39695  0.273753  ...             -2.519416
IdentitySynthesizer            0.82440  0.659250  ...             -1.705487
...                                ...       ...  ...                   ...
my_synthesizer_function        0.64865  0.210103  ...             -1.964966
```

Further details about all the arguments and possibilities that the `benchmark` function offers
can be found in the [BENCHMARK.md](BENCHMARK.md) document.

## Using the SDGym Synthesizers

Apart from the benchmark functionality, **SDGym** implements a collection of **Synthesizers** which
are either custom demo synthesizers or re-implementations of synthesizers that have been presented
in third party publications. Further details about these **Synthesizers** and their performance
can be found in the [SYNTHESIZERS.md](SYNTHESIZERS.md) document.

Here's a short example about how to use one of them, the `IndependentSynthesizer`, to model and
sample the `adult` dataset.

```python3
from sdgym import load_dataset
from sdgym.synthesizers import IndependentSynthesizer

data, categorical_columns, ordinal_columns = load_dataset('adult')

synthesizer = IndependentSynthesizer()
synthesizer.fit(data, categorical_columns, ordinal_columns)

sampled = synthesizer.sample(3)
```

This will return a numpy matrix of sampeld data with the same columns as the original data and
as many rows as we have requested:

```
array([[5.1774925e+01, 0.0000000e+00, 5.3538445e+04, 6.0000000e+00,
        8.9999313e+00, 2.0000000e+00, 1.0000000e+00, 3.0000000e+00,
        2.0000000e+00, 1.0000000e+00, 3.7152294e-04, 1.9912617e-04,
        1.0767025e+01, 0.0000000e+00, 0.0000000e+00],
       [6.4843109e+01, 0.0000000e+00, 2.6462553e+05, 1.2000000e+01,
        8.9993210e+00, 1.0000000e+00, 0.0000000e+00, 1.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 5.3685449e-06, 1.9797031e-03,
        2.2253288e+01, 0.0000000e+00, 0.0000000e+00],
       [6.5659584e+01, 5.0000000e+00, 3.6158912e+05, 8.0000000e+00,
        9.0010223e+00, 0.0000000e+00, 1.2000000e+01, 3.0000000e+00,
        0.0000000e+00, 0.0000000e+00, 1.0562389e-03, 0.0000000e+00,
        3.9998917e+01, 0.0000000e+00, 0.0000000e+00]], dtype=float32)
```

# Related Projects

## SDV

[SDV](https://github.com/HDI-Project/SDV), for Synthetic Data Vault, is the end-user library for
synthesizing data in development under the [HDI Project](https://hdi-dai.lids.mit.edu/).
SDV allows you to easily model and sample relational datasets using Copulas thought a simple API.
Other features include anonymization of Personal Identifiable Information (PII) and preserving
relational integrity on sampled records.

## TGAN

[TGAN](https://github.com/sdv-dev/TGAN) is a GAN based model for synthesizing tabular data.
It's also developed by the [MIT's Data to AI Lab](https://dai-lab.github.io/) and is under
active development.
