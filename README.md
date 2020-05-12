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
for tabular data. SDGym is a project of the [Data to AI Laboratory](https://dai.lids.mit.edu/) at MIT.

## What is a Synthetic Data Generator?

A **Synthetic Data Generator** is a Python function (or class method) that takes as input some
data, which we call the *real* data, learns a model from it, and outputs new *synthetic* data that
has similar mathematical properties as the *real* one.

Please refer to the [synthesizers documentation](SYNTHESIZERS.md) for instructions about how to
implement your own Synthetic Data Generator and integrate with SDGym. You can also read about how
to use the ones included in **SDGym** and see the current [leaderboard](SYNTHESIZERS.md#leaderboard).

## Benchmark datasets

**SDGym** evaluates the performance of **Synthetic Data Generators** using datasets
that are in three families:

* Simulated data generated using Gaussian Mixtures
* Simulated data generated using Bayesian Networks
* Real world datasets

Further details about how these datasets were generated can be found in the [Modeling Tabular
data using Conditional GAN](https://arxiv.org/abs/1907.00503) paper and in the [datasets
documentation](DATASETS.md).

## Current Leaderboard

This is a summary of the current SDGym leaderboard, showing the number of datasets in which
each Synthesizer obtained the best score.

The complete scores table can be found in the [synthesizers document](SYNTHESIZERS.md#leaderboard)
and it can also be downloaded as a CSV file form here: [sdgym/leaderboard.csv](sdgym/leaderboard.csv)

Detailed leaderboard results for all the releases are available [in this Google Docs Spreadsheet](
https://docs.google.com/spreadsheets/d/1iNJDVG_tIobcsGUG5Gn4iLa565vVhz2U/edit).

### Gaussian Mixture Simulated Data

| Synthesizer         |   0.2.1 |   0.2.0 |
|---------------------|---------|---------|
| CLBNSynthesizer     |       0 |       1 |
| CTGANSynthesizer    |       0 |       1 |
| MedganSynthesizer   |       0 |       0 |
| PrivBNSynthesizer   |       0 |       0 |
| TVAESynthesizer     |       5 |       4 |
| TableganSynthesizer |       1 |       0 |
| VEEGANSynthesizer   |       0 |       0 |

### Bayesian Networks Simulated Data

| Synthesizer         |   0.2.1 |   0.2.0 |
|---------------------|---------|---------|
| CLBNSynthesizer     |       0 |       0 |
| CTGANSynthesizer    |       0 |       0 |
| MedganSynthesizer   |       4 |       1 |
| PrivBNSynthesizer   |       3 |       6 |
| TVAESynthesizer     |       1 |       3 |
| TableganSynthesizer |       0 |       0 |
| VEEGANSynthesizer   |       0 |       0 |

### Real World Datasets

| Synthesizer         |   0.2.1 |   0.2.0 |
|---------------------|---------|---------|
| CLBNSynthesizer     |       0 |       0 |
| CTGANSynthesizer    |       3 |       3 |
| MedganSynthesizer   |       0 |       0 |
| PrivBNSynthesizer   |       0 |       0 |
| TVAESynthesizer     |       5 |       5 |
| TableganSynthesizer |       0 |       0 |
| VEEGANSynthesizer   |       0 |       0 |


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

## Benchmarking your own synthesizer

All you need to do in order to use the SDGym Benchmark, is to import and call the `sdgym.benchmark`
function passing it your synthesizer function:

```python3
from sdgym import benchmark

scores = benchmark(synthesizers=my_synthesizer_function)
```

* You can learn how to create your own synthesizer function [here](SYNTHESIZERS.md).
* You can learn about different arguments for benchmark function [here](BENCHMARK.md).

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

## Benchmarking the SDGym Synthesizers

If you want to run the SDGym benchmark on the SDGym Synthesizers you can directly pass the
corresponding class, or a list of classes, to the `benchmark` function.

For example, if you want to run the complete benchmark suite to evaluate all the existing
synthesizers you can run (this will take a lot of time to run!):

```python3
from sdgym.synthesizers import (
    CLBNSynthesizer, CTGANSynthesizer, IdentitySynthesizer, IndependentSynthesizer,
    MedganSynthesizer, PrivBNSynthesizer, TableganSynthesizer, TVAESynthesizer,
    UniformSynthesizer, VEEGANSynthesizer)

all_synthesizers = [
    CLBNSynthesizer,
    IdentitySynthesizer,
    IndependentSynthesizer,
    MedganSynthesizer,
    PrivBNSynthesizer,
    TableganSynthesizer,
    CTGANSynthesizer,
    TVAESynthesizer,
    UniformSynthesizer,
    VEEGANSynthesizer,
]
scores = benchmark(synthesizers=all_synthesizers)
```

For further details about all the arguments and possibilities that the `benchmark` function offers
please refer to the [benchmark documentation](BENCHMARK.md)

# Additional References

* Datasets used in SDGym are detailed [here](DATASETS.md).
* How to write a synthesizer is detailed [here](SYNTHESIZERS.md).
* How to use benchmark function is detailed [here](BENCHMARK.md).
* Detailed leaderboard results for all the releases are available [here](
https://docs.google.com/spreadsheets/d/1iNJDVG_tIobcsGUG5Gn4iLa565vVhz2U/edit).

# Related Projects

## SDV

[SDV](https://github.com/HDI-Project/SDV), for Synthetic Data Vault, is the end-user library for
synthesizing data in development under the [HDI Project](https://hdi-dai.lids.mit.edu/).
SDV allows you to easily model and sample relational datasets using Copulas through a simple API.
Other features include anonymization of Personal Identifiable Information (PII) and preserving
relational integrity on sampled records.

## CTGAN

[CTGAN](https://github.com/sdv-dev/CTGAN) is the GAN based model for synthesizing tabular data
presented in the [Modeling Tabular data using Conditional GAN](https://arxiv.org/abs/1907.00503)
paper. It's also developed by the [MIT's Data to AI Lab](https://dai-lab.github.io/) and is under
active development.

## TGAN

[TGAN](https://github.com/sdv-dev/TGAN) is another GAN based model for synthesizing tabular data.
It's also developed by the [MIT's Data to AI Lab](https://dai-lab.github.io/) and is under
active development.
