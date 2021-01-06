<p align="left">
  <a href="https://dai.lids.mit.edu">
    <img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt="DAI-Lab" />
  </a>
  <i>An Open Source Project from the <a href="https://dai.lids.mit.edu">Data to AI Lab, at MIT</a></i>
</p>

[![Development Status](https://img.shields.io/badge/Development%20Status-2%20--%20Pre--Alpha-yellow)](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
[![Travis](https://travis-ci.org/sdv-dev/SDGym.svg?branch=master)](https://travis-ci.org/sdv-dev/SDGym)
[![PyPi Shield](https://img.shields.io/pypi/v/sdgym.svg)](https://pypi.python.org/pypi/sdgym)
[![Downloads](https://pepy.tech/badge/sdgym)](https://pepy.tech/project/sdgym)

<img align="center" width=30% src="docs/resources/header.png">

Benchmarking framework for Synthetic Data Generators

* Website: https://sdv.dev
* Documentation: https://sdv.dev/SDV
* Repository: https://github.com/sdv-dev/SDGym
* License: [MIT](https://github.com/sdv-dev/SDGym/blob/master/LICENSE)
* Development Status: [Pre-Alpha](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)

# Overview

Synthetic Data Gym (SDGym) is a framework to benchmark the performance of synthetic data
generators based on [SDV](https://github.com/sdv-dev/SDV) and [SDMetrics](
https://github.com/sdv-dev/SDMetrics).

SDGym is a part of the [The Synthetic Data Vault](https://sdv.dev/) project.

## What is a Synthetic Data Generator?

A **Synthetic Data Generator** is a Python function (or class method) that takes as input some
data, which we call the *real* data, learns a model from it, and outputs new *synthetic* data that
has the same structure and similar mathematical properties as the *real* one.

Please refer to the [synthesizers documentation](SYNTHESIZERS.md) for instructions about how to
implement your own Synthetic Data Generator and integrate with SDGym. You can also read about how
to use the ones included in **SDGym** and see the current [leaderboard](SYNTHESIZERS.md#leaderboard).

## Benchmark datasets

**SDGym** evaluates the performance of **Synthetic Data Generators** using *single table*,
*multi table* and *timeseries* datasets stored as CSV files alongside an [SDV Metadata](
https://sdv.dev/SDV/user_guides/relational/relational_metadata.html) JSON file.

Further details about the list of available datasets and how to add your own datasets to
the collection can be found in the [datasets documentation](DATASETS.md).


# Install

**SDGym** can also be installed using the following commands:

**Using `pip`:**

```bash
pip install sdgym
```

**Using `conda`:**

```bash
conda install -c sdv-dev -c conda-forge sdgym
```

For more installation options please visit the [SDGym installation Guide](INSTALL.md)

# Usage

## Benchmarking your own synthesizer

To benchmark your own synthesizer function import `sdgym` and call it passing your synthesizer
function and the settings that you want to use for the evaluation.

For example, if we want to evaluate a simple synthesizer function in the `census` dataset
we can execute:

```python3
import numpy as np
import sdgym

def my_synthesizer_function(real_data, metadata):
    """dummy synthesizer that just returns a permutation of the real data."""
    return {name: table.sample(len(table)) for name, table in real_data.items()}

scores = sdgym.run(synthesizers=my_synthesizer_function, datasets=['census'])
```

* You can learn how to create your own synthesizer function [here](SYNTHESIZERS.md).
* You can learn about different arguments for `sdgym.run` function [here](BENCHMARK.md).

The output of the `sdgym.run` function will be a `pd.DataFrame` containing the results obtained
by your synthesizer on each dataset.

## Benchmarking the SDGym Synthesizers

If you want to run the SDGym benchmark on the SDGym Synthesizers you can directly pass the
corresponding class, or a list of classes, to the `sdgym.run` function.

For example, if you want to run the complete benchmark suite to evaluate all the existing
synthesizers you can run (this will take a lot of time to run!):

```python
from sdgym.synthesizers import (
    CLBN, CopulaGAN, CTGAN, HMA1, Identity, Independent,
    MedGAN, PAR, PrivBN, SDV, TableGAN, TVAE,
    Uniform, VEEGAN)

all_synthesizers = [
    CLBN,
    CTGAN,
    CopulaGAN,
    HMA1,
    Identity,
    Independent,
    MedGAN,
    PAR,
    PrivBN,
    SDV,
    TVAE,
    TableGAN,
    Uniform,
    VEEGAN,
]
scores = sdgym.run(synthesizers=all_synthesizers)
```

For further details about all the arguments and possibilities that the `benchmark` function offers
please refer to the [benchmark documentation](BENCHMARK.md)

# Additional References

* Datasets used in SDGym are detailed [here](DATASETS.md).
* How to write a synthesizer is detailed [here](SYNTHESIZERS.md).
* How to use benchmark function is detailed [here](BENCHMARK.md).
* Detailed leaderboard results for all the releases are available [here](
https://docs.google.com/spreadsheets/d/1iNJDVG_tIobcsGUG5Gn4iLa565vVhz2U/edit).

# The Synthetic Data Vault

<p>
  <a href="https://sdv.dev">
    <img width=30% src="https://github.com/sdv-dev/SDV/blob/master/docs/images/SDV-Logo-Color-Tagline.png?raw=true">
  </a>
  <p><i>This repository is part of <a href="https://sdv.dev">The Synthetic Data Vault Project</a></i></p>
</p>

* Website: https://sdv.dev
* Documentation: https://sdv.dev/SDV
