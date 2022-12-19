<div align="center">
<br/>
<p align="center">
    <i>This repository is part of <a href="https://sdv.dev">The Synthetic Data Vault Project</a>, a project from <a href="https://datacebo.com">DataCebo</a>.</i>
</p>

[![Development Status](https://img.shields.io/badge/Development%20Status-2%20--%20Pre--Alpha-yellow)](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
[![Travis](https://travis-ci.org/sdv-dev/SDGym.svg?branch=master)](https://travis-ci.org/sdv-dev/SDGym)
[![PyPi Shield](https://img.shields.io/pypi/v/sdgym.svg)](https://pypi.python.org/pypi/sdgym)
[![Downloads](https://pepy.tech/badge/sdgym)](https://pepy.tech/project/sdgym)

<div align="left">
<br/>
<p align="center">
<a href="https://github.com/sdv-dev/SDGym">
<img align="center" width=40% src="https://github.com/sdv-dev/SDV/blob/master/docs/images/SDGym-DataCebo.png"></img>
</a>
</p>
</div>

</div>

# Overview

Synthetic Data Gym (SDGym) is a framework to benchmark the performance of synthetic data
generators based on [SDV](https://github.com/sdv-dev/SDV) and [SDMetrics](
https://github.com/sdv-dev/SDMetrics).

| Important Links                               |                                                                      |
| --------------------------------------------- | -------------------------------------------------------------------- |
| :computer: **[Website]**                      | Check out the SDV Website for more information about the project.    |
| :orange_book: **[SDV Blog]**                  | Regular publshing of useful content about Synthetic Data Generation. |
| :book: **[Documentation]**                    | Quickstarts, User and Development Guides, and API Reference.         |
| :octocat: **[Repository]**                    | The link to the Github Repository of this library.                   |
| :scroll: **[License]**                        | The entire ecosystem is published under the MIT License.             |
| :keyboard: **[Development Status]**           | This software is in its Pre-Alpha stage.                             |
| [![][Slack Logo] **Community**][Community]    | Join our Slack Workspace for announcements and discussions.          |
| [![][MyBinder Logo] **Tutorials**][Tutorials] | Run the SDV Tutorials in a Binder environment.                       |

[Website]: https://sdv.dev
[SDV Blog]: https://sdv.dev/blog
[Documentation]: https://sdv.dev/SDV
[Repository]: https://github.com/sdv-dev/SDGym
[License]: https://github.com/sdv-dev/SDGym/blob/master/LICENSE
[Development Status]: https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha
[Slack Logo]: https://github.com/sdv-dev/SDV/blob/master/docs/images/slack.png
[Community]: https://bit.ly/sdv-slack-invite
[MyBinder Logo]: https://github.com/sdv-dev/SDV/blob/master/docs/images/mybinder.png
[Tutorials]: https://mybinder.org/v2/gh/sdv-dev/SDV/master?filepath=tutorials

## What is a Synthetic Data Generator?

A **Synthetic Data Generator** is a Python function (or method) that takes as input some
data, which we call the *real* data, learns a model from it, and outputs new *synthetic* data that
has the same structure and similar mathematical properties as the *real* one.

Please refer to the [synthesizers documentation](SYNTHESIZERS.md) for instructions about how to
implement your own Synthetic Data Generator and integrate with SDGym. You can also read about how
to use the ones already included in **SDGym** and see how to run them.

## Benchmark datasets

**SDGym** evaluates the performance of **Synthetic Data Generators** using *single table*,
*multi table* and *timeseries* datasets stored as CSV files alongside an [SDV Metadata](
https://sdv.dev/SDV/user_guides/relational/relational_metadata.html) JSON file.

Further details about the list of available datasets and how to add your own datasets to
the collection can be found in the [datasets documentation](DATASETS.md).

# Install

**SDGym** can be installed using the following commands:

**Using `pip`:**

```bash
pip install sdgym
```

**Using `conda`:**

```bash
conda install -c pytorch -c conda-forge sdgym
```

For more installation options please visit the [SDGym installation Guide](INSTALL.md)

# Usage

## Benchmarking your own Synthesizer

SDGym evaluates **Synthetic Data Generators**, which are Python functions (or classes) that take
as input some data, which we call the *real* data, learn a model from it, and output new
*synthetic* data that has the same structure and similar mathematical properties as the *real* one.

As an example, let use define a synthesizer function that applies the [GaussianCopula model from SDV
](https://sdv.dev/SDV/user_guides/single_table/gaussian_copula.html) with `gaussian` distribution.

```python3
import numpy as np
from sdv.tabular import GaussianCopula


def create_gaussian_copula(real_data, metadata):
    gc = GaussianCopula(default_distribution='gaussian')
    table_name = metadata.get_tables()[0]
    gc.fit(real_data[table_name])
    num_rows = len(real_data[table_name])
    return (table_name, num_rows, gc)

def sample_gaussian_copula(synthesizer, num_samples):
    table_name, num_rows, gc = synthesizer
    return {table_name: gc.sample(num_rows)}
```

|:information_source: You can learn how to create your own synthesizer function [here](SYNTHESIZERS.md).|
|:-|

We can now try to evaluate this function on the `asia` and `alarm` datasets:

```python3
import sdgym

scores = sdgym.benchmark_single_table(
    synthesizers=(create_gaussian_copula, sample_gaussian_copula), sdv_datasets=['asia', 'alarm'])
```

|:information_source: You can learn about different arguments for `sdgym.run` function [here](BENCHMARK.md).|
|:-|

The output of the `sdgym.run` function will be a `pd.DataFrame` containing the results obtained
by your synthesizer on each dataset.

| synthesizer     | dataset | modality     | metric          |      score | metric_time | model_time |
|-----------------|---------|--------------|-----------------|------------|-------------|------------|
| gaussian_copula | asia    | single-table | BNLogLikelihood |  -2.842690 |    2.762427 |   0.752364 |
| gaussian_copula | alarm   | single-table | BNLogLikelihood | -20.223178 |    7.009401 |   3.173832 |

## Benchmarking the SDGym Synthesizers

If you want to run the SDGym benchmark on the SDGym Synthesizers you can directly pass the
corresponding class, or a list of classes, to the `sdgym.run` function.

For example, if you want to run the complete benchmark suite to evaluate all the existing
synthesizers you can run (:warning: this will take a lot of time to run!):

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

---


<div align="center">
<a href="https://datacebo.com"><img align="center" width=40% src="https://github.com/sdv-dev/SDV/blob/master/docs/images/DataCebo.png"></img></a>
</div>
<br/>
<br/>

[The Synthetic Data Vault Project](https://sdv.dev) was first created at MIT's [Data to AI Lab](
https://dai.lids.mit.edu/) in 2016. After 4 years of research and traction with enterprise, we
created [DataCebo](https://datacebo.com) in 2020 with the goal of growing the project.
Today, DataCebo is the proud developer of SDV, the largest ecosystem for
synthetic data generation & evaluation. It is home to multiple libraries that support synthetic
data, including:

* 🔄 Data discovery & transformation. Reverse the transforms to reproduce realistic data.
* 🧠 Multiple machine learning models -- ranging from Copulas to Deep Learning -- to create tabular,
  multi table and time series data.
* 📊 Measuring quality and privacy of synthetic data, and comparing different synthetic data
  generation models.

[Get started using the SDV package](https://sdv.dev/SDV/getting_started/install.html) -- a fully
integrated solution and your one-stop shop for synthetic data. Or, use the standalone libraries
for specific needs.
