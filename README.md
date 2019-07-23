<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“SDGym” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>


[![Travis](https://travis-ci.org/DAI-Lab/SDGym.svg?branch=master)](https://travis-ci.org/DAI-Lab/SDGym)
[![PyPi Shield](https://img.shields.io/pypi/v/sdgym.svg)](https://pypi.python.org/pypi/sdgym)
<!--[![Coverage Status](https://codecov.io/gh/DAI-Lab/SDGym/branch/master/graph/badge.svg)](https://codecov.io/gh/DAI-Lab/SDGym)-->
<!--[![Downloads](https://pepy.tech/badge/sdgym)](https://pepy.tech/project/sdgym)-->

# SDGym - Synthetic Data Gym

- License: MIT
- Documentation: https://DAI-Lab.github.io/SDGym/
- Homepage: https://github.com/DAI-Lab/SDGym

# Overview

Synthetic Data Gym (SDGym) is a framework to benchmark the performance of synthetic data generators
for non-temporal tabular data. SDGym is based on the paper [Modeling Tabular data using Conditional
GAN](https://arxiv.org/abs/1907.00503), and the project is part of the [Data to AI
Laboratory](https://dai.lids.mit.edu/) at MIT.

The benchmarking of a synthesizer is a process in which different datasets are generated by your
synthesizer. Then, each couple of real and synthetic data is evaluated with multiple scores.

# Install

## Requirements

**SDGym** has been developed and tested on [Python 3.5, and 3.6](https://www.python.org/downloads/)

Also, although it is not strictly required, the usage of a [virtualenv](https://virtualenv.pypa.io/en/latest/)
is highly recommended in order to avoid interfering with other software installed in the system
where **SDGym** is run.

These are the minimum commands needed to create a virtualenv using python3.6 for **SDGym**:

```bash
pip install virtualenv
virtualenv -p $(which python3.6) sdgym-venv
```

Afterwards, you have to execute this command to have the virtualenv activated:

```bash
source sdgym-venv/bin/activate
```

Remember about executing it every time you start a new console to work on **SDGym**!

## Install with pip

After creating the virtualenv and activating it, we recommend using
[pip](https://pip.pypa.io/en/stable/) in order to install **SDGym**:

```bash
pip install sdgym
```

This will pull and install the latest stable release from [PyPi](https://pypi.org/).

## Install from source

Alternatively, with your virtualenv activated, you can clone the repository and install it from
source by running `make install` on the `stable` branch:

```bash
git clone git@github.com:DAI-Lab/SDGym.git
cd SDGym
git checkout stable
make install
```

## Install for Development

If you want to contribute to the project, a few more steps are required to make the project ready
for development.

First, please head to [the GitHub page of the project](https://github.com/DAI-Lab/SDGym)
and make a fork of the project under you own username by clicking on the **fork** button on the
upper right corner of the page.

Afterwards, clone your fork and create a branch from master with a descriptive name that includes
the number of the issue that you are going to work on:

```bash
git clone git@github.com:{your username}/SDGym.git
cd SDGym
git branch issue-xx-cool-new-feature master
git checkout issue-xx-cool-new-feature
```

Finally, install the project with the following command, which will install some additional
dependencies for code linting and testing.

```bash
make install-develop
```

Make sure to use them regularly while developing by running the commands `make lint` and
`make test`.

## How to benchmark your synthesizer?

In order to use **SDGym** you will need a synthesizer function.
This is a function that takes as input a numpy matrix with real data and outputs another numpy
matrix with the same shape filled with synthesized data.

Also, alongside the real data, some additional variables informing about the column contents
will be passed, which means that the exact signature of the function will be like this:


```python
def my_synthesizer_function(
    real_data: numpy.ndarray,
    categorical_columns: list,
    ordinal_columns: list
) -> syntehtesized_data: numpy.ndarray
```

If your synthesizer implements a different interface, you can wrap it in a function like this:

```python
def my_synthesizer_function(real_data):
    # do all necessary steps here
    return synthesized_data
```

This function should contain inside it all the parameters and arguments needed to use your
synthesizer and call it to generate the new synthesized data based on the real data that is
being passed.

## What data should you synthesizer work with?

As we mentioned in the section before, the main input of **SDGym** is a synthesizer to be
benchmarked, which is expected to be a function that has as unique input and output a table of
data.

The inputs for your synthesizer funciton should be:

* `real_data`: a 2D `numpy.ndarray` with the real data the your synthesizer will attempt to imitate.
* `categorical_columns`: a `list` with the indexes of any columns that should be considered
  categorical independently on their type.
* `ordinal_columns`: a `list` with the indexes of any integer columns that should be treated as
  ordinal values.

And the output should be a single 2D `numpy.ndarray` with the exact same shape as the `real_data`
matrix.

# Quickstart

In this short tutorial we will guide you through a series of steps that will help you getting
started with **SDGym** by exploring its Python API.

## 1. Load the synthesizer

The first step is loading our synthesizer function.

```python
from my_package.my_module import my_synthesizer_function
```

## 2. Run

Now we can run the `benchmark` function to test our model:

```python
from sdgym import benchmark

scores = benchmark(my_synthesizer_function)
```

The output of the `benchmark` function will be a `pd.DataFrame` containing all the scores
computed by the different evaluators:

```
   accuracy  distance        f1                          name  step dataset  iter
0    0.7980       0.0  0.658206  Decision Tree (max_depth=20)     0   adult     0
1    0.8588       0.0  0.673601       Adaboost (estimator=50)     0   adult     0
2    0.7949       0.0  0.661048           Logistic Regression     0   adult     0
3    0.8453       0.0  0.668949                      MLP (50)     0   adult     0
0    0.7982       0.0  0.645343  Decision Tree (max_depth=20)     0   adult     1
1    0.8573       0.0  0.675313       Adaboost (estimator=50)     0   adult     1
2    0.8020       0.0  0.666891           Logistic Regression     0   adult     1
3    0.8540       0.0  0.680245                      MLP (50)     0   adult     1
0    0.7988       0.0  0.653820  Decision Tree (max_depth=20)     0   adult     2
1    0.8604       0.0  0.679817       Adaboost (estimator=50)     0   adult     2
2    0.7936       0.0  0.658956           Logistic Regression     0   adult     2
3    0.8445       0.0  0.675704                      MLP (50)     0   adult     2
```

# What's next?

For more details about **SDGym** and all its possibilities and features, please check the
[documentation site](https://DAI-Lab.github.io/SDGym/).

There you can learn more about
[how to contribute to SDGym](https://HDI-Project.github.io/SDGym/community/contributing.html)
in order to help us developing new features or cool ideas.

# Credits

SDGym is an open source project from the Data to AI Lab at MIT which has been built and maintained
over the years by the following team:

- Lei Xu <leonard.xu.thu@gmail.com>
- Kalyan Veeramachaneni <kalyan@csail.mit.edu>
- Manuel Alvarez <manuel@pythiac.com>
- Carles Sala <csala@csail.mit.edut>

## Citing SDGym

## Related Projects

### SDV

[SDV](https://github.com/HDI-Project/SDV), for Synthetic Data Vault, is the end-user library for
synthesizing data in development under the [HDI Project](https://hdi-dai.lids.mit.edu/).
SDV allows you to easily model and sample relational datasets using Copulas thought a simple API.
Other features include anonymization of Personal Identifiable Information (PII) and preserving
relational integrity on sampled records.

### TGAN

[TGAN](https://github.com/DAI-Lab/TGAN) is a GAN based model for synthesizing tabular data.
It's also developed by the [MIT's Data to AI Lab](https://dai-lab.github.io/) and is under
active development.