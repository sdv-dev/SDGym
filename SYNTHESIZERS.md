# Synthetic Data Generators

**SDGym** evaluates the performance of **Synthesizers**.

A Synthesizer is a Python function (or class method) that takes as input a numpy matrix with some
data, which we call the *real* data, and outputs another numpy matrix with the same shape, filled
with new *synthetic* data that has similar mathematical properties as the *real* one.

The complete list of inputs of the synthesizer is:

* `real_data`: a 2D `numpy.ndarray` with the real data the synthesizer will attempt to imitate.
* `categorical_columns`: a `list` with the indexes of any columns that should be considered
  categorical independently on their type.
* `ordinal_columns`: a `list` with the indexes of any integer columns that should be treated as
  ordinal values.

|**Note**: Columns that are not listed in either lists above are assumed to be numerical.|
|:-|

And the output is a single 2D `numpy.ndarray` with the exact same shape as the `real_data`
matrix.

```python
def synthesizer_function(real_data: numpy.ndarray, categorical_columns: list[int],
                         ordinal_columns: list[int]) -> numpy.ndarray:
    ...
    # do all necessary steps to learn from the real data
    # and produce new synthetic data that resembles it
    ...
    return synthetic_data
```

## SDGym Synthesizers

Apart from the benchmark functionality, SDGym implements a collection of Synthesizers which are
either custom demo synthesizers or re-implementations of synthesizers that have been presented
in other publications.

These Synthesizers are written as Python classes that can be imported from the `sdgym.synthesizers`
module and have the following methods:

* `fit`: Fits the synthesizer on the data. Expects the following arguments:
    * `data (numpy.ndarray)`: 2 dimensional Numpy matrix with the real data to learn from.
    * `categorical_columns (list or tuple)`: List of indexes of the columns that are categorical
      within the dataset.
    * `ordinal_columns (list or tuple)`: List of indexes of the columns that are ordinal within
      the dataset.
* `sample`: Generates new data resembling the original dataset. Expects the following arguments:
    * `n_samples (int)`: Number of samples to generate.
* `fit_sample`: Fits the synthesizer on the dataset and then samples as many rows as there were in
  the original dataset. It expects the same arguments as the `fit` method, and is ready to be
  directly passed to the `benchmark` function in order to evaluate the synthesizer performance.

This is the list of all the Synthesizers currently implemented, with references to the
corresponding publications when applicable.

|Name|Description|Reference|
|:--|:--|:--|
|[IdentitySynthesizer](sdgym/synthesizers/identity.py)|The synthetic data is the same as training data.||
|[UniformSynthesizer](sdgym/synthesizers/uniform.py)|Each column in the synthetic data is sampled independently and uniformly.||
|[IndependentSynthesizer](sdgym/synthesizers/independent.py)|Each column in the synthetic data is sampled independently. Continuous columns are modeled by Gaussian mixture model. Discrete columns are sampled from the PMF of training data.||
|[CLBNSynthesizer](sdgym/synthesizers/clbn.py)||[2]|
|[PrivBNSynthesizer](sdgym/synthesizers/privbn.py)||[3]|
|[TableganSynthesizer](sdgym/synthesizers/tablegan.py)||[4]|
|[VEEGANSynthesizer](sdgym/synthesizers/veegan.py)||[5]|
|[TVAESynthesizer](sdgym/synthesizers/tvae.py)||[1]|
|[CTGANSynthesizer](sdgym/synthesizers/ctgan.py)||[1]|

## Leaderboard

This is the leaderboard with the scores that the SDGym Synthesizer obtained on the SDGym Benchmark,
which is also available for download as a CSV file here: [leaderboard.csv](sdgym/leaderboard.csv).

### Gaussian Mixture Simulated Data

|                   |grid/syn_likelihood|grid/test_likelihood|gridr/syn_likelihood|gridr/test_likelihood|ring/syn_likelihood|ring/test_likelihood|
|-------------------|-------------------|--------------------|--------------------|---------------------|-------------------|--------------------|
|IdentitySynthesizer|              -3.06|               -3.06|               -3.06|                -3.07|              -1.7 |               -1.7 |
|CLBNSynthesizer    |              -3.68|               -8.62|               -3.76|               -11.6 |              -1.75|               -1.7 |
|PrivBNSynthesizer  |              -4.33|              -21.67|               -3.98|               -13.88|              -1.82|               -1.71|
|MedganSynthesizer  |             -10.04|              -62.93|               -9.45|               -72   |              -2.32|              -45.16|
|VEEGANSynthesizer  |              -9.81|               -4.79|              -12.51|                -4.94|              -7.85|               -2.92|
|TableganSynthesizer|              -8.7 |               -4.99|               -9.64|                -4.7 |              -6.38|               -2.66|
|TVAESynthesizer    |              -2.86|              -11.26|               -3.41|                -3.2 |              -1.68|               -1.79|
|CTGANSynthesizer   |              -5.63|               -3.69|               -8.11|                -4.31|              -3.43|               -2.19|

### Bayesian Networks Simulated Data

|                   |asia/syn_likelihood|asia/test_likelihood|alarm/syn_likelihood|alarm/test_likelihood|child/syn_likelihood|child/test_likelihood|insurance/syn_likelihood|insurance/test_likelihood|
|-------------------|-------------------|--------------------|--------------------|---------------------|--------------------|---------------------|------------------------|-------------------------|
|IdentitySynthesizer|              -2.23|               -2.24|               -10.3|                -10.3|               -12  |                -12  |                   -12.8|                    -12.9|
|CLBNSynthesizer    |              -2.44|               -2.27|               -12.4|                -11.2|               -12.6|                -12.3|                   -15.2|                    -13.9|
|PrivBNSynthesizer  |              -2.28|               -2.24|               -11.9|                -10.9|               -12.3|                -12.2|                   -14.7|                    -13.6|
|MedganSynthesizer  |              -2.81|               -2.59|               -10.9|                -14.2|               -14.2|                -15.4|                   -16.4|                    -16.4|
|VEEGANSynthesizer  |              -8.11|               -4.63|               -17.7|                -14.9|               -17.6|                -17.8|                   -18.2|                    -18.1|
|TableganSynthesizer|              -3.64|               -2.77|               -12.7|                -11.5|               -15  |                -13.3|                   -16  |                    -14.3|
|TVAESynthesizer    |              -2.31|               -2.27|               -11.2|                -10.7|               -12.3|                -12.3|                   -14.7|                    -14.2|
|CTGANSynthesizer   |              -2.56|               -2.31|               -14.2|                -12.6|               -13.4|                -12.7|                   -16.5|                    -14.8|

### Real World Datasets

|                   |adult/f1|census/f1|credit/f1|covtype/macro_f1|intrusion/macro_f1|mnist12/accuracy|mnist28/accuracy| news/r2|
|-------------------|--------|---------|---------|----------------|------------------|----------------|----------------|--------|
|IdentitySynthesizer|   0.669|    0.494|    0.72 |           0.652|             0.862|           0.886|           0.916| 0.14   |
|CLBNSynthesizer    |   0.334|    0.31 |    0.409|           0.319|             0.384|           0.741|           0.176|-6.28   |
|PrivBNSynthesizer  |   0.414|    0.121|    0.185|           0.27 |             0.384|           0.117|           0.081|-4.49   |
|MedganSynthesizer  |   0.375|    0    |    0    |           0.093|             0.299|           0.091|           0.104|-8.8    |
|VEEGANSynthesizer  |   0.235|    0.094|    0    |           0.082|             0.261|           0.194|           0.136|-6.5e+06|
|TableganSynthesizer|   0.492|    0.358|    0.182|           0    |             0    |           0.1  |           0    |-3.09   |
|TVAESynthesizer    |   0.626|    0.377|    0.098|           0.433|             0.511|           0.793|           0.794|-0.2    |
|CTGANSynthesizer   |   0.601|    0.391|    0.672|           0.324|             0.528|           0.394|           0.371|-0.43   |


## Compile C++ dependencies

Some of the third party synthesizers that SDGym offers, like the `PrivBNSynthesizer`, require
dependencies written in C++ need to be compiled.

In order to be able to use them, please do:

1. Clone or download the SDGym repository to your local machine:

```bash
git clone git@github.com:sdv-dev/SDGym.git
cd SDGym
```

2. make sure to have installed all the necessary dependencies to compile C++. In Linux
distributions based on Ubuntu, this can be done with the following command:

```bash
sudo apt-get install build-essential
```

3. Trigger the C++ compilation:

```bash
make compile
```

## Using the SDGym Synthesizers

In order to use the synthesizer classes included in **SDGym**, you need to create an instance
of them, pass the real data to their `fit` method, and then generate new data using its
`sample` method.

Here's an example about how to use the `IdentitySynthesizer` to model and sample the `adult`
dataset.

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

## Benchmarking the SDGym Synthesizers

If you want to re-evaluate the performance of any of the SDGym synthesizers, all you need to
do is pass its class directly to the `benchmark` function:

```python3
from sdgym import benchmark
from sdgym.synthesizers import CTGANSynthesizer

leaderboard = benchmark(synthesizers=CTGANSynthesizer)
```

Alternatively, if you wanted to change any of the default hyperparameters, you can generate an
instance with the desired values and pass it to the function.

```python3
synthesizer = CTGANSynthesizer(epochs=10)
leaderboard = benchmark(synthesizers=synthesizer)
```

Finally, if you want to run the complete benchmark suite to re-evaluate all the existing
synthesizers you can simply pass the list of them to the function:

> :warning: **WARNING**: This takes a lot of time to run!

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
leaderboard = benchmark(synthesizers=all_synthesizers)
```

## How to add your own Synthesizer to SDGym?

Coming soon!

## References

[1] Lei Xu, Maria Skoularidou, Alfredo Cuesta-Infante, Kalyan Veeramachaneni. "Modeling tabular data using conditional gan." (2019) [(pdf)](https://papers.nips.cc/paper/8953-modeling-tabular-data-using-conditional-gan.pdf)

[2] C. Chow, Cong Liu. "Approximating discrete probability distributions with dependence trees." (1968) [(pdf)](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.133.9772&rep=rep1&type=pdf)

[3] Jun  Zhang, Graham Cormode, Cecilia M. Procopiuc, Divesh Srivastava, and Xiaokui Xiao. "Privbayes: Private data release via bayesian networks." (2017) [(pdf)](https://dl.acm.org/doi/pdf/10.1145/3134428)

[4] Noseong Park, Mahmoud Mohammadi, Kshitij Gorde, Sushil Jajodia, Hongkyu Park, Youngmin Kim. "Data synthesis based on generative adversarial networks." (2018) [(pdf)](https://dl.acm.org/ft_gateway.cfm?id=3242929&type=pdf)

[5] Akash Srivastava, Lazar Valkov, Chris Russell, Michael U. Gutmann, Charles Sutton. "Veegan: Reducing mode collapse in gans using implicit variational learning." (2017) [(pdf)](https://papers.nips.cc/paper/6923-veegan-reducing-mode-collapse-in-gans-using-implicit-variational-learning.pdf)
