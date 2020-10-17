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
|[IndependentSynthesizer](sdgym/synthesizers/independent.py)|Each column in the synthetic data is sampled independently. Continuous columns are modeled by Gaussian mixture model. Discrete columns are sampled from the PMF of training data.||
|[UniformSynthesizer](sdgym/synthesizers/uniform.py)|Each column in the synthetic data is sampled independently and uniformly.||
|[CLBNSynthesizer](sdgym/synthesizers/clbn.py)||[2]|
|[CopulaGAN](sdgym/synthesizers/sdv.py)|[sdv.tabular.CopulaGAN](https://sdv.dev/SDV/user_guides/single_table/copulagan.html)||
|[CTGAN](sdgym/synthesizers/sdv.py)|[sdv.tabular.CTGAN](https://sdv.dev/SDV/user_guides/single_table/ctgan.html)||
|[CTGANSynthesizer](sdgym/synthesizers/ctgan.py)||[1]|
|[GaussianCopulaCategorical](sdgym/synthesizers/sdv.py)|[sdv.tabular.GaussianCopula](https://sdv.dev/SDV/user_guides/single_table/gaussian_copula.html) using a CategoricalTransformer||
|[GaussianCopulaCategoricalFuzzy](sdgym/synthesizers/sdv.py)|[sdv.tabular.GaussianCopula](https://sdv.dev/SDV/user_guides/single_table/gaussian_copula.html) using a CategoricalTransformer with `fuzzy=True`||
|[GaussianCopulaOneHot](sdgym/synthesizers/sdv.py)|[sdv.tabular.GaussianCopula](https://sdv.dev/SDV/user_guides/single_table/gaussian_copula.html) using a OneHotEncodingTransformer||
|[PrivBNSynthesizer](sdgym/synthesizers/privbn.py)||[3]|
|[TVAESynthesizer](sdgym/synthesizers/tvae.py)||[1]|
|[TableganSynthesizer](sdgym/synthesizers/tablegan.py)||[4]|
|[VEEGANSynthesizer](sdgym/synthesizers/veegan.py)||[5]|

## Leaderboard

This is the leaderboard with the scores that the SDGym Synthesizer obtained on the SDGym Benchmark,
which is also available for download as a CSV file here: [leaderboard.csv](sdgym/leaderboard.csv).

### Gaussian Mixture Simulated Data

| Synthesizer                    |   grid/syn_likelihood |   grid/test_likelihood |   gridr/syn_likelihood |   gridr/test_likelihood | ring/syn_likelihood   | ring/test_likelihood   |
|--------------------------------|-----------------------|------------------------|------------------------|-------------------------|-----------------------|------------------------|
| CLBNSynthesizer                |              -3.88316 |               -9.20214 |               -4.00839 |                -7.4328  | -1.76507342195328     | -47.1579361217972      |
| CTGAN                          |              -8.76064 |               -5.06297 |               -8.30975 |                -5.04831 | -6.59132388962251     | -2.66528147128561      |
| CTGANSynthesizer               |              -8.91839 |               -5.08856 |               -8.32358 |                -5.02729 | -7.1305860438021      | -2.70407618202863      |
| CopulaGAN                      |              -8.19017 |               -5.14136 |               -8.16277 |                -5.00547 | -6.20616566233844     | -2.80004284451614      |
| GaussianCopulaCategorical      |              -7.2354  |               -4.51105 |               -7.16437 |                -4.5434  | -3.19691185753935     | -2.15092960090006      |
| GaussianCopulaCategoricalFuzzy |              -7.34413 |               -4.56575 |               -7.14453 |                -4.55408 | -3.17900097117842     | -2.15414841923683      |
| GaussianCopulaOneHot           |              -7.26805 |               -4.51438 |               -7.19225 |                -4.55071 | -3.20762438182436     | -2.15449339858511      |
| MedganSynthesizer              |              -5.82946 |              -90.342   |               -7.37063 |              -141.407   | -2.77956647170292     | -149.766976540331      |
| PrivBNSynthesizer              |              -3.9944  |               -8.30844 |               -4.07166 |                -7.12135 | N/E                   | N/E                    |
| TVAESynthesizer                |              -3.26779 |               -5.6578  |               -3.86723 |                -3.70828 | -1.57984123031383     | -1.93999679554369      |
| TableganSynthesizer            |              -6.99216 |               -5.33074 |               -6.99889 |                -4.82922 | -4.74019037169834     | -2.53367543172312      |
| VEEGANSynthesizer              |              -8.64686 |             -423.573   |              -11.4585  |                -8.90848 | -16.8306340686327     | -6.35495995600412      |

### Bayesian Networks Simulated Data

| Synthesizer                    |   asia/syn_likelihood |   asia/test_likelihood |   alarm/syn_likelihood |   alarm/test_likelihood |   child/syn_likelihood |   child/test_likelihood |   insurance/syn_likelihood |   insurance/test_likelihood |
|--------------------------------|-----------------------|------------------------|------------------------|-------------------------|------------------------|-------------------------|----------------------------|-----------------------------|
| CLBNSynthesizer                |              -2.40255 |               -2.2738  |              -12.4588  |                -11.1878 |               -12.6259 |                -12.3067 |                   -15.1668 |                    -13.9176 |
| CTGAN                          |              -4.18796 |               -2.45544 |              -15.882   |                -13.0976 |               -14.3535 |                -12.8404 |                   -17.026  |                    -14.9725 |
| CTGANSynthesizer               |              -2.69344 |               -2.31181 |              -15.2204  |                -12.928  |               -13.8082 |                -12.8125 |                   -16.5983 |                    -14.8406 |
| CopulaGAN                      |              -3.95607 |               -2.40401 |              -15.6881  |                -13.0529 |               -14.2472 |                -12.9165 |                   -16.9585 |                    -14.9577 |
| GaussianCopulaCategorical      |              -2.24812 |               -3.61216 |              -12.9064  |                -15.5749 |               -16.4019 |                -15.5461 |                   -17.8379 |                    -16.5784 |
| GaussianCopulaCategoricalFuzzy |              -2.82965 |               -3.10891 |              -14.5215  |                -14.5651 |               -16.8951 |                -15.4053 |                   -18.0283 |                    -16.5174 |
| GaussianCopulaOneHot           |              -2.31389 |               -3.22664 |              -15.4761  |                -15.6663 |               -14.4828 |                -15.3061 |                   -17.8437 |                    -17.9075 |
| MedganSynthesizer              |              -1.57422 |               -5.96556 |               -7.83896 |                -13.2552 |               -11.1125 |                -12.987  |                   -13.8838 |                    -15.0777 |
| PrivBNSynthesizer              |              -2.29359 |               -2.2447  |              -12.1537  |                -11.1396 |               -12.3601 |                -12.1877 |                   -14.7018 |                    -13.6371 |
| TVAESynthesizer                |              -2.29348 |               -2.26632 |              -11.4418  |                -10.7605 |               -12.4594 |                -12.296  |                   -14.2951 |                    -14.2367 |
| TableganSynthesizer            |              -3.40013 |               -2.72081 |              -12.6888  |                -11.5424 |               -15.0168 |                -13.3929 |                   -16.1781 |                    -14.3188 |
| VEEGANSynthesizer              |             -11.4923  |               -5.95271 |              -18.3861  |                -18.2109 |               -17.3143 |                -17.6815 |                   -18.3258 |                    -18.1134 |

### Real World Datasets

| Synthesizer                    | adult/f1          |   census/f1 |   credit/f1 |   covtype/macro_f1 | intrusion/macro_f1   | mnist12/accuracy   | mnist28/accuracy   | news/r2            |
|--------------------------------|-------------------|-------------|-------------|--------------------|----------------------|--------------------|--------------------|--------------------|
| CLBNSynthesizer                | 0.305082698225735 |   0.285714  |   0.440643  |          0.329636  | 0.385243072199282    | 0.7201             | 0.16315            | -6.47183249120821  |
| CTGAN                          | 0.608178843850075 |   0.328815  |   0.659771  |          0.31662   | 0.540598170677063    | 0.122316666666667  | 0.13775            | -0.069598047933615 |
| CTGANSynthesizer               | 0.602407290015437 |   0.378843  |   0.523338  |          0.329751  | 0.510841888605536    | 0.14675            | 0.112466666666667  | -0.512953590553904 |
| CopulaGAN                      | 0.606637996037602 |   0.387663  |   0.717449  |          0.326941  | 0.516588482874166    | 0.159016666666667  | 0.149233333333333  | -0.055570339707436 |
| GaussianCopulaCategorical      | N/E               |   0         |   0         |          0.14002   | 0.176792311617512    | 0.124766666666667  | N/E                | -5.02909628931276  |
| GaussianCopulaCategoricalFuzzy | 0.257747943010586 |   0.0535276 |   0.0103442 |          0.138726  | 0.256446581500444    | 0.1684             | 0.176016666666667  | -8.65514335573062  |
| GaussianCopulaOneHot           | 0.198041838748949 |   0.13283   |   0         |          0.182262  | 0.331105948082712    | 0.485383333333333  | 0.493066666666667  | -36.9430938013929  |
| MedganSynthesizer              | 0.243345396168715 |   0.136471  |   0.024403  |          0.0911309 | 0.274575506845598    | 0.376666666666667  | 0.10455            | -5.6014619622007   |
| PrivBNSynthesizer              | 0.428731508524732 |   0.244719  |   0.010973  |          0.214713  | 0.382637188907442    | N/E                | N/E                | N/E                |
| TVAESynthesizer                | 0.618866173537051 |   0.382321  |   0         |          0.456447  | 0.432752384609416    | 0.779116666666667  | 0.769233333333333  | -0.244782783556202 |
| TableganSynthesizer            | 0.352537209716839 |   0.27212   |   0.270299  |          0         | N/E                  | 0.0872             | 0.082916666666667  | -5.82076678008137  |
| VEEGANSynthesizer              | 0.162141533929063 |   0.0518164 |   0.16603   |          0.0959155 | 0.180796928285852    | 0.374683333333333  | 0.155633333333333  | -319605180.253642  |

## Compile C++ dependencies

Some of the third party synthesizers that SDGym offers, like the `PrivBNSynthesizer`, require
dependencies written in C++ that need to be compiled before they can be used.

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
