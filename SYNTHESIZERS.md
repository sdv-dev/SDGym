# Synthetic Data Generators

**SDGym** evaluates the performance of **Synthetic Data Generators**, also called **Synthesizers**.

A Synthesizer is a Python function (or class method) that takes as input a `dict` with table
names and `pandas.DataFrame` instances, which we call the *real* data, and outputs another
`dict` with the same shape entries and new `pandas.DataFrame` instances, filled with new
*synthetic* data that has the same format and mathematical properties as the *real* data.

The complete list of inputs of the synthesizer is:

* `real_data`: a `dict` containing table names as keys and `pandas.DataFrame` instances as values.
* `metadata`: an instance of an `sdv.Metadata` with information about the dataset.

And the output is a new `dict` with the same tables that the `real_data` contains.

```python
def synthesizer_function(real_data: dict[str, pandas.DataFrame],
                         metadata: sdv.Metadata) -> real_data: dict[str, pandas.DataFrame]:
    ...
    # do all necessary steps to learn from the real data
    # and produce new synthetic data that resembles it
    ...
    return synthetic_data
```

## SDGym Synthesizers

Apart from the benchmark functionality, SDGym implements a collection of Baseline Synthesizers
which are either trivial baseline synthesizers or integrations of synthesizers found in other
libraries.

These Synthesizers are written as Python classes that can be imported from the `sdgym.synthesizers`
module and have a `fit_sample` method with the signature indicated above, which can be directly
passed to the `sdgym.run` function to benchmark them.

This is the list of all the Synthesizers currently implemented, with references to the
corresponding publications when applicable.

| Name                                                        | Description                                                                                                                                     | Reference |
|:------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------|:----------|
| [Identity](sdgym/synthesizers/identity.py)                  | The synthetic data is the same as training data.                                                                                                |           |
| [Independent](sdgym/synthesizers/independent.py)            | Each column sampled independently. Continuous columns use Gaussian Mixture Model and discrete columns use the PMF of training data.             |           |
| [Uniform](sdgym/synthesizers/uniform.py)                    | Each column in the synthetic data is sampled independently and uniformly.                                                                       |           |
| [CLBN](sdgym/synthesizers/clbn.py)                          |                                                                                                                                                 | [2]       |
| [CopulaGAN](sdgym/synthesizers/sdv.py)                      | [sdv.tabular.CopulaGAN](https://sdv.dev/SDV/user_guides/single_table/copulagan.html)                                                            |           |
| [CTGAN](sdgym/synthesizers/sdv.py)                          | [sdv.tabular.CTGAN](https://sdv.dev/SDV/user_guides/single_table/ctgan.html)                                                                    | [1]       |
| [GaussianCopulaCategorical](sdgym/synthesizers/sdv.py)      | [sdv.tabular.GaussianCopula](https://sdv.dev/SDV/user_guides/single_table/gaussian_copula.html) using a CategoricalTransformer                  |           |
| [GaussianCopulaCategoricalFuzzy](sdgym/synthesizers/sdv.py) | [sdv.tabular.GaussianCopula](https://sdv.dev/SDV/user_guides/single_table/gaussian_copula.html) using a CategoricalTransformer with `fuzzy=True`|           |
| [GaussianCopulaOneHot](sdgym/synthesizers/sdv.py)           | [sdv.tabular.GaussianCopula](https://sdv.dev/SDV/user_guides/single_table/gaussian_copula.html) using a OneHotEncodingTransformer               |           |
| [HMA1](sdgym/synthesizers/sdv.py)                           | [sdv.relational.HMA1](https://sdv.dev/SDV/user_guides/relational/hma1.html)                                                                     | [7]       |
| [MedGAN](sdgym/synthesizers/medgan.py)                      |                                                                                                                                                 | [6]       |
| [PAR](sdgym/synthesizers/sdv.py)                            | [sdv.timeseries.PAR](https://sdv.dev/SDV/user_guides/timeseries/par.html)                                                                       |           |
| [PrivBN](sdgym/synthesizers/privbn.py)                      |                                                                                                                                                 | [3]       |
| [TVAE](sdgym/synthesizers/tvae.py)                          |                                                                                                                                                 | [1]       |
| [TableGAN](sdgym/synthesizers/tablegan.py)                  |                                                                                                                                                 | [4]       |
| [SDV](sdgym/synthesizers/sdv.py)                            | [sdv.SDV](https://sdv.dev/SDV/getting_started/quickstart.html)                                                                                  | [7]       |
| [VEEGAN](sdgym/synthesizers/veegan.py)                      |                                                                                                                                                 | [5]       |

## Benchmarking the SDGym Synthesizers

If you want to re-evaluate the performance of any of the SDGym synthesizers, all you need to
do is pass its class directly to the `benchmark` function:

```python3
from sdgym import benchmark
from sdgym.synthesizers import CTGAN

leaderboard = benchmark(synthesizers=CTGAN)
```

If you want to run the complete benchmark suite to re-evaluate all the existing
synthesizers you can simply pass the list of them to the function:

|:warning: **WARNING**: This takes a lot of time to run!|
|:-|

```python3
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

## References

[1] Lei Xu, Maria Skoularidou, Alfredo Cuesta-Infante, Kalyan Veeramachaneni. "Modeling tabular data using conditional gan." (2019) [(pdf)](https://papers.nips.cc/paper/8953-modeling-tabular-data-using-conditional-gan.pdf)

[2] C. Chow, Cong Liu. "Approximating discrete probability distributions with dependence trees." (1968) [(pdf)](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.133.9772&rep=rep1&type=pdf)

[3] Jun  Zhang, Graham Cormode, Cecilia M. Procopiuc, Divesh Srivastava, and Xiaokui Xiao. "Privbayes: Private data release via bayesian networks." (2017) [(pdf)](https://dl.acm.org/doi/pdf/10.1145/3134428)

[4] Noseong Park, Mahmoud Mohammadi, Kshitij Gorde, Sushil Jajodia, Hongkyu Park, Youngmin Kim. "Data synthesis based on generative adversarial networks." (2018) [(pdf)](https://dl.acm.org/ft_gateway.cfm?id=3242929&type=pdf)

[5] Akash Srivastava, Lazar Valkov, Chris Russell, Michael U. Gutmann, Charles Sutton. "VEEGAN: Reducing mode collapse in gans using implicit variational learning." (2017) [(pdf)](https://papers.nips.cc/paper/6923-veegan-reducing-mode-collapse-in-gans-using-implicit-variational-learning.pdf)

[6] Karim Armanious, Chenming Jiang, Marc Fischer, Thomas Küstner, Konstantin Nikolaou, Sergios Gatidis, Bin Yang. "MedGAN: Medical Image Translation using GANs" (2018) [(pdf)](https://arxiv.org/pdf/1806.06397.pdf)

[7] Neha Patki, Roy Wedge, Kalyan Veeramachaneni. "The Synthetic Data Vault" (2018) [(pdf)](https://dai.lids.mit.edu/wp-content/uploads/2018/03/SDV.pdf)
