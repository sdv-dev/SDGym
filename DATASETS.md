# SDGym Datasets

The datasets used for the SDGym benchmarking process are grouped in three families:

* Simulated data generated using Gaussian Mixtures
    * grid
    * gridr
    * ring
* Simulated data generated using Bayesian Networks
    * asia
    * alarm
    * child
    * insurance
* Real world datasets
    * adult
    * census
    * covtype
    * credit
    * intrusion
    * mnist12
    * mnist28
    * news

Further details about how these datasets were generated can be found in the [Modeling Tabular
data using Conditional GAN](https://arxiv.org/abs/1907.00503) paper, and the code used to
generate the simulated ones can be found inside the [sdgym/utils](sdgym/utils) folder of
this repository.

## Using the datasets

All the datasets can also be found for download inside the [sgdym S3 bucket](
http://sdgym.s3.amazonaws.com/index.html) in the form of an `.npz` numpy matrix archive and
a `.json` metadata file that contains information about the dataset structure and their columns.

In order to load these datasets in the same format as they will be passed to your synthesizer
function you can use the `sdgym.load_dataset` function passing the name of the dataset to load.

In this example, we will load the `adult` dataset:

```python3
from sdgym import load_dataset

data, categorical_columns, ordinal_columns = load_dataset('adult')
```

This will return a numpy matrix with the data that will be passed to your synthesizer function,
as well as the list of indexes for the categorical and ordinal columns.

## How to add your own dataset to SDGym?

Coming soon!
