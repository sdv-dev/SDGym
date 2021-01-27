# Benchmark

The main component of the **SDGym** project is the `sdgym.run` function.

As shown in the [Readme](README.md) quickstart, in the most simple usage scenario a *synthesizer
function* can be directly passed to it indicating the name of the datasets in which you want
to evaluate your synthesizer function:

```python3
In [1]: import numpy as np
   ...: from sdv.tabular import GaussianCopula
   ...:
   ...:
   ...: def my_synthesizer_function(real_data, metadata):
   ...:     gc = GaussianCopula(default_distribution='gaussian')
   ...:     table_name = metadata.get_tables()[0]
   ...:     gc.fit(real_data[table_name])
   ...:     return {table_name: gc.sample()}

In [2]: import sdgym

In [3]: scores = sdgym.run(synthesizers=my_synthesizer_function)
```

This will evaluate the *synthesizer function* on all the available datasets and, produce one or
more scores for each one of them, and present them in a table.

```python3
In [4]: scores
Out[4]:
               synthesizer dataset      modality  iteration           metric error      score  metric_time  model_time
0  my_synthesizer_function    asia  single-table          0  BNLogLikelihood  None  -2.834019     2.769234    0.738452
0  my_synthesizer_function   alarm  single-table          0  BNLogLikelihood  None -20.264935     7.157158    3.183285
```

Let's see what other arguments we can use to control the behavior of the `sdgym.run` function.

## Arguments for the sdgym.run function

The `sdgym.run` function accepts the following arguments:

* `synthesizers (function, class, list, tuple or dict)`:
    The synthesizer or synthesizers to evaluate. It can be a single synthesizer
    (function or method or class), or an iterable of synthesizers, or a dict
    containing synthesizer names as keys and synthesizers as values. If the input
    is not a dict, synthesizer names will be extracted from the given object.
* `datasets (list[str])`:
    Names of the datasets to use for the benchmark. Defaults to all the ones available.
* `datasets_path (str)`::
    Path to where the datasets can be found. If not given, use the default path.
* `modalities (list[str])`::
    Filter datasets by the given modalities. If not given, filter datasets by the
    synthesizer modalities.
* `metrics (list[str])`::
    List of metrics to apply.
* `bucket (str)`::
    Name of the bucket from which the datasets must be downloaded if not found locally.
* `iterations (int)`::
    Number of iterations to perform over each dataset and synthesizer. Defaults to 3.
* `workers (int or str)`::
    If ``workers`` is given as an integer value other than 0 or 1, a multiprocessing
    Pool is used to distribute the computation across the indicated number of workers.
    If the string ``dask`` is given, the computation is distributed using ``dask``.
    In this case, setting up the ``dask`` cluster and client is expected to be handled
    outside of this function.
* `cache_dir (str)`::
    If a ``cache_dir`` is given, intermediate results are stored in the indicated directory
    as CSV files as they get computted. This allows inspecting results while the benchmark
    is still running and also recovering results in case the process does not finish
    properly. Defaults to ``None``.
* `show_progress (bool)`::
    Whether to use tqdm to keep track of the progress. Defaults to ``True``.
* `timeout (int)`::
    Maximum number of seconds to wait for each dataset to
    finish the evaluation process. If not passed, wait until
    all the datasets are done.
* `output_path (str)`::
    If an ``output_path`` is given, the generated leaderboard will be stored in the
    indicated path as a CSV file. The given path must be a complete path including
    the ``.csv`` filename.

### Synthesizers

#### Synthesizer Classes

The most basic scenario is to pass a synthesizer function, but the sdgym.run function
can also be used to evaluate any `Synthesizer` class, as far as it is a subclass of
`sdgym.synthesizers.BaseSynthesizer`.

For example, if we want to evaluate the `Independent` we can do so by passing the class
directly to the sdgym.run function:

```python3
In [5]: from sdgym.synthesizers import Independent

In [6]: scores = sdgym.run(synthesizers=Independent)
```

#### Evaluating multiple Synthesizers

The `sdgym.run` function can be used to evaluate more than one Synthesizer at a time.

In order to do this, all you need to do is pass a list of functions instead of a single
object.

For example, if we want to evaluate our synthesizer function and also the `Independent`
we can pass both of them inside a list:

```python3
In [7]: synthesizers = [my_synthesizer_function, Independent]

In [8]: scores = sdgym.run(synthesizers=synthesizers)
```

Or, if we wanted to evaluate all the SDGym Synthesizers at once (note that this takes a lot of time
to run!), we could just pass all the subclasses of `Baseline`:

```python3
In [9]: from sdgym.synthesizers import Baseline

In [10]: scores = sdgym.run(Baseline.get_subclasses())
```

#### Customizing the Synthesizer names.

Sometimes we might want to customize the name that we give to the function or class that we are
passing to the benchmark, so they show up nicer in the output leaderboard.

In order to do this, all we have to do is pass a dict instead of a single object or a list,
putting the names as keys and the functions or classes as the values:

```python3
In [11]: synthesizers = {
    ...:     'My Synthesizer': my_synthesizer_function,
    ...:     'SDGym Independent': Independent
    ...: }

In [12]: scores = sdgym.run(synthesizers=synthesizers)
```

### Datasets

By default, the sdgym.run function will run on [all the SDGym Datasets](DATASETS.md).

However, this takes a lot of time to run, and sometimes we will be interested in only a few of
them.

If we want to restrict the datasets used, we can simply pass a list with their names as strings:

For example, if we want to evaluate our synthesizer function on only the `adult` and `intrusion`
datasets we can do:

```python3
In [13]: datasets = ['adult', 'intrusion']

In [14]: scores = sdgym.run(my_synthesizer_function, datasets=datasets)
```

### Iterations

By default, the SDGym benchmark evaluates each synthesizer on each datasets exactly once.
However, in some cases you may want to run each synthesizer and dataset combination multiple times
and then average the obtained scores.

The number of evaluations performed on each synthesizer and dataset combination can be altered
by passing a different value to the `iterations` argument:

```python3
In [16]: scores = sdgym.run(my_synthesizer_function, iterations=10)
```
