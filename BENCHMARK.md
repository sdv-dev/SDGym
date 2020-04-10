# Benchmark

The main component of the **SDGym** project is the `sdgym.benchmark` function.

As shown in the [Readme](README.md) quickstart, in the most simple usage scenario a *synthesizer
function* can be directly passed to it:

```python3
from sdgym import benchmark

scores = benchmark(synthesizers=my_synthesizer_function)
```

This will evaluate the *synthesizer function* on all the datasets, produce one or more scores
for each one of them, and present them in a table alongside the scores previously obtained
by all the SDGym Synthesizers:

```
                        adult/accuracy  adult/f1  ...  ring/test_likelihood
IndependentSynthesizer         0.56530  0.134593  ...             -1.958888
UniformSynthesizer             0.39695  0.273753  ...             -2.519416
IdentitySynthesizer            0.82440  0.659250  ...             -1.705487
...                                ...       ...  ...                   ...
my_synthesizer_function        0.64865  0.210103  ...             -1.964966
```

However, the `benchmark` function supports a number of additional arguments that allows us
to configure its behavior to better adapt it to our needs.

## Arguments for the benchmark function

The `benchmark` function accepts the following arguments:

* `synthesizers (function, class, list, tuple or dict)`:
    The synthesizer or synthesizers to evaluate. It can be a single synthesizer
    (function or method or class), or an iterable of synthesizers, or a dict
    containing synthesizer names as keys and synthesizers as values. If the input
    is not a dict, synthesizer names will be extracted from the given object.
* `datasets (list[str])`:
    Names of the datasets to use for the benchmark. Defaults to all the ones available.
* `iterations (int)`:
    Number of iterations to perform over each dataset and synthesizer. Defaults to 3.
* `add_leaderboard (bool)`:
    Whether to append the obtained scores to the previous leaderboard or not. Defaults
    to `True`.
* `leaderboard_path (str)`:
    Path to where the leaderboard is stored. Defaults to the leaderboard included
    with the package, which contains the scores obtained by the SDGym synthesizers.
* `replace_existing (bool)`:
    Whether to replace old scores or keep them in the returned leaderboard. Defaults
    to `True`.

### Synthesizers

#### Synthesizer Classes

The most basic scenario is to pass a synthesizer function, but the benchmark function
can also be used to evaluate any `Synthesizer` class, as far as it is a subclass of
`sdgym.synthesizers.BaseSynthesizer`.

For example, if we want to evaluate the `IndependentSynthesizer` we can do so by passing the
class directly to the benchmark function:

```python3
from sdgym.synthesizers import IndependentSynthesizer

scores = benchmark(IndependentSynthesizer)
```

#### Evaluating multiple Synthesizers

The benchmark function can be used to evaluate more than one Synthesizer at a time.

In order to do this, all you need to do is pass a list of functions instead of a single
object.

For example, if we want to evaluate our synthesizer function and also the `IndependentSynthesizer`
we can pass both of them inside a list:

```python3
synthesizers = [
    my_synthesizer_function,
    IndependentSynthesizer
]
scores = benchmark(synthesizers=synthesizers)
```

Or, if we wanted to evaluate all the SDGym Synthesizers at once (note that this takes a lot of time
to run!), we could just pass all the subclasses of `BaseSynthesizer`:

```python3
from sdgym.synthesizers import BaseSynthesizer

scores = benchmark(BaseSynthesizers.__subclasses__())
```

#### Customizing the Synthesizer names.

Sometimes we might want to customize the name that we give to the function or class that we are
passing to the benchmark, so they show up nicer in the output leaderboard.

In order to do this, all we have to do is pass a dict instead of a single object or a list,
putting the names as keys and the functions or classes as the values:

```python3
synthesizers = {
    'My Synthesizer': my_synthesizer_function,
    'SDGym Independent': IndependentSynthesizer
}
scores = benchmark(synthesizers=synthesizers)
```

## Datasets

By default, the benchmark function will run on [all the SDGym Datasets](DATASETS.md).

However, this takes a lot of time to run, and sometimes we will be interested in only a few of
them.

If we want to restrict the datasets used, we can simply pass a list with their names as strings:

For example, if we want to evaluate our synthesizer function on only the `adult` and `intrusion`
datasets we can do:

```python3
datasets = ['adult', 'intrusion']
scores = benchmark(my_synthesizer_function, datasets=datasets)
```

## Iterations

By default, and in order to reduce the noise in the results, the SDGym benchmark evaluates each
synthesizer on each datasets exactly 3 times and then averages the obtained scores.

The number of evaluates performed on each synthesizer and dataset combination can be altered
by passing a different value to the `iterations` argument:

```python3
scores = benchmark(my_synthesizer_function, iterations=10)
```

## Leaderboard

When the benchmark is run, SDGym returns the obtained results alongside the internal leaderboard
of scores obtained by the [SDGym Synthesizers](SYNTHESIZERS.md).

This behavior can be optionally disabled by passing `add_leaderboard=False`.

```python3
scores = benchmark(my_synthesizer_function, add_leaderboard=False)
```

Also, alternatively, if you want to maintain your own version of the leaderboard, you can pass
the path to your leaderboard CSV file:

```python3
scores = benchmark(my_synthesizer_function, leaderboard_path='path/to/my/leaderboard.csv')
```

Finally, if the name of the synthesizer that is being evaluated already exists in the leaderboard
table, it is dropped, so each synthesizer appears only once. This behavior can be disabled by
passing `replace_existing=False`.
