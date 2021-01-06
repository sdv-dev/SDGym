# History

## v0.2.2 - 2020-10-17

This version adds a rework of the benchmark function and a few new synthesizers.

### New Features

* New CLI with `run`, `make-leaderboard` and `make-summary` commands
* Parallel execution via Dask or Multiprocessing
* Download datasets without executing the benchmark
* Support for python from 3.6 to 3.8

### New Synthesizers

* `sdv.tabular.CTGAN`
* `sdv.tabular.CopulaGAN`
* `sdv.tabular.GaussianCopulaOneHot`
* `sdv.tabular.GaussianCopulaCategorical`
* `sdv.tabular.GaussianCopulaCategoricalFuzzy`

## v0.2.1 - 2020-05-12

New updated leaderboard and minor improvements.

### New Features

* Add parameters for PrivBNSynthesizer - [Issue #37](https://github.com/sdv-dev/SDGym/issues/37) by @csala

## v0.2.0 - 2020-04-10

New Becnhmark API and lots of improved documentation.

### New Features

* The benchmark function now returns a complete leaderboard instead of only one score
* Class Synthesizers can be directly passed to the benchmark function

### Bug Fixes

* One hot encoding errors in the Independent, VEEGAN and Medgan Synthesizers.
* Proper usage of the `eval` mode during sampling.
* Fix improperly configured datasets.

## v0.1.0 - 2019-08-07

First release to PyPi
