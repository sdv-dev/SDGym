# History

## v0.4.0 - 2021-06-17
This release fixed a bug where passing a `json` file as configuration for a multi-table synthesizer crashed the model.
It also adds a number of fixes and enhancements, including: (1) a function and CLI command to list the available synthesizer names,
(2) a curate set of dependencies and making `Gretel` into an optional dependency, (3) updating `Gretel` to use temp directories,
(4) using `nvidia-smi` to get the number of gpus and (5) multiple `dockerfile` updates to improve functionality.

### Issues closed

* Bug when using JSON configuration for multiple multi-table evaluation - [Issue #115](https://github.com/sdv-dev/SDGym/issues/115) by @pvk-developer
* Use nvidia-smi to get number of gpus - [PR #113](https://github.com/sdv-dev/SDGym/issues/113) by @katxiao
* List synthesizer names - [Issue #82](https://github.com/sdv-dev/SDGym/issues/82) by @fealho
* Use nvidia base for dockerfile - [PR #108](https://github.com/sdv-dev/SDGym/issues/108) by @katxiao
* Add Makefile target to install gretel and ydata - [PR #107](https://github.com/sdv-dev/SDGym/issues/107) by @katxiao
* Curate dependencies and make Gretel optional - [PR #106](https://github.com/sdv-dev/SDGym/issues/106) by @csala
* Update gretel checkpoints to use temp directory - [PR #105](https://github.com/sdv-dev/SDGym/issues/105) by @katxiao
* Initialize variable before reference - [PR #104](https://github.com/sdv-dev/SDGym/issues/104) by @katxiao

## v0.4.0 - 2021-06-17

This release adds new synthesizers for Gretel and ydata, and creates a Docker image for SDGym.
It also includes enhancements to the accepted SDGym arguments, adds a summary command to aggregate
metrics, and adds the normalized score to the benchmark results.

### New Features

* Add normalized score to benchmark results - [Issue #102](https://github.com/sdv-dev/SDGym/issues/102) by @katxiao
* Add max rows and max columns args - [Issue #96](https://github.com/sdv-dev/SDGym/issues/96) by @katxiao
* Automatically detect number of workers - [Issue #97](https://github.com/sdv-dev/SDGym/issues/97) by @katxiao
* Add summary function and command - [Issue #92](https://github.com/sdv-dev/SDGym/issues/92) by @amontanez24
* Allow jobs list/JSON to be passed - [Issue #93](https://github.com/sdv-dev/SDGym/issues/93) by @fealho
* Add ydata to sdgym - [Issue #90](https://github.com/sdv-dev/SDGym/issues/90) by @fealho
* Add dockerfile for sdgym - [Issue #88](https://github.com/sdv-dev/SDGym/issues/88) by @katxiao
* Add Gretel to SDGym synthesizer - [Issue #87](https://github.com/sdv-dev/SDGym/issues/87) by @amontanez24

## v0.3.1 - 2021-05-20

This release adds new features to store results and cache contents into an S3 bucket
as well as a script to collect results from a cache dir and compile a single results
CSV file.

### Issues closed

* Collect cached results from s3 bucket - [Issue #85](https://github.com/sdv-dev/SDGym/issues/85) by @katxiao
* Store cache contents into an S3 bucket - [Issue #81](https://github.com/sdv-dev/SDGym/issues/81) by @katxiao
* Store SDGym results into an S3 bucket - [Issue #80](https://github.com/sdv-dev/SDGym/issues/80) by @katxiao
* Add a way to collect cached results - [Issue #79](https://github.com/sdv-dev/SDGym/issues/79) by @katxiao
* Allow reading datasets from private s3 bucket - [Issue #74](https://github.com/sdv-dev/SDGym/issues/74) by @katxiao
* Typos in the sdgym.run function docstring documentation - [Issue #69](https://github.com/sdv-dev/SDGym/issues/69) by @sbrugman

## v0.3.0 - 2021-01-27

Major rework of the SDGym functionality to support a collection of new features:

* Add relational and timeseries model benchmarking.
* Use SDMetrics for model scoring.
* Update datasets format to match SDV metadata based storage format.
* Centralize default datasets collection in the `sdv-datasets` S3 bucket.
* Add options to download and use datasets from different S3 buckets.
* Rename synthesizers to baselines and adapt to the new metadata format.
* Add model execution and metric computation time logging.
* Add optional synthetic data and error traceback caching.

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
