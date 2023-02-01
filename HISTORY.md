# History

## v0.6.0 - 2021-02-01
This release introduces methods for benchmarking single table data and creating custom synthesizers, which can be based on existing SDGym-defined synthesizers or on user-defined functions. This release also adds support for Python 3.10 and drops support for Python 3.6.

### New Features
* Benchmarking progress bar should update on one line - Issue [#204](https://github.com/sdv-dev/SDGym/issues/204) by @katxiao
* Support local additional datasets folder with zip files - Issue [#186](https://github.com/sdv-dev/SDGym/issues/186) by @katxiao
* Enforce that each synthesizer is unique in benchmark_single_table - Issue [#190](https://github.com/sdv-dev/SDGym/issues/190) by @katxiao
* Simplify the file names inside the detailed_results_folder - Issue [#191](https://github.com/sdv-dev/SDGym/issues/191) by @katxiao
* Use SDMetrics silent report generation - Issue [#179](https://github.com/sdv-dev/SDGym/issues/179) by @katxiao
* Remove arguments in get_available_datasets - Issue [#197](https://github.com/sdv-dev/SDGym/issues/197) by @katxiao
* Accept metadata.json as valid metadata file - Issue [#194](https://github.com/sdv-dev/SDGym/issues/194) by @katxiao
* Check if file or folder exists before writing benchmarking results - Issue [#196](https://github.com/sdv-dev/SDGym/issues/196) by @katxiao
* Rename benchmarking argument "evaluate_quality" to "compute_quality_score" - Issue [#195](https://github.com/sdv-dev/SDGym/issues/195) by @katxiao
* Add option to disable sdmetrics in benchmarking - Issue [#182](https://github.com/sdv-dev/SDGym/issues/182) by @katxiao
* Prefix remote bucket with 's3' - Issue [#183](https://github.com/sdv-dev/SDGym/issues/183) by @katxiao
* Benchmarking error handling - Issue [#177](https://github.com/sdv-dev/SDGym/issues/177) by @katxiao
* Allow users to specify custom synthesizers' display names - Issue [#174](https://github.com/sdv-dev/SDGym/issues/174) by @katxiao
* Update benchmarking results columns - Issue [#172](https://github.com/sdv-dev/SDGym/issues/172) by @katxiao
* Allow custom datasets - Issue [#166](https://github.com/sdv-dev/SDGym/issues/166) by @katxiao
* Use new datasets s3 bucket - Issue [#161](https://github.com/sdv-dev/SDGym/issues/161) by @katxiao
* Create benchmark_single_table method - Issue [#151](https://github.com/sdv-dev/SDGym/issues/151) by @katxiao
* Update summary metrics - Issue [#134](https://github.com/sdv-dev/SDGym/issues/134) by @katxiao
* Benchmark individual methods - Issue [#159](https://github.com/sdv-dev/SDGym/issues/159) by @katxiao
* Add method to create a sdv variant synthesizer - Issue [#152](https://github.com/sdv-dev/SDGym/issues/152) by @katxiao
* Add method to generate a multi table synthesizer - Issue [#149](https://github.com/sdv-dev/SDGym/issues/149) by @katxiao
* Add method to create single table synthesizers - Issue [#148](https://github.com/sdv-dev/SDGym/issues/148) by @katxiao
* Updating existing synthesizers to new API - Issue [#154](https://github.com/sdv-dev/SDGym/issues/154) by @katxiao

### Bug Fixes
* Pip encounters dependency issues with ipython - Issue [#187](https://github.com/sdv-dev/SDGym/issues/187) by @katxiao
* IndependentSynthesizer is printing out ConvergeWarning too many times - Issue [#192](https://github.com/sdv-dev/SDGym/issues/192) by @katxiao
* Size values in benchmarking results seems inaccurate - Issue [#184](https://github.com/sdv-dev/SDGym/issues/184) by @katxiao
* Import error in the example for benchmarking the synthesizers - Issue [#139](https://github.com/sdv-dev/SDGym/issues/139) by @katxiao
* Updates and bugfixes - Issue [#132](https://github.com/sdv-dev/SDGym/issues/132) by @csala

### Maintenance
* Update README - Issue [#203](https://github.com/sdv-dev/SDGym/issues/203) by @katxiao
* Support Python Versions >=3.7 and <3.11 - Issue [#170](https://github.com/sdv-dev/SDGym/issues/170) by @katxiao
* SDGym Package Maintenance Updates documentation  - Issue [#163](https://github.com/sdv-dev/SDGym/issues/163) by @katxiao
* Remove YData - Issue [#168](https://github.com/sdv-dev/SDGym/issues/168) by @katxiao
* Update to newest SDV - Issue [#157](https://github.com/sdv-dev/SDGym/issues/157) by @katxiao
* Update slack invite link. - Issue [#144](https://github.com/sdv-dev/SDGym/issues/144) by @pvk-developer
* updating workflows to work with windows - Issue [#136](https://github.com/sdv-dev/SDGym/issues/136) by @amontanez24
* Update conda dependencies - Issue [#130](https://github.com/sdv-dev/SDGym/issues/130) by @katxiao

## v0.5.0 - 2021-12-13
This release adds support for Python 3.9, and updates dependencies to accept the latest versions when possible.

### Issues closed

* Add support for Python 3.9 - [Issue #127](https://github.com/sdv-dev/SDGym/issues/127) by @katxiao
* Add pip check worflow - [Issue #124](https://github.com/sdv-dev/SDGym/issues/124) by @pvk-developer
* Fix meta.yaml dependencies - [PR #119](https://github.com/sdv-dev/SDGym/pull/119) by @fealho
* Upgrade dependency ranges - [Issue #118](https://github.com/sdv-dev/SDGym/issues/118) by @katxiao

## v0.4.1 - 2021-08-20
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
