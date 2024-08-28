# History

## v0.9.1 - 2024-08-29

### Bugs Fixed

* `AttributeError` when running custom synthesizer with timeout - Issue [#335](https://github.com/sdv-dev/SDGym/issues/335) by @fealho

## v0.9.0 - 2024-08-07

This release enables the diagnostic score to be computed in a benchmarking run. It also renames the `IndependentSynthesizer` to `ColumnSynthesizer`. Finally, it fixes a bug so that the time for all metrics will now be used to compute the `Evaluate_Time` column in the results.

### Bugs Fixed

* Cap numpy to less than 2.0.0 until SDGym supports - Issue [#313](https://github.com/sdv-dev/SDGym/issues/313) by @gsheni
* The returned `Evaluate_Time` does not include results from all metrics - Issue [#310](https://github.com/sdv-dev/SDGym/issues/310) by @lajohn4747

### New Features

* Rename `IndependentSynthesizer` to `ColumnSynthesizer` - Issue [#319](https://github.com/sdv-dev/SDGym/issues/319) by @lajohn4747
* Allow the ability to compute diagnostic score in a benchmarking run - Issue [#311](https://github.com/sdv-dev/SDGym/issues/311) by @lajohn4747

## v0.8.0 - 2024-06-07

This release adds support for both Python 3.11 and 3.12! It also drops support for Python 3.7.

This release adds a new parameter to `benchmark_single_table` called `run_on_ec2`. When enabled, it will launch a `t2.medium` ec2 instance on the user's AWS account using the credentials they specify in environment variables. The benchmarking will then run on this instance. The `output_filepath` must be provided and must be in the format `{s3_bucket_name}/{path_to_file}` when `run_on_ec2` is enabled.

### Documentation

* Docs for AWS integration are incorrect - Issue [#304](https://github.com/sdv-dev/SDGym/issues/304) by @srinify

### Maintenance

* Add support for Python 3.11 - Issue [#250](https://github.com/sdv-dev/SDGym/issues/250) by @fealho
* Remove anyio usage - Issue [#252](https://github.com/sdv-dev/SDGym/issues/252) by @lajohn4747
* Drop support for Python 3.7 - Issue [#254](https://github.com/sdv-dev/SDGym/issues/254) by @R-Palazzo
* Switch default branch from master to main - Issue [#257](https://github.com/sdv-dev/SDGym/issues/257) by @R-Palazzo
* Transition from using setup.py to pyproject.toml to specify project metadata - Issue [#266](https://github.com/sdv-dev/SDGym/issues/266) by @R-Palazzo
* Remove bumpversion and use bump-my-version - Issue [#267](https://github.com/sdv-dev/SDGym/issues/267) by @R-Palazzo
* Switch to using ruff for Python linting and code formatting - Issue [#268](https://github.com/sdv-dev/SDGym/issues/268) by @gsheni
* Add dependency checker - Issue [#277](https://github.com/sdv-dev/SDGym/issues/277) by @lajohn4747
* Add bandit workflow - Issue [#282](https://github.com/sdv-dev/SDGym/issues/282) by @R-Palazzo
* Cleanup automated PR workflows - Issue [#286](https://github.com/sdv-dev/SDGym/issues/286) by @R-Palazzo
* Add support for Python 3.12 - Issue [#288](https://github.com/sdv-dev/SDGym/issues/288) by @fealho
* Only run unit and integration tests on oldest and latest python versions for macos - Issue [#294](https://github.com/sdv-dev/SDGym/issues/294) by @R-Palazzo
* Bump verions SDV, SDMetrics and RDT - Issue [#298](https://github.com/sdv-dev/SDGym/issues/298)

### Bugs Fixed

* The `UniformSynthesizer` should follow the sdtypes in metadata (not the data's dtypes)  - Issue [#248](https://github.com/sdv-dev/SDGym/issues/248) by @lajohn4747
* Fix minimum version workflow when pointing to github branch - Issue [#280](https://github.com/sdv-dev/SDGym/issues/280) by @R-Palazzo
* Passing synthesizer as string fails if run_on_ec2 is enabled - Issue [#306](https://github.com/sdv-dev/SDGym/issues/306) by @lajohn4747

### New Features

* Add run_on_ec2 flag to benchmark_single_table - Issue [#265](https://github.com/sdv-dev/SDGym/issues/265) by @lajohn4747
* Remove FastML Synthesizer - Issue [#292](https://github.com/sdv-dev/SDGym/issues/292) by @lajohn4747

## v0.7.0 - 2023-06-13

This release adds support for SDV 1.0 and PyTorch 2.0!

### New Features

* Add functions to top level import - Issue [#229](https://github.com/sdv-dev/SDGym/issues/229) by @fealho
* Cleanup SDGym to the new SDV 1.0 metadata and synthesizers - Issue [#212](https://github.com/sdv-dev/SDGym/issues/212) by @fealho

### Bugs Fixed

* limit_dataset_size causes sdgym to crash - Issue [#231](https://github.com/sdv-dev/SDGym/issues/231) by @fealho
* benchmark_single_table crashes with metadata dict - Issue [#232](https://github.com/sdv-dev/SDGym/issues/232) by @fealho
* Passing None as synthesizers runs all of them - Issue [#233](https://github.com/sdv-dev/SDGym/issues/233) by @fealho
* timeout parameter causes sdgym to crash - Issue [#234](https://github.com/sdv-dev/SDGym/issues/234) by @pvk-developer
* SDGym is not working with latest torch - Issue [#210](https://github.com/sdv-dev/SDGym/issues/210) by @amontanez24
* Fix sdgym --help - Issue [#206](https://github.com/sdv-dev/SDGym/issues/206) by @katxiao

### Internal

* Increase code style lint - Issue [#123](https://github.com/sdv-dev/SDGym/issues/123) by @fealho
* Remove code support for synthesizers that are not strings/classes - PR [#236](https://github.com/sdv-dev/SDGym/pull/236) by @fealho
* Code Refactoring - Issue [#215](https://github.com/sdv-dev/SDGym/issues/215) by @fealho

### Maintenance

* Remove pomegranate - Issue [#230](https://github.com/sdv-dev/SDGym/issues/230) by @amontanez24

## v0.6.0 - 2023-02-01
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
