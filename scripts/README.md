# SDGym Scripts

This folder contains a collection of bash scripts and configuration files used to run the entire
SDGym benchmarking suite on the [MIT SuperCloud](https://supercloud.mit.edu/) infrastructure.

## Overview

The contents of this folder consist on the following elements:

* `config`: Folder that contains configuration files for different tasks, which indicate
  the synthesizers and data modalities to run and the resources needed.
* `run.sh`: A bash script that interprets the `config` files and launches SDGym with the
  corresponding settings.
* `submit.sh`: A bash script that submits tasks to the MIT SuperCloud job queue using the
  `run.sh` script and the configuration files.

### Prerequisites

In order to use these scripts you will need to [have been given an account in the MIT SuperCloud](
https://supercloud.mit.edu/requesting-account).
If you want to also upload the results to the `sdgym` S3 bucket you will need to also have an AWS
IAM user with write permissions to the bucket and have configured the [.aws/credentials file](
https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html) accordingly.

## Usage

In order to run `SDGym` using these scripts, please follow these steps:

1. Log into MIT SuperCloud launch server

2. Clone and enter the SDGym repository

```bash
git clone https://github.com/sdv-dev/SDGym
cd SDGym
```

3. Install SDGym inside a virtualenv or conda env (**NOTE**: you may need to activate the conda module)

```bash
conda create -y -n sdgym
conda activate sdgym
make install
```

4. Enter the `scripts` folder and execute `submit.sh` passing the config files you want to run

```bash
cd scripts
./submit.sh config/identity.conf config/...
```

> **NOTE**: The first time the script is run it will download all the available datasets in the
`datasets` folder within the `scripts` folder. Subsequent runs will skip this step if the
`datasets` folder is found.

After this, you can verify that the tasks have been properly submitted running `LLstat`, and
that a folder called `runs/<current-date-and-time>` has been created inside the `scripts` folder.

## Uploading results to S3

After all the tasks have finished (`LLstat` should show no running tasks), you can collect
the results and upload them to the `sdgym` S3 bucket using the following steps:

1. Enter the `scripts/runs` folder.

```bash
cd SDGym/scripts/runs
```

2. Create a `tar.gz` file with the run results:

```bash
tar cvzf <run-date-and-time>.tar.gz <run-date-and-time>
```

3. Upload the `tar.gz` file that you just created to S3

```bash
aws s3 cp <run-date-and-time>.tar.gz s3://sdgym/runs
```

## Collecting results as a single CSV file

While running, SDGym will generate a large collection of CSV files containing results from
the different synthesizers and datasets.

In order to collect all these CSVs into a single table you can use the `collect.py` python
script passing it the path to the results folder and the path of the new CSV file to generate:

1. Enter the `scripts` folder.

```bash
cd SDGym/scripts
```

2. Call `python collect.py` passing the path to the run folder and the output csv path:

```bash
python collect.py runs/<run-date-and-time> output.csv
```
