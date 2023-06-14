# SDGym Datasets

**SDGym** uses SDV datasets to benchmark the **Synthesizers** which are in three data modalities:

* Single Table Datasets: Datasets that contain only one table with no inter-row dependencies.
* Multi Table Datasets: Datasets that contain more than one table, potentially with relationships
  between them.
* Time Series Datasets: Datasets that contain a single table that represents sequences of rows.

## Dataset Format

The **SDV Datasets** are comprised of two elements:

* A `metadata.json` file which describes the data found in the dataset. This file follows the
  [SDV Single Table Metadata schema](https://docs.sdv.dev/sdv/single-table-data/data-preparation/single-table-metadata-api)
* A collection of `CSV` files stored in a format which can be loaded by the `pandas.read_csv`
  function without any additional arguments than the csv path.

## Using the datasets

All the datasets can also be found for download inside the [sdv-datasets S3 bucket](
http://sdv-demo-datasets.s3.amazonaws.com) in the form of a `.zip` file that contains
both the `metadata_v1.json` and the `CSV` file collection.

In order to load these datasets in the same format as they will be passed to your synthesizer
you can use the `sdgym.load_dataset` function passing the name of the dataset to load.

In this example, we will load the `adult` dataset:

```python3
In [1]: from sdgym.datasets import load_dataset

In [2]: data, metadata = load_dataset('adult')
```

Afterwards, you can load the tables from the dataset passing the loaded `metadata` to the
`sdgym.load_tables` function:

```python3
In [4]: from sdgym.datasets import load_tables

In [5]: tables = load_tables(metadata)
```

This will return a `dict` containing the tables loaded as `pandas.DataFrames`.

```python3
In [6]: tables
Out[6]:
{'adult':        age  workclass  fnlwgt     education  education-num  ... capital-gain capital-loss hours-per-week native-country  label
 0       27    Private  177119  Some-college             10  ...            0            0             44  United-States  <=50K
 1       27    Private  216481     Bachelors             13  ...            0            0             40  United-States  <=50K
 2       25    Private  256263    Assoc-acdm             12  ...            0            0             40  United-States  <=50K
 3       46    Private  147640       5th-6th              3  ...            0         1902             40  United-States  <=50K
 4       45    Private  172822          11th              7  ...            0         2824             76  United-States   >50K
 ...    ...        ...     ...           ...            ...  ...          ...          ...            ...            ...    ...
 32556   43  Local-gov   33331       Masters             14  ...            0            0             40  United-States   >50K
 32557   44    Private   98466          10th              6  ...            0            0             35  United-States  <=50K
 32558   23    Private   45317  Some-college             10  ...            0            0             40  United-States  <=50K
 32559   45  Local-gov  215862     Doctorate             16  ...         7688            0             45  United-States   >50K
 32560   25    Private  186925  Some-college             10  ...         2597            0             48  United-States  <=50K

 [32561 rows x 15 columns]}
```

## Getting the list of all the datasets

If you want to obtain the list of all the available datasets you can use the
`sdgym.get_available_datasets` function:

```python
In [7]: from sdgym import get_available_datasets

In [8]: get_available_datasets()
Out[8]:
              dataset_name     size_MB  num_tables
0                   KRK_v1    0.072128           1
1                    adult    3.907448           1
2                    alarm    4.520128           1
3                     asia    1.280128           1
4                   census   98.165608           1
5          census_extended    4.949400           1
6                    child    3.200128           1
7                  covtype  255.645408           1
8                   credit   68.353808           1
9       expedia_hotel_logs    0.200128           1
10          fake_companies    0.001280           1
11       fake_hotel_guests    0.032628           1
12                    grid    0.320128           1
13                   gridr    0.320128           1
14               insurance    3.340128           1
15               intrusion  162.039016           1
16                 mnist12   81.200128           1
17                 mnist28  439.600128           1
18                    news   18.712096           1
19                    ring    0.320128           1
20      student_placements    0.026358           1
21  student_placements_pii    0.028078           1
```
