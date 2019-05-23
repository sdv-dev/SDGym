# SDGym

Synthetic Data Gym: A framework to benchmark the performance of synthetic data generators for
non-temporal tabular data.

# Getting started

## Installation

To install `SDGym` you only need to fork the repository, clone it and install its requirements

```
git clone git@github.com:$YOUR_USERNAME/SDGym.git
cd SDGym/
pip install -r requirements.txt
```

After installing the requirements, we just need to build the models that are not in python

```
sudo apt install build-essential
cd privbayes
make
```

## Data requirements

### Input Format

The input for all the synthesizers includecd in `SDGym` is a couple of files:

- A `npz` file containing two tables, `train` and `test`, where each is a `numpy.ndarray`.
All continous columns are stored as is, while categorical and ordinal columns are stored
using integers, altought the dtype will be float because numpy does not support mixed types.

- A `json` file containing the metadata for the dataset, that is, information about the columns,
like the max and minimum values on continous columns or the mapping from integer to string in
categorical columns.

```
[
	{
		'name': None or str
		'type': 'Ordinal' or 'Categorical' or 'Continuous'

		# if Ordinal or Categorical
		'size': integer
		'i2s': list of str

		# if Continuous
		'min': float
		'max': float
	},
	...
]

```

### Output Format

The results from `SDGym` are stored in the `output` folder with the following structure:

```
output
   __results__
       $MODEL.json	# Raw scores for model $MODEL
       ...

   __summaries__
      result.csv	# Table summary of the results
      barchart_$MODEL	# Bar chart for model $MODEL
      ...
```


### Demo Datasets

`SDGym` includes a few datasets to use for development or demonstration purposes. These datasets
have been preprocessed to be ready to use with `SDGym`, following the requirements specified in
the [Input Format](#input-format) section.

These datasets can be downloaded from [here](https://s3.amazonaws.com/sdgym/SDGymBenchmarkData.zip).
After downloading them, you just need to unzip their contents into a folder named `data` at the
root of `SDGym`.

You can also execute the following commands from the root of the repository:
```
curl https://s3.amazonaws.com/sdgym/SDGymBenchmarkData.zip -o data.zip
mkdir data
unzip data.zip -d data/
```

Have below the list of included datasets and their original source:

- MINIST28: Use flatten 28\*28 pixels into 784 binary columns with an extra label column.
- MINIST12: Reshape 28\*28 pixels into 12\*12 binary columns with an extra label column.
- Credit: Kaggle credit card fraud dataset. https://www.kaggle.com/mlg-ulb/creditcardfraud
- Adult: Adult Dataset. https://archive.ics.uci.edu/ml/datasets/adult
- Census: KDD Census dataset https://archive.ics.uci.edu/ml/datasets/Census-Income+(KDD)
- News: Online News Popularity Dataset (Regression) https://archive.ics.uci.edu/ml/datasets/online+news+popularity
- Covertype: Covertype Dataset (8 continuous + 40 binary + 1 multi) https://archive.ics.uci.edu/ml/datasets/Covertype
- Intrusion: network intrusion detector kdd99 https://archive.ics.uci.edu/ml/datasets/kdd+cup+1999+data

### Simulated data

- Bivariate

	- Gaussian Ring: Gaussian Mixtures arranged in a ring.

	- Gaussian Grid: Gaussian Mixtures arranged in a grid.

	<table>
	<tr>
	<td>
	<img src="https://i.imgur.com/oiGdICB.png" width="100%">
	</td>
	<td>
	<img src="https://i.imgur.com/EaAm2SO.png" width="100%">
	</td>
	</tr>
	</table>

- Multivariate Structured Data: Generate samples from some pre-specified common causal strutures.
	<table>
	<tr>
	<th>Chain</th>
	<th>Tree</th>
	</tr>
	<tr>
	<td>
	<figure>
	<img src="https://i.imgur.com/5fmjxOA.png" width = "100" height = "200">
	</figure>
	</td>
	<td>
	<figure>
	<img src="https://imgur.com/oXSV8AK.png" width = "170" height = "100">
	</figure>
	</td>
	</tr>

	<tr>
	<th>Fully Connected</th>
	<th>General</th>
	</tr>

	<tr>
	<td>
	<figure>
	<img src="https://imgur.com/nN3Ouzj.png" width = "170" height = "200">
	</figure>
	</td>
	<td>
	<figure>
	<img src="https://imgur.com/5RzIFsh.png" width = "170" height = "200">
	</figure>
	</td>
	</tr>
	</table>

## Quickstart

After installing the requirements and preparing the datasets, you only need to run the following
commands to evaluate a synthesizer:

```
python3 -m launcher SYNTHESIZER
```

* `SYNTHESIZER`: Name of the synthesizer you want to evaluate.

  Available synthesizers: [bgmvae, bgmwgan, clbn, identity, independent, medgan, privbn, uniform, veegan]

Optional arguments:

* `--datasets`: A list of datasets to evaluate the synthesizer with.
  If the argument is not present or the datasets are not specified it defaults to all datasets.

  Available datasets: [ asia, alarm, child,
insurance, grid, gridr, ring, adult, credit, census, news, covtype, intrusion, mnist12, mnist28]

* `--force`: Wheter or no overwritte results.
* `--repeat`(int): Number of copies to generate for each dataset.


## Summary Examples

<table>
<tr>
<td>
<img src="https://i.imgur.com/2m4FSQ4.jpg" width="100%">
</td>
<tr>
	<td>
	<img src="https://i.imgur.com/1plUKXZ.jpg" width="100%">
	</td>
	<td>
	<img src="https://imgur.com/s1Xgu3g.jpg" width="100%">
	</td>
</tr>
<tr>
	<td>
	<img src="https://i.imgur.com/vdRCO77.jpg" width="100%">
	</td>
	<td>
	<img src="https://imgur.com/bCG8NCs.jpg" width="100%">
	</td>
</tr>
<tr>
	<td>
	<img src="https://imgur.com/Sa5fUTs.jpg" width="100%">
	</td>
	<td>
	<img src="https://imgur.com/vgri2hd.jpg" width="100%">
	</td>
</tr>

<tr>
	<td>
	<img src="https://i.imgur.com/3k7X6VW.jpg" width="100%">
	</td>
	<td>
	<img src="https://imgur.com/mGC4f8h.jpg" width="100%">
	</td>
</tr>

</table>

