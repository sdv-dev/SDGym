# SDGym: Synthetic Data Generation Gym

This is a benchmark focusing on evaluating the performance of synthetic data generators on generating tabular non-series data.

All data are stored on Dropbox.

## Data Format

To run this benchmark, you should first download all the data from dropbox and store them in `data` folder. So the whole working folder should look like

```
SyntheticDataBenchmark
 |-data/
 |  |
 |  |--real/
 |  |  |
 |  |  |--rdata1.npz
 |  |  |
 |  |  |--rdata1.json
 |  |  |
 |  |  |-- ...
 |  |
 |  |--simulate/
 |      |
 |      |--sdata1.npz
 |      |
 |      |--sdata1.json
 |      |
 |      |--...
 |
 |-synthetic_data_benchmark/
 |-output/
```

Each dataset comes with a npz data file and a json meta data. The meta file looks like

```
[
	{
		'name': None or str
		'type': 'Ordinal' or 'Categorical' or 'Continuous'

		# if Oridinal or Categorical:
		'size': integer
		'i2s': list of str

		#if Continuous:
		'min': float
		'max': float
	},
	...
]

```

The npz file includes 2 tables `train` and `test`. Each is a numpy array of (1~8 bytes) float or integer depending on the data. All continous columns are stored as is while categorical and oridinal columns are stored as integers (column be float because numpy array does not support mixed types). The mapping from integer to string is stored in meta file.

## Benchmark Framework

- Preprocess and get clean synthetic and real data sets. All code should goto `synthetic_data_benchmark /sdata_maker` and `synthetic_data_benchmark/rdata_maker`. (Once done, everything will be uploaded to S3, so that data are fixed for future use.
- `synthetic_data_benchmark/synthesizer/` contains several baseline synthesizers. Each synthesizer can be easily excuted on one or more datasets multiple times by
	-`> python3 -m synthetic_data_benchmark.synthesizer.xxx_synthesizer --repeat 3 [dataset1, dataset2, ...]`
	- Excution results are stored in `output/` as several npz files. Each one is one version of synthetic data.
- Evaluators
	- `synthetic_data_benchmark/evaluator/evaluate.py` evaluates the output of one synthesizer on all datasets and store in a json file.
		- `> python3 -m synthetic_data_benchmark.evaluator.evaluate output/XxxSynthesizer`
		- Results are stored as json file in `output/__result__`.
	- `synthetic_data_benchmark/evaluator/summary.py` generates the summary of results in `output/__result__`.
		- `> python3 evaluator/summary.py`
		- Outputs are stored as images and tables in `output/__summary__`.


## Summary Examples

<table>
<tr>
<td>
<img src="misc/coverage.jpg" width="100%">
</td>
<tr>
	<td>
	<img src="misc/mnist12.jpg" width="100%">
	</td>
	<td>
	<img src="misc/mnist28.jpg" width="100%">
	</td>
</tr>
<tr>
	<td>
	<img src="misc/adult.jpg" width="100%">
	</td>
	<td>
	<img src="misc/census.jpg" width="100%">
	</td>
</tr>
<tr>
	<td>
	<img src="misc/credit.jpg" width="100%">
	</td>
	<td>
	<img src="misc/covtype.jpg" width="100%">
	</td>
</tr>

<tr>
	<td>
	<img src="misc/intrusion.jpg" width="100%">
	</td>
	<td>
	<img src="misc/news.jpg" width="100%">
	</td>
</tr>

</table>

## List of datasets and metric


### Simulated data

- Bivariate

	- Gaussian Ring: Gaussian Mixtures arranged in a ring.

	- Gaussian Grid: Gaussian Mixtures arranged in a grid.
	<table>
	<tr>
	<td>
	<img src="misc/gaussian_ring.png" width="100%">
	</td>
	<td>
	<img src="misc/gaussian_grid.png" width="100%">
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
	<img src="misc/chain.png" width = "100" height = "200">
	</figure>
	</td>
	<td>
	<figure>
	<img src="misc/tree.png" width = "170" height = "100">
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
	<img src="misc/fc.png" width = "170" height = "200">
	</figure>
	</td>
	<td>
	<figure>
	<img src="misc/general.png" width = "170" height = "200">
	</figure>
	</td>
	</tr>
	</table>


### Real data
- MINIST28: Use flatten 28\*28 pixels into 784 binary columns with an extra label column.
- MINIST12: Reshape 28\*28 pixels into 12\*12 binary columns with an extra label column.
- Credit: Kaggle credit card fr dataset. `https://www.kaggle.com/mlg-ulb/creditcardfraud`
- Adult: Adult Data Set. `https://archive.ics.uci.edu/ml/datasets/adult`
- Census: KDD Census data set `https://archive.ics.uci.edu/ml/datasets/Census-Income+(KDD)`
- News: Online News Popularity Data Set (Regression) https://archive.ics.uci.edu/ml/datasets/online+news+popularity
- Covertype: Covertype Data Set (8 continuous + 40 binary + 1 multi) `https://archive.ics.uci.edu/ml/datasets/Covertype`
- intrusion: network intrusion detector kdd99 `https://archive.ics.uci.edu/ml/datasets/kdd+cup+1999+data`
