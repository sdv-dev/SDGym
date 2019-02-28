# Synthetic Data Generator Benchmark

This benchmark focuses on evaluating the performance of synthetic data generators on generating tabular non-series data.

All data are stored on Dropbox. 

## Meta Format
meta data is a JSON file. 

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

## List of datasets and metric


### Synthetic

- 2D Ring
- 2D Grid
- high D Gaussian


### Real
- MINIST28: Use flatten 28\*28 pixels into 784 binary columns with an extra label column. 
- MINIST12: Reshape 28\*28 pixels into 12\*12 binary columns with an extra label column. 


- Covertype (8 continuous + 40 binary + 1 multi) `https://archive.ics.uci.edu/ml/datasets/Covertype`
- KDD Census data set `https://archive.ics.uci.edu/ml/datasets/Census-Income+%28KDD%29`
- KDD98 DNA `https://archive.ics.uci.edu/ml/datasets/KDD+Cup+1998+Data`
- Statlog (German Credit Data) Data Set 
- Blood Transfusion Service Center Data Set
- Tic-Tac-Toe https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame

