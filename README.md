# MDSINE2

This reposity contains code used to run the MDSINE2 (Microbial Dynamical Systems INference Engine 2). A robust and scalable model for microbiome time series dynamics


## Description of the software

## Documentation
[documentaion link](https://htmlpreview.github.io/?https://raw.githubusercontent.com/gerberlab/MDSINE2/master/docs/mdsine2/index.html)


## Dependencies (Python 3.7.3)


 * biopython==1.78
 * ete3==3.1.2
 * numpy==1.19.4
 * pandas==1.14
 * matplotlib==3.3.1
 * numba==0.48
 * sklearn==0.0
 * seaborn==0.11.0
 * psutil==5.7.3
 * h5py==2.9.0
 * networkx==2.3

## Installation

If you already have Python 3.7.3 Clone this directory, `cd` into mdsine and type
```bash
pip install .
```
This installs the package `mdsine2` and all of the dependencies listed above.

For a fresh install of Python 3.7.3 and MDSINE2 with a linked Jupyter kernel all from the command line one can take the followings steps
```bash
conda create -n mdsine2 -c conda-forge python=3.7.3 jupyterlab
conda activate mdsine2
python -m ipykernel install --user --name mdsine2 --display-name "mdsine2"
git clone https://github.com/gerberlab/MDSINE2
pip install MDSINE2/.
 ``` 

## Tutorials
 ---
 Tutorials on how to use the package can be found in the `tutorials` directory.

## Datasets
---
```python
import mdsine2 as md2
```
#### Gibson dataset

The Gibson dataset that was used in ########## can be obtained using
```python
study = md2.dataset.gibson()
```
Which returns an `md2.Study` object that contains all of the data from both the Healthy and Ulcerative Colitis cohorts. To obtain a `md2.Study` object of a single cohort, simply:
```python
healthy_cohort = md2.dataset.gibson(healthy=True)
ulcerative_colitis_cohort = md2.dataset.gibson(healthy=False)
```
To retrieve the raw data used to construct the `md2.Study` object:
```python
dfs = md2.dataset.gibson(as_df=True)
taxonomy = dfs['taxonomy']
qpcr = dfs['qpcr']
reads = dfs['reads']
metadata = dfs['metadata']
```
where `taxonomy`, `qpcr`, `reads`, and `metadata` are `pandas.DataFrame` objects that contain the raw data. If you additionally specify the `healthy` parameter in `md2.dataset.gibson`, it will only retrieve the raw data for that specific cohort.

#### Parsing your own dataset
---
To parse your own data, refer to `tutorials/parsing_data.md`.

