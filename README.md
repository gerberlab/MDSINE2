# MDSINE2

This repository contains the MDSINE2 (Microbial Dynamical Systems INference Engine 2) package. A python implementation of a robust and scalable Bayesian model for learning  microbial dynamics.
MDSINE2 extends the generalized Lotka-Volterra (gLV) model to include automatically learned interaction modules, which we define as groups of taxa that share common interaction structure (i.e., are promoted or inhibited by the same taxa outside the module) and have a common response to external perturbations (e.g., antibiotics).

There is another repo for the paper associated with this model. If this is your first time using MDSINE2 we suggest you go to the companion repo  https://github.com/gerberlab/MDSINE2_Paper

## Description of inputs and outputs

MDSINE2 takes as inputs microbial abundances from two data modalities, reads from sequencing and qPCR for quantification of total bacterial load. Of note, because our model is fully Bayesian, MDSINE2 returns confidence measures on all aspects of the model (e.g., Bayes Factors). See [mathematical model](#underlying-model-and-parameters) for more details.

<p align="center">
<img src="/figures/github2.svg" width="700" />
</p>

## Documentation
[documentation link](https://htmlpreview.github.io/?https://raw.githubusercontent.com/gerberlab/MDSINE2/master/docs/mdsine2/index.html)

## Installation

#### Dependencies (Python 3.7.3)

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

#### Option 1 Simple installation of just the 'MDSINE2' package

clone the repository, `cd` into mdsine, and then `pip install`
```bash
git clone https://github.com/gerberlab/MDSINE2
pip install MDSINE2/.
```
This installs the package `MDSINE2` and all of the dependencies listed above.

#### Option 2 Create a `conda` environment with MDSINE2 and jupyterlab

For a fresh install of Python 3.7.3 and MDSINE2 with a linked Jupyter kernel all from the command line one can take the followings steps
```bash
conda create -n mdsine2 -c conda-forge python=3.7.3 jupyterlab
conda activate mdsine2
python -m ipykernel install --user --name mdsine2 --display-name "mdsine2"
git clone https://github.com/gerberlab/MDSINE2
pip install MDSINE2/.
 ```


## Underlying model and parameters
<p align="center">
<img src="/figures/github1.svg" width="600" />
</p>

Key model parameters
- $i$: taxon number
- $k$: time index
- $s$: replicate index (for different time series: mouse, subject, etc)
- $y_{s,i}(k)$: reads taxon $i$, replicate $s$, time index $k$
- $Q_{j,s}(k)$: qPCR replicate $j$, replicate $s$, time index $k$
- $x_{s,i}(k)$: latent state for microbial abundance
- $c_i$: cluster assignment for taxon $i$
- $b_{c_i,c_j}$: interaction parameter for taxon $j$ to taxon $i$
- $z^{(b)}_{c_i,c_j}$: interaction indicator for taxon $j$ to taxon $i$
- $\gamma_{p,c_i}$: perturbation $p$ affect on cluster $c_i$ members
- $z^{(\gamma)}_{p,c_i}$: perturbation indicator for $\gamma_{p,c_i}$
- $a_{i,1}$: growth rate of taxa $i$
- $a_{i,2}$: self limiting term for taxa $i$

Inference is performed using MCMC with Gibbs and collapsed Gibbs sampling.

## Tutorials

We recommend heading on over to the github repo for the paper https://github.com/gerberlab/MDSINE2_Paper that has detailed examples for working with `MDSINE2` as well as example data
