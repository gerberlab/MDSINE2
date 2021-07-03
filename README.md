# MDSINE2

This repository contains the MDSINE2 (Microbial Dynamical Systems INference Engine 2) package. A python implementation of a robust and scalable Bayesian model for learning  microbial dynamics.
MDSINE2 extends the generalized Lotka-Volterra (gLV) model to include automatically learned interaction modules, which we define as groups of taxa that share common interaction structure (i.e., are promoted or inhibited by the same taxa outside the module) and have a common response to external perturbations (e.g., antibiotics).

There is another repo for the paper associated with this model. If this is your first time using MDSINE2 we suggest you go to the companion repo  https://github.com/gerberlab/MDSINE2_Paper

## Description of inputs and outputs

MDSINE2 takes as inputs microbial abundances from two data modalities, reads from sequencing and qPCR for quantification of total bacterial load. The output from the model are the traces of the posterior samples for all the [model](#underlying-model-and-parameters) parameters. Of note, because our model is fully Bayesian, MDSINE2 returns confidence measures on all aspects of the model (e.g., Bayes Factors). See [model](#underlying-model-and-parameters) for more details.

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

#### Option 1 Simple installation of just the MDSINE2

clone the repository and then `pip install`
```bash
git clone https://github.com/gerberlab/MDSINE2
pip install MDSINE2/.
```
This installs the package MDSINE2 and all of the dependencies listed above.

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



## Tutorials

We recommend heading on over to the github repo for the paper https://github.com/gerberlab/MDSINE2_Paper that has detailed examples for working with MDSINE2 as well as example data
