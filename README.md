# MDSINE2

This repository contains the MDSINE2 (Microbial Dynamical Systems INference Engine 2) package. A python implementation of a robust and scalable Bayesian model for learning  microbial dynamics.
MDSINE2 extends the generalized Lotka-Volterra (gLV) model to include automatically learned interaction modules, which we define as groups of taxa that share common interaction structure (i.e., are promoted or inhibited by the same taxa outside the module) and have a common response to external perturbations (e.g., antibiotics).

There is an [associated repo](https://github.com/gerberlab/MDSINE2_Paper) for the [pre-print](https://doi.org/10.1101/2021.12.14.469105) that introduces and applies this model to a densely sampled gnotobiotic time series of healthy and dysbiotic microbiomes along with [google colab tutorials](https://github.com/gerberlab/MDSINE2_Paper/tree/master/google_colab) exploring the model, data, and paper results. 

- Main Paper (Pre-print): ["Intrinsic instability of the dysbiotic microbiome revealed through dynamical systems inference at scale"](https://doi.org/10.1101/2021.12.14.469105)<br />
  <a href="https://doi.org/10.1101/2021.12.14.469105"><img alt="" src="https://img.shields.io/badge/bioRχiv%20DOI-10.1101/2021.12.14.46910-blue?style=flat"/></a>
- Associated GitHub repo for the paper: ["MDSINE2_Paper"](https://github.com/gerberlab/MDSINE2_Paper)<br />
  <a href="https://github.com/gerberlab/MDSINE2_Paper"><img alt="" src="https://img.shields.io/badge/GitHub-MDSINE2%20Paper-blue?style=flat&logo=github"/></a>
- Folder containing [tutorials as notebooks exploring the model, data and paper](https://github.com/gerberlab/MDSINE2_Paper/tree/master/google_colab) that can be opened directly in Google Colab<br />
<a href="https://github.com/gerberlab/MDSINE2_Paper/tree/master/google_colab"><img alt="" src="https://img.shields.io/badge/Jupyter Notebooks-MDSINE2%20Tutorials-blue?style=flat&logo=jupyter"/></a>
- The mathematics behind this model are detailed in the [supplemental text for the pre-print](https://www.biorxiv.org/content/biorxiv/early/2021/12/16/2021.12.14.469105/DC1/embed/media-1.pdf)<br />
<a href="https://www.biorxiv.org/content/biorxiv/early/2021/12/16/2021.12.14.469105/DC1/embed/media-1.pdf"><img alt="" src="https://img.shields.io/badge/PDF-MDSINE2%20Mathematics-blue?style=flat&logo=adobeacrobatreader"/></a>

### References
Pre-print
```bibtex
@article {Gibson2021.12.14.469105,
	author = {Gibson, Travis E and Kim, Younhun and Acharya, Sawal and Kaplan, David E and DiBenedetto, Nicholas and Lavin, Richard and Berger, Bonnie and Allegretti, Jessica R and Bry, Lynn and Gerber, Georg K},
	title = {Intrinsic instability of the dysbiotic microbiome revealed through dynamical systems inference at scale},
	year = {2021},
	doi = {10.1101/2021.12.14.469105},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2021/12/16/2021.12.14.469105},
	journal = {bioRxiv}}
```
ICML conference paper 
```bibtex
@InProceedings{pmlr-v80-gibson18a,
  title = 	 {Robust and Scalable Models of Microbiome Dynamics},
  author =       {Gibson, Travis and Gerber, Georg},
  booktitle = 	 {Proceedings of the 35th International Conference on Machine Learning},
  pages = 	 {1763--1772},
  year = 	 {2018},
  editor = 	 {Dy, Jennifer and Krause, Andreas},
  volume = 	 {80},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {10--15 Jul},
  publisher =    {PMLR},
  url = 	 {https://proceedings.mlr.press/v80/gibson18a.html}}
```

## Installation

#### Option 1: Simple installation using pip

clone the repository and then `pip install`
```
git clone https://github.com/gerberlab/MDSINE2
pip install MDSINE2/.
```
This installs the package MDSINE2 and all of the dependencies listed in `requirements.txt`.

#### Option 2: Conda environment with MDSINE2 and jupyterlab

An alternative is to install MDSINE2 through conda.
```
git clone https://github.com/gerberlab/MDSINE2
cd MDSINE2
conda env create -f conda_recipe.yml 
```

## Documentation
MDSINE2 is implemented as a python library and as a command line interface (CLI).
The library is importable using python's import command: `import mdsine2`, and the CLI is accessed using the command `mdsine2`.
The classes and methods' documentation can be found here: 

[documentation link](https://htmlpreview.github.io/?https://raw.githubusercontent.com/gerberlab/MDSINE2/master/docs/mdsine2/index.html)

We also provide some comand-line interfaces for the core features. 
While these are more limited in functionality than directly using the python package, it does allow bash scripting
for tasks generalizable to arbitrary datasets.
See [Features and Examples](#features-and-examples) for more details.


## Description of inputs and outputs
MDSINE2 takes as inputs microbial abundances from two data modalities, reads from sequencing and qPCR for 
quantification of total bacterial load. The output from the model are the traces of the posterior samples 
for all the [model](#Underlying-model-and-parameters) parameters (growth rates, module assignments, interaction 
indicators and strengths, perturbation indicators and strengths ...).

<p align="center">
<img src="/figures/github2.svg" />
</p>

## Underlying model and parameters
A complete description of the model is given as supplemental text in [doi], or for a direct link click [here]().

## Tutorials

We recommend going to ['MDSINE2_Paper/google_colab'](https://github.com/gerberlab/MDSINE2_Paper/tree/master/google_colab) repository that has detailed examples for working with MDSINE2 and exploring real data


## Features and Examples

MDSINE2 has the following core features, implemented as a python package but also interfacable using the command line.
We **strongly** encourage users to start with the [tutorials](#tutorials) to get started, using the following sections
as a high-level overview only.
For more detailed specifications on the python functions, refer to the documentation.
For command line specifications, use the `--help` option (e.g. `mdsine2 infer --help`).

### Input processing and Visualization

Process the raw input files (TSV format) and process them into python `mdsine2.Study` objects.

#### A) Read count information and qPCR
    
*python example:*
```python
import mdsine2 as md2
study = md2.dataset.parse(name="gibson_dataset", reads="counts.tsv", (...))
```

*command-line example:*
```bash
> mdsine2 parse --name "gibson_dataset" --reads counts.tsv (...)
```
    
#### B) Time-series plots of qPCR, abundances and alpha-diversity of each subject.

*command-line example:*
```bash
> mdsine2 plot-subjects -i dataset.pkl -o . -t phylum
```

### MCMC inference using MDSINE2's model.

MDSINE2's primary function is to implement an MCMC algorithm for learning gLV parameters for many taxa.
To do this, we fit some parameters as a preliminary step (negative-binomial disperson parameters) and then pass
this as input into the main MCMC algorithm.

#### A) Learn the negative binomial parameters from data.

*python example:*
```python
import mdsine2 as md2
study = md2.Study.load("dataset.pkl")
params = md2.config.NegBinConfig(...)
mcmc = md2.negbin.build_graph(params=params, graph_name=study.name, subjset=study)
md2.negbin.run_graph(mcmc)
```

*command-line example:*
```bash
mdsine2 infer-negbin --input dataset.pkl (...)
```

#### B) Run MDSINE2's MCMC algorithm.

*python example:*
```python
import mdsine2 as md2
study = md2.Study.load("dataset.pkl")
params = md2.config.MDSINE2ModelConfig(...)
mcmc = md2.initialize_graph(params=params, graph_name=study.name, subjset=study)
md2.run_graph(mcmc)
```

*command-line example:*
```bash
> mdsine2 infer --input dataset.pkl (...)
```

### Visualization of posterior distribution from MCMC samples

Using the results of MDSINE2's MCMC output (a collection of posterior samples), visualize the posterior distribution of
the parameters. This includes a visual summary of the gLV parameters for each taxa, a heatmap of
co-clustering likelihoods, and multiple heatmaps of interactions between taxa.

*python example:*
```python
from mdsine2.names import STRNAMES
mcmc.graph[STRNAMES.PRIOR_MEAN_SELF_INTERACTIONS].visualize(path='si_mean.pdf', section="posterior")
```

*command-line example:*
```bash
> mdsine2 visualize-posterior --chain output/mcmc.pkl --section posterior (...)
```

### Computation and visualization of forward-simulation of learned gLV model for each OTU.

*python example:*
```python
dyn = md2.model.gLVDynamicsSingleClustering(growth=g, interactions=A, perturbations=p, (...))
x = md2.integrate(dynamics=dyn, initial_conditions=x, dt=dt, n_days=n_days, (...))
np.save("fwsim.npy", x)
```

*command-line example:*
```bash
> mdsine2 forward-simulate (TODO) -i output/mcmc.pkl --study dataset.pkl --plot all (...)
```

### Visualize the phylogenetic placement of OTUs.

We provide a tool to draw phylogenetic placements -- rendered into PDF as local subtrees induced by leaves
within a specified radius of each taxa -- provided by the user (e.g. produced by FastTree + pplacer).

*command-line example:*
```bash
> mdsine2 render-phylogeny --study dataset.pkl --tree my_tree.nhx --output-basepath phylo/ (...)
```
    
### Visualize the learned network of interactions

In addition to the visualizations from 3.3 which draws a heatmap of interactions, gLV interactions can be drawn
as a network of nodes (modules of taxa) and edges (signed interactions). 
This functionality takes as input the posterior samples from a "fixed-cluster" inference run, and generates a file 
interpretable by Cytoscape, showing modules of taxa and the interactions between them.

*python example:*
```python
mcmc = md2.BaseMCMC.load("fixed_clustering/mcmc.pkl")
md2.write_fixed_clustering_as_json(
    mcmc=mcmc,
    output_path="fixed_module_interactions.json"
)
```

*command-line example:*
```bash
> mdsine2 interaction-to-cytoscape -i fixed_clustering/mcmc.pkl -o fixed_clustering/fixed_module_interactions.json 
```

### Compute and visualize keystoneness metric

As a downstream analysis, we compute "keystoneness" which quantifies the amount of influence that each module has
on the rest of the system.

*command-line example:*
```bash
> mdsine2 extract-abundances -s dataset.pkl -t 19 -o initial.tsv
> mdsine2 evaluate-keystoneness -f fixed_clustering/mcmc.pkl -s dataset.pkl -i initial.tsv (...)
```
