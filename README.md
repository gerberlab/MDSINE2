# MDSINE2

This repository contains the MDSINE2 (Microbial Dynamical Systems INference Engine 2) package. A python implementation of a robust and scalable Bayesian model for learning  microbial dynamics.
MDSINE2 extends the generalized Lotka-Volterra (gLV) model to include automatically learned interaction modules, which we define as groups of taxa that share common interaction structure (i.e., are promoted or inhibited by the same taxa outside the module) and have a common response to external perturbations (e.g., antibiotics).

There is another repo for the paper associated with this model. If this is your first time using MDSINE2 we suggest you go to the companion repo  https://github.com/gerberlab/MDSINE2_Paper

## 1. Description of inputs and outputs

MDSINE2 takes as inputs microbial abundances from two data modalities, reads from sequencing and qPCR for 
quantification of total bacterial load. The output from the model are the traces of the posterior samples 
for all the [model](#4-Underlying-model-and-parameters) parameters (growth rates, module assignments, interaction 
indicators and strengths, perturbation indicators and strengths ...). Of note, because our model is fully Bayesian, 
MDSINE2 returns confidence measures on all aspects of the model (e.g., Bayes Factors). 
See [model](#4-Underlying-model-and-parameters) for more details.

<p align="center">
<img src="/figures/github2.svg" />
</p>

## 2. Documentation
MDSINE2 is implemented as a python library, which is importable using python's import command: `import mdsine2`.
The classes and methods' documentation can be found here: 

[documentation link](https://htmlpreview.github.io/?https://raw.githubusercontent.com/gerberlab/MDSINE2/master/docs/mdsine2/index.html)

We also provide some comand-line interfaces for the core features. 
While these are more limited in functionality than directly using the python package, it does allow bash scripting
for tasks generalizable to arbitrary datasets.
See [Features and Examples](#6-features-and-examples) for more details.


## 3. Installation

#### Dependencies (Python 3.7.3)

 * biopython==1.78
 * ete3==3.1.2
 * numpy==1.19.4
 * pandas==1.14
 * matplotlib==3.3.1
 * numba==0.52
 * sklearn==0.0
 * seaborn==0.11.0
 * psutil==5.7.3
 * h5py==2.9.0
 * networkx==2.3

#### Option 1: Simple installation using pip

clone the repository and then `pip install`
```
git clone https://github.com/gerberlab/MDSINE2
pip install MDSINE2/.
```
This installs the package MDSINE2 and all of the dependencies listed above.

#### Option 2: Conda environment with MDSINE2 and jupyterlab

An alternative is to install MDSINE2 through conda and a linked jupyter kernel (useful for data exploration).
```
conda create -n mdsine2 -c conda-forge python=3.7.3 jupyterlab
conda activate mdsine2
python -m ipykernel install --user --name mdsine2 --display-name "mdsine2"
git clone https://github.com/gerberlab/MDSINE2
pip install MDSINE2/.
```

To install without the Jupyter kernel, follow these shortened instructions:
```
conda create -n mdsine2 -c conda-forge python=3.7.3
conda activate mdsine2
git clone https://github.com/gerberlab/MDSINE2
pip install MDSINE2/.
```


## 4. Underlying model and parameters
<p align="center">
<img src="/figures/github1.svg" width="600" />
</p>



## 5. Tutorials

We recommend reading the MDSINE2_Paper repository that has detailed examples for working with MDSINE2 as well as 
example data:
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="161" height="20" role="img" aria-label="GitHub: MDSINE2 Paper"><title>GitHub: MDSINE2 Paper</title><linearGradient id="s" x2="0" y2="100%"><stop offset="0" stop-color="#bbb" stop-opacity=".1"/><stop offset="1" stop-opacity=".1"/></linearGradient><clipPath id="r"><rect width="161" height="20" rx="3" fill="#fff"/></clipPath><g clip-path="url(#r)"><rect width="64" height="20" fill="#555"/><rect x="64" width="97" height="20" fill="#007ec6"/><rect width="161" height="20" fill="url(#s)"/></g><g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" text-rendering="geometricPrecision" font-size="110"><image x="5" y="3" width="14" height="14" xlink:href="data:image/svg+xml;base64,PHN2ZyBmaWxsPSJ3aGl0ZXNtb2tlIiByb2xlPSJpbWciIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48dGl0bGU+R2l0SHViPC90aXRsZT48cGF0aCBkPSJNMTIgLjI5N2MtNi42MyAwLTEyIDUuMzczLTEyIDEyIDAgNS4zMDMgMy40MzggOS44IDguMjA1IDExLjM4NS42LjExMy44Mi0uMjU4LjgyLS41NzcgMC0uMjg1LS4wMS0xLjA0LS4wMTUtMi4wNC0zLjMzOC43MjQtNC4wNDItMS42MS00LjA0Mi0xLjYxQzQuNDIyIDE4LjA3IDMuNjMzIDE3LjcgMy42MzMgMTcuN2MtMS4wODctLjc0NC4wODQtLjcyOS4wODQtLjcyOSAxLjIwNS4wODQgMS44MzggMS4yMzYgMS44MzggMS4yMzYgMS4wNyAxLjgzNSAyLjgwOSAxLjMwNSAzLjQ5NS45OTguMTA4LS43NzYuNDE3LTEuMzA1Ljc2LTEuNjA1LTIuNjY1LS4zLTUuNDY2LTEuMzMyLTUuNDY2LTUuOTMgMC0xLjMxLjQ2NS0yLjM4IDEuMjM1LTMuMjItLjEzNS0uMzAzLS41NC0xLjUyMy4xMDUtMy4xNzYgMCAwIDEuMDA1LS4zMjIgMy4zIDEuMjMuOTYtLjI2NyAxLjk4LS4zOTkgMy0uNDA1IDEuMDIuMDA2IDIuMDQuMTM4IDMgLjQwNSAyLjI4LTEuNTUyIDMuMjg1LTEuMjMgMy4yODUtMS4yMy42NDUgMS42NTMuMjQgMi44NzMuMTIgMy4xNzYuNzY1Ljg0IDEuMjMgMS45MSAxLjIzIDMuMjIgMCA0LjYxLTIuODA1IDUuNjI1LTUuNDc1IDUuOTIuNDIuMzYuODEgMS4wOTYuODEgMi4yMiAwIDEuNjA2LS4wMTUgMi44OTYtLjAxNSAzLjI4NiAwIC4zMTUuMjEuNjkuODI1LjU3QzIwLjU2NSAyMi4wOTIgMjQgMTcuNTkyIDI0IDEyLjI5N2MwLTYuNjI3LTUuMzczLTEyLTEyLTEyIi8+PC9zdmc+"/><text aria-hidden="true" x="415" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" textLength="370">GitHub</text><text x="415" y="140" transform="scale(.1)" fill="#fff" textLength="370">GitHub</text><text aria-hidden="true" x="1115" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" textLength="870">MDSINE2 Paper</text><text x="1115" y="140" transform="scale(.1)" fill="#fff" textLength="870">MDSINE2 Paper</text></g></svg>


## 6. Features and Examples

MDSINE2 has the following core features, implemented as a python package but also interfacable using the command line.
We **strongly** encourage users to start with the [tutorials](#5-tutorials) to get started, using the following sections
as a high-level overview only.
For more detailed specifications on the python functions, refer to the documentation.
For command line specifications, use the `--help` option (e.g. `mdsine2 infer --help`).

### 6.1 Input processing and Visualization

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

### 6.2 MCMC inference using MDSINE2's model.

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

### 6.3 Visualization of posterior distribution from MCMC samples

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

### 6.4 Computation and visualization of forward-simulation of learned gLV model for each OTU.

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

### 6.5 Visualize the phylogenetic placement of OTUs.

We provide a tool to draw phylogenetic placements -- rendered into PDF as local subtrees induced by leaves
within a specified radius of each taxa -- provided by the user (e.g. produced by FastTree + pplacer).

*command-line example:*
```bash
> mdsine2 render-phylogeny --study dataset.pkl --tree my_tree.nhx --output-basepath phylo/ (...)
```
    
### 6.6 Visualize the learned network of interactions

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

### 6.7 Compute and visualize keystoneness metric

As a downstream analysis, we compute "keystoneness" which quantifies the amount of influence that each module has
on the rest of the system.

*command-line example:*
```bash
> mdsine2 evaluate-keystoneness -i fixed_clustering/mcmc.pkl --initial-condition-path initial.tsv (...)
```
