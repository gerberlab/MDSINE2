import mdsine2 as md2
import numpy as np
import os
import pandas as pd
from mdsine2.names import STRNAMES
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("agg")


def run_inference(datasetloc, outputloc, nburnin, nsamp, perturbation_on=True, time_mask_on=True):
    study_name = "test"  

    # load dataset
    if perturbation_on is True:
        study = md2.dataset.parse(name=study_name,
                                  taxonomy=os.path.join(datasetloc, 'taxonomy.tsv'),
                                  reads=os.path.join(datasetloc, f'reads.tsv'),
                                  qpcr=os.path.join(datasetloc, f'qpcr.tsv'),
                                  metadata=os.path.join(datasetloc, f'meta.tsv'),
                                  perturbations=os.path.join(datasetloc, "perturbations.tsv"))
    else:
        study = md2.dataset.parse(name=study_name,
                                  taxonomy=os.path.join(datasetloc, 'taxonomy.tsv'),
                                  reads=os.path.join(datasetloc, f'reads.tsv'),
                                  qpcr=os.path.join(datasetloc, f'qpcr.tsv'),
                                  metadata=os.path.join(datasetloc, f'meta.tsv'))

    results_dir = outputloc

    # negbin dispersion parameters
    a0 = 1e-10
    a1 = 5e-2

    basepath = outputloc

    params = md2.config.MDSINE2ModelConfig(
        basepath=str(basepath),
        seed=0,
        burnin=nburnin,
        n_samples=nsamp,
        negbin_a0=a0, negbin_a1=a1,
        checkpoint=1
    )

    # initialize with no clusters
    params.INITIALIZATION_KWARGS[STRNAMES.CLUSTERING]['value_option'] = 'no-clusters'

    # zero-inflation options
    if time_mask_on is True:
        params.INITIALIZATION_KWARGS[STRNAMES.ZERO_INFLATION]['value_option'] = "custom"
        params.ZERO_INFLATION_TRANSITION_POLICY = 'ignore'
        params.ZERO_INFLATION_DATA_PATH = os.path.join(datasetloc, "time_mask.tsv")


    # initilize the graph
    mcmc = md2.initialize_graph(params=params, graph_name=study.name, subjset=study)

    # perform inference
    mcmc = md2.run_graph(mcmc, crash_if_error=True)


if __name__ == "__main__":
    def create_dir(filename):
        if not os.path.exists(filename):
            os.mkdir(filename)
            print("Directory " , filename ,  " Created ")
        else:    
            print("Directory " , filename ,  " already exists")
        return

    pert = False
    tmask = True
    n_burnin = 100
    n_samp = 500

    dataset_path = "./example_data/" 
    output_path = "./inference_results/"
    create_dir(output_path)

    run_inference(dataset_path, output_path, n_burnin, n_samp, perturbation_on=pert, time_mask_on=tmask)
