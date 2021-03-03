"""
Contains logic for initializing the clustering of an instantiated MDSINE2 model, by first learning perturbations
on individual OTUs (no interactions), then grouping the OTUs by perturbation sign.
"""
import os
from collections import defaultdict
import numpy as np

from mdsine2 import initialize_graph, Study, MDSINE2ModelConfig, run_graph
from mdsine2.pylab import BaseMCMC
from mdsine2.names import STRNAMES

from mdsine2.logger import logger


def sign_str(x: float):
    if x < 0:
        return "-"
    if x > 0:
        return "+"
    else:
        return "0"


def initialize_mdsine_from_perturbations(
        mcmc: BaseMCMC,
        cfg: MDSINE2ModelConfig,
        study: Study,
        n_samples: int=1000,
        burnin: int=0,
        checkpoint: int=500):
    '''
    Trains the model by first learning perturbations (no interactions)
    and divides up the OTUs into clusters based on perturbation magnitudes/signs.
    :param mcmc: The BaseMCMC object
    '''

    # ========= Run a copy of the model with interactions disabled.
    cfg_proxy = cfg.copy()
    cfg_proxy.OUTPUT_BASEPATH = os.path.join(cfg.OUTPUT_BASEPATH, "cluster-init")
    cfg_proxy.N_SAMPLES = n_samples
    cfg_proxy.BURNIN = burnin
    cfg_proxy.CHECKPOINT = checkpoint

    # Disable clustering.
    cfg_proxy.INITIALIZATION_KWARGS[STRNAMES.CLUSTERING]['value_option'] = 'no-clusters'
    cfg_proxy.LEARN[STRNAMES.CLUSTERING] = False
    cfg_proxy.INITIALIZATION_KWARGS[STRNAMES.PERT_INDICATOR_PROB] = {
        'value_option': 'manual',
        'value': 1,
        'hyperparam_option': 'manual',
        'a': 10000,
        'b': 1,
        'delay': 0
    }
    cfg_proxy.LEARN[STRNAMES.PERT_INDICATOR_PROB] = False

    mcmc_proxy = initialize_graph(params=cfg_proxy, graph_name=study.name, subjset=study)
    run_graph(mcmc_proxy, crash_if_error=True)

    # ========= Divide up OTUs based on perturbation signs.
    result_clustering = defaultdict(list)
    otus_with_perturbation_signs = [
        (
            next(iter(cluster.members)),
            "".join([sign_str(pert.magnitude.value[ckey]) for pert in mcmc_proxy.graph.perturbations])
        )
        for ckey, cluster in mcmc_proxy.graph[STRNAMES.CLUSTERING].clustering.clusters.items()
    ]

    for otu, pert_sign in otus_with_perturbation_signs:
        result_clustering[pert_sign].append(otu)

    # ========= Save the result into original chain.
    result_clustering_arr = [
        c for _, c in result_clustering.items()
    ]

    cids = mcmc.graph[STRNAMES.CLUSTERING].value.fromlistoflists(result_clustering_arr)

    # Perturbation initialize to mean value across constituents of cluster.
    for cid, cluster in zip(cids, result_clustering_arr):
        for pert, pert_proxy in zip(mcmc.graph.perturbations, mcmc_proxy.graph.perturbations):
            pert_value = np.mean([
                pert_proxy.magnitude.value[pert_proxy.clustering.idx2cid[taxa_id]] for taxa_id in cluster
            ])
            logger.info("Initializing cluster {} with perturbation ({})={}".format(
                cluster,
                pert.name,
                pert_value
            ))
            pert.magnitude.value[cid] = pert_value
