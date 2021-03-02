"""
Contains logic for initializing the clustering of an instantiated MDSINE2 model, by first learning perturbations
on individual OTUs (no interactions), then grouping the OTUs by perturbation sign.
"""
import os
from collections import defaultdict

from mdsine2 import initialize_graph, Study, MDSINE2ModelConfig, run_graph
from mdsine2.pylab import BaseMCMC
from mdsine2.names import STRNAMES


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
    mcmc.graph[STRNAMES.CLUSTERING].value.fromlistoflists([
        c for _, c in result_clustering.items()
    ])
