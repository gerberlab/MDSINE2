"""
Run keystoneness analysis using the specified MDSINE2 output.
"""

import argparse
import csv
from typing import List, Union

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from mdsine2 import BaseMCMC

import mdsine2 as md2
from mdsine2.names import STRNAMES
from .base import CLIModule
from .helpers.fwsim_helper import run_forward_sim

from mdsine2.logger import logger


class KeystonenessCLI(CLIModule):
    def __init__(self, subcommand="evaluate-keystoneness"):
        super().__init__(
            subcommand=subcommand,
            docstring=__doc__
        )

    def create_parser(self, parser: argparse.ArgumentParser):
        # Inputs
        parser.add_argument('--fixed-cluster-mcmc-path', '-f', type=str, dest='mcmc_path',
                            required=True,
                            help='Path of saved MDSINE2.BaseMCMC chain (fixed-clustering inference)')
        parser.add_argument('--study', '-s', dest='study', type=str, required=True,
                            help="The path to the relevant Study object containing the input data (subjects, taxa).")

        # Optional:
        parser.add_argument('--initial-conditions', '-i', type=str, dest='initial_condition_path',
                            required=True,
                            help='The path to a file specifying the initial conditions. File will be interpreted as a '
                                 'two-column TSV file (Taxa name, Absolute abundance).')

        # Outputs
        parser.add_argument('--output-path', '-o', type=str, dest='out_path',
                            required=True,
                            help='This is where you are saving the posterior renderings')

        # Simulation params
        parser.add_argument('--n-days', type=int, dest='n_days', required=False,
                            help='Total number of days to simulate for', default=180)
        parser.add_argument('--simulation-dt', '-dt', type=float, dest='dt', required=False,
                            help='Timesteps we go in during forward simulation', default=0.01)
        parser.add_argument('--sim-max', dest='sim_max', type=float, required=False,
                            help='Maximum value of abundance.', default=1e20)
        parser.add_argument('--limit-of-detection', dest='limit_of_detection', required=False,
                            help='If any of the taxa have a 0 abundance at the start, then we ' \
                                 'set it to this value.', default=1e5, type=float)

    def main(self, args: argparse.Namespace):
        study = md2.Study.load(args.study)
        mcmc = md2.BaseMCMC.load(args.mcmc_path)

        logger.info(f"Loading initial conditions from {args.initial_condition_path}")
        initial_conditions_master = load_initial_conditions(study, args.initial_condition_path)

        out_path = Path(args.out_path)
        out_path.parent.mkdir(exist_ok=True, parents=True)

        df_entries = []

        # Baseline
        compute_keystoneness_of_cluster(
            mcmc,
            None,
            initial_conditions_master,
            args.n_days,
            args.dt,
            args.sim_max,
            df_entries
        )

        # Cluster exclusion
        for cluster_idx, cluster in enumerate(mcmc.graph[STRNAMES.CLUSTERING_OBJ]):
            initial_conditions = exclude_cluster_from(initial_conditions_master, cluster)
            compute_keystoneness_of_cluster(
                mcmc,
                cluster_idx,
                initial_conditions,
                args.n_days,
                args.dt,
                args.sim_max,
                df_entries
            )

        df = pd.DataFrame(df_entries)
        del df_entries

        df.to_csv(str(out_path), sep='\t')


def exclude_cluster_from(initial_conditions_master: np.ndarray, cluster):
    initial_conditions = np.copy(initial_conditions_master)
    for oidx in cluster.members:
        initial_conditions_master[oidx] = 0.0
    return initial_conditions


def compute_keystoneness_of_cluster(
        mcmc: BaseMCMC,
        cluster_idx: Union[int, None],
        initial_conditions: np.ndarray,
        n_days: int,
        dt: float,
        sim_max: float,
        df_entries: List,
):
    taxa = mcmc.graph.data.taxa

    # forward simulate and add results to dataframe.
    if cluster_idx is None:
        tqdm_disp = "Keystoneness Simulations (Baseline)"
    else:
        tqdm_disp = f"Keystoneness Simulations (Cluster {cluster_idx})"

    for gibbs_idx, fwsim in tqdm(do_fwsims(
        mcmc, initial_conditions, n_days, dt, sim_max
    ), total=mcmc.n_samples, desc=tqdm_disp):
        for entry in fwsim_entries(taxa, cluster_idx, fwsim, gibbs_idx):
            df_entries.append(entry)


def fwsim_entries(taxa, excluded_cluster_idx, fwsim, gibbs_idx):
    stable_states = fwsim[:, -50:].mean(axis=1)
    for otu in taxa:
        yield {
            "ExcludedCluster": str(excluded_cluster_idx),
            "OTU": otu.name,
            "SampleIdx": gibbs_idx,
            "StableState": stable_states[otu.idx]
        }


def do_fwsims(mcmc,
              initial_conditions: np.ndarray,
              n_days,
              dt: float,
              sim_max
              ):

    # Forward simulate if necessary
    # -----------------------------
    logger.info('Forward simulating')

    # Load the rest of the parameters
    growth = mcmc.graph[STRNAMES.GROWTH_VALUE].get_trace_from_disk(section="posterior")
    self_interactions = mcmc.graph[STRNAMES.SELF_INTERACTION_VALUE].get_trace_from_disk(section="posterior")
    interactions = mcmc.graph[STRNAMES.INTERACTIONS_OBJ].get_trace_from_disk(section="posterior")
    interactions[np.isnan(interactions)] = 0
    self_interactions = -np.absolute(self_interactions)
    for i in range(self_interactions.shape[1]):
        interactions[:, i, i] = self_interactions[:, i]

    num_samples = mcmc.n_samples

    # Do the forward sim.
    for gibb in range(num_samples):
        gibbs_step_sim = run_forward_sim(
            growth=growth[gibb],
            interactions=interactions[gibb],
            initial_conditions=initial_conditions.reshape(-1, 1),
            perturbations=None,
            perturbations_start=[],
            perturbations_end=[],
            dt=dt,
            sim_max=sim_max,
            n_days=n_days
        )
        yield gibb, gibbs_step_sim


def load_initial_conditions(study: md2.Study, initial_condition_path: str) -> np.ndarray:
    taxa_to_abundance = {}
    with open(initial_condition_path, "r") as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for row in rd:
            if len(row) != 2:
                raise ValueError(
                    "Input file for initial_condition_path must be a two-column format. "
                    "Found {} columns instead.".format(
                        len(row)
                    )
                )
            taxa_to_abundance[row[0]] = float(row[1])

    abundances = np.zeros(len(study.taxa), dtype=np.float)
    for tidx, taxa in enumerate(study.taxa):
        try:
            abundances[tidx] = taxa_to_abundance[taxa.name]
        except KeyError:
            raise KeyError("Could not find initial condition value for taxa `{}`.".format(taxa.name))
    return abundances.reshape(-1, 1)
