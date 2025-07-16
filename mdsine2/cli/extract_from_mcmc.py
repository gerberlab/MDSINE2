"""
Run MDSINE2 inference

Author: David Kaplan
Date: 11/30/20
MDSINE2 version: 4.0.6

Fixed Clustering
----------------
To run inference with fixed clustering, use the parameter `--fixed-clustering` where
this is the location of the MCMC object ob the inference. This will automatically set the
parameters for the clustering intialization and set learning turned off for the
cluster assignments.

Priors on the sparsity
----------------------
Set the sparsity of the prior indicator of the interactions and perturbations with the
arguments `--interaction-ind-prior` and `perturbation-ind-prior`.
"""

import argparse
from pathlib import Path
from typing import Tuple, List
import scipy.stats

import numpy as np
from mdsine2.pylab.inference import TraceNotFoundException
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

from .base import CLIModule
import mdsine2 as md2
from mdsine2.logger import logger
from mdsine2.names import STRNAMES


def extract_interactions(mcmc: md2.BaseMCMC) -> np.ndarray:
    interactions = mcmc.graph[STRNAMES.INTERACTIONS_OBJ].get_trace_from_disk(section='posterior')
    interactions[np.isnan(interactions)] = 0

    self_interactions = mcmc.graph[STRNAMES.SELF_INTERACTION_VALUE].get_trace_from_disk(section='posterior')
    self_interactions = -np.absolute(self_interactions)

    for i in range(self_interactions.shape[1]):
        interactions[:, i, i] = self_interactions[:, i]
    return interactions


def extract_growth(mcmc: md2.BaseMCMC) -> np.ndarray:
    return mcmc.graph[STRNAMES.GROWTH_VALUE].get_trace_from_disk(section='posterior')


def extract_perts(mcmc: md2.BaseMCMC, pert_name: str) -> np.ndarray:
    perts = mcmc.graph.perturbations[pert_name].get_trace_from_disk(section='posterior')
    perts[np.isnan(perts)] = 0.
    return perts


def extract_concentrations(mcmc: md2.BaseMCMC) -> np.ndarray:
    return mcmc.graph[STRNAMES.CONCENTRATION].get_trace_from_disk(section='posterior')


def extract_process_variance(mcmc: md2.BaseMCMC) -> np.ndarray:
    return mcmc.graph[STRNAMES.PROCESSVAR].get_trace_from_disk(section='posterior')


def extract_clustering(mcmc: md2.BaseMCMC) -> Tuple[np.ndarray, np.ndarray]:
    clustering = mcmc.graph[STRNAMES.CLUSTERING_OBJ]
    n_clusters = clustering.n_clusters.get_trace_from_disk(section='posterior')
    coclusters = clustering.coclusters.get_trace_from_disk(section='posterior')
    return n_clusters, coclusters


def compute_r_hat(samples: List[np.ndarray]) -> np.ndarray:
    """
    This implements the "split R-hat" from Gelman's BDA3.
    (originally published in Vehtari et al 2019: https://arxiv.org/pdf/1903.08008.pdf)
    """

    # dim 0 enumerates different chains, dim1 enumerates sample in each chain.
    # shape (M, N, *)
    samples = np.stack(samples, axis=0)

    # Split each chain in half.
    n = samples.shape[1]
    first_half = samples[:, :(n // 2)]
    second_half = samples[:, (n // 2):]
    samples = np.concatenate([first_half, second_half], axis=0)

    # Compute B and W.
    samples_per_chain = samples.shape[1]
    within_chain_mean = np.mean(samples, axis=1)  # shape (M, *)
    B = samples_per_chain * np.var(within_chain_mean, ddof=1, axis=0)  # shape (*), note the "ddof=1" for sample var.
    W = np.mean(
        np.var(samples, ddof=1, axis=1),  # shape (M), note the "ddof=1" for sample var.
        axis=0
    )  # shape (*)

    wt = 1 / samples_per_chain
    chain_var = ((1 - wt) * W) + (wt * B)
    return np.sqrt(chain_var / W)


class ExtractPosteriorCLI(CLIModule):
    """
    A command-line utility tool to extract the posterior distribution MCMC samples from (one or possibly multiple-seeded) MDSINE2 output.
    Outputs the MCMC samples as a large numpy array, and calculates R-hat values for growth rates, process variance, and concentration parameters.
    """
    def __init__(self, subcommand="extract-posterior"):
        super().__init__(
            subcommand=subcommand,
            docstring=self.__doc__
        )

    def create_parser(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            '--input', '-i', type=str, action='append', dest='input',
            required=True,
            help='Path to an MCMC pickle (repeat to merge traces from multiple MCMC pickle files).'
        )
        parser.add_argument(
            '--out-dir', '-o', type=str, dest='out_dir', required=True,
            help='The path to output to.'
        )

    def main(self, args: argparse.Namespace):
        mcmc_paths = [Path(x) for x in args.input]
        logger.info("Loading from:")
        for mcmc_path in mcmc_paths:
            logger.info(mcmc_path)

        out_dir = Path(args.out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)

        """
        To save memory, loop through each MCMC object one-by-one and reload from disk as needed.
        (Note: increases runtime since file pointers must be re-opened several times!)
        """

        # Interactions
        interaction_traces = np.concatenate([
            extract_interactions(md2.BaseMCMC.load(str(mcmc_path)))
            for mcmc_path in tqdm(mcmc_paths, desc='Interactions')
        ])
        np.save(str(out_dir / 'interactions.npy'), interaction_traces)
        del interaction_traces

        # Growths
        growth_traces = [
            extract_growth(md2.BaseMCMC.load(str(mcmc_path)))
            for mcmc_path in tqdm(mcmc_paths, desc='Growth Rates')
        ]
        r_hats = compute_r_hat(growth_traces)
        np.save(str(out_dir / 'growth.npy'), np.concatenate(growth_traces))
        np.save(str(out_dir / 'growth_rhat.npy'), r_hats)
        del growth_traces
        del r_hats

        # Concentration parameter
        conc_params = [
            extract_concentrations(md2.BaseMCMC.load(str(mcmc_path)))
            for mcmc_path in tqdm(mcmc_paths, desc='Concentrations')
        ]
        r_hat = compute_r_hat(conc_params)
        np.save(str(out_dir / 'concentration_rhat.npy'), r_hat)
        del conc_params
        del r_hat

        # Process Variance parameter
        procvar_params = [
            extract_process_variance(md2.BaseMCMC.load(str(mcmc_path)))
            for mcmc_path in tqdm(mcmc_paths, desc='Process Variance')
        ]
        r_hat = compute_r_hat(procvar_params)
        np.save(str(out_dir / 'procvar_rhat.npy'), r_hat)
        del procvar_params
        del r_hat

        # Perts
        mcmc0 = md2.BaseMCMC.load(str(mcmc_paths[0]))
        pert_names = []
        for pert in mcmc0.graph.perturbations:
            pert_names.append(pert.name)
        print("Found perturbations: {}".format(pert_names))
        del mcmc0

        perturbations = {
            pert_name: np.concatenate([
                extract_perts(
                    md2.BaseMCMC.load(str(mcmc_path)),
                    pert_name
                )
                for mcmc_path in tqdm(mcmc_paths, desc=f'Perturbation ({pert_name})')
            ])
            for pert_name in pert_names
        }
        pert_rhats = {
            pert_name: compute_r_hat([
                extract_perts(
                    md2.BaseMCMC.load(str(mcmc_path)),
                    pert_name
                )
                for mcmc_path in tqdm(mcmc_paths, desc=f'Perturbation ({pert_name}) R-hat')
            ])
            for pert_name in pert_names
        }
        np.savez(str(out_dir / 'perturbations.npz'), **perturbations)
        np.savez(str(out_dir / 'perturbations_rhat.npz'), **pert_rhats)
        del perturbations

        # Coclustering
        try:
            n_clusters_all = []
            coclustering_all = []
            for mcmc_path in tqdm(mcmc_paths, desc='Clustering'):
                n_clusters, coclustering = extract_clustering(md2.BaseMCMC.load(str(mcmc_path)))
                n_clusters_all.append(n_clusters)
                coclustering_all.append(coclustering)

            n_clusters_all = np.concatenate(n_clusters_all)
            coclustering_all = np.concatenate(coclustering_all, axis=0)
            np.save(str(out_dir / 'n_clusters.npy'), n_clusters_all)
            np.save(str(out_dir / 'coclusters.npy'), coclustering_all)

            # Agglomerated modules
            A = 1 - np.mean(coclustering_all, axis=0)
            n = scipy.stats.mode(n_clusters_all)[0]
            if isinstance(n, np.ndarray):  # for scipy backwards compatibility, older versions "n" is still an array.
                n = n[0]
            linkage = 'complete'
            c = AgglomerativeClustering(
                n_clusters=n,
                affinity='precomputed',
                linkage=linkage
            )
            agglom = c.fit_predict(A)
            np.save(str(out_dir / "n_clusters.npy"), n_clusters_all)
            np.save(str(out_dir / "agglomeration.npy"), agglom)
        except TraceNotFoundException:
            print("Trace for clustering doesn't exist. Skipping.")
            pass
