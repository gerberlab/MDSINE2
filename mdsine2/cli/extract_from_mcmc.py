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
from typing import Dict, Tuple

import numpy as np

from .base import CLIModule
import mdsine2 as md2
from mdsine2 import Clustering
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


def extract_perts(mcmc: md2.BaseMCMC, pert_name: str) -> Dict[str, np.ndarray]:
    perts = mcmc.graph.perturbations[pert_name].get_trace_from_disk(section='posterior')
    perts[np.isnan(perts)] = 0.
    return perts


def extract_clustering(mcmc: md2.BaseMCMC) -> Tuple[np.ndarray, np.ndarray]:
    clustering = mcmc.graph[STRNAMES.CLUSTERING_OBJ]
    n_clusters = clustering.n_clusters.get_trace_from_disk(section='posterior')
    coclusters = md2.summary(clustering.coclusters, section='posterior')['mean']
    return n_clusters, coclusters


class ExtractPosteriorCLI(CLIModule):
    def __init__(self, subcommand="extract-posterior"):
        super().__init__(
            subcommand=subcommand,
            docstring=__doc__
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
        out_dir = Path(args.out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)

        """
        To save memory, loop through each MCMC object one-by-one and reload from disk as needed.
        (Note: increases runtime since file pointers must be re-opened several times!)
        """

        # Interactions
        interaction_traces = np.concatenate([
            extract_interactions(md2.BaseMCMC.load(str(mcmc_path)))
            for mcmc_path in mcmc_paths
        ])
        np.save(str(out_dir / 'interactions.npy'), interaction_traces)
        del interaction_traces

        # Growths
        growth_traces = np.concatenate([
            extract_growth(md2.BaseMCMC.load(str(mcmc_path)))
            for mcmc_path in mcmc_paths
        ])
        np.save(str(out_dir / 'growth.npy'), growth_traces)
        del growth_traces

        # Perts
        mcmc0 = md2.BaseMCMC.load(str(mcmc_paths[0]))
        pert_names = []
        for pert in mcmc0.graph.perturbations:
            pert_names.append(pert.name)
        del mcmc0

        perturbations = {
            np.concatenate([
                extract_perts(
                    md2.BaseMCMC.load(str(mcmc_path)), pert_name
                )
                for mcmc_path in mcmc_paths
            ])
            for pert_name in pert_names
        }
        np.savez(str(out_dir / 'perturbations.npy'), **perturbations)
        del perturbations

        n_clusters_all = []
        coclustering_all = []
        total_samples = 0
        for mcmc_path in mcmc_paths:
            n_clusters, coclustering, n_posteriors = extract_clustering(md2.BaseMCMC.load(str(mcmc_path)))
            n_clusters_all.append(n_clusters)
            n_samples = n_clusters.shape[0]
            coclustering_all.append(coclustering * n_samples)
            total_samples += n_samples
        np.savez(str(out_dir / 'n_clusters.npy'), np.concatenate(n_clusters_all))
        np.savez(
            str(out_dir / 'coclusters.npy'),
            np.sum(
                (1 / total_samples) * np.stack(coclustering_all, axis=0),
                axis=0
            )
        )
