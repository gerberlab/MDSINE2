"""
Run keystoneness analysis using the specified MDSINE2 output.
"""

import argparse
import csv
from typing import List

import numpy as np
import pandas as pd
from pathlib import Path

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
        parser.add_argument('--fixed-clustering-mcmc', '-i', type=str, dest='mcmc_path',
                            required=True,
                            help='Path of saved MDSINE2.BaseMCMC chain (fixed-clustering inference)')
        parser.add_argument('--study', type=str, required=True,
                            help="The path to the relevant Study object containing the input data (subjects, taxa).")
        parser.add_argument('--initial-condition-path', type=str, dest='initial_condition_path',
                            required=True,
                            help='The path to a file specifying the initial conditions. File will be interpreted as a '
                                 'two-column TSV file (Taxa name, Absolute abundance).')

        # Outputs
        parser.add_argument('--output-dir', '-o', type=str, dest='out_dir',
                            required=True,
                            help='This is where you are saving the posterior renderings')

        # Simulation params
        parser.add_argument('--n-days', type=int, dest='n_days', required=False,
                            help='Total umber of days to simulate for', default=180)
        parser.add_argument('--simulation-dt', type=float, dest='dt', required=False,
                            help='Timesteps we go in during forward simulation', default=0.01)
        parser.add_argument('--sim-max', dest='sim_max', type=float, required=False,
                            help='Maximum value', default=1e20)

    def main(self, args: argparse.Namespace):
        mcmc = md2.BaseMCMC.load(args.mcmc_path)
        study = md2.Study.load(args.study)
        initial_conditions = load_initial_conditions(study, args.initial_condition_path)

        clustering = mcmc.graph[STRNAMES.CLUSTERING_OBJ]
        growth = mcmc.graph[STRNAMES.GROWTH_VALUE].get_trace_from_disk(section="posterior")
        self_interactions = mcmc.graph[STRNAMES.SELF_INTERACTION_VALUE].get_trace_from_disk(section="posterior")
        interactions = mcmc.graph[STRNAMES.INTERACTIONS_OBJ].get_trace_from_disk(section="posterior")
        interactions[np.isnan(interactions)] = 0
        self_interactions = -np.absolute(self_interactions)
        for i in range(self_interactions.shape[1]):
            interactions[:, i, i] = self_interactions[:, i]

        n_samples = growth.shape[0]

        # Baseline run (no taxa excluded)
        logger.info("Evaluating baseline forward-sim.")
        baseline_steady_state = run_forward_simulations(n_samples,
                                                        initial_conditions,
                                                        growth, interactions,
                                                        args.dt, args.sim_max, args.n_days)

        # Exclude one taxa at a time.
        altered_steady_states = []
        for cidx, cluster in enumerate(clustering):
            logger.info("Evaluating forward-sim with cluster #{} (ID `{}`) excluded.".format(
                cidx + 1,
                cluster.id
            ))
            altered_initial_conditions = np.copy(initial_conditions)
            for tidx in cluster.members:
                altered_initial_conditions[tidx] = 0.0
            altered_steady_state = run_forward_simulations(n_samples,
                                                           altered_initial_conditions,
                                                           growth, interactions,
                                                           args.dt, args.sim_max, args.n_days)
            altered_steady_states.append(altered_steady_state)

        out_dir = Path(args.out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)

        logger.info("Aggregating data. (Output dir = {}).".format(str(out_dir)))
        aggregate_dataframes(study, clustering, baseline_steady_state, altered_steady_states, out_dir)


def run_forward_simulations(n_samples,
                            initial_conditions,
                            growth,
                            interactions,
                            dt,
                            sim_max,
                            n_days) -> np.ndarray:
    steady_states = []
    for sample_idx in range(n_samples):
        traj = run_forward_sim(
            growth=growth[sample_idx],
            interactions=interactions[sample_idx],
            initial_conditions=initial_conditions,
            perturbations=None, perturbations_start=[], perturbations_end=[],
            dt=dt,
            sim_max=sim_max,
            n_days=n_days,
        )
        steady_states.append(traj[:, -50:].mean(axis=1))
    return np.stack(steady_states)


def aggregate_dataframes(study, clustering,
                         baseline_steady_state: np.ndarray,
                         altered_steady_states: List[np.ndarray],
                         out_dir: Path):
    # Convert to pandas and store to disk.
    baseline_df = pd.DataFrame([
        {
            "Taxa": taxa.name,
            "SampleIdx": sample_idx,
            "SteadyState": baseline_steady_state[sample_idx, tidx]
        }
        for sample_idx in baseline_steady_state.shape[0]
        for tidx, taxa in enumerate(study.taxa)
    ])

    altered_df = pd.DataFrame([
        {
            "ExcludedCluster": cluster.cidx + 1,
            "Taxa": taxa.name,
            "SampleIdx": sample_idx,
            "SteadyState": altered_steady_states[cidx][sample_idx, tidx]
        }
        for sample_idx in baseline_steady_state.shape[0]
        for tidx, taxa in enumerate(study.taxa)
        for cidx, cluster in enumerate(clustering)
        if taxa.idx not in cluster.members
    ])

    baseline_df.to_csv(out_dir / "baseline_sim.csv")
    altered_df.to_csv(out_dir / "cluster_exclusion_sim.csv")

    # Compute keystoneness.
    merged_df = altered_df.merge(
        baseline_df,
        left_on=["SampleIdx", "Taxa"],
        right_on=["SampleIdx", "Taxa"],
        suffixes=["", "Base"],
        how="inner"
    )

    eps = 1e5
    merged_df["SteadyStateDiff"] = np.log10(merged_df["SteadyState"] + eps) - np.log10(
        merged_df["SteadyStateBase"] + eps)

    ky_df = merged_df[["ExcludedCluster", "SampleIdx", "SteadyStateDiff"]].groupby(
        ["ExcludedCluster", "SampleIdx"]  # Aggregate over taxa
    ).mean().groupby(
        level=0  # Aggregate over MCMC samples
    ).mean()

    ky_df.rename(columns={
        "SteadyStateDiff": "Keystoneness"
    })
    ky_df.to_csv(out_dir / "keystoneness.csv")


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
