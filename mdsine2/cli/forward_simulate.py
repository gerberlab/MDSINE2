"""
Run a basic forward simulation using the parameters learned using `mdsine2 infer`.

Author: David Kaplan
Date: 12/01/20
MDSINE2 version: 4.0.6

Input format
------------
There are two different input formats that you can pass in:
    1) MDSINE2.BaseMCMC pickle
    2) Folder of numpy arrays
For most users, passing in the MDSINE2.BaseMCMC file is the way to go. Users
might want to pass in the folder only if they are running jobs in parallel
for this simulation and have many different jobs accessing the data at once.

You can load the MCMC chain from either a `mcmc.pkl` file or from a folder.
If you load it from a folder, then it must have the following structure
folder/
    growth.npy # np.ndarray(n_gibbs, n_taxa)
    interactions.npy # np.ndarray(n_gibbs, n_taxa, n_taxa)
    perturbations.pkl # Optional, dictionary
        (name of perturbation) str -> dict
            'values' -> np.ndarray (n_gibbs, n_taxa)

Forward simulation
------------------
The default simulation is a full time prediction where we start from the first timepoint
and simulate until the last timepoint. Another thing you can do with this is only
forward simulate from a subjset of the times. Specify the start time with
`--start` and the number of days to forward simulate with with `--n-days`. If you
additionally want to save all of the intermediate times within start and end, include
the flag `--save-intermediate-times`.
"""

import argparse
import numpy as np
from pathlib import Path

import mdsine2 as md2
from mdsine2.names import STRNAMES
from mdsine2.logger import logger
from mdsine2.cli.base import CLIModule
from mdsine2.cli.helpers.fwsim_helper import run_forward_sim, plot_fwsim_comparison


class ForwardSimulationCLI(CLIModule):
    def __init__(self, subcommand="forward-simulate"):
        super().__init__(
            subcommand=subcommand,
            docstring=__doc__
        )

    def create_parser(self, parser: argparse.ArgumentParser):
        parser.add_argument('--input-mcmc', '-i', type=str, dest='input_mcmc',
                            required=True,
                            help='Location of input containing MDSINE2.BaseMCMC chain')
        parser.add_argument('--study', type=str, dest='study',
                            required=True,
                            help='The Study pickle object containing input data (including subjects and perturbations)')
        parser.add_argument('--subject', type=str, dest='subject',
                            required=True,
                            help='The subject to use for generating initial conditions from.')
        parser.add_argument('--simulation-dt', type=float, dest='dt',
                            required=False, default=0.01,
                            help='Timesteps we go in during forward simulation')
        parser.add_argument('--limit-of-detection', dest='limit_of_detection',
                            required=False, default=1e5,
                            help='If any of the taxa have a 0 abundance at the start, then we set it to this value.')
        parser.add_argument('--sim-max', dest='sim_max',
                            required=False, default=1e20,
                            help='Maximum value for abundances.')
        parser.add_argument('--output-path', '-o', type=str, dest='out_path',
                            required=True,
                            help='This is where you are saving the posterior forward simulation. (stored in numpy format)')
        parser.add_argument('--gibbs-subsample', type=int,
                            required=False, default=1,
                            help='The number of gibbs samples to skip. A value of n indicates that one out of every '
                                 'n samples will be used.')

        parser.add_argument('--plot', type=str,
                            required=False, default="None",
                            help='If specified, will render plots of chosen taxa to PDF in addition to saving the '
                                 'values to numpy arrays. These plots will be saved to the same directory as the '
                                 'resulting numpy file. (Default: `none`).'
                                 '\nAvailable options: `none`, `all`, `<comma_separated_taxa_names>`.'
                                 '\nExample: `--plot all`, or `--plot OTU_1,OTU_3,OTU_10`')

    def main(self, args: argparse.Namespace):
        out_path = Path(args.out_path)
        out_path.parent.mkdir(exist_ok=True, parents=True)

        study = md2.Study.load(args.study)
        subj_name = args.subject

        try:
            subject = study[subj_name]
        except KeyError:
            logger.error("Unknown subject `{}`. Available subjects: [{}]".format(
                subj_name,
                ",".join([subj.name for subj in study])
            ))
            exit(1)

        mcmc = md2.BaseMCMC.load(args.input_mcmc)
        n_days = np.ceil(subject.times[-1] - subject.times[0])

        limit_of_detection = args.limit_of_detection

        # ======= gLV parameters
        logger.info("Loading gLV parameter values.")
        growth = mcmc.graph[STRNAMES.GROWTH_VALUE].get_trace_from_disk()
        self_interactions = mcmc.graph[STRNAMES.SELF_INTERACTION_VALUE].get_trace_from_disk()
        interactions = mcmc.graph[STRNAMES.INTERACTIONS_OBJ].get_trace_from_disk()
        interactions[np.isnan(interactions)] = 0
        self_interactions = -np.absolute(self_interactions)
        for i in range(self_interactions.shape[1]):
            interactions[:, i, i] = self_interactions[:, i]

        # ======= Perturbations
        if mcmc.graph.perturbations is not None:
            perturbations = []
            perturbations_start = []
            perturbations_end = []
            for pert in mcmc.graph.perturbations:
                logger.info('Loading perturbation `{}`'.format(pert.name))
                pert_value = pert.get_trace_from_disk()
                pert_value[np.isnan(pert_value)] = 0.

                perturbations.append(pert_value)
                perturbations_start.append(pert.starts[subj_name])
                perturbations_end.append(pert.ends[subj_name])
        else:
            logger.info('No perturbations found.')
            perturbations = None
            perturbations_start = []
            perturbations_end = []

        # ======= Initial conditions
        M = subject.matrix()['abs']

        initial_conditions = M[:, 0]
        if np.any(initial_conditions == 0):
            logger.info('{} of {} taxa have abundance zero at the start. Setting to {}'.format(
                np.sum(initial_conditions == 0),
                initial_conditions.shape[0],
                limit_of_detection
            ))
            initial_conditions[initial_conditions == 0] = limit_of_detection
        initial_conditions = initial_conditions.reshape(-1, 1)

        # ======= Perform forward simulations
        gibbs_samples = list(range(0, growth.shape[0], args.gibbs_subsample))
        logger.info("Running forward simulations on {} MCMC samples.".format(len(gibbs_samples)))

        fwsims = []
        for gibbs_idx in gibbs_samples:
            fwsim = run_forward_sim(
                growth=growth[gibbs_idx],
                interactions=interactions[gibbs_idx],
                initial_conditions=initial_conditions,
                perturbations=[p[gibbs_idx] for p in perturbations],
                perturbations_start=perturbations_start,
                perturbations_end=perturbations_end,
                dt=args.dt,
                sim_max=args.sim_max,
                n_days=n_days
            )
            fwsims.append(fwsim)

        fwsims = np.stack(fwsims)
        times = np.array([args.dt * i for i in range(fwsims.shape[-1])]) + subject.times[0]
        np.save(str(out_path), fwsims)
        logger.info("Saved forward simulations to {}.".format(str(out_path)))

        if args.plot.lower() == "all":
            taxa_to_plot = list(study.taxa)
        elif args.plot.lower() == "none":
            taxa_to_plot = []
        else:
            tokens = args.plot.split(",")
            taxa_to_plot = [
                study.taxa[token.strip()] for token in tokens
            ]

        if len(taxa_to_plot) > 0:
            logger.info("Plotting {} taxa.".format(len(taxa_to_plot)))
            for taxa in taxa_to_plot:
                plot_out_path = out_path.parent / "fwsim_{}.pdf".format(taxa.name)
                plot_fwsim_comparison(
                    taxa=taxa,
                    taxa_trajectory=fwsims[:, taxa.idx, :],
                    trajectory_times=times,
                    subject=subject,
                    out_path=plot_out_path,
                    mcmc_display_method="quantiles",
                    figsize=(10, 8),
                    ylim=(1e5, 1e12)
                )
