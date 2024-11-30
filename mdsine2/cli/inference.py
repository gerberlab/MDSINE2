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

import os
import time
import argparse
from .base import CLIModule
import mdsine2 as md2
from mdsine2.logger import logger
from mdsine2.names import STRNAMES


class InferenceCLI(CLIModule):
    def __init__(self, subcommand="infer"):
        super().__init__(
            subcommand=subcommand,
            docstring=__doc__
        )

    def create_parser(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            '--input', '-i', type=str, dest='input',
            required=True,
            help='This is the dataset to do inference with.'
        )
        parser.add_argument(
            '--fixed-clustering', type=str, dest='fixed_clustering',
            required=False, default=None,
            help='Specify a file with this argument to run fixed-clustering mode inference.'
                 'If extension is .pkl, then the argument will be treated as an MCMC inference using normal-mode inference, from which consensus clusters will be computed.'
                 'If extension is .npy, then the argument will be treated as a clustering numpy array, meaning a (N_TAXA)-length array of integers. Each taxa will be assigned a cluster ID grouped by these integer values.'
        )
        parser.add_argument(
            '--nomodules', action='store_true', dest='nomodules',
            help='If flag is provided, then run inference without learning modules.'
        )
        parser.add_argument(
            '--negbin', type=str, dest='negbin', nargs='+',
            required=True,
            help='If there is a single argument, then this is the MCMC object that was run to ' \
                 'learn a0 and a1. If there are two arguments passed, these are the a0 and a1 ' \
                 'of the negative binomial dispersion parameters. Example: ' \
                 '--negbin /path/to/negbin/mcmc.pkl. Example: ' \
                 '--negbin 0.0025 0.025'
        )
        parser.add_argument(
            '--seed', '-s', type=int, dest='seed',
            required=True,
            help='This is the seed to initialize the inference with'
        )
        parser.add_argument(
            '--burnin', '-nb', type=int, dest='burnin',
            required=True,
            help='How many burn-in Gibb steps for Markov Chain Monte Carlo (MCMC)'
        )
        parser.add_argument(
            '--n-samples', '-ns', type=int, dest='n_samples',
            required=True,
            help='Total number Gibb steps to perform during MCMC inference'
        )
        parser.add_argument(
            '--checkpoint', '-c', type=int, dest='checkpoint',
            required=True,
            help='How often to write the posterior to disk. Note that `--burnin` and ' \
                 '`--n-samples` must be a multiple of `--checkpoint` (e.g. checkpoint = 100, ' \
                 'n_samples = 600, burnin = 300)'
        )
        parser.add_argument(
            '--basepath', '--output-basepath', '-b', type=str, dest='basepath',
            required=True,
            help='This is folder to save the output of inference'
        )
        parser.add_argument(
            '--multiprocessing', '-mp', type=int, dest='mp',
            help='If 1, run the inference with multiprocessing. Else run on a single process',
            default=0
        )
        parser.add_argument(
            '--rename-study', type=str, dest='rename_study',
            required=False, default=None,
            help='Specify the name of the study to set'
        )
        parser.add_argument(
            '--interaction-ind-prior', '-ip', type=str, dest='interaction_prior',
            required=True,
            help='Prior of the indicator of the interactions.'
        )
        parser.add_argument(
            '--perturbation-ind-prior', '-pp', type=str, dest='perturbation_prior',
            required=True,
            help='Prior of the indicator of the perturbations'
        )

        parser.add_argument(
            '--log-every', type=int, default=100, dest='log_every',
            required=False,
            help='<Optional> Tells the inference loop to print debug messages every k iterations.'
        )
        parser.add_argument(
            '--benchmark', action='store_true', dest='benchmark',
            help='If flag is set, then logs (at INFO level) the update() runtime of each component at the end.'
        )

        parser.add_argument(
            '--interaction-mean-loc', type=float, dest='interaction_mean_loc',
            required=False, help='The loc parameter for the interaction strength prior mean.', default=0.0
        )
        parser.add_argument(
            '--interaction-var-dof', type=float, dest='interaction_var_dof',
            required=False, help='The dof parameter for the interaction strength prior var.', default=2.01
        )
        parser.add_argument(
            '--interaction-var-rescale', type=float, dest='interaction_var_rescale',
            required=False,
            help='Controls the scale parameter for the interaction strength prior var, using the formula [SCALE]*E^2',
            default=1.0
        )
        parser.add_argument(
            '-r', '--resume',
            required=False, default=False, action='store_true',
            help='If set, tries to check for an existing MCMC trace and resume from where it left off.'
        )

    def main(self, args: argparse.Namespace):
        # 1) load dataset
        logger.info('Loading dataset {}'.format(args.input))
        study = md2.Study.load(args.input)
        if args.rename_study is not None:
            if args.rename_study.lower() != 'none':
                study.name = args.rename_study
        md2.seed(args.seed)

        # 2) Load the model parameters
        os.makedirs(args.basepath, exist_ok=True)
        basepath = os.path.join(args.basepath, study.name)
        os.makedirs(basepath, exist_ok=True)

        # Load the negative binomial parameters
        if len(args.negbin) == 1:
            negbin = md2.BaseMCMC.load(args.negbin[0])
            a0 = md2.summary(negbin.graph[STRNAMES.NEGBIN_A0])['mean']
            a1 = md2.summary(negbin.graph[STRNAMES.NEGBIN_A1])['mean']

        elif len(args.negbin) == 2:
            a0 = float(args.negbin[0])
            a1 = float(args.negbin[1])
        else:
            raise ValueError('Argument `negbin`: there must be only one or two arguments.')

        logger.info('Setting a0 = {:.4E}, a1 = {:.4E}'.format(a0, a1))

        # 3) Begin inference
        params = md2.config.MDSINE2ModelConfig(
            basepath=basepath, seed=args.seed,
            burnin=args.burnin, n_samples=args.n_samples, negbin_a1=a1,
            negbin_a0=a0, checkpoint=args.checkpoint)
        # Run with multiprocessing if necessary
        if args.mp:
            params.MP_FILTERING = 'full'
            params.MP_CLUSTERING = 'full-4'

        # Change parameters if there is fixed clustering
        if args.fixed_clustering and args.nomodules:
            logger.error("Can't use both `fixed_clustering` and `nomodules` mode; only one can be chosen at a time.")
            exit(1)
        if args.fixed_clustering:
            params.LEARN[STRNAMES.CLUSTERING] = False
            params.LEARN[STRNAMES.CONCENTRATION] = False
            params.INITIALIZATION_KWARGS[STRNAMES.CLUSTERING]['value_option'] = 'fixed-clustering'
            params.INITIALIZATION_KWARGS[STRNAMES.CLUSTERING]['value'] = args.fixed_clustering
            params.INITIALIZATION_KWARGS[STRNAMES.CLUSTER_INTERACTION_INDICATOR_PROB]['N'] = 'fixed-clustering'
            params.INITIALIZATION_KWARGS[STRNAMES.PERT_INDICATOR_PROB]['N'] = 'fixed-clustering'
        elif args.nomodules:
            params.LEARN[STRNAMES.CLUSTERING] = False
            params.LEARN[STRNAMES.CONCENTRATION] = False
            params.INITIALIZATION_KWARGS[STRNAMES.CLUSTERING]['value_option'] = 'no-clusters'
            params.INITIALIZATION_KWARGS[STRNAMES.CLUSTER_INTERACTION_INDICATOR_PROB]['N'] = 'fixed-clustering'
            params.INITIALIZATION_KWARGS[STRNAMES.PERT_INDICATOR_PROB]['N'] = 'fixed-clustering'

        # Set the sparsities
        params.INITIALIZATION_KWARGS[STRNAMES.CLUSTER_INTERACTION_INDICATOR_PROB]['hyperparam_option'] = \
            args.interaction_prior
        params.INITIALIZATION_KWARGS[STRNAMES.PERT_INDICATOR_PROB]['hyperparam_option'] = \
            args.perturbation_prior

        # Set interaction str priors
        if args.interaction_mean_loc != 0.0:
            params.INITIALIZATION_KWARGS[STRNAMES.PRIOR_MEAN_INTERACTIONS]['loc_option'] = 'manual'
            params.INITIALIZATION_KWARGS[STRNAMES.PRIOR_MEAN_INTERACTIONS]['loc'] = args.interaction_mean_loc

        if args.interaction_var_dof != None:
            params.INITIALIZATION_KWARGS[STRNAMES.PRIOR_VAR_INTERACTIONS]['dof_option'] = 'manual'
            params.INITIALIZATION_KWARGS[STRNAMES.PRIOR_VAR_INTERACTIONS]['dof'] = args.interaction_var_dof

        params.INITIALIZATION_KWARGS[STRNAMES.PRIOR_VAR_INTERACTIONS]['scale_option'] = 'inflated-median'
        params.INITIALIZATION_KWARGS[STRNAMES.PRIOR_VAR_INTERACTIONS]['inflation_factor'] = 1e4 * args.interaction_var_rescale

        # Change the cluster initialization to no clustering if there are less than 30 clusters
        if len(study.taxa) <= 30:
            logger.info(
                'Since there is less than 30 taxa, we set the initialization of the clustering to `no-clusters`')
            params.INITIALIZATION_KWARGS[STRNAMES.CLUSTERING]['value_option'] = 'no-clusters'

        # Try to see if we should resume.
        if args.resume:
            from pathlib import Path
            from mdsine2 import BaseMCMC

            # Check for existing pickle file. If not, run in default mode.
            target_pickle_file = Path(params.MODEL_PATH) / "mcmc.pkl"
            if target_pickle_file.exists():
                mcmc = BaseMCMC.load(str(target_pickle_file))
                growth_posterior = mcmc.graph[STRNAMES.GROWTH_VALUE].get_trace_from_disk(section='posterior')
                n_samples_done = growth_posterior.shape[0]
                resume_from_mcmc_index = n_samples_done
                del mcmc
                del growth_posterior
            else:
                resume_from_mcmc_index = None


        mcmc = md2.initialize_graph(params=params, graph_name=study.name, subjset=study, continue_inference=resume_from_mcmc_index)
        mdata_fname = os.path.join(params.MODEL_PATH, 'metadata.txt')
        params.make_metadata_file(fname=mdata_fname)

        start_time = time.time()
        mcmc = md2.run_graph(mcmc, crash_if_error=True, log_every=args.log_every, benchmarking=args.benchmark)

        # Record how much time inference took
        t = time.time() - start_time
        t = t / 3600  # Convert to hours

        f = open(mdata_fname, 'a')
        f.write('\n\nTime for inference: {} hours'.format(t))
        f.close()
