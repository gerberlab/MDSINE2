"""
Learn the Negative Binomial dispersion parameters that parameterize the noise
of the reads. We learn these parameters with replicate measurements of the reads.

Author: David Kaplan
Date: 11/30/20
MDSINE2 version: 4.0.6
"""
import os
import mdsine2 as md2
import argparse
from .base import CLIModule


class NegBinCLI(CLIModule):
    """ Perform inference using the calibration model, for tuning MDSINE2 hyperparameters. """
    def __init__(self, subcommand="infer-negbin"):
        super().__init__(
            subcommand=subcommand,
            docstring=self.__doc__
        )

    def create_parser(self, parser: argparse.ArgumentParser):
        parser.add_argument('--input', '-i', type=str, dest='input',
                            required=True,
                            help='This is the dataset to do inference with.')
        parser.add_argument('--seed', '-s', type=int, dest='seed',
                            required=True,
                            help='This is the seed to initialize the inference with')
        parser.add_argument('--burnin', '-nb', type=int, dest='burnin',
                            required=True,
                            help='How many burn-in Gibb steps for Markov Chain Monte Carlo (MCMC)')
        parser.add_argument('--n-samples', '-ns', type=int, dest='n_samples',
                            required=True,
                            help='Total number Gibb steps to perform during MCMC inference')
        parser.add_argument('--checkpoint', '-c', type=int, dest='checkpoint',
                            required=True,
                            help='How often to write the posterior to disk. Note that `--burnin` and ' \
                                 '`--n-samples` must be a multiple of `--checkpoint` (e.g. checkpoint = 100, ' \
                                 'n_samples = 600, burnin = 300)')
        parser.add_argument('--basepath', '-b', type=str, dest='basepath',
                            required=True,
                            help='This is folder to save the output of inference')
        parser.add_argument('--multiprocessing', '-mp', action='store_true', dest="mp",
                            help='If flag is set, run the inference with multiprocessing. Else run on a single process')

        parser.add_argument('--log-every', type=int, required=False, default=100,
                            help='<Optional> Tells the inference loop to print debug messages '
                                 '(if logging level is set to DEBUG) every <LOG_EVERY> iterations.'
                                 '(Default: 100)')

    def main(self, args: argparse.Namespace):
        study = md2.Study.load(args.input)
        os.makedirs(args.basepath, exist_ok=True)
        basepath = os.path.join(args.basepath, study.name)
        os.makedirs(basepath, exist_ok=True)

        # 1) Load the parameters
        params = md2.config.NegBinConfig(
            seed=args.seed,
            burnin=args.burnin,
            n_samples=args.n_samples,
            checkpoint=args.checkpoint,
            basepath=basepath
        )
        if args.mp:
            params.MP_FILTERING = 'full'
        else:
            params.MP_FILTERING = 'debug'

        # 2) Perform inference
        mcmc = md2.negbin.build_graph(params=params, graph_name=study.name, subjset=study)
        mcmc = md2.negbin.run_graph(mcmc, crash_if_error=True, log_every=args.log_every)
        mcmc.save()
        study.save(os.path.join(params.MODEL_PATH, md2.config.SUBJSET_FILENAME))
