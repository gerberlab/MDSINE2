"""
Learn the Negative Binomial dispersion parameters that parameterize the noise
of the reads. We learn these parameters with replicate measurements of the reads.

Author: David Kaplan
Date: 11/30/20
MDSINE2 version: 4.0.6
"""
import os
import argparse
import mdsine2 as md2
from mdsine2.names import STRNAMES
import matplotlib.pyplot as plt

from .base import CLIModule


class NegBinVisualizationCLI(CLIModule):
    def __init__(self, subcommand="visualize-negbin"):
        super().__init__(
            subcommand=subcommand,
            docstring=__doc__
        )

    def create_parser(self, parser: argparse.ArgumentParser):
        parser.add_argument('--chain', '-c', type=str, dest='chain',
                            help='This is the MCMC object that inference was performed on. This is most likely' \
                                 'the `mcmc.pkl` file that is in the output folder of `step_3_infer_negbin`')
        parser.add_argument('--output-basepath', '-o', type=str, dest='basepath',
                            help='This is the folder to save the output.')

    def main(self, args: argparse.Namespace):
        basepath = args.basepath
        os.makedirs(basepath, exist_ok=True)

        mcmc = md2.BaseMCMC.load(args.chain)
        fig = md2.negbin.visualize_learned_negative_binomial_model(mcmc)
        fig.tight_layout()
        path = os.path.join(basepath, 'learned_model.pdf')
        plt.savefig(path)
        plt.close()

        f = open(os.path.join(basepath, 'a0a1.txt'), 'w')
        mcmc.graph[STRNAMES.NEGBIN_A0].visualize(
            path=os.path.join(basepath, 'a0.pdf'),
            f=f, section='posterior')
        mcmc.graph[STRNAMES.NEGBIN_A1].visualize(
            path=os.path.join(basepath, 'a1.pdf'),
            f=f, section='posterior')
        f.close()
        print('Plotting filtering')
        mcmc.graph[STRNAMES.FILTERING].visualize(
            basepath=basepath, section='posterior')
