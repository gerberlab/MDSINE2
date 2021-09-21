"""
Plot the QPCR and relative abundance levels.
"""
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

import mdsine2 as md2
from .base import CLIModule
from mdsine2.plots import *


class PlotAbundanceCLI(CLIModule):
    def __init__(self, subcommand="plot-abundances"):
        super().__init__(
            subcommand=subcommand,
            docstring=__doc__
        )

    def create_parser(self, parser: argparse.ArgumentParser):
        parser.add_argument('--input', '-i', type=str, dest='input',
                            required=True,
                            help='This is the path to the dataset (a pickled Study object) to do inference with.')
        parser.add_argument('--taxlevel', '-t', type=str, dest='taxlevel',
                            required=False,
                            default='family',
                            choices=TAXLEVEL_INTS,
                            help='The taxonomic level at which to group together organisms for plotting.')
        parser.add_argument('--width', type=int, default=35, required=False)
        parser.add_argument('--height', type=int, default=20, required=False)
        parser.add_argument('--format', '-f', type=str, default='png', required=False)
        parser.add_argument('--output', '-o', dest='output', type=str, required=True,
                            help='The path to which the image will be output to.')

    def main(self, args: argparse.Namespace):
        subjset = md2.Study.load(args.input)
        df = get_df(subjset, taxlevel=args.taxlevel)

        fig = plt.figure(figsize=(args.width, args.height))
        gs = fig.add_gridspec(6, 1, hspace=0.75)

        ax_qpcr = fig.add_subplot(gs[:2, :])
        ax_rel = fig.add_subplot(gs[2:, :])

        colors = list(sns.color_palette('muted', n_colors=df.shape[0]))
        plot_qpcr(subjset, ax_qpcr)
        plot_rel(df, ax=ax_rel, color_set=colors)
        plt.savefig(args.output, format=args.format)
