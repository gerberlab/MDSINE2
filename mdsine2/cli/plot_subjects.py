"""
Plot the QPCR and relative abundance levels.
"""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import mdsine2 as md2
from .base import CLIModule


class PlotSubjectCLI(CLIModule):
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
                            choices=['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species', 'asv'],
                            help='The taxonomic level at which to group together organisms for plotting.')
        parser.add_argument('--width', type=int, default=35, required=False)
        parser.add_argument('--height', type=int, default=20, required=False)
        parser.add_argument('--format', '-f', type=str, default='png', required=False)
        parser.add_argument('--outdir', '-o', dest='outdir', type=str, required=True,
                            help='The path to which the image will be output to.')

    def main(self, args: argparse.Namespace):
        subjset = md2.Study.load(args.input)

        for subject in subjset:
            fig, ax = plt.subplots(figsize=(15, 7))
            md2.visualization.taxonomic_distribution_over_time(
                subject, taxlevel=args.taxlevel, label_formatter='%(${})s'.format(args.taxlevel), ax=ax
            )
            out_path = Path(args.outdir) / "rel_abund_{}.{}".format(subject.name, args.format)
            plt.savefig(out_path, format=args.format)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(15, 7))
            md2.visualization.alpha_diversity_over_time(subject, metric=md2.diversity.alpha.normalized_entropy, ax=ax)
            out_path = Path(args.outdir) / "alpha_diversity_{}.{}".format(subject.name, args.format)
            plt.savefig(out_path, format=args.format)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(15, 7))
            md2.visualization.abundance_over_time(subject, dtype='qpcr', yscale_log=True, ax=ax)
            out_path = Path(args.outdir) / "qpcr_{}.{}".format(subject.name, args.format)
            plt.savefig(out_path, format=args.format)
            plt.close(fig)
