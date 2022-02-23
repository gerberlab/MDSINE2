"""
Run keystoneness analysis using the specified MDSINE2 output.
"""

import argparse
import csv
from pathlib import Path

import mdsine2 as md2
from .base import CLIModule

from mdsine2.logger import logger


class ExtractAbundancesCLI(CLIModule):
    def __init__(self, subcommand="extract-abundances"):
        super().__init__(
            subcommand=subcommand,
            docstring=__doc__
        )

    def create_parser(self, parser: argparse.ArgumentParser):
        # Inputs
        parser.add_argument('--study', '-s', dest='study', type=str, required=True,
                            help="The path to the relevant Study object containing the input data (subjects, taxa).")
        parser.add_argument('--time-index', '-t', dest='time_index', type=int, required=True,
                            help="The timepoint index (0-indexed) to extract the abundance profile from.")

        # Outputs
        parser.add_argument('--output-path', '-o', type=str, dest='out_path',
                            required=True,
                            help='This is where you are saving the posterior renderings')

        # Simulation params
        parser.add_argument('--limit-of-detection', dest='limit_of_detection', required=False,
                            help='If any of the taxa have a 0 abundance at the start, then we ' \
                                 'set it to this value.', default=1e5, type=float)

    def main(self, args: argparse.Namespace):
        study = md2.Study.load(args.study)
        limit_of_detection = args.limit_of_detection

        M = study.matrix(dtype='abs', agg='mean', times='intersection', qpcr_unnormalize=True)
        day20_state = M[:, args.time_index]

        abundances = day20_state
        abundances[abundances < limit_of_detection] = limit_of_detection

        Path(args.out_path).parent.mkdir(exist_ok=True, parents=True)
        with open(args.out_path, "w") as fd:
            wd = csv.writer(fd, delimiter='\t', quotechar='"')
            for taxa_idx, taxa in enumerate(study.taxa):
                wd.writerow([taxa.name, abundances[taxa_idx]])
        logger.info(f"Saved abundances from timepoint index {args.time_index} to {args.out_path}.")
