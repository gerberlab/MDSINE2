"""
Render the learned interaction structure from the fixed-clustering run as a Cytoscape-readable JSON file.
"""
import argparse
import mdsine2 as md2

from .base import CLIModule


class InteractionToCytoscapeCLI(CLIModule):
    def __init__(self, subcommand="interaction-to-cytoscape"):
        super().__init__(
            subcommand=subcommand,
            docstring=__doc__
        )

    def create_parser(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--input-fixed-cluster-samples", "-i", dest="input_chain",
            required=True,
            help="The path to the MCMC pickle file, output by `mdsine2 infer` in fixed-clustering mode."
        )
        parser.add_argument(
            "--out_path", "-o", dest="output_path",
            required=True,
            help="The desired output path. "
                 "The resulting file will be in JSON format, so a .json extension is recommended."
        )

    def main(self, args: argparse.Namespace):
        mcmc = md2.BaseMCMC.load(args.input_chain)

        md2.write_fixed_clustering_as_json(
            mcmc=mcmc,
            output_path=args.output_path
        )
