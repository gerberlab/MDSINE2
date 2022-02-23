import sys
import argparse
from typing import Dict


class CLIModule:
    def __init__(self, subcommand: str, docstring: str):
        self.subcommand = subcommand
        self.docstring = docstring

    def create_parser(self, parser: argparse.ArgumentParser):
        raise NotImplementedError()

    def main(self, args: argparse.Namespace):
        raise NotImplementedError()


def dispatch(cli_mapping: Dict[str, CLIModule]):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subcommand')

    for subcommand, cli_module in cli_mapping.items():
        cli_module.create_parser(subparsers.add_parser(subcommand))

    args = parser.parse_args()

    try:
        cli_module = cli_mapping[args.subcommand]
        cli_module.main(args)
    except KeyError:
        print("Supported commands: {cmds}".format(
            prog=sys.argv[0],
            cmds=",".join(list(cli_mapping.keys()))
        ))
        exit(1)
