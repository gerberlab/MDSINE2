import argparse
from typing import List


class CLIModule:
    def __init__(self, subcommand: str, docstring: str):
        self.subcommand = subcommand
        self.docstring = docstring

    def create_parser(self, parser: argparse.ArgumentParser):
        raise NotImplementedError()

    def main(self, args: argparse.Namespace):
        raise NotImplementedError()


def dispatch(clis: List[CLIModule]):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subcommand')

    for cli_module in clis:
        cli_module.create_parser(
            subparsers.add_parser(
                cli_module.subcommand,
                description=cli_module.docstring
            )
        )

    args = parser.parse_args()

    cli_mapping = {cli.subcommand: cli for cli in clis}
    if args.subcommand not in cli_mapping:
        print("Subcommand `{in_cmd}` not found. Supported commands: {cmds}".format(
            in_cmd=args.subcommand,
            cmds=",".join(cli_mapping.keys())
        ))
        exit(1)

    cli_module = cli_mapping[args.subcommand]
    cli_module.main(args)

