"""
Parse the input TSV files into a python mdsine2.Study object. Saves this object into a pickle file.

Author : David Kaplan
Date: 11/30/20

Input tables
------------
Metadata
    Holds the metadata for each sample. Columns:
    sampleID : str
        Name of the sample
    subject : str
        Name of the subject this sample belongs to
    time : float
        Timepoint of the sample

Perturbations
    These are the perturbations for each subject. Columns
    name : str
        Name of the perturbation
    start, end : float
        Start and end of the perturbation
    subject : str
        This is the subject this perturbation corresponds to. Subject must
        be contained in the metadata file as well.

qPCR
    These are the qPCR measurements for each sample. Columns:
    sampleID : str
        Name of the sampe
    measurement1, ... : float
        Rest of the columns are the replicate measurements

Counts
    These are the counts for each taxon. Columns:
    name : str
        Name of the taxon
    `sampleID`s
        Each sampleID has its own count

Taxonomy
    This is the taxonomy name for each taxon in counts. Columns:
    name : str
        Name of the taxon. This corresponds to the name in `counts`
    sequence : str
        Sequence associated with the taxon
    kingdom : str
        Kingdom taxonomic classification
    phylum : str
        Phylum taxonomic classification
    class : str
        Class taxonomic classification
    order : str
        Order taxonomic classification
    family : str
        Family taxonomic classification
    genus : str
        Genus taxonomic classification
    species : str
        Species taxonomic classification
"""

import argparse
from .base import CLIModule
import mdsine2 as md2


class InputParseCLI(CLIModule):
    def __init__(self, subcommand="parse-input"):
        super().__init__(
            subcommand=subcommand,
            docstring=__doc__
        )

    def create_parser(self, parser: argparse.ArgumentParser):
        parser.add_argument('--name', '-n', type=str, dest='name', required=True,
                            help='Name of the dataset')
        parser.add_argument('--taxonomy', '-t', type=str, dest='taxonomy', required=True,
                            help='This is the table showing the sequences and the taxonomy for each ASV or OTU')
        parser.add_argument('--metadata', '-m', type=str, dest='metadata', required=True,
                            help='This is the metadata table')
        parser.add_argument('--reads', '-r', type=str, dest='reads', required=True,
                            help='This is the reads table', default=None)
        parser.add_argument('--qpcr', '-q', type=str, dest='qpcr', required=True,
                            help='This is the qPCR table', default=None)
        parser.add_argument('--perturbations', '-p', type=str, dest='perturbations', required=False,
                            help='(Optional) This is the perturbation table')
        parser.add_argument('--sep', '-s', type=str, dest='sep', required=False, default='\t',
                            help='This is the separator for the tables')
        parser.add_argument('--outfile', '-o', type=str, dest='outfile', required=True,
                            help='This is where you want to save the parsed dataset')

    def main(self, args: argparse.Namespace):
        study = md2.dataset.parse(
            name=args.name,
            metadata=args.metadata,
            taxonomy=args.taxonomy,
            reads=args.reads,
            qpcr=args.qpcr,
            perturbations=args.perturbations,
            sep=args.sep
        )
        study.save(args.outfile)
