"""
Preprocess (aggregate and filter) the input Study objects.

Methodology
-----------
1) Load the dataset (mdsine2.Study object)
2) Aggregate the ASVs into OTUs using the aligned 16S v4 rRNA sequences using a threshold on Hamming distance.
   Once we agglomerate them together we set the sequences to the original sequence (unaligned).
3) Calculate the consensus sequences
4) Preprocess Taxa objects (ASVs) into OTUs.
5) Remove selected timepoints.
"""
import argparse
import numpy as np
from pathlib import Path

from Bio import SeqIO, SeqRecord, Seq
import mdsine2 as md2
from mdsine2.logger import logger

from .base import CLIModule


class AggregationCLI(CLIModule):
    def __init__(self, subcommand="aggregate"):
        super().__init__(
            subcommand=subcommand,
            docstring=__doc__
        )

    def create_parser(self, parser: argparse.ArgumentParser):
        parser.add_argument('--out_path', '-o', type=str, dest='out_path',
                            required=True,
                            help='This is where you want to save the parsed dataset.')
        parser.add_argument('--hamming-distance', '-hd', type=int, dest='hamming_distance',
                            required=True, default=2,
                            help='This is the hamming radius to aggregate ASV sequences.')
        parser.add_argument('--rename-prefix', '-rp', type=str, dest='rename_prefix',
                            required=False, default=None,
                            help='(Optional) This is the prefix you are renaming the aggregate taxa to. ' \
                                 'If nothing is provided, then they will not be renamed')
        parser.add_argument('--sequences', '-s', type=str, dest='sequences',
                            required=True,
                            help='(Optional) A fasta file listing the target sequences for each ASV (possibly aligned/trimmed). '
                                 'If specified, the sequences in this file replaces the sequences stored in the Study object. '
                                 'All gaps will be removed and a copy of the sequences will be saved with the same filename with a .fa extension.')
        parser.add_argument('--remove-timepoints', dest='remove_timepoints', nargs='+',
                            required=False, default=None,
                            type=float, help='(Optional) a list of timepoints to remove.')
        parser.add_argument('--max-n-species', '-ms', dest='max_n_species', type=int,
                            required=False, default=2,
                            help='Maximum number of species assignments to have in the name')
        parser.add_argument('--study_path', '-d', dest='dataset_dir', type=str,
                            required=True,
                            help='The path to a pickled Study object.')

    def main(self, args: argparse.Namespace):
        out_path = Path(args.out_path)
        out_path.parent.mkdir(exist_ok=True)

        # 1) Load the study
        study = md2.Study.load(args.study_path)

        # 2) Set the sequences for each taxon
        #    Remove all taxa that are not contained in that file
        #    Remove the gaps
        if args.sequences is not None:
            logger.info('Replacing sequences with the file {}'.format(args.sequences))
            seqs = SeqIO.to_dict(SeqIO.parse(args.sequences, format='fasta'))
            to_delete = []
            for taxon in study.taxa:
                if taxon.name not in seqs:
                    to_delete.append(taxon.name)
            for name in to_delete:
                logger.info('Deleting {} because it was not in {}'.format(
                    name, args.sequences))
            study.pop_taxa(to_delete)

            M = []
            for taxon in study.taxa:
                seq = list(str(seqs[taxon.name].seq))
                M.append(seq)
            M = np.asarray(M)
            gaps = M == '-'
            n_gaps = np.sum(gaps, axis=0)
            idxs = np.where(n_gaps == 0)[0]
            logger.info(
                'There are {} positions where there are no gaps out of {}. Setting those to the sequences'.format(
                    len(idxs), M.shape[1]
                )
            )
            M = M[:, idxs]
            for i, taxon in enumerate(study.taxa):
                taxon.sequence = ''.join(M[i])
                print("Len {}: {}".format(len(taxon.sequence), taxon.sequence))

        # Aggregate with specified hamming distance
        if args.hamming_distance is not None:
            logger.info('Aggregating taxa with a hamming distance of {}'.format(args.hamming_distance))
            study = md2.aggregate_items(subjset=study, hamming_dist=args.hamming_distance)

            # Get the maximum distance of all the OTUs
            m = -1
            for taxon in study.taxa:
                if md2.isotu(taxon):
                    for aname in taxon.aggregated_taxa:
                        for bname in taxon.aggregated_taxa:
                            if aname == bname:
                                continue
                            aseq = taxon.aggregated_seqs[aname]
                            bseq = taxon.aggregated_seqs[bname]
                            d = md2.diversity.beta.hamming(aseq, bseq)
                            if d > m:
                                m = d
            logger.info('Maximum distance within an OTU: {}'.format(m))

        # 3) compute consensus sequences
        study.taxa.generate_consensus_seqs(threshold=0.65, noconsensus_char='N')

        # 4) Rename taxa
        if args.rename_prefix is not None:
            print('Renaming taxa with prefix {}'.format(args.rename_prefix))
            study.taxa.rename(prefix=args.rename_prefix, zero_based_index=False)

        # 5) Remove timepoints
        if args.remove_timepoints is not None:
            study.pop_times(args.remove_timepoints)

        # 6) Save the study set and sequences
        study.save(out_path)

        ret = []
        for taxon in study.taxa:
            ret.append(
                SeqRecord.SeqRecord(seq=Seq.Seq(taxon.sequence), id=taxon.name, description='')
            )
        SeqIO.write(
            ret,
            str(out_path.with_suffix('.fa')),
            format='fasta-2line'
        )
