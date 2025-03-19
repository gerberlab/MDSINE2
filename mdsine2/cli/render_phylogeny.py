"""
Plot the phylogenetic subtree for each taxon.
"""

import argparse
import os
import pandas as pd
import numpy as np
from .base import CLIModule

import ete3
from ete3 import TreeStyle
import mdsine2 as md2
from mdsine2.logger import logger


class PhylogenyRenderCLI(CLIModule):
    """ Render a phylogenetic subtree for each taxon. """
    def __init__(self, subcommand="render-phylogeny"):
        super().__init__(
            subcommand=subcommand,
            docstring=self.__doc__
        )

    def create_parser(self, parser: argparse.ArgumentParser):
        parser.add_argument('--output-basepath', '-o', type=str, dest='basepath',
                            required=True,
                            help='This is where you want to save the parsed dataset.')
        parser.add_argument('--study', '-s', type=str, dest='study',
                            required=True,
                            help='Dataset that contains all of the information')
        parser.add_argument('--tree', '-t', type=str, dest='tree',
                            required=True,
                            help='Path to newick tree file.')
        parser.add_argument('--seq-info', type=str, dest='seq_info',
                            required=True,
                            help='Maps the sequence id of the reference sequences identifying info')
        parser.add_argument('--sep', type=str, dest='sep',
                            required=False, default='\t',
                            help='The separator character for the input TSV file.')
        parser.add_argument('--family-radius-factor', type=float, dest='family_radius_factor',
                            required=False, default=1.5,
                            help='How much to multiply the radius of each family')
        parser.add_argument('--top', type=int, dest='top',
                            required=False, default=None,
                            help='Plot only up to this number.')

    def main(self, args: argparse.Namespace):
        basepath = args.basepath
        os.makedirs(basepath, exist_ok=True)
        study = md2.Study.load(args.study)

        # Get the median phylogenetic distance within each family of the reference seqs
        # ------------------------------------------------------------------------------
        import treeswift

        # Make the distance matrix - this is a 2D dict
        logger.info('Making distance matrix (this may take a minute)')
        tree = treeswift.read_tree_newick(args.tree)
        M = tree.distance_matrix(leaf_labels=True)
        df_distance_matrix = pd.DataFrame(M)
        node_dict = tree.label_to_node()

        # Get the families of the reference trees
        logger.info('Read sequence info file')
        df_seqs = pd.read_csv(args.seq_info, sep='\t', index_col=0)

        logger.info('get families of reference seqs')
        ref_families = {}
        for i, seq in enumerate(df_seqs.index):
            lineage = df_seqs['Lineage'][seq].split('; ')
            # print(lineage)
            l = len(lineage)
            if l != 7:
                continue
            family = lineage[-2].lower()
            if family not in ref_families:
                ref_families[family] = []
            ref_families[family].append(seq)

        # Get the distance of every taxon to the respective family in the reference seqs
        d = {}
        percents = []
        not_found = set([])
        for i, taxon in enumerate(study.taxa):
            if i % 100 == 0:
                logger.info('{}/{} - {}'.format(i, len(study.taxa), np.mean(percents)))
                percents = []

            if taxon.tax_is_defined('family'):
                family = taxon.taxonomy['family'].lower()
                if family not in d:
                    d[family] = []

                if family not in ref_families:
                    logger.debug('{} NOT IN REFERENCE TREE'.format(family))
                    continue
                refs = ref_families[family]

                aaa = 0
                for ref in refs:
                    try:
                        dist = tree.distance_between(node_dict[taxon.name], node_dict[ref])
                        d[family].append(dist)
                        aaa += 1
                    except Exception as e:
                        not_found.add(ref)
                        # logger.debug('no worked - {}, {}'.format(taxon.name, ref))
                        continue
                percents.append(aaa / len(refs))
            else:
                print('family is not defined for', taxon.name)

        # make a text file indicating the intra family distances
        fname = os.path.join(basepath, 'family_distances.txt')
        f = open(fname, 'w')
        family_dists = {}
        for family in d:
            f.write(family + '\n')
            arr = np.asarray(d[family])
            summ = md2.summary(arr)
            family_dists[family] = summ['median']
            for k, v in summ.items():
                f.write('\t{}: {}\n'.format(k, v))

        f.write('total\n')
        arr = []
        for ele in d.values():
            arr += ele
        arr = np.asarray(arr)
        summ = md2.summary(arr)
        for k, v in summ.items():
            f.write('\t{}:{}\n'.format(k, v))

        # Set the default radius to global median
        default_radius = args.family_radius_factor * summ['median']
        f.write('default radius set to {}% of global median ({})'.format(
            100 * args.family_radius_factor, default_radius))
        f.close()

        def my_layout_fn(node):
            if node.is_leaf():
                if 'OTU' in node.name:
                    node.img_style["bgcolor"] = "#9db0cf"
                    node.name = md2.taxaname_for_paper(taxon=study.taxa[node.name], taxa=study.taxa)
                else:
                    if node.name in df_seqs.index:
                        # replace the sequence name with the species id
                        node.name = df_seqs['Species'][node.name]
                    else:
                        print('Node {} not found'.format(node.name))

        # Make the phylogenetic subtrees for each OTU
        # -------------------------------------------
        if args.top is None:
            top = len(study.taxa)
        else:
            top = args.top

        i = 0
        names = df_distance_matrix.index
        fname = os.path.join(basepath, 'table.tsv')
        f = open(fname, 'w')
        for iii, taxon in enumerate(study.taxa):
            if iii >= top:
                break
            logger.info('\n\nLooking at {}, {}'.format(i, taxon))
            logger.info('-------------------------')

            f.write('{}\n'.format(taxon.name))
            tree = ete3.Tree(args.tree)
            # Get the all elements within `radius`

            row = df_distance_matrix[taxon.name].to_numpy()
            idxs = np.argsort(row)

            names_found = False
            mult = 1.
            title = taxon.name
            while not names_found:
                if mult == 3:
                    # title += '\nTOO MANY, BREAKING'
                    break
                if taxon.tax_is_defined('family'):
                    family = taxon.taxonomy['family'].lower()
                    if family_dists[family] < 1e-2:
                        radius = default_radius * mult
                        # title += '\nfamily defined but not ref: {} Median family distance: {:.4f}'.format(taxon.taxonomy['family'], radius)
                    else:
                        radius = family_dists[family] * mult
                        # title += '\nfamily defined: Median {} distance: {:.4f}'.format(taxon.taxonomy['family'], radius)
                else:
                    radius = default_radius * mult * args.family_radius_factor
                    mmm = mult * 100 * args.family_radius_factor
                    # title += '\nfamily not defined: {}% Median family distance: {:.4f}'.format(mmm, radius)

                names_to_keep = []
                for idx in idxs:
                    if row[idx] > radius:
                        break
                    if 'OTU' in names[idx]:
                        continue
                    names_to_keep.append(names[idx])

                if len(names_to_keep) > 5:
                    print(len(names_to_keep), ' found for ', taxon.name)
                    names_found = True
                else:
                    print('expand radius')
                    # title += '\n`{}` reference seqs within radius, expanding radius by 25%'.format(len(names_to_keep))
                    mult += .25

            suffix_taxa = {
                'genus': '*',
                'family': '**',
                'order': '***',
                'class': '****',
                'phylum': '*****',
                'kingdom': '******'}
            title += '\nTaxonomic Key for ASV\n'
            i = 0
            for k, v in suffix_taxa.items():
                if i == 2:
                    title += '\n'
                    i = 0
                title += '{}: {}'.format(v, k)
                if i == 0:
                    title += ', '
                i += 1

            # print(names_to_keep)

            # Make subtree of just these names
            names_to_keep.append(taxon.name)
            tree.prune(names_to_keep, preserve_branch_length=False)

            ts = TreeStyle()
            ts.layout_fn = my_layout_fn
            ts.title.add_face(ete3.TextFace(title, fsize=15), column=1)
            fname = os.path.join(basepath, '{}.pdf'.format(taxon.name))
            tree.render(fname, tree_style=ts)
        f.close()
