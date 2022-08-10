from typing import Iterator, Any, Union, List

import pandas as pd

from mdsine2.pylab import Saveable
from mdsine2.pylab import util as plutil
from mdsine2.pylab import diversity
from mdsine2.logger import logger

from .constants import *
from .util import CustomOrderedDict
from .cluster import ClusterItem, Clusterable


class Taxon(ClusterItem):
    """Wrapper class for a single Taxon

    Parameters
    ----------
    name : str
        Name given to the Taxon
    sequence : str
        Base Pair sequence
    idx : int
        The index that the asv occurs
    """
    def __init__(self, name: str, idx: int, sequence: str = None):
        ClusterItem.__init__(self, name=name)
        self.sequence = sequence
        self.idx = idx
        # Initialize the taxonomies to nothing
        self.taxonomy = {
            'kingdom': DEFAULT_TAXLEVEL_NAME,
            'phylum': DEFAULT_TAXLEVEL_NAME,
            'class': DEFAULT_TAXLEVEL_NAME,
            'order': DEFAULT_TAXLEVEL_NAME,
            'family': DEFAULT_TAXLEVEL_NAME,
            'genus': DEFAULT_TAXLEVEL_NAME,
            'species': DEFAULT_TAXLEVEL_NAME,
            'asv': self.name}
        self.id = id(self)

    def __eq__(self, val: Any) -> bool:
        """Compares different taxa between each other. Checks all of the attributes but the id

        Parameters
        ----------
        val : any
            This is what we are checking if they are equivalent
        """
        if not isinstance(val, Taxon):
            return False
        if self.name != val.name:
            return False
        if self.sequence != val.sequence:
            return False
        for k,v in self.taxonomy.items():
            if v != val.taxonomy[k]:
                return False
        return True

    def __str__(self) -> str:
        return 'Taxon\n\tid: {}\n\tidx: {}\n\tname: {}\n' \
            '\ttaxonomy:\n\t\tkingdom: {}\n\t\tphylum: {}\n' \
            '\t\tclass: {}\n\t\torder: {}\n\t\tfamily: {}\n' \
            '\t\tgenus: {}\n\t\tspecies: {}'.format(
            self.id, self.idx, self.name,
            self.taxonomy['kingdom'], self.taxonomy['phylum'],
            self.taxonomy['class'], self.taxonomy['order'],
            self.taxonomy['family'], self.taxonomy['genus'],
            self.taxonomy['species'])

    def set_taxonomy(self, tax_kingdom: str=None, tax_phylum: str=None, tax_class: str=None,
        tax_order: str=None, tax_family: str=None, tax_genus: str=None, tax_species: str=None):
        """Sets the taxonomy of the parts that are specified

        Parameters
        ----------
        tax_kingdom, tax_phylum, tax_class, tax_order, tax_family, tax_genus : str
            'kingdom', 'phylum', 'class', 'order', 'family', 'genus'
            Name of the taxon for each respective level
        """
        if tax_kingdom is not None and tax_kingdom != '' and plutil.isstr(tax_kingdom):
            self.taxonomy['kingdom'] = tax_kingdom
        if tax_phylum is not None and tax_phylum != '' and plutil.isstr(tax_phylum):
            self.taxonomy['phylum'] = tax_phylum
        if tax_class is not None and tax_class != '' and plutil.isstr(tax_class):
            self.taxonomy['class'] = tax_class
        if tax_order is not None and tax_order != '' and plutil.isstr(tax_order):
            self.taxonomy['order'] = tax_order
        if tax_family is not None and tax_family != '' and plutil.isstr(tax_family):
            self.taxonomy['family'] = tax_family
        if tax_genus is not None and tax_genus != '' and plutil.isstr(tax_genus):
            self.taxonomy['genus'] = tax_genus
        if tax_species is not None and tax_species != '' and plutil.isstr(tax_species):
            self.taxonomy['species'] = tax_species
        return self

    def get_lineage(self, level: str=None) -> Iterator[str]:
        """Returns a tuple of the lineage in order from Kingdom to the level
        indicated. Default value for level is `asv`.
        Parameters
        ----------
        level : str, Optional
            The taxonomic level you want the lineage until
            If nothing is provided, it returns the entire taxonomic lineage
            Example:
                level = 'class'
                returns a tuple of (kingdom, phylum, class)
        Returns
        -------
        str
        """
        a =  (self.taxonomy['kingdom'], self.taxonomy['phylum'], self.taxonomy['class'],
            self.taxonomy['order'], self.taxonomy['family'], self.taxonomy['genus'],
            self.taxonomy['species'], self.taxonomy['asv'])

        if level is None:
            a = a
        if level == 'asv':
            a = a
        elif level == 'species':
            a = a[:-1]
        elif level == 'genus':
            a = a[:-2]
        elif level == 'family':
            a = a[:-3]
        elif level == 'order':
            a = a[:-4]
        elif level == 'class':
            a = a[:-5]
        elif level == 'phylum':
            a = a[:-6]
        elif level == 'kingdom':
            a = a[:-7]
        else:
            raise ValueError('level `{}` was not recognized'.format(level))

        return a

    def get_taxonomy(self, level: str) -> str:
        """Get the taxonomy at the level specified

        Parameters
        ----------
        level : str
            This is the level to get
            Valid responses: 'kingdom', 'phylum', 'class', 'order', 'family', 'genus'

        Returns
        -------
        str
        """
        return self.get_lineage(level=level)[-1]

    def tax_is_defined(self, level: str) -> bool:
        """Whether or not the taxon is defined at the specified taxonomic level

        Parameters
        ----------
        level : str
            This is the level to get
            Valid responses: 'kingdom', 'phylum', 'class', 'order', 'family', 'genus'

        Returns
        -------
        bool
        """
        try:
            tax = self.taxonomy[level]
        except:
            raise KeyError('`tax` ({}) not defined. Available taxs: {}'.format(level,
                list(self.taxonomy.keys())))
        return (type(tax) != float) and (tax != DEFAULT_TAXLEVEL_NAME) and (tax != '')


class OTU(Taxon):
    """Aggregates of Taxon objects

    NOTE: For self consistency, let the class TaxaSet initialize this object.

    Parameters
    ----------
    anchor, other : mdsine2.Taxon, mdsine2.OTU
        These are the taxa/Aggregates that you're joining together. The anchor is
        the one you are setting the sequeunce and taxonomy to
    """
    def __init__(self, components: List[Taxon], idx: int):
        self.components = components

        Taxon.__init__(
            self,
            name=f"OTU_{idx+1}",
            idx=idx,
            sequence=self.generate_consensus_seq()
        )
        # self.aggregated_taxa = agg1 + agg2 # list
        # self.aggregated_seqs = agg1_seq # dict: taxon.name (str) -> sequence (str)
        # self.aggregated_taxonomies = _agg_taxa # dict: taxon.name (str) -> (dict: tax level (str) -> taxonomy (str))
        # for k,v in agg2_seq.items():
        #     self.aggregated_seqs[k] = v
        #
        # self.taxonomy = anchor.taxonomy

    def __str__(self) -> str:
        return 'OTU\n\tid: {}\n\tidx: {}\n\tname: {}\n' \
            '\tAggregates: {}\n' \
            '\ttaxonomy:\n\t\tkingdom: {}\n\t\tphylum: {}\n' \
            '\t\tclass: {}\n\t\torder: {}\n\t\tfamily: {}\n' \
            '\t\tgenus: {}\n\t\tspecies: {}'.format(
            self.id, self.idx, self.name, [taxa.name for taxa in self.components],
            self.taxonomy['kingdom'], self.taxonomy['phylum'],
            self.taxonomy['class'], self.taxonomy['order'],
            self.taxonomy['family'], self.taxonomy['genus'],
            self.taxonomy['species'])

    def generate_consensus_seq(self, threshold: float=0.65, noconsensus_char: str='N') -> str:
        """Generate the consensus sequence for the OTU given the sequences
        of all the contained ASVs

        Parameters
        ----------
        threshold : float
            This is the threshold for consensus (0 < threshold <= 1)
        noconsensus_char : str
            This is the character to set base if no consensus base is found
            at the respective position.

        NOTE
        ----
        Situation where all of the sequences are not the same length is not implemented
        """
        if not plutil.isstr(noconsensus_char):
            raise TypeError('`noconsensus_char` ({}) must be a str'.format(
                type(noconsensus_char)))
        if not plutil.isnumeric(threshold):
            raise TypeError('`threshold` ({}) must be a numeric'.format(threshold))
        if threshold < 0 or threshold > 1:
            raise ValueError('`threshold` ({}) must be 0 <= thresold <= 1'.format(threshold))

        # Check if all of the sequences are the same length
        agg_seqs = [taxa.sequence for taxa in self.components]
        l = None
        for seq in agg_seqs:
            if l is None:
                l = len(seq)
            if len(seq) != l:
                raise NotImplementedError('Unaligned sequences not implemented yet')

        # Generate the consensus base for each base position
        consensus_seq = []
        for i in range(l):

            # Count the number of times each base occurs at position `i`
            found = {}
            for seq in agg_seqs:
                base = seq[i]
                if base not in found:
                    found[base] = 1
                else:
                    found[base] += 1

            # Set the base
            if len(found) == 1:
                # Every sequence agrees on this base. Set
                consensus_seq.append(list(found.keys())[0])
            else:
                consensus_ratio = -1
                consensus_base = 'N'

                for base in found:
                    this_ratio = 1 - (found[base]/len(agg_seqs))
                    if this_ratio > consensus_ratio:
                        consensus_percent = consensus_ratio
                        consensus_base = base

                # Set the consensus base if it passes the threshold
                if consensus_ratio >= threshold:
                    logger.debug('Consensus found for taxon {} in position {} as {}, found ' \
                        '{}'.format(self.name, i, consensus_base, found))
                    consensus_seq.append(consensus_base)
                else:
                    logger.debug('No consensus for taxon {} in position {}. Consensus: {}, found {}'.format(
                        self.name, i, consensus_ratio, found
                    ))
                    consensus_seq.append(noconsensus_char)

        # Check for errors with consensus sequence
        for seq in agg_seqs:
            perc_dist = diversity.beta.hamming(
                seq,
                consensus_seq,
                ignore_char=noconsensus_char
            ) / l
            if perc_dist > 0.03:
                logger.warning('Taxon {} has a hamming distance > 3% ({}) to the generated ' \
                    'consensus sequence {} from individual sequence {}. Check that sequences ' \
                    'make sense'.format(self.name, perc_dist, consensus_seq, seq))

        # Set the consensus sequence as the OTU's sequence
        return ''.join(c for c in consensus_seq if c != '-')

    def generate_consensus_taxonomy(self, consensus_table: pd.DataFrame=None):
        """Set the taxonomy of the OTU to the consensus taxonomy of the.

        If one of the ASVs is defined at a lower level than another ASV, use
        that taxonomy. If ASVs' taxonomies disagree at the species level, use the
        union of all the species.

        Disagreeing taxonomy
        --------------------
        If the taxonomy of the ASVs differ on a taxonomic level other than species, we use an alternate
        way of naming the OTU. The input `consensus_table` is a `pandas.DataFrame` object showing the
        taxonomic classification of an OTU. You would get this table by running RDP on the consensus
        sequence.

        If the consensus table is not given, then we specify the lowest level that they agree. If the
        consensus table is given, then we use the taxonomy specified in that table.

        Examples
        --------
        ```
        Input:
         kingdom          phylum                class        order             family  genus       species      asv
        Bacteria  Proteobacteria  Alphaproteobacteria  Rhizobiales  Bradyrhizobiaceae  Bosea  massiliensis  ASV_722
        Bacteria  Proteobacteria  Alphaproteobacteria  Rhizobiales  Bradyrhizobiaceae  Bosea            NA  ASV_991

        Output:
         kingdom          phylum                class        order             family  genus       species
        Bacteria  Proteobacteria  Alphaproteobacteria  Rhizobiales  Bradyrhizobiaceae  Bosea  massiliensis
        ```

        ```
        Input:
         kingdom          phylum           class              order              family            genus                 species      asv
        Bacteria  Actinobacteria  Actinobacteria  Bifidobacteriales  Bifidobacteriaceae  Bifidobacterium                      NA  ASV_283
        Bacteria  Actinobacteria  Actinobacteria  Bifidobacteriales  Bifidobacteriaceae  Bifidobacterium                      NA  ASV_302
        Bacteria  Actinobacteria  Actinobacteria  Bifidobacteriales  Bifidobacteriaceae  Bifidobacterium    adolescentis/faecale  ASV_340
        Bacteria  Actinobacteria  Actinobacteria  Bifidobacteriales  Bifidobacteriaceae  Bifidobacterium  choerinum/pseudolongum  ASV_668

        Ouput:
         kingdom          phylum           class              order              family            genus                                      species
        Bacteria  Actinobacteria  Actinobacteria  Bifidobacteriales  Bifidobacteriaceae  Bifidobacterium  adolescentis/faecale/choerinum/pseudolongum
        ```

        Parameters
        ----------
        consensus_table : pd.DataFrame
            Table for resolving conflicts
        """
        # Check that all the taxonomies have the same lineage
        set_to_na = False
        set_from_table = False

        for tax_level in TAX_LEVELS:
            if set_to_na:
                self.taxonomy[tax_level] = DEFAULT_TAXLEVEL_NAME
                continue
            if set_from_table:
                if tax_level not in consensus_table.columns:
                    self.taxonomy[tax_level] = DEFAULT_TAXLEVEL_NAME
                else:
                    self.taxonomy[tax_level] = consensus_table[tax_level][self.name]
                continue
            if tax_level == 'asv':
                continue

            labels = set(taxon.taxonomy[tax_level] for taxon in self.components)
            if DEFAULT_TAXLEVEL_NAME in labels and tax_level == 'species':
                labels.remove(DEFAULT_TAXLEVEL_NAME)

            if len(labels) == 0:
                # No taxonomy found at this level
                self.taxonomy[tax_level] = DEFAULT_TAXLEVEL_NAME
            elif len(labels) == 1:
                # All taxonomies agree
                self.taxonomy[tax_level] = next(iter(labels))
            else:
                # All taxonomies do not agree
                if tax_level == 'species':
                    # Take the union of the species
                    self.taxonomy[tax_level] = '/'.join(sorted(labels))
                else:
                    # This means that the taxonomy is different on a level different than
                    logger.warning('{} taxonomy does not agree'.format(self.name))
                    logger.warning(''.join(taxon.name for taxon in self.components))

                    if consensus_table is not None:
                        # Set from the table
                        self.taxonomy[tax_level] = consensus_table[tax_level][self.name]
                        set_from_table = True
                    else:
                        # Set this taxonomic level and everything below it to NA
                        self.taxonomy[tax_level] = DEFAULT_TAXLEVEL_NAME
                        set_to_na = True



# class OTU(Taxon):
#     """Aggregates of Taxon objects
#
#     NOTE: For self consistency, let the class TaxaSet initialize this object.
#
#     Parameters
#     ----------
#     anchor, other : mdsine2.Taxon, mdsine2.OTU
#         These are the taxa/Aggregates that you're joining together. The anchor is
#         the one you are setting the sequeunce and taxonomy to
#     """
#     def __init__(self, anchor: Union[Taxon, 'OTU'], other: Union[Taxon, 'OTU']):
#         name = anchor.name + '_agg'
#         Taxon.__init__(self, name=name, idx=anchor.idx, sequence=anchor.sequence)
#
#         _agg_taxa = {}
#
#         if isinstance(anchor):
#             if other.name in anchor.aggregated_taxa:
#                 raise ValueError('`other` ({}) already aggregated with anchor ' \
#                     '({}) ({})'.format(other.name, anchor.name, anchor.aggregated_taxa))
#             agg1 = anchor.aggregated_taxa
#             agg1_seq = anchor.aggregated_seqs
#             for k,v in anchor.aggregated_taxonomies.items():
#                 _agg_taxa[k] = v
#         else:
#             agg1 = [anchor.name]
#             agg1_seq = {anchor.name: anchor.sequence}
#             _agg_taxa[anchor.name] = anchor.taxonomy
#
#         if isotu(other):
#             if anchor.name in other.aggregated_taxa:
#                 raise ValueError('`anchor` ({}) already aggregated with other ' \
#                     '({}) ({})'.format(anchor.name, other.name, other.aggregated_taxa))
#             agg2 = other.aggregated_taxa
#             agg2_seq = other.aggregated_seqs
#             for k,v in other.aggregated_taxonomies.items():
#                 _agg_taxa[k] = v
#         else:
#             agg2 = [other.name]
#             agg2_seq = {other.name: other.sequence}
#             _agg_taxa[other.name] = other.taxonomy
#
#         self.aggregated_taxa = agg1 + agg2 # list
#         self.aggregated_seqs = agg1_seq # dict: taxon.name (str) -> sequence (str)
#         self.aggregated_taxonomies = _agg_taxa # dict: taxon.name (str) -> (dict: tax level (str) -> taxonomy (str))
#         for k,v in agg2_seq.items():
#             self.aggregated_seqs[k] = v
#
#         self.taxonomy = anchor.taxonomy
#
#     def __str__(self) -> str:
#         return 'OTU\n\tid: {}\n\tidx: {}\n\tname: {}\n' \
#             '\tAggregates: {}\n' \
#             '\ttaxonomy:\n\t\tkingdom: {}\n\t\tphylum: {}\n' \
#             '\t\tclass: {}\n\t\torder: {}\n\t\tfamily: {}\n' \
#             '\t\tgenus: {}\n\t\tspecies: {}'.format(
#             self.id, self.idx, self.name, self.aggregated_taxa,
#             self.taxonomy['kingdom'], self.taxonomy['phylum'],
#             self.taxonomy['class'], self.taxonomy['order'],
#             self.taxonomy['family'], self.taxonomy['genus'],
#             self.taxonomy['species'])
#
#     def generate_consensus_seq(self, threshold: float=0.65, noconsensus_char: str='N'):
#         """Generate the consensus sequence for the OTU given the sequences
#         of all the contained ASVs
#
#         Parameters
#         ----------
#         threshold : float
#             This is the threshold for consensus (0 < threshold <= 1)
#         noconsensus_char : str
#             This is the character to set base if no consensus base is found
#             at the respective position.
#
#         NOTE
#         ----
#         Situation where all of the sequences are not the same length is not implemented
#         """
#         if not plutil.isstr(noconsensus_char):
#             raise TypeError('`noconsensus_char` ({}) must be a str'.format(
#                 type(noconsensus_char)))
#         if not plutil.isnumeric(threshold):
#             raise TypeError('`threshold` ({}) must be a numeric'.format(threshold))
#         if threshold < 0 or threshold > 1:
#             raise ValueError('`threshold` ({}) must be 0 <= thresold <= 1'.format(threshold))
#
#         # Check if all of the sequences are the same length
#         agg_seqs = [seq for seq in self.aggregated_seqs.values()]
#         l = None
#         for seq in agg_seqs:
#             if l is None:
#                 l = len(seq)
#             if len(seq) != l:
#                 raise NotImplementedError('Unaligned sequences not implemented yet')
#
#         # Generate the consensus base for each base position
#         consensus_seq = []
#         for i in range(l):
#
#             # Count the number of times each base occurs at position `i`
#             found = {}
#             for seq in agg_seqs:
#                 base = seq[i]
#                 if base not in found:
#                     found[base] = 1
#                 else:
#                     found[base] += 1
#
#             # Set the base
#             if len(found) == 1:
#                 # Every sequence agrees on this base. Set
#                 consensus_seq.append(list(found.keys())[0])
#             else:
#                 # Get the maximum consensus
#                 consensus_percent = -1
#                 consensus_base = None
#                 for base in found:
#                     consensus = 1 - (found[base]/len(agg_seqs))
#                     if consensus > consensus_percent:
#                         consensus_percent = consensus
#                         consensus_base = base
#
#                 # Set the consensus base if it passes the threshold
#                 if consensus_percent >= threshold:
#                     logger.debug('Consensus found for taxon {} in position {} as {}, found ' \
#                         '{}'.format(self.name, i, consensus_base, found))
#                     consensus_seq.append(consensus_base)
#                 else:
#                     logger.debug('No consensus for taxon {} in position {}. Consensus: {}' \
#                         ', found {}'.format(self.name, i, consensus, found))
#                     consensus_seq.append(noconsensus_char)
#
#         # Check for errors with consensus sequence
#         for seq in agg_seqs:
#             perc_dist = diversity.beta.hamming(
#                 seq,
#                 consensus_seq,
#                 ignore_char=noconsensus_char
#             ) / l
#             if perc_dist > 0.03:
#                 logger.warning('Taxon {} has a hamming distance > 3% ({}) to the generated ' \
#                     'consensus sequence {} from individual sequence {}. Check that sequences ' \
#                     'make sense'.format(self.name, perc_dist, consensus_seq, seq))
#
#         # Set the consensus sequence as the OTU's sequence
#         self.sequence = ''.join(c for c in consensus_seq if c != '-')
#
#     def generate_consensus_taxonomy(self, consensus_table: pd.DataFrame=None):
#         """Set the taxonomy of the OTU to the consensus taxonomy of the.
#
#         If one of the ASVs is defined at a lower level than another ASV, use
#         that taxonomy. If ASVs' taxonomies disagree at the species level, use the
#         union of all the species.
#
#         Disagreeing taxonomy
#         --------------------
#         If the taxonomy of the ASVs differ on a taxonomic level other than species, we use an alternate
#         way of naming the OTU. The input `consensus_table` is a `pandas.DataFrame` object showing the
#         taxonomic classification of an OTU. You would get this table by running RDP on the consensus
#         sequence.
#
#         If the consensus table is not given, then we specify the lowest level that they agree. If the
#         consensus table is given, then we use the taxonomy specified in that table.
#
#         Examples
#         --------
#         ```
#         Input:
#          kingdom          phylum                class        order             family  genus       species      asv
#         Bacteria  Proteobacteria  Alphaproteobacteria  Rhizobiales  Bradyrhizobiaceae  Bosea  massiliensis  ASV_722
#         Bacteria  Proteobacteria  Alphaproteobacteria  Rhizobiales  Bradyrhizobiaceae  Bosea            NA  ASV_991
#
#         Output:
#          kingdom          phylum                class        order             family  genus       species
#         Bacteria  Proteobacteria  Alphaproteobacteria  Rhizobiales  Bradyrhizobiaceae  Bosea  massiliensis
#         ```
#
#         ```
#         Input:
#          kingdom          phylum           class              order              family            genus                 species      asv
#         Bacteria  Actinobacteria  Actinobacteria  Bifidobacteriales  Bifidobacteriaceae  Bifidobacterium                      NA  ASV_283
#         Bacteria  Actinobacteria  Actinobacteria  Bifidobacteriales  Bifidobacteriaceae  Bifidobacterium                      NA  ASV_302
#         Bacteria  Actinobacteria  Actinobacteria  Bifidobacteriales  Bifidobacteriaceae  Bifidobacterium    adolescentis/faecale  ASV_340
#         Bacteria  Actinobacteria  Actinobacteria  Bifidobacteriales  Bifidobacteriaceae  Bifidobacterium  choerinum/pseudolongum  ASV_668
#
#         Ouput:
#          kingdom          phylum           class              order              family            genus                                      species
#         Bacteria  Actinobacteria  Actinobacteria  Bifidobacteriales  Bifidobacteriaceae  Bifidobacterium  adolescentis/faecale/choerinum/pseudolongum
#         ```
#
#         Parameters
#         ----------
#         consensus_table : pd.DataFrame
#             Table for resolving conflicts
#         """
#         # Check that all the taxonomies have the same lineage
#         set_to_na = False
#         set_from_table = False
#         for tax in TAX_LEVELS:
#             if set_to_na:
#                 self.taxonomy[tax] = DEFAULT_TAXLEVEL_NAME
#                 continue
#             if set_from_table:
#                 if tax not in consensus_table.columns:
#                     self.taxonomy[tax] = DEFAULT_TAXLEVEL_NAME
#                 else:
#                     self.taxonomy[tax] = consensus_table[tax][self.name]
#                 continue
#             if tax == 'asv':
#                 continue
#             consensus = []
#             for taxonname in self.aggregated_taxa:
#                 if tax == 'species':
#                     aaa = self.aggregated_taxonomies[taxonname][tax].split('/')
#                 else:
#                     aaa = [self.aggregated_taxonomies[taxonname][tax]]
#                 for bbb in aaa:
#                     if bbb in consensus:
#                         continue
#                     else:
#                         consensus.append(bbb)
#             if DEFAULT_TAXLEVEL_NAME in consensus:
#                 if tax == "species":
#                     consensus.remove(DEFAULT_TAXLEVEL_NAME)
#
#             if len(consensus) == 0:
#                 # No taxonomy found at this level
#                 self.taxonomy[tax] = DEFAULT_TAXLEVEL_NAME
#             elif len(consensus) == 1:
#                 # All taxonomies agree
#                 self.taxonomy[tax] = consensus[0]
#             else:
#                 # All taxonomies do not agree
#                 if tax == 'species':
#                     # Take the union of the species
#                     self.taxonomy[tax] = '/'.join(consensus)
#                 else:
#                     # This means that the taxonomy is different on a level different than
#                     logger.critical('{} taxonomy does not agree'.format(self.name))
#                     logger.critical(str(self))
#                     for taxonname in self.aggregated_taxonomies:
#                         logger.warning('{}'.format(list(self.aggregated_taxonomies[taxonname].values())))
#
#                     if consensus_table is not None:
#                         # Set from the table
#                         self.taxonomy[tax] = consensus_table[tax][self.name]
#                         set_from_table = True
#
#                     else:
#                         # Set this taxonomic level and everything below it to NA
#                         self.taxonomy[tax] = DEFAULT_TAXLEVEL_NAME
#                         set_to_na = True
#
#
#
# class TaxaSet(Saveable, Clusterable):
#     """Wraps a set of `` objects. You can get the  object via the
#      id,  name.
#     Provides functionality for aggregating sequeunces and getting subsets for lineages.
#
#     Aggregating/Deaggregating
#     -------------------------
#     s that are aggregated together to become OTUs are used because sequences are
#     very close together. This class provides functionality for aggregating taxa together
#     (`mdsine2.TaxaSet.aggregate_items`) and to deaggregate a specific name from an aggregation
#     (`mdsine2.TaxaSet.deaggregate_item`). If this object is within a `mdsine2.Study` object,
#     MAKE SURE TO CALL THE AGGREGATION FUNCTIONS FROM THE `mdsine2.Study` OBJECT
#     (`mdsine2.Study.aggregate_items`, `mdsine2.Study.deaggregate_item`) so that the reads
#     for the agglomerates and individual taxa can be consistent with the TaxaSet.
#
#     Parameters
#     ----------
#     taxonomy_table : pandas.DataFrame
#         This is the table defining the set. If this is specified, then it is passed into
#         TaxaSet.parse
#
#     See also
#     --------
#     mdsine2.TaxaSet.parse
#     """
#
#     def __init__(self, taxonomy_table: pd.DataFrame=None):
#         self.taxonomy_table = taxonomy_table
#         self.ids = CustomOrderedDict() # Effectively a dictionary (id (int) -> OTU or Taxon)
#         self.names = CustomOrderedDict() # Effectively a dictionary (name (int) -> OTU or Taxon)
#         self.index = [] # List (index (int) -> OTU or Taxon)
#         self._len = 0
#
#         # Add all of the taxa from the dataframe if necessary
#         if taxonomy_table is not None:
#             self.parse(taxonomy_table=taxonomy_table)
#
#     def __contains__(self, key: Union[Taxon, OTU_Fixed, str, int]) -> bool:
#         try:
#             _ = self.__getitem__(key)
#             return True
#         except:
#             return False
#
#     def __getitem__(self, key: Union[Taxon, str, int]):
#         """Get a Taxon/OTU by either its sequence, name, index, or id
#
#         Parameters
#         ----------
#         key : str, int
#             Key to reference the Taxon
#         """
#         if isinstance(key, Taxon):
#             return key
#         if key in self.ids:
#             return self.ids[key]
#         elif plutil.isint(key):
#             return self.index[key]
#         elif key in self.names:
#             return self.names[key]
#         else:
#             raise IndexError('`{}` ({}) was not found as a name, sequence, index, or id'.format(
#                 key, type(key)))
#
#     def __iter__(self) -> Union[Taxon, OTU_Fixed]:
#         """Returns each Taxa obejct in order
#         """
#         for taxon in self.index:
#             yield taxon
#
#     def __len__(self) -> int:
#         """Return the number of taxa in the TaxaSet
#         """
#         return self._len
#
#     @property
#     def n_taxa(self) -> int:
#         """Alias for __len__
#         """
#         return self._len
#
#     def reset(self):
#         """Reset the system
#         """
#         self.taxonomy_table = None
#         self.ids = CustomOrderedDict()
#         self.names = CustomOrderedDict()
#         self.index = []
#         self._len = 0
#
#     def parse(self, taxonomy_table: pd.DataFrame):
#         """Parse a taxonomy table
#
#         `taxonomy_table`
#         ----------------
#         This is a dataframe that contains the taxonomic information for each Taxon.
#         The columns that must be included are:
#             'name' : name of the taxon
#             'sequence' : sequence of the taxon
#         All of the taxonomy specifications are optional:
#             'kingdom' : kingdom taxonomy
#             'phylum' : phylum taxonomy
#             'class' : class taxonomy
#             'family' : family taxonomy
#             'genus' : genus taxonomy
#             'species' : species taxonomy
#
#         Note that if the `name` column is not in the columns, this assumes that the
#         OTU names are the index already.
#
#         Parameters
#         ----------
#         taxonomy_table : pandas.DataFrame, Optional
#             DataFrame containing the required information (Taxonomy, sequence).
#             If nothing is passed in, it will be an empty TaxaSet
#         """
#         logger.info('TaxaSet parsng new taxonomy table. Resetting')
#         self.taxonomy_table = taxonomy_table
#         self.ids = CustomOrderedDict()
#         self.names = CustomOrderedDict()
#         self.index = []
#         self._len = 0
#
#         self.taxonomy_table = taxonomy_table
#         taxonomy_table = taxonomy_table.rename(str.lower, axis='columns')
#         if 'name' not in taxonomy_table.columns:
#             logger.info('No `name` found - assuming index is the name')
#         else:
#             taxonomy_table = taxonomy_table.set_index('name')
#         if SEQUENCE_COLUMN_LABEL not in taxonomy_table.columns:
#             raise ValueError('`"{}"` ({}) not found as a column in `taxonomy_table`'.format(
#                 SEQUENCE_COLUMN_LABEL, taxonomy_table.columns))
#
#         for tax in TAX_LEVELS[:-1]:
#             if tax not in taxonomy_table.columns:
#                 logger.info('Adding in `{}` column'.format(tax))
#                 taxonomy_table = taxonomy_table.insert(-1, tax,
#                     [DEFAULT_TAXLEVEL_NAME for _ in range(len(taxonomy_table.index))])
#
#         for i, name in enumerate(taxonomy_table.index):
#             seq = taxonomy_table[SEQUENCE_COLUMN_LABEL][name]
#             taxon = Taxon(name=name, sequence=seq, idx=self._len)
#             taxon.set_taxonomy(
#                 tax_kingdom=taxonomy_table.loc[name]['kingdom'],
#                 tax_phylum=taxonomy_table.loc[name]['phylum'],
#                 tax_class=taxonomy_table.loc[name]['class'],
#                 tax_order=taxonomy_table.loc[name]['order'],
#                 tax_family=taxonomy_table.loc[name]['family'],
#                 tax_genus=taxonomy_table.loc[name]['genus'],
#                 tax_species=taxonomy_table.loc[name]['species'])
#
#             self.ids[taxon.id] = taxon
#             self.names[taxon.name] = taxon
#             self.index.append(taxon)
#             self._len += 1
#
#         self.ids.update_order()
#         self.names.update_order()
#
#     def add_taxon(self, name: str, sequence: Iterator[str]=None):
#         """Adds a taxon to the set
#
#         Parameters
#         ----------
#         name : str
#             This is the name of the taxon
#         sequence : str
#             This is the sequence of the taxon
#         """
#         taxon = Taxon(name=name, sequence=sequence, idx=self._len)
#         self.ids[taxon.id] = taxon
#         self.names[taxon.name] = taxon
#         self.index.append(taxon)
#
#         # update the order of the taxa
#         self.ids.update_order()
#         self.names.update_order()
#         self._len += 1
#
#         return self
#
#     def del_taxon(self, taxon: Union[Taxon, OTU_Fixed, str, int]):
#         """Deletes the taxon from the set.
#
#         Parameters
#         ----------
#         taxon : str, int, Taxon
#             Can either be the name, sequence, or the ID of the taxon
#         """
#         # Get the ID
#         taxon = self[taxon]
#         oidx = self.ids.index[taxon.id]
#
#         # Delete the taxon from everything
#         # taxon = self[taxon]
#         self.ids.pop(taxon.id, None)
#         self.names.pop(taxon.name, None)
#         self.index.pop(oidx)
#
#         # update the order of the taxa
#         self.ids.update_order()
#         self.names.update_order()
#
#         # Update the indices of the taxa
#         # Since everything points to the same object we only need to do it once
#         for aidx, taxon in enumerate(self.index):
#             taxon.idx = aidx
#
#         self._len -= 1
#         return self
#
#     def taxonomic_similarity(self,
#         oid1: Union[Taxon, OTU_Fixed, str, int],
#         oid2: Union[Taxon, OTU_Fixed, str, int]) -> float:
#         """Calculate the taxonomic similarity between taxon1 and taxon2
#         Iterates through most broad to least broad taxonomic level and
#         returns the fraction that are the same.
#
#         Example:
#             taxon1.taxonomy = (A,B,C,D)
#             taxon2.taxonomy = (A,B,E,F)
#             similarity = 0.5
#
#             taxon1.taxonomy = (A,B,C,D)
#             taxon2.taxonomy = (A,B,C,F)
#             similarity = 0.75
#
#             taxon1.taxonomy = (A,B,C,D)
#             taxon2.taxonomy = (A,B,C,D)
#             similarity = 1.0
#
#             taxon1.taxonomy = (X,Y,Z,M)
#             taxon2.taxonomy = (A,B,E,F)
#             similarity = 0.0
#
#         Parameters
#         ----------
#         oid1, oid2 : str, int
#             The name, id, or sequence for the taxon
#         """
#         if oid1 == oid2:
#             return 1
#         taxon1 = self[oid1].get_lineage()
#         taxon2 = self[oid2].get_lineage()
#         i = 0
#         for a in taxon1:
#             if a == taxon2[i]:
#                 i += 1
#             else:
#                 break
#         return i/8 # including asv
#
#     # DEPRECATED
#     def aggregate_items(self, anchor: Union[Taxon, OTU_Fixed, str, int], other: Union[Taxon, OTU_Fixed, str, int]):
#         """Create an OTU with the anchor `anchor` and other taxon  `other`.
#         The aggregate takes the sequence and the taxonomy from the anchor.
#
#         Parameters
#         ----------
#         anchor, other : str, int, mdsine2.Taxon, mdsine2.OTU
#             These are the Taxa/Aggregates that you're joining together. The anchor is
#             the one you are setting the sequeunce and taxonomy to
#
#         Returns
#         -------
#         mdsine2.OTU
#             This is the new aggregated taxon containing anchor and other
#         """
#         anchor = self[anchor]
#         other = self[other]
#
#         agg = OTU_Fixed(anchor=anchor, other=other)
#
#         self.index[agg.idx] = agg
#         self.index.pop(other.idx)
#
#         self.ids = CustomOrderedDict()
#         self.names = CustomOrderedDict()
#
#         for idx, taxon in enumerate(self.index):
#             taxon.idx = idx
#             self.ids[taxon.id] = taxon
#             self.names[taxon.name] = taxon
#
#         # update the order of the taxa
#         self.ids.update_order()
#         self.names.update_order()
#
#         self._len = len(self.index)
#         return agg
#
#     def deaggregate_item(self, agg: Union[Taxon, OTU_Fixed, str, int], other: str) -> Taxon:
#         """Deaggregate the sequence `other` from OTU `agg`.
#         `other` is then appended to the end
#
#         Parameters
#         ----------
#         agg : OTU, str
#             This is an OTU with multiple sequences contained. Must
#             have the name `other` in there
#         other : str
#             This is the name of the taxon that should be taken out of `agg`
#
#         Returns
#         -------
#         mdsine2.Taxon
#             This is the deaggregated taxon
#         """
#         agg = self[agg]
#         if not isotu(agg):
#             raise TypeError('`agg` ({}) must be an OTU'.format(type(agg)))
#         if not plutil.isstr(other):
#             raise TypeError('`other` ({}) must be a str'.format(type(other)))
#         if other not in agg.aggregated_taxa:
#             raise ValueError('`other` ({}) is not contained in `agg` ({}) ({})'.format(
#                 other, agg.name, agg.aggregated_taxa))
#
#         other = Taxon(name=other, sequence=agg.aggregated_seqs[other], idx=self._len)
#         other.taxonomy = agg.aggregated_taxonomies[other.name]
#         agg.aggregated_seqs.pop(other.name, None)
#         agg.aggregated_taxa.remove(other.name)
#         agg.aggregated_taxonomies.pop(other.name, None)
#
#         self.index.append(other)
#         self.ids[other.id] = other
#         self.names[other.name] = other
#
#         self.ids.update_order()
#         self.names.update_order()
#         self._len += 1
#         return other
#
#     def rename(self, prefix: str, zero_based_index: bool=False):
#         """Rename the contents based on their index:
#
#         Example
#         -------
#         Names before in order:
#         [Taxon_22, Taxon_9982, TUDD_8484]
#
#         Calling taxa.rename(prefix='OTU')
#         New names:
#         [OTU_1, OTU_2, OTU_3]
#
#         Calling taxa.rename(prefix='OTU', zero_based_index=True)
#         New names:
#         [OTU_0, OTU_1, OTU_2]
#
#         Parameters
#         ----------
#         prefix : str
#             This is the prefix of the new taxon. The name of the taxa will change
#             to `'{}_{}'.format(prefix, index)`
#         zero_based_index : bool
#             If this is False, then we start the enumeration of the taxa from 1
#             instead of 0. If True, then the enumeration starts at 0
#         """
#         if not plutil.isstr(prefix):
#             raise TypeError('`prefix` ({}) must be a str'.format(type(prefix)))
#         if not plutil.isbool(zero_based_index):
#             raise TypeError('`zero_based_index` ({}) must be a bool'.format(
#                 type(zero_based_index)))
#
#         offset = 0
#         if not zero_based_index:
#             offset = 1
#
#         self.names = CustomOrderedDict()
#         for taxon in self.index:
#             newname = prefix + '_{}'.format(int(taxon.idx + offset))
#             taxon.name = newname
#             self.names[taxon.name] = taxon
#
#     def generate_consensus_seqs(self, threshold: float=0.65, noconsensus_char: str='N'):
#         """Generate the consensus sequence for all of the taxa given the sequences
#         of all the contained ASVs of the respective OTUs
#
#         Parameters
#         ----------
#         threshold : float
#             This is the threshold for consensus (0 < threshold <= 1)
#         noconsensus_char : str
#             This is the character to replace
#         """
#         for taxon in self:
#             if isotu(taxon):
#                 taxon.generate_consensus_seq(
#                     threshold=threshold,
#                     noconsensus_char=noconsensus_char)
#
#     def generate_consensus_taxonomies(self, consensus_table: pd.DataFrame=None):
#         """Generates the consensus taxonomies for all of the OTUs within the TaxaSet.
#         For details on the algorithm - see `OTU.generate_consensus_taxonomy`
#
#         See Also
#         --------
#         mdsine2.pylab.base.OTU.generate_consensus_taxonomy
#         """
#         for taxon in self:
#             if isotu(taxon):
#                 taxon.generate_consensus_taxonomy(consensus_table=consensus_table)
#
#     def write_taxonomy_to_csv(self, path: str=None, sep:str='\t') -> pd.DataFrame:
#         """Write the taxon names, sequences, and taxonomy to a table. If a path
#         is passed in, then write to that table
#
#         Parameters
#         ----------
#         path : str
#             This is the location to save the metadata file
#         sep : str
#             This is the separator of the table
#         """
#         columns = ['name', 'sequence', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
#         data = []
#
#         for taxon in self:
#             temp = [taxon.name, taxon.sequence]
#             for taxlevel in TAX_LEVELS[:-1]:
#                 temp.append(taxon.taxonomy[taxlevel])
#             data.append(temp)
#
#         df = pd.DataFrame(data, columns=columns)
#         if path is not None:
#             df.to_csv(path, sep=sep, index=False, header=True)
#         return df
#
#     def make_random(self, n_taxa: int):
#         """Reset the TaxaSet so that it is composed of `n_taxa` random `Taxon` objects.
#         You would use this function for debugging or testing.
#
#         Parameters
#         ----------
#         n_taxa : int
#             Number of taxa to initialize the `TaxaSet` object with
#         """
#         import random
#
#         self.reset()
#         letters = ['A', 'T', 'G', 'C']
#         for i in range(n_taxa):
#             seq = ''.join(random.choice(letters) for _ in range(50))
#             self.add_taxon(
#                 name='Taxon_{}'.format(i+1),
#                 sequence=seq)


class TaxaSet(Clusterable):
    """Wraps a set of `` objects. You can get the  object via the
     id,  name.
    Provides functionality for aggregating sequeunces and getting subsets for lineages.

    Aggregating/Deaggregating
    -------------------------
    s that are aggregated together to become OTUs are used because sequences are
    very close together. This class provides functionality for aggregating taxa together
    (`mdsine2.TaxaSet.aggregate_items`) and to deaggregate a specific name from an aggregation
    (`mdsine2.TaxaSet.deaggregate_item`). If this object is within a `mdsine2.Study` object,
    MAKE SURE TO CALL THE AGGREGATION FUNCTIONS FROM THE `mdsine2.Study` OBJECT
    (`mdsine2.Study.aggregate_items`, `mdsine2.Study.deaggregate_item`) so that the reads
    for the agglomerates and individual taxa can be consistent with the TaxaSet.

    Parameters
    ----------
    taxonomy_table : pandas.DataFrame
        This is the table defining the set. If this is specified, then it is passed into
        TaxaSet.parse

    See also
    --------
    mdsine2.TaxaSet.parse
    """

    def __init__(self, taxonomy_table: pd.DataFrame=None):
        self.taxonomy_table = taxonomy_table
        self.ids = CustomOrderedDict() # Effectively a dictionary (id (int) -> OTU or Taxon)
        self.names = CustomOrderedDict() # Effectively a dictionary (name (int) -> OTU or Taxon)
        self.index = [] # List (index (int) -> OTU or Taxon)
        self._len = 0

        # Add all of the taxa from the dataframe if necessary
        if taxonomy_table is not None:
            self.parse(taxonomy_table=taxonomy_table)

    def __contains__(self, key: Union[Taxon, OTU, str, int]) -> bool:
        try:
            self[key]
            return True
        except:
            return False

    def __getitem__(self, key: Union[Taxon, OTU, str, int]):
        """Get a Taxon/OTU by either its sequence, name, index, or id

        Parameters
        ----------
        key : str, int
            Key to reference the Taxon
        """
        if isinstance(key, Taxon):
            return key
        if key in self.ids:
            return self.ids[key]
        elif plutil.isint(key):
            return self.index[key]
        elif key in self.names:
            return self.names[key]
        else:
            raise IndexError('`{}` ({}) was not found as a name, sequence, index, or id'.format(
                key, type(key)))

    def __iter__(self) -> Union[Taxon, OTU]:
        """Returns each Taxa obejct in order
        """
        for taxon in self.index:
            yield taxon

    def __len__(self) -> int:
        """Return the number of taxa in the TaxaSet
        """
        return self._len

    @property
    def n_taxa(self) -> int:
        """Alias for __len__
        """
        return self._len

    def reset(self):
        """Reset the system
        """
        self.taxonomy_table = None
        self.ids = CustomOrderedDict()
        self.names = CustomOrderedDict()
        self.index = []
        self._len = 0

    def parse(self, taxonomy_table: pd.DataFrame):
        """Parse a taxonomy table

        `taxonomy_table`
        ----------------
        This is a dataframe that contains the taxonomic information for each Taxon.
        The columns that must be included are:
            'name' : name of the taxon
            'sequence' : sequence of the taxon
        All of the taxonomy specifications are optional:
            'kingdom' : kingdom taxonomy
            'phylum' : phylum taxonomy
            'class' : class taxonomy
            'family' : family taxonomy
            'genus' : genus taxonomy
            'species' : species taxonomy

        Note that if the `name` column is not in the columns, this assumes that the
        OTU names are the index already.

        Parameters
        ----------
        taxonomy_table : pandas.DataFrame, Optional
            DataFrame containing the required information (Taxonomy, sequence).
            If nothing is passed in, it will be an empty TaxaSet
        """
        logger.info('TaxaSet parsng new taxonomy table. Resetting')
        self.taxonomy_table = taxonomy_table
        self.ids = CustomOrderedDict()
        self.names = CustomOrderedDict()
        self.index = []
        self._len = 0

        self.taxonomy_table = taxonomy_table
        taxonomy_table = taxonomy_table.rename(str.lower, axis='columns')
        if 'name' not in taxonomy_table.columns:
            logger.info('No `name` found - assuming index is the name')
        else:
            taxonomy_table = taxonomy_table.set_index('name')
        if SEQUENCE_COLUMN_LABEL not in taxonomy_table.columns:
            raise ValueError('`"{}"` ({}) not found as a column in `taxonomy_table`'.format(
                SEQUENCE_COLUMN_LABEL, taxonomy_table.columns))

        for tax in TAX_LEVELS[:-1]:
            if tax not in taxonomy_table.columns:
                logger.info('Adding in `{}` column'.format(tax))
                taxonomy_table = taxonomy_table.insert(-1, tax,
                    [DEFAULT_TAXLEVEL_NAME for _ in range(len(taxonomy_table.index))])

        for i, name in enumerate(taxonomy_table.index):
            seq = taxonomy_table[SEQUENCE_COLUMN_LABEL][name]
            taxon = Taxon(name=name, sequence=seq, idx=self._len)
            taxon.set_taxonomy(
                tax_kingdom=taxonomy_table.loc[name]['kingdom'],
                tax_phylum=taxonomy_table.loc[name]['phylum'],
                tax_class=taxonomy_table.loc[name]['class'],
                tax_order=taxonomy_table.loc[name]['order'],
                tax_family=taxonomy_table.loc[name]['family'],
                tax_genus=taxonomy_table.loc[name]['genus'],
                tax_species=taxonomy_table.loc[name]['species'])

            self.ids[taxon.id] = taxon
            self.names[taxon.name] = taxon
            self.index.append(taxon)
            self._len += 1

        self.ids.update_order()
        self.names.update_order()

    def add(self, taxon: Taxon):
        self.ids[taxon.id] = taxon
        self.names[taxon.name] = taxon
        self.index.append(taxon)

        # update the order of the taxa
        self.ids.update_order()
        self.names.update_order()
        self._len += 1

    def add_taxon(self, name: str, sequence: Iterator[str]=None):
        """Adds a taxon to the set

        Parameters
        ----------
        name : str
            This is the name of the taxon
        sequence : str
            This is the sequence of the taxon
        """
        self.add(Taxon(name=name, sequence=sequence, idx=self._len))

    def del_taxon(self, taxon: Union[Taxon, OTU, str, int]):
        """Deletes the taxon from the set.

        Parameters
        ----------
        taxon : str, int, Taxon
            Can either be the name, sequence, or the ID of the taxon
        """
        # Get the ID
        taxon = self[taxon]
        oidx = self.ids.index[taxon.id]

        # Delete the taxon from everything
        # taxon = self[taxon]
        self.ids.pop(taxon.id, None)
        self.names.pop(taxon.name, None)
        self.index.pop(oidx)

        # update the order of the taxa
        self.ids.update_order()
        self.names.update_order()

        # Update the indices of the taxa
        # Since everything points to the same object we only need to do it once
        for aidx, taxon in enumerate(self.index):
            taxon.idx = aidx

        self._len -= 1
        return self

    def taxonomic_similarity(self,
        oid1: Union[Taxon, OTU, str, int],
        oid2: Union[Taxon, OTU, str, int]) -> float:
        """Calculate the taxonomic similarity between taxon1 and taxon2
        Iterates through most broad to least broad taxonomic level and
        returns the fraction that are the same.

        Example:
            taxon1.taxonomy = (A,B,C,D)
            taxon2.taxonomy = (A,B,E,F)
            similarity = 0.5

            taxon1.taxonomy = (A,B,C,D)
            taxon2.taxonomy = (A,B,C,F)
            similarity = 0.75

            taxon1.taxonomy = (A,B,C,D)
            taxon2.taxonomy = (A,B,C,D)
            similarity = 1.0

            taxon1.taxonomy = (X,Y,Z,M)
            taxon2.taxonomy = (A,B,E,F)
            similarity = 0.0

        Parameters
        ----------
        oid1, oid2 : str, int
            The name, id, or sequence for the taxon
        """
        if oid1 == oid2:
            return 1
        taxon1 = self[oid1].get_lineage()
        taxon2 = self[oid2].get_lineage()
        i = 0
        for a in taxon1:
            if a == taxon2[i]:
                i += 1
            else:
                break
        return i/8 # including asv

    def aggregate_items(self, groupings: List[List[Taxon]]) -> 'OTUTaxaSet':
        """Create an OTU with the anchor `anchor` and other taxon  `other`.
        The aggregate takes the sequence and the taxonomy from the anchor.

        Returns
        -------
        mdsine2.OTU
            This is the new aggregated taxon containing anchor and other
        """
        other = OTUTaxaSet()
        for gidx, grouping in enumerate(groupings):
            otu = OTU(component=grouping, idx=gidx)
            other.add(otu)
        return other

    def rename(self, prefix: str, zero_based_index: bool=False):
        """Rename the contents based on their index:

        Example
        -------
        Names before in order:
        [Taxon_22, Taxon_9982, TUDD_8484]

        Calling taxa.rename(prefix='OTU')
        New names:
        [OTU_1, OTU_2, OTU_3]

        Calling taxa.rename(prefix='OTU', zero_based_index=True)
        New names:
        [OTU_0, OTU_1, OTU_2]

        Parameters
        ----------
        prefix : str
            This is the prefix of the new taxon. The name of the taxa will change
            to `'{}_{}'.format(prefix, index)`
        zero_based_index : bool
            If this is False, then we start the enumeration of the taxa from 1
            instead of 0. If True, then the enumeration starts at 0
        """
        if not plutil.isstr(prefix):
            raise TypeError('`prefix` ({}) must be a str'.format(type(prefix)))
        if not plutil.isbool(zero_based_index):
            raise TypeError('`zero_based_index` ({}) must be a bool'.format(
                type(zero_based_index)))

        offset = 0
        if not zero_based_index:
            offset = 1

        self.names = CustomOrderedDict()
        for taxon in self.index:
            newname = prefix + '_{}'.format(int(taxon.idx + offset))
            taxon.name = newname
            self.names[taxon.name] = taxon

    def write_taxonomy_to_csv(self, path: str=None, sep:str='\t') -> pd.DataFrame:
        """Write the taxon names, sequences, and taxonomy to a table. If a path
        is passed in, then write to that table

        Parameters
        ----------
        path : str
            This is the location to save the metadata file
        sep : str
            This is the separator of the table
        """
        columns = ['name', 'sequence', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
        data = []

        for taxon in self:
            temp = [taxon.name, taxon.sequence]
            for taxlevel in TAX_LEVELS[:-1]:
                temp.append(taxon.taxonomy[taxlevel])
            data.append(temp)

        df = pd.DataFrame(data, columns=columns)
        if path is not None:
            df.to_csv(path, sep=sep, index=False, header=True)
        return df

    def make_random(self, n_taxa: int):
        """Reset the TaxaSet so that it is composed of `n_taxa` random `Taxon` objects.
        You would use this function for debugging or testing.

        Parameters
        ----------
        n_taxa : int
            Number of taxa to initialize the `TaxaSet` object with
        """
        import random

        self.reset()
        letters = ['A', 'T', 'G', 'C']
        for i in range(n_taxa):
            seq = ''.join(random.choice(letters) for _ in range(50))
            self.add_taxon(
                name='Taxon_{}'.format(i+1),
                sequence=seq)


class OTUTaxaSet(TaxaSet):
    def __init__(self, taxonomy_table: pd.DataFrame = None):
        super().__init__(taxonomy_table=taxonomy_table)

    def generate_consensus_seqs(self, threshold: float=0.65, noconsensus_char: str='N'):
        """Generate the consensus sequence for all of the taxa given the sequences
        of all the contained ASVs of the respective OTUs

        Parameters
        ----------
        threshold : float
            This is the threshold for consensus (0 < threshold <= 1)
        noconsensus_char : str
            This is the character to replace
        """
        for taxon in self:
            taxon.generate_consensus_seq(
                threshold=threshold,
                noconsensus_char=noconsensus_char)

    def generate_consensus_taxonomies(self, consensus_table: pd.DataFrame=None):
        """Generates the consensus taxonomies for all of the OTUs within the TaxaSet.
        For details on the algorithm - see `OTU.generate_consensus_taxonomy`

        See Also
        --------
        mdsine2.pylab.base.OTU.generate_consensus_taxonomy
        """
        for taxon in self:
            taxon.generate_consensus_taxonomy(consensus_table=consensus_table)

    def deaggregate_item(self, agg: Union[OTU, str, int], other: str) -> Taxon:
        """Deaggregate the sequence `other` from OTU `agg`.
        `other` is then appended to the end

        Parameters
        ----------
        agg : OTU, str
            This is an OTU with multiple sequences contained. Must
            have the name `other` in there
        other : str
            This is the name of the taxon that should be taken out of `agg`

        Returns
        -------
        mdsine2.Taxon
            This is the deaggregated taxon
        """
        agg = self.__getitem__(agg)
        if not plutil.isstr(other):
            raise TypeError('`other` ({}) must be a str'.format(type(other)))
        if other not in agg.aggregated_taxa:
            raise ValueError('`other` ({}) is not contained in `agg` ({}) ({})'.format(
                other, agg.name, agg.aggregated_taxa))

        other = Taxon(name=other, sequence=agg.aggregated_seqs[other], idx=self._len)
        other.taxonomy = agg.aggregated_taxonomies[other.name]
        agg.aggregated_seqs.pop(other.name, None)
        agg.aggregated_taxa.remove(other.name)
        agg.aggregated_taxonomies.pop(other.name, None)

        self.index.append(other)
        self.ids[other.id] = other
        self.names[other.name] = other

        self.ids.update_order()
        self.names.update_order()
        self._len += 1
        return other
