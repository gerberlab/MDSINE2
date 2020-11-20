import pandas as pd
from os.path import dirname, join
import os
import sys
from ete3 import Tree

from .pylab import ASVSet, Study

__all__ = ['gibson', 'parse']

class _Gibson:
    '''Wrapper for the functionality of loading the gibson dataset. Called through
    `mdsine2.dataset.gibson`. See 
    '''
    _HEALTHY_SUBJECTS = set(['2', '3', '4', '5'])
    _UC_SUBJECTS = set(['6', '7', '8', '9', '10'])

    @staticmethod
    def load_taxonomy(species_assignment):
        '''Load the taxonomy assignment

        Parameters
        ----------
        species_assignment : str, None
            How to assign the species
            If 'silva', only use the Silva species assignment
            If 'rdp', only use RDP 138 assignment
            If 'both', combine both the RDP and Silva species assignment
            If None, have no species assignment and just return the taxonomy provided by DADA2
        '''
        if species_assignment == 'silva':
            path = join(dirname(__file__),'datasets/gibson_dataset' ,'silva_species.tsv')
            taxonomy = pd.read_csv(path, sep='\t', index_col=0)
        elif species_assignment == 'rdp':
            path = join(dirname(__file__),'datasets/gibson_dataset' ,'rdp_species.tsv')
            taxonomy = pd.read_csv(path, sep='\t', index_col=0)
        elif species_assignment == 'both':
            rdp = join(dirname(__file__),'datasets/gibson_dataset' ,'rdp_species.tsv')
            rdp = pd.read_csv(rdp, sep='\t', index_col=0)
            silva = join(dirname(__file__),'datasets/gibson_dataset' ,'silva_species.tsv')
            silva = pd.read_csv(silva, sep='\t', index_col=0)

            rdp.columns = rdp.columns.str.lower()
            silva.columns = silva.columns.str.lower()

            data = []

            for iii, aname in enumerate(rdp.index):

                rdp_spec = rdp['species'][aname]
                silva_spec = silva['species'][aname]
                tmp = rdp.iloc[iii, :-1].to_list()

                if type(rdp_spec) == float:
                    rdp_spec = 'NA'
                if type(silva_spec) == float:
                    silva_spec = 'NA'
                rdp_spec = rdp_spec.split('/')
                silva_spec = silva_spec.split('/')

                both = list(set(rdp_spec + silva_spec))
                if 'NA' in both and len(both) > 1:
                    both.remove('NA')

                if len(both) > 2:
                    both = 'NA'
                else:
                    both = '/'.join(both)
                tmp.append(both)
                data.append(tmp)

            taxonomy = pd.DataFrame(data, columns=rdp.columns, index=rdp.index)

        elif species_assignment is None:
            taxonomy = join(dirname(__file__),'datasets/gibson_dataset' ,'rdp_species.tsv')
            taxonomy = pd.read_csv(taxonomy, sep='\t')
            taxonomy['species'][:] = 'NA'
        else:
            raise ValueError('`species_assignment` ({}) is not recognized.'.format(species_assignment))
        return taxonomy
    
    @staticmethod
    def load_perturbations():
        '''Load the perturbations for Gibson dataset as a `pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
        '''
        path = join(dirname(__file__),'datasets/gibson_dataset' ,'perturbations.tsv')
        df = pd.read_csv(path, sep='\t')
        return df

    @staticmethod
    def load_qpcr_masses(dset=None):
        '''Load qpcr masses table into a `pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
        '''
        path = join(dirname(__file__),'datasets/gibson_dataset' ,'qpcr.tsv')
        df = pd.read_csv(path, sep='\t')
        df = df.set_index('sampleID')

        if dset is not None:
            keep = _Gibson._get_sampleids(eles=df.index, dset=dset)
            df = df.loc[keep]
        return df

    @staticmethod
    def load_reads(dset=None):
        '''Load reads table into a `pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
        '''
        path = join(dirname(__file__),'datasets/gibson_dataset' ,'counts.tsv')
        df = pd.read_csv(path, sep='\t', index_col=0)

        if dset is not None:
            keep = _Gibson._get_sampleids(eles=df.columns, dset=dset)
            df = df[keep]
        return df

    @staticmethod
    def load_sampleid(dset=None):
        '''Load sample ID table into a `pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
        '''
        path = join(dirname(__file__),'datasets/gibson_dataset' ,'metadata.tsv')
        df = pd.read_csv(path, sep='\t')
        df = df.set_index('sampleID')

        if dset is not None:
            keep = _Gibson._get_sampleids(eles=df.index, dset=dset)
            df = df.loc[keep]
        return df

    @staticmethod
    def _get_sampleids(eles, dset):
        keep = []

        for sampleid in eles:

            mid = sampleid.split('-')[0]
            if mid == '1':
                continue

            if dset == 'inoculum' and _Gibson._is_inoc_sampleid(sampleid):
                keep.append(sampleid)
            elif dset == 'replicates' and _Gibson._is_negbin_sampleid(sampleid):
                keep.append(sampleid)
            elif dset == 'healthy' and _Gibson._is_healthy_sampleid(sampleid):
                keep.append(sampleid)
            elif dset == 'uc' and _Gibson._is_uc_sampleid(sampleid):
                keep.append(sampleid)

        return keep

    @staticmethod
    def _is_healthy_sampleid(sampleid):
        '''Checks whether the sample id is for healty or UC

        Parameters
        ----------
        sampleid : str
            Sample ID

        Returns
        -------
        bool
        '''
        sid = sampleid.split('-')[0]
        return sid in _Gibson._HEALTHY_SUBJECTS

    @staticmethod
    def _is_uc_sampleid(sampleid):
        '''Checks whether the sample id is for healty or UC

        Parameters
        ----------
        sampleid : str
            Sample ID

        Returns
        -------
        bool
        '''
        sid = sampleid.split('-')[0]
        return sid in _Gibson._UC_SUBJECTS

    @staticmethod
    def _is_negbin_sampleid(sampleid):
        '''Checks whether the sample id is for learning the negative
        binomial dispersion parameters

        Parameters
        ----------
        sampleid : str
            Sample ID

        Returns
        -------
        bool
        '''
        return 'M2-D' in sampleid

    @staticmethod
    def _is_inoc_sampleid(sampleid):
        '''Checks whether the sample id is from the inoculum

        Parameters
        ----------
        sampleid : str
            Sample ID

        Returns
        -------
        bool
        '''
        return 'inoculum' in sampleid

    @staticmethod
    def gibson_phylogenetic_tree(with_reference_sequences):
        '''Load phylogenetic tree into an `ete3.Tree` object.

        Parameters
        ----------
        with_reference_sequences : bool
            If True, load the phylogenetic tree with the reference sequeunces from
            placement. If False, only include the ASVs and not the reference 
            sequences

        Returns
        -------
        ete3.Tree
        '''
        if with_reference_sequences:
            path = join(dirname(__file__),'datasets/gibson_dataset' ,'phylogenetic_tree_with_reference.nhx')
        else:
            path = join(dirname(__file__),'datasets/gibson_dataset' ,'phylogenetic_tree_w_branch_len_preserved.nhx')

        return Tree(path)


def gibson(dset=None, as_df=False, with_perturbations=True, species_assignment='both'):
    '''Load the Gibson dataset.

    Returns either a `mdsine2.Study` object or the `pandas.DataFrame` objects that
    that comprise the Gibson dataset.

    Parameters
    ----------
    dset : str, None
        If 'healthy', return the Healthy cohort.
        If 'UC', return the Ulcerative Colitis cohort.
        If 'replicates', return the replicate samples used for learning negative binomial
        dispersion parameters
        If 'inoculum', return the samples used for the inoculum
        If None, return all the data.
    as_df : bool
        If True, return the four dataframes that make up the data as a dict (str->pandas.DataFrame)
        dict: 
            'taxonomy' -> Taxonomy table
            'reads' -> Reads table
            'qpcr' -> qPCR table
            'metadata' -> metadata table
    with_perturbations : bool
        If True, load in the perturbations. Otherwise do not load them
    species_assignment : str, None
        How to assign the species
        If 'silva', only use the Silva species assignment
        If 'rdp', only use RDP 138 assignment
        If 'both', combine both the RDP and Silva species assignment
        If None, have no species assignment and just return the taxonomy provided by DADA2

    Returns
    -------
    mdsine2.Study OR dict
    '''
    # Load the taxonomy assignment
    taxonomy = _Gibson.load_taxonomy(species_assignment=species_assignment)
    metadata = _Gibson.load_sampleid(dset=dset)
    reads = _Gibson.load_reads(dset=dset)
    qpcr = _Gibson.load_qpcr_masses(dset=dset)
    if with_perturbations:
        perturbations = _Gibson.load_perturbations()
    else:
        perturbations = None

    if as_df:
        return {'metadata': metadata, 'taxonomy': taxonomy, 
            'reads': reads, 'qpcr':qpcr, 'perturbations': perturbations}
    else:
        asvs = ASVSet(taxonomy_table=taxonomy)
        study = Study(asvs=asvs, name=dset)
        study.parse(
            metadata=metadata,
            reads=reads,
            qpcr=qpcr,
            perturbations=perturbations)
        return study

def parse(metadata, reads=None, qpcr=None, perturbations=None, sep='\t'):
    '''Parse a dataset. Acts as a wrapper for `mdsine2.Study.parse`
    '''
    pass

