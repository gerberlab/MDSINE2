import pandas as pd
from os.path import dirname, join
import os
import sys
from ete3 import Tree

from .pylab import ASVSet, Study

_HEALTHY_SUBJECTS = set(['2', '3', '4', '5'])
_UC_SUBJECTS = set(['6', '7', '8', '9', '10'])

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
    module_path = dirname(__file__)
    if with_reference_sequences:
        path = join(dirname(__file__),'datasets/gibson_dataset' ,'phylogenetic_tree_with_reference.nhx')
    else:
        path = join(dirname(__file__),'datasets/gibson_dataset' ,'phylogenetic_tree_w_branch_len_preserved.nhx')

    return Tree(path)

def gibson(dset=None, as_df=False, species_assignment='both'):
    '''Load the Gibson dataset.

    Returns either a `mdsine2.Study` object or the `pandas.DataFrame` objects that
    that comprise the Gibson dataset.

    TODO: inline documentation about options, need to explain healthy, dfs, and taxonomy

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

    metadata = _load_sampleid(dset=dset)
    if dset in ['inoculum', 'replicates']:
        # Remove the perturbations
        metadata = metadata[[col for col in metadata.columns if 'perturbation:' not in col]]
    reads = _load_reads(dset=dset)
    qpcr = _load_qpcr_masses(dset=dset)

    if as_df:
        return {'metadata': metadata, 'taxonomy': taxonomy, 
            'reads': reads, 'qpcr':qpcr}
    else:

        # print(taxonomy.head())
        # print(metadata.head())
        # print(reads.head())
        # print(qpcr.head())

        asvs = ASVSet(taxonomy_table=taxonomy)
        study = Study(asvs=asvs)
        study.parse_samples(
            metadata=metadata,
            reads=reads,
            qpcr=qpcr)
        return study

def _load_qpcr_masses(dset=None):
    '''Load qpcr masses table into a `pandas.DataFrame`.

    Returns
    -------
    pandas.DataFrame
    '''
    path = join(dirname(__file__),'datasets/gibson_dataset' ,'qpcr.tsv')
    df = pd.read_csv(path, sep='\t')
    df = df.set_index('sampleID')

    if dset is not None:
        keep = _get_sampleids(eles=df.index, dset=dset)
        df = df.loc[keep]
    return df

def _load_reads(dset=None):
    '''Load reads table into a `pandas.DataFrame`.

    Returns
    -------
    pandas.DataFrame
    '''
    path = join(dirname(__file__),'datasets/gibson_dataset' ,'counts.tsv')
    df = pd.read_csv(path, sep='\t', index_col=0)

    if dset is not None:
        keep = _get_sampleids(eles=df.columns, dset=dset)
        df = df[keep]
    return df

def _load_sampleid(dset=None):
    '''Load sample ID table into a `pandas.DataFrame`.

    Returns
    -------
    pandas.DataFrame
    '''
    path = join(dirname(__file__),'datasets/gibson_dataset' ,'metadata.tsv')
    df = pd.read_csv(path, sep='\t')
    df = df.set_index('sampleID')

    if dset is not None:
        keep = _get_sampleids(eles=df.index, dset=dset)
        df = df.loc[keep]
    return df

def _get_sampleids(eles, dset):
    keep = []

    for sampleid in eles:

        mid = sampleid.split('-')[0]
        if mid == '1':
            continue

        if dset == 'inoculum' and _is_inoc_sampleid(sampleid):
            keep.append(sampleid)
        elif dset == 'replicates' and _is_negbin_sampleid(sampleid):
            keep.append(sampleid)
        elif dset == 'healthy' and _is_healthy_sampleid(sampleid):
            keep.append(sampleid)
        elif dset == 'uc' and _is_uc_sampleid(sampleid):
            keep.append(sampleid)

    return keep

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
    return sid in _HEALTHY_SUBJECTS

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
    return sid in _UC_SUBJECTS

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