import pandas as pd
from ete3 import Tree
from os.path import dirname, join
import os

_HEALTHY_SUBJECTS = set(['2', '3', '4', '5'])
_UC_SUBJECTS = set(['6', '7', '8', '9', '10'])


def load_phylogenetic_tree(with_reference_sequences):
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
        path = join(dirname(__file__),'gibson_dataset' ,'phylogenetic_tree_with_reference.nhx')
    else:
        path = join(dirname(__file__),'gibson_dataset' ,'phylogenetic_tree_w_branch_len_preserved.nhx')

    return Tree(path)

def load_qpcr_masses(healthy=None):
    '''Load qpcr masses table into a `pandas.DataFrame`.

    Parameters
    ----------
    healthy : bool, None
        Whether or not to return the healthy cohort. 
        If True only return the healthy cohort.
        If False only return the ulcerative colitis cohort.
        If None then return everything

    Returns
    -------
    pandas.DataFrame
    '''
    path = join(dirname(__file__),'gibson_dataset' ,'qpcr.tsv')
    df = pd.read_csv(path, sep='\t')
    df = df.set_index('sampleID')

    if healthy is not None:
        keep = []

        for sampleid in df.index:
            if healthy and _is_healthy_sampleid(sampleid):
                keep.append(sampleid)
            elif (not healthy) and (not _is_healthy_sampleid(sampleid)):
                keep.append(sampleid)

        df = df.loc[keep]
    return df

def load_reads(healthy=None):
    '''Load reads table into a `pandas.DataFrame`.

    Parameters
    ----------
    healthy : bool, None
        Whether or not to return the healthy cohort. 
        If True only return the healthy cohort.
        If False only return the ulcerative colitis cohort.
        If None then return everything

    Returns
    -------
    pandas.DataFrame
    '''
    path = join(dirname(__file__),'gibson_dataset' ,'reads.tsv')
    df = pd.read_csv(path, sep='\t')
    df = df.set_index('name')

    if healthy is not None:
        keep = []

        for sampleid in df.columns:
            if healthy and _is_healthy_sampleid(sampleid):
                keep.append(sampleid)
            elif (not healthy) and (not _is_healthy_sampleid(sampleid)):
                keep.append(sampleid)

        df = df[keep]
    return df

def load_sampleid(healthy=None):
    '''Load sample ID table into a `pandas.DataFrame`.

    Parameters
    ----------
    healthy : bool, None
        Whether or not to return the healthy cohort. 
        If True only return the healthy cohort.
        If False only return the ulcerative colitis cohort.
        If None then return everything

    Returns
    -------
    pandas.DataFrame
    '''
    path = join(dirname(__file__),'gibson_dataset' ,'sampleid.tsv')
    df = pd.read_csv(path, sep='\t')
    df = df.set_index('sampleID')

    if healthy is not None:
        keep = []

        for sampleid in df.index:
            if healthy and _is_healthy_sampleid(sampleid):
                keep.append(sampleid)
            elif (not healthy) and (not _is_healthy_sampleid(sampleid)):
                keep.append(sampleid)

        df = df.loc[keep]
    return df

def load_taxonomy():
    '''Load taxonomy table into a `pandas.DataFrame`.

    Returns
    -------
    pandas.DataFrame
    '''
    path = join(dirname(__file__),'gibson_dataset' ,'taxonomy.tsv')
    return pd.read_csv(path, sep='\t')

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
