import pandas as pd
import os
from mdsine2.logger import logger
from typing import Dict, Union
from .pylab import TaxaSet, Study

__all__ = ['load_gibson', 'parse']

def load_gibson(dset: str=None, as_df: bool=False, with_perturbations: bool=True, species_assignment: str='both',
    load_local: str=None, max_n_species: int=2) -> Union[Dict[str, pd.DataFrame], Study]:
    '''Load the Gibson dataset.
    Returns either a `mdsine2.Study` object or the `pandas.DataFrame` objects that
    that comprise the Gibson dataset.

    Tries to load the dataset from Github. If there is no internet connection, then we
    can read it from a local path specfied with the parameter `load_local`.

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
    load_local : str, None
        If specified, this is the local base path with all the files to load from. it will
        expect the respective file names. One may want to use this if they cannot get access
        to the internet
    max_n_species : int
        This is the maximum number of species assignments allowed before the lowest taxonomic
        assignment gets bumped up to Genus.

    Returns
    -------
    mdsine2.Study OR dict
    '''
    # Load the taxonomy assignment
    logger.debug('Downloading taxonomy')
    taxonomy = _Gibson.load_taxonomy(species_assignment=species_assignment, load_local=load_local,
        max_n_species=max_n_species)
    logger.debug('Downloading metadata')
    metadata = _Gibson.load_sampleid(dset=dset, load_local=load_local)
    logger.debug('Downloading reads')
    reads = _Gibson.load_reads(dset=dset, load_local=load_local)
    logger.debug('Downloading qpcr')
    qpcr = _Gibson.load_qpcr_masses(dset=dset, load_local=load_local)
    if with_perturbations:
        logger.debug('Downloading peturbations')
        perturbations = _Gibson.load_perturbations(dset=dset, load_local=load_local)
    else:
        perturbations = None

    if as_df:
        return {'metadata': metadata, 'taxonomy': taxonomy, 
            'reads': reads, 'qpcr':qpcr, 'perturbations': perturbations}
    else:
        taxa = TaxaSet(taxonomy_table=taxonomy)
        study = Study(taxa=taxa, name=dset)
        study.parse(
            metadata=metadata,
            reads=reads,
            qpcr=qpcr,
            perturbations=perturbations)
        return study

def parse(name: str, metadata: str, taxonomy: str, reads: str=None, qpcr: str=None, 
    perturbations: str=None, sep: str='\t') -> Study:
    '''Parse a dataset. Acts as a wrapper for `mdsine2.Study.parse`

    Parameters
    ----------
    name : str
        This is the name of the study
    metadata : str
        This is the location for the metadata table
    taxonomy : str
        This is the location for the taxonomy table
    reads : str
        This is the location for the reads table
    qpcr : str
        This is the location for the qPCR table
    perturbations : str
        This is the location for the perturbations table
    sep : str
        This is the separator for each table
    
    Returns
    -------
    mdsine2.Study
    '''
    taxonomy = pd.read_csv(taxonomy, sep=sep)
    taxa = TaxaSet()
    taxa.parse(taxonomy_table=taxonomy)
    study = Study(taxa, name=name)

    metadata = pd.read_csv(metadata, sep=sep)
    if reads is not None:
        reads = pd.read_csv(reads, sep=sep)
    if qpcr is not None:
        qpcr = pd.read_csv(qpcr, sep=sep)
    if perturbations is not None:
        perturbations = pd.read_csv(perturbations, sep=sep)
    
    return study.parse(metadata=metadata, reads=reads, qpcr=qpcr, perturbations=perturbations)

class _Gibson:
    '''Wrapper for the functionality of loading the gibson dataset. Called through
    `mdsine2.dataset.gibson`. See 
    '''
    _HEALTHY_SUBJECTS = set(['2', '3', '4', '5'])
    _UC_SUBJECTS = set(['6', '7', '8', '9', '10'])

    _URL_PATH = 'https://raw.githubusercontent.com/gerberlab/MDSINE2_Paper/master/datasets/gibson/'

    _URL_READS = 'counts.tsv'
    _URL_RDP_TAX = 'rdp_species.tsv'
    _URL_SILVA_TAX = 'silva_species.tsv'
    _URL_PERTS = 'perturbations.tsv'
    _URL_QPCR = 'qpcr.tsv'
    _URL_METADATA = 'metadata.tsv'

    @staticmethod
    def load_taxonomy(species_assignment, load_local=None, max_n_species=2):
        '''Load the taxonomy assignment
        Parameters
        ----------
        species_assignment : str, None
            How to assign the species
            If 'silva', only use the Silva species assignment
            If 'rdp', only use RDP 138 assignment
            If 'both', combine both the RDP and Silva species assignment
            If None, have no species assignment and just return the taxonomy provided by DADA2
        load_local : str, None  
            This is the local path if we need to load it locally. Otherwise we download from
            github
        max_n_species : int
            This is the maximum number of species assignments allowed before the lowest taxonomic
            assignment gets bumped up to Genus.

        Returns
        -------
        pandas.DataFrame
        '''
        if species_assignment == 'silva':
            if load_local is None:
                path = _Gibson._URL_PATH + _Gibson._URL_SILVA_TAX
            else:
                logger.debug('Load local')
                path = os.path.join(load_local, _Gibson._URL_SILVA_TAX)
            taxonomy = pd.read_csv(path, sep='\t', index_col=0)
        elif species_assignment == 'rdp':
            if load_local is None:
                path = _Gibson._URL_PATH + _Gibson._URL_RDP_TAX
            else:
                logger.debug('Load local')
                path = os.path.join(load_local, _Gibson._URL_RDP_TAX)
            taxonomy = pd.read_csv(path, sep='\t', index_col=0)
        elif species_assignment == 'both':
            if load_local is None:
                path = _Gibson._URL_PATH + _Gibson._URL_RDP_TAX
            else:
                logger.debug('Load local')
                path = os.path.join(load_local, _Gibson._URL_RDP_TAX)
            rdp = pd.read_csv(path, sep='\t', index_col=0)
            
            if load_local is None:
                path = _Gibson._URL_PATH + _Gibson._URL_SILVA_TAX
            else:
                logger.debug('Load local')
                path = os.path.join(load_local, _Gibson._URL_SILVA_TAX)
            silva = pd.read_csv(path, sep='\t', index_col=0)

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

                if len(both) > max_n_species:
                    both = 'NA'
                else:
                    both = '/'.join(both)
                tmp.append(both)
                data.append(tmp)

            taxonomy = pd.DataFrame(data, columns=rdp.columns, index=rdp.index)

        elif species_assignment is None:
            taxonomy = _Gibson._URL_PATH + _Gibson._URL_RDP_TAX
            taxonomy = pd.read_csv(taxonomy, sep='\t')
            taxonomy['species'][:] = 'NA'
        else:
            raise ValueError('`species_assignment` ({}) is not recognized.'.format(species_assignment))
        return taxonomy
    
    @staticmethod
    def load_perturbations(dset=None, load_local=None):
        '''Load the perturbations for Gibson dataset as a `pandas.DataFrame`.

        Parameters
        ----------
        dset : str
            This is the dataset to load.
        load_local : str, None  
            This is the local path if we need to load it locally. Otherwise we download from
            github

        Returns
        -------
        pandas.DataFrame
        '''
        if load_local is None:
            path = _Gibson._URL_PATH + _Gibson._URL_PERTS
        else:
            logger.debug('Load local')
            path = os.path.join(load_local, _Gibson._URL_PERTS)
        df = pd.read_csv(path, sep='\t')

        if dset is None:
            pass
        elif dset == 'inoculum':
            df = None
        elif dset == 'replicates':
            df = None
        elif dset == 'healthy':
            row_to_keep = []
            for i, subj in enumerate(df['subject']):
                if str(subj) in _Gibson._HEALTHY_SUBJECTS:
                    row_to_keep.append(i)
            df = df.iloc[row_to_keep, :]
        elif dset == 'uc':
            row_to_keep = []
            for i, subj in enumerate(df['subject']):
                if str(subj) in _Gibson._UC_SUBJECTS:
                    row_to_keep.append(i)
            df = df.iloc[row_to_keep, :]
        else:
            raise ValueError('`dset` ({}) not recognized'.format(dset))
        return df

    @staticmethod
    def load_qpcr_masses(dset=None, load_local=None):
        '''Load qpcr masses table into a `pandas.DataFrame`.

        Parameters
        ----------
        dset : str
            This is the dataset to load.
        load_local : str, None  
            This is the local path if we need to load it locally. Otherwise we download from
            github

        Returns
        -------
        pandas.DataFrame
        '''
        if load_local is None:
            path = _Gibson._URL_PATH + _Gibson._URL_QPCR
        else:
            logger.debug('Load local')
            path = os.path.join(load_local, _Gibson._URL_QPCR)
        
        df = pd.read_csv(path, sep='\t')
        df = df.set_index('sampleID')

        if dset is not None:
            keep = _Gibson._get_sampleids(eles=df.index, dset=dset)
            df = df.loc[keep]
        return df

    @staticmethod
    def load_reads(dset=None, load_local=None):
        '''Load reads table into a `pandas.DataFrame`.

        Parameters
        ----------
        dset : str
            This is the dataset to load.
        load_local : str, None  
            This is the local path if we need to load it locally. Otherwise we download from
            github

        Returns
        -------
        pandas.DataFrame
        '''
        if load_local is None:
            path = _Gibson._URL_PATH + _Gibson._URL_READS
        else:
            logger.debug('Load local')
            path = os.path.join(load_local, _Gibson._URL_READS)
        df = pd.read_csv(path, sep='\t', index_col=0)

        if dset is not None:
            keep = _Gibson._get_sampleids(eles=df.columns, dset=dset)
            df = df[keep]
        return df

    @staticmethod
    def load_sampleid(dset=None, load_local=None):
        '''Load sample ID table into a `pandas.DataFrame`.

        Parameters
        ----------
        dset : str
            This is the dataset to load.
        load_local : str, None  
            This is the local path if we need to load it locally. Otherwise we download from
            github

        Returns
        -------
        pandas.DataFrame
        '''
        if load_local is None:
            path = _Gibson._URL_PATH + _Gibson._URL_METADATA
        else:
            path = os.path.join(load_local, _Gibson._URL_METADATA)
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
