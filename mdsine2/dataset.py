from .pylab import TaxaSet, Study
import pandas as pd

__all__ = ['parse']

def parse(name, metadata, taxonomy, reads=None, qpcr=None, perturbations=None, sep='\t'):
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
    taxas = TaxaSet()
    taxas.parse(taxonomy_table=taxonomy)
    study = Study(taxas, name=name)

    metadata = pd.read_csv(metadata, sep=sep)
    if reads is not None:
        reads = pd.read_csv(reads, sep=sep)
    if qpcr is not None:
        qpcr = pd.read_csv(qpcr, sep=sep)
    if perturbations is not None:
        perturbations = pd.read_csv(perturbations, sep=sep)
    
    return study.parse(metadata=metadata, reads=reads, qpcr=qpcr, perturbations=perturbations)


