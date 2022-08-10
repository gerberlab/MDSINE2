from typing import Union
import numpy as np
import pandas as pd

from .constants import *
from mdsine2.pylab import util as plutil
from mdsine2.logger import logger


def condense_matrix_with_taxonomy(M: Union[pd.DataFrame, np.ndarray], taxa: 'TaxaSet', fmt: str) -> pd.DataFrame:
    '''Condense the specified matrix M thats on the asv level
    to a taxonomic label specified with `fmt`. If `M`
    is a pandas.DataFrame then we assume the index are the Taxon
    names. If `M` is a numpy.ndarray, then we assume that the
    order of the matrix mirrors the order of the taxa. `fmt` is
    passed through `pylab.base.taxaname_formatter` to get the label.

    Parameters
    ----------
    M : numpy.ndarray, pandas.DataFrame
        Matrix to condense
    taxa : pylab.base.TaxaSet
        Set of Taxa with the relevant taxonomic information
    taxlevel : str
        This is the taxonomic level to condense to

    Returns
    -------
    pandas.DataFrame
        The index are the taxonomic classes. If M was a pandas.DataFrame, then
        the columns in M correspond to these columns. If `M` was a
        numpy.ndarray, then the order of the columsn correspond and no names
        are sent.
    '''
    if type(M) == pd.DataFrame:
        for idx in M.index:
            if idx not in taxa:
                raise ValueError('row `{}` not found in taxa'.format(idx))
        names = M.index
    elif plutil.isarray(M):
        if M.shape[0] != len(taxa):
            raise ValueError('Number of rows in M ({}) not equal to number of taxa ({})'.format(
                M.shape[0], len(taxa)))
        names = taxa.names.order
    else:
        raise TypeError('`M` ({}) type not recognized'.format(type(M)))

    # Get the rows that correspond to each row
    d = {}
    for row, name in enumerate(names):
        taxon = taxa[name]
        tax = taxaname_formatter(format=fmt, taxon=taxon, taxa=taxa)
        if tax not in d:
            d[tax] = []
        d[tax].append(row)

    # Add all of the rows for each taxon
    Ms = ()
    index = []
    columns = None
    if not plutil.isarray(M):
        columns = M.columns
    for taxname, rows, in d.items():
        index.append(taxname)
        if plutil.isarray(M):
            temp = np.sum(M[rows, ...], axis=0).reshape(1,-1)
        else:
            temp = np.sum(M.iloc[rows], axis=0).reshape(1,-1)
        Ms = Ms + (temp, )
    matrix = np.vstack(Ms)
    df = pd.DataFrame(matrix, index=index, columns=columns)
    df = df.sort_index(axis='index')
    return df


def taxaname_for_paper(taxon: Union["Taxon", "OTU"], taxa: "TaxaSet") -> str:
    '''Makes the name in the format needed for the paper

    Parameters
    ----------
    taxon : pylab.base.Taxon/pylab.base.OTU
        This is the taxon we are making the name for
    taxa : pylab.base.TaxaSet
        This is the TaxaSet object that contains the taxon objects

    Returns
    -------
    str
    '''
    taxon = taxa[taxon]
    if taxon.tax_is_defined('species'):
        species = taxon.taxonomy['species']
        species = species.split('/')
        if len(species) >= 3:
            species = species[:2]
        species = '/'.join(species)
        label = taxaname_formatter(
            format='%(genus)s {spec} %(name)s'.format(
                spec=species),
            taxon=taxon, taxa=taxa)
    elif taxon.tax_is_defined('genus'):
        label = taxaname_formatter(
            format='* %(genus)s %(name)s',
            taxon=taxon, taxa=taxa)
    elif taxon.tax_is_defined('family'):
        label = taxaname_formatter(
            format='** %(family)s %(name)s',
            taxon=taxon, taxa=taxa)
    elif taxon.tax_is_defined('order'):
        label = taxaname_formatter(
            format='*** %(order)s %(name)s',
            taxon=taxon, taxa=taxa)
    elif taxon.tax_is_defined('class'):
        label = taxaname_formatter(
            format='**** %(class)s %(name)s',
            taxon=taxon, taxa=taxa)
    elif taxon.tax_is_defined('phylum'):
        label = taxaname_formatter(
            format='***** %(phylum)s %(name)s',
            taxon=taxon, taxa=taxa)
    elif taxon.tax_is_defined('kingdom'):
        label = taxaname_formatter(
            format='****** %(kingdom)s %(name)s',
            taxon=taxon, taxa=taxa)
    else:
        logger.debug('Something went wrong - no taxnonomy: {}'.format(str(taxon)))
        label = 'NA {}'.format(taxa[taxon].name)

    return label


def taxaname_formatter(format: str, taxon: Union["Taxon", "OTU"], taxa: "TaxaSet") -> str:
    '''Format the label of a taxon. Specify the taxon by its index in the TaxaSet `taxa`.

    If `format == mdsine.TAXANAME_PAPER_FORMAT`, then we call the function
    `taxaname_for_paper`.

    Example:
        taxon is an Taxon object at index 0 where:
        ```
        taxon.genus = 'A'
        taxon.id = 1234532
        ```
        In[1]
        ```
        >>> taxaname_formatter(
            format='%(genus)s: %(index)s',
            taxon=1234532,
            taxa=taxa)
        'A: 0'
        ```
        In[2]
        ```
        >>> taxaname_formatter(
            format='%(genus)s: %(genus)s',
            taxon=1234532,
            taxa=taxa)
        'A: A'
        ```
        In[3]
        ```
        >>> taxaname_formatter(
            format='%(index)s',
            taxon=1234532,
            taxa=taxa)
        '0'
        ```
        In[4]
        ```
        >>> taxaname_formatter(
            format='%(geNus)s: %(genus)s',
            taxon=1234532,
            taxa=taxa)
        '%(geNus)s: A'
        ```

    Parameters
    ----------
    format : str
        - This is the format for us to do the labels. Options:
            - '%(paperformat)s'
                * Return the `taxaname_for_paper`
            - '%(name)s'
                * Name of the taxon (pylab.base..name)
            - '%(id)s'
                * ID of the taxon (pylab.base..id)
            - '%(index)s'
                * The order that this appears in the TaxaSet
            - '%(species)s'
                * `'species'` taxonomic classification of the taxon
            - '%(genus)s'
                * `'genus'` taxonomic classification of the taxon
            - '%(family)s'
                * `'family'` taxonomic classification of the taxon
            - '%(class)s'
                * `'class'` taxonomic classification of the taxon
            - '%(order)s'
                * `'order'` taxonomic classification of the taxon
            - '%(phylum)s'
                * `'phylum'` taxonomic classification of the taxon
            - '%(kingdom)s'
                * `'kingdom'` taxonomic classification of the taxon
    taxon : str, int, Taxon, OTU
        Taxon/OTU object or identifier (name, ID, index)
    taxa : pylab.base.TaxaSet
        Dataset containing all of the information for the taxa

    Returns
    -------
    str
    '''
    if format == TAXANAME_PAPER_FORMAT:
        return taxaname_for_paper(taxon=taxon, taxa=taxa)
    taxon = taxa[taxon]
    index = taxon.idx
    label = format.replace(NAME_FORMATTER, str(taxon.name))
    label = label.replace(ID_FORMATTER, str(taxon.id))
    label = label.replace(INDEX_FORMATTER,  str(index))

    if PAPER_FORMATTER in label:
        label = label.replace(PAPER_FORMATTER, '%(temp)s')
        label = label.replace('%(temp)s', taxaname_for_paper(taxon=taxon, taxa=taxa))

    for i in range(len(TAX_LEVELS)-1):
        taxlevel = TAX_LEVELS[i]
        fmt = TAXFORMATTERS[i]
        try:
            label = label.replace(fmt, str(taxon.get_taxonomy(taxlevel)))
        except:
            logger.critical('taxon: {}'.format(taxon))
            logger.critical('fmt: {}'.format(fmt))
            logger.critical('label: {}'.format(label))
            raise

    return label


class CustomOrderedDict(dict):
    """Order is an initialized version of self.keys() -> much more efficient
    index maps the key to the index in order:
    - order (list)
        - same as a numpy version of the keys in order
    - index (dict)
        - Maps the key to the index that it was inserted in
    """

    def __init__(self, *args, **kwargs):
        """Extension of the OrderedDict

        Paramters
        ---------
        args, kwargs : Arguments
            These are extra arguments to initialize the baseline OrderedDict
        """
        dict.__init__(self, *args, **kwargs)
        self.order = None
        self.index = None

    def update_order(self):
        """This will update the reverse dictionary based on the index. It will
        also redo the indexes if a taxon was deleted
        """
        self.order = np.array(list(self.keys()))
        self.index = {}
        for i, taxon in enumerate(self.order):
            self.index[taxon] = i
