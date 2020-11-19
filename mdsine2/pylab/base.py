'''These are base classes that are used throughout the rest of Pylab

Difference between Saveable and Traceable
-----------------------------------------
`Saveable` defines functions (`save` and `load`) to save the class object in 
memory as a Pickle. Traceable defines functions and attributes that allows the
trace of the object during inference to be saved on disk DURING inference. An 
object can both be `Saveable` and `Traceable`.

How perturbations are switched on and off
-----------------------------------------
'The time ahead prediction must be included in the perturbation' - Travis

Example: Pertubtion period (2,5) - this is **3** doses

                   
perturbation on    |-->|-->|-->
Days           1   2   3   4   5   6

`d1` indicates the perturbation parameter that gets added for the day that it
should be included in.

x2 = x1 + ...
x3 = x2 + ... + d1
x4 = x3 + ... + d1
x5 = x4 + ... + d1
x6 = x5 + ...

The perturbation periods that are given are in the format (start, end).
For the above example our perturbation period would be (2, 5). Thus, we should do
inclusion/exclusion brackets such that:

(start, end]
    - The first day is inclusive
    - Last day is exclusive
'''

import numpy as np
import collections
import pickle
import scipy.spatial.distance
import pandas as pd
import logging
import ete3
import os
import os.path

from . import util as plutil
from .errors import NeedToImplementError

# Constants
DEFAULT_TAXA_NAME = 'NA'
SEQUENCE_COLUMN_LABEL = 'sequence'
TAX_IDXS = {'kingdom': 0, 'phylum': 1, 'class': 2,  'order': 3, 'family': 4, 
    'genus': 5, 'species': 6, 'asv': 7}
_TAX_REV_IDXS = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species', 'asv']


def isqpcrdata(x):
    '''Checks whether the input is a subclass of qPCRData

    Parameters
    ----------
    x : any
        Input instance to check the type of qPCRData
    
    Returns
    -------
    bool
        True if `x` is of type qPCRData, else False
    '''
    return x is not None and issubclass(x.__class__, qPCRdata)

def isasvset(x):
    '''Checks whether the input is a subclass of ASVSet

    Parameters
    ----------
    x : any
        Input instance to check the type of ASVSet
    
    Returns
    -------
    bool
        True if `x` is of type ASVSet, else False
    '''
    return x is not None and issubclass(x.__class__, ASVSet)

def issavable(x):
    '''Checks whether the input is a subclass of Savable

    Parameters
    ----------
    x : any
        Input instance to check the type of Savable
    
    Returns
    -------
    bool
        True if `x` is of type Savable, else False
    '''
    return x is not None and issubclass(x.__class__, Saveable)

def isclusterable(x):
    '''Determines whether the input is a subclass of Clusterable

    Parameters
    ----------
    x : any
        Input instance to check the type of Clusterable
    
    Returns
    -------
    bool
        True if `x` is of type Clusterable, else False
    '''
    return x is not None and issubclass(x.__class__, Clusterable)

def istraceable(x):
    '''Checks whether the input is a subclass of Traceable

    Parameters
    ----------
    x : any
        Input instance to check the type of Traceable
    
    Returns
    -------
    bool
        True if `x` is of type Traceable, else False
    '''
    return x is not None and issubclass(x.__class__, Traceable)

def isasv(x):
    '''Checks whether the input is a subclass of ASV

    Parameters
    ----------
    x : any
        Input instance to check the type of ASV
    
    Returns
    -------
    bool
        True if `x` is of type ASV, else False
    '''
    return x is not None and issubclass(x.__class__, ASV)

def isaggregatedasv(x):
    '''Checks whether the input is a subclass of AggregateASV

    Parameters
    ----------
    x : any
        Input instance to check the type of ASV
    
    Returns
    -------
    bool
        True if `x` is of type ASV, else False
    '''
    return issubclass(x.__class__, AggregateASV)

def isasvtype(x):
    '''Checks whether the input is a subclass of AggregateASV or ASV

    Parameters
    ----------
    x : any
        Input instance to check the type of AggregateASV or ASV
    
    Returns
    -------
    bool
        True if `x` is of type AggregateASV or ASV, else False
    '''
    return isasv(x) or isaggregatedasv(x)

def issubject(x):
    '''Checks whether the input is a subclass of Subject

    Parameters
    ----------
    x : any
        Input instance to check the type of Subject
    
    Returns
    -------
    bool
        True if `x` is of type Subject, else False
    '''
    return x is not None and issubclass(x.__class__, Subject)

def isstudy(x):
    '''Checks whether the input is a subclass of Study

    Parameters
    ----------
    x : any
        Input instance to check the type of Study
    
    Returns
    -------
    bool
        True if `x` is of type Study, else False
    '''
    return x is not None and issubclass(x.__class__, Study)

def isperturbation(x):
    '''Checks whether the input is a subclass of BasePerturbation

    Parameters
    ----------
    x : any
        Input instance to check the type of BasePerturbation
    
    Returns
    -------
    bool
        True if `x` is of type BasePerturbation, else False
    '''
    return x is not None and issubclass(x.__class__, BasePerturbation)

def condense_matrix_with_taxonomy(M, asvs, fmt):
    '''Condense the specified matrix M thats on the asv level
    to a taxonomic label specified with `fmt`. If `M` 
    is a pandas.DataFrame then we assume the index are the ASV
    names. If `M` is a numpy.ndarray, then we assume that the 
    order of the matrix mirrors the order of the asvs. `fmt` is
    passed through `pylab.util.plutil.asvname_formatter` to get the label.

    Parameters
    ----------
    M : numpy.ndarray, pandas.DataFrame
        Matrix to condense
    asvs : pylab.base.ASVSet
        Set of ASVs with the relevant taxonomic information
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
    if not isasvset(asvs):
        raise TypeError('`asvs` ({}) must be a pylab.base.ASVSet'.format(type(asvs)))
    if not plutil.isstr(fmt):
        raise TypeError('`fmt` ({}) must be a str'.format(type(fmt)))

    if type(M) == pd.DataFrame:
        for idx in M.index:
            if idx not in asvs:
                raise ValueError('row `{}` not found in asvs'.format(idx))
        names = M.index
    elif plutil.isarray(M):
        if M.shape[0] != len(asvs):
            raise ValueError('Number of rows in M ({}) not equal to number of asvs ({})'.format(
                M.shape[0], len(asvs)))
        names = asvs.names.order
    else:
        raise TypeError('`M` ({}) type not recognized'.format(type(M)))

    # Get the rows that correspond to each row
    d = {}
    for row, name in enumerate(names):
        asv = asvs[name]
        tax = plutil.asvname_formatter(format=fmt, asv=asv, asvs=asvs)
        if tax not in d:
            d[tax] = []
        d[tax].append(row)

    # Add all of the rows for each taxa
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


class Saveable:
    '''Implements baseline saving classes with pickle for classes
    '''
    def save(self, filename=None):
        '''Pickle the object

        Paramters
        ---------
        filename : str
            This is the location to store the file. Overrides the location if
            it is set using `pylab.base.Saveable.set_save_location`. If None
            it means that we are using the file location set in 
            set_location. 
        '''
        if filename is None:
            if not hasattr(self, '_save_loc'):
                raise TypeError('`filename` must be specified if you have not ' \
                    'set the save location')
            filename = self._save_loc
        
        try:
            with open(str(filename), 'wb') as output:  # Overwrites any existing file.
                pickle.dump(self, output, protocol=pickle.HIGHEST_PROTOCOL)
        except:
            os.system('rm {}'.format(filename))
            with open(str(filename), 'wb') as output:  # Overwrites any existing file.
                pickle.dump(self, output, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        '''Unpickle the object

        Paramters
        ---------
        filename : str
            This is the location of the file to unpickle
        '''
        with open(str(filename), 'rb') as handle:
            b = pickle.load(handle)
        
        # redo the filename to the new path if it has a save location
        if not hasattr(b, '_save_loc'):
            filename = os.path.abspath(filename)
            b._save_loc = filename

        return b

    def set_save_location(self, filename):
        '''Set the save location for the object.

        Internally converts this to the absolute path

        Parameters
        ----------
        filename : str
            This is the path to set it to
        '''
        if not plutil.isstr(filename):
            raise TypeError('`filename` ({}) must be a str'.format(type(filename)))
        filename = os.path.abspath(filename)
        self._save_loc = filename

    def get_save_location(self):
        try:
            return self._save_loc
        except:
            raise AttributeError('Save location is not set.')


class Traceable:
    '''Defines the functionality for a Node to interact with the Graph tracer object
    '''

    @property
    def initialization_value(self):
        if hasattr(self, '_init_value'):
            return self._init_value
        else:
            return None

    def set_trace(self):
        '''Initialize the trace arrays for the variable in the Tracer object. 

        It will initialize a buffer the size of the checkpoint size in Tracer
        '''
        raise NotImplementedError('User needs to define this function')

    def add_trace(self):
        '''Adds the current value to the trace. If the buffer is full
        it will end it to disk
        '''
        raise NotImplementedError('User needs to define this function')

    def get_trace_from_disk(self, *args, **kwargs):
        '''Returns the entire trace (after burnin) writen on the disk. NOTE: This may/may not 
        include the samples in the local buffer trace and could be very large

        Returns
        -------
        np.ndarray
        '''
        return self.G.tracer.get_trace(name=self.name, *args, **kwargs)

    def overwrite_entire_trace_on_disk(self, data, **kwargs):
        '''Overwrites the entire trace of the variable with the given data.

        Parameters
        ----------
        data : np.ndarray
            Data you are overwriting the trace with.
        '''
        self.G.tracer.overwrite_entire_trace_on_disk(
            name=self.name, data=data, dtype=self.dtype, **kwargs)

    def get_iter(self):
        '''Get the number of iterations saved to the hdf5 file of the variable

        Returns
        -------
        int
        '''
        return self.G.tracer.get_iter(name=self.name)


class BasePerturbation:
    '''Base perturbation class.

    We assume that the `start` is when the first perturbation happend (affects the next time point) and 
    `end` is the last time that it gets affected

    Paramters
    ---------
    start : float, int
        This is the start of the perturbation (it will afect the next time point)
    end : float, int
        This is the end of the perturbation (this is the last time point the 
        perturbation will affect)
    name : str, None
        This is the name of the perturabtion. If nothing is given then it will be None
    '''
    def __init__(self, start, end, name=None):
        if not plutil.isnumeric(start):
            raise TypeError('`start` ({}) must be a numeric'.format(type(start)))
        if not plutil.isnumeric(end):
            raise TypeError('`end` ({}) must be a numeric'.format(type(end)))
        if end < start:
            raise ValueError('`end` ({}) must be >= `start` ({})'.format(end, start))
        if name is not None:
            if not plutil.isstr(name):
                raise TypeError('`name` ({}) must be a str'.format(type(name)))
        self.start = start
        self.end = end
        self.name = name

    def isactive(self, time):
        '''Returns a `bool` if the perturbation is on at time `time`.

        Parameters
        ----------
        time : float, int
            Time to check
        '''
        return time > self.start and time <= self.end

    def timetuple(self):
        '''Returns the time tuple of the start and end

        Paramters
        ---------
        None

        Returns
        -------
        2-tuple
            (start,end) as floats
        '''
        return (self.start, self.end)

    def __str__(self):
        return 'Perturbation\n\tstart: {}\n\tend:{}'.format(
            self.start, self.end)


class ClusterItem:
    '''These are single points that get clustered

    It must have the parameter
        'name'
    '''
    def __init__(self, name):
        self.name = name

    def cluster_str(self):
        return self.name


class ASV(ClusterItem):
    '''Wrapper class for a single ASV

    Parameters
    ----------
    name : str
        Name given to the ASV 
    sequence : str
        Base Pair sequence
    idx : int
        The index that the asv occurs
    '''
    def __init__(self, name, idx, sequence=None):
        ClusterItem.__init__(self, name=name)
        self.sequence = sequence
        self.idx = idx
        # Initialize the taxonomies to nothing
        self.taxonomy = {
            'kingdom': DEFAULT_TAXA_NAME,
            'phylum': DEFAULT_TAXA_NAME,
            'class': DEFAULT_TAXA_NAME,
            'order': DEFAULT_TAXA_NAME,
            'family': DEFAULT_TAXA_NAME,
            'genus': DEFAULT_TAXA_NAME,
            'species': DEFAULT_TAXA_NAME,
            'asv': self.name}
        self.id = id(self)

    def __getitem__(self,key):
        return self.taxonomy[key.lower()]

    def __eq__(self, val):
        '''Compares different ASVs between each other. Checks all of the attributes but the id

        Parameters
        ----------
        val : any
            This is what we are checking if they are equivalent
        '''
        if type(val) != ASV:
            return False
        if self.name != val.name:
            return False
        if self.sequence != val.sequence:
            return False
        for k,v in self.taxonomy.items():
            if v != val.taxonomy[k]:
                return False
        return True

    def __str__(self):
        return 'ASV\n\tid: {}\n\tidx: {}\n\tname: {}\n' \
            '\ttaxonomy:\n\t\tkingdom: {}\n\t\tphylum: {}\n' \
            '\t\tclass: {}\n\t\torder: {}\n\t\tfamily: {}\n' \
            '\t\tgenus: {}\n\t\tspecies: {}'.format(
            self.id, self.idx, self.name,
            self.taxonomy['kingdom'], self.taxonomy['phylum'],
            self.taxonomy['class'], self.taxonomy['order'],
            self.taxonomy['family'], self.taxonomy['genus'],
            self.taxonomy['species'])

    def set_taxonomy(self, tax_kingdom=None, tax_phylum=None, tax_class=None,
        tax_order=None, tax_family=None, tax_genus=None, tax_species=None):
        '''Sets the taxonomy of the parts that are specified

        Parameters
        ----------
        tax_kingdom, tax_phylum, tax_class, tax_order, tax_family, tax_genus : str
            'kingdom', 'phylum', 'class', 'order', 'family', 'genus'
            Name of the taxa for each respective level
        '''
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

    def get_lineage(self, level=None):
        '''Returns a tuple of the lineage in order from Kingdom to the level
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
        '''
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
    
    def get_taxonomy(self, level):
        '''Get the taxonomy at the level specified

        Parameters
        ----------
        level : str
            This is the level to get
            Valid responses: 'kingdom', 'phylum', 'class', 'order', 'family', 'genus'

        Returns
        -------
        str
        '''
        return self.get_lineage(level=level)[-1]

    def tax_is_defined(self, level):
        '''Whether or not the ASV is defined at the specified taxonomic level

        Parameters
        ----------
        level : str
            This is the level to get
            Valid responses: 'kingdom', 'phylum', 'class', 'order', 'family', 'genus'
        
        Returns
        -------
        bool
        '''
        try:
            tax = self.taxonomy[level]
        except:
            raise KeyError('`tax` ({}) not defined. Available taxs: {}'.format(level, 
                list(self.taxonomy.keys())))
        return (type(tax) != float) and (tax != DEFAULT_TAXA_NAME) and (tax != '')


class AggregateASV(ASV):
    '''Aggregates of ASV objects

    NOTE: For self consistency, let the class ASVSet initialize this object.

    Parameters
    ----------
    anchor, other : mdsine2.ASV, mdsine2.AggregateASV
        These are the ASVs/Aggregates that you're joining together. The anchor is
        the one you are setting the sequeunce and taxonomy to
    '''
    def __init__(self, anchor, other):
        name = anchor.name + '_agg'
        ASV.__init__(self, name=name, idx=anchor.idx, sequence=anchor.sequence)

        _agg_taxas = {}

        if isaggregatedasv(anchor):
            if other.name in anchor.aggregated_asvs:
                raise ValueError('`other` ({}) already aggregated with anchor ' \
                    '({}) ({})'.format(other.name, anchor.name, anchor.aggregated_asvs))
            agg1 = anchor.aggregated_asvs
            agg1_seq = anchor.aggregated_seqs
            for k,v in anchor.aggregated_taxonomies.items():
                _agg_taxas[k] = v
        else:
            agg1 = [anchor.name]
            agg1_seq = {anchor.name: anchor.sequence}
            _agg_taxas[anchor.name] = anchor.taxonomy

        if isaggregatedasv(other):
            if anchor.name in other.aggregated_asvs:
                raise ValueError('`anchor` ({}) already aggregated with other ' \
                    '({}) ({})'.format(anchor.name, other.name, other.aggregated_asvs))
            agg2 = other.aggregated_asvs
            agg2_seq = other.aggregated_seqs
            for k,v in other.aggregated_taxonomies.items():
                _agg_taxas[k] = v
        else:
            agg2 = [other.name]
            agg2_seq = {other.name: other.sequence}
            _agg_taxas[other.name] = other.taxonomy

        self.aggregated_asvs = agg1 + agg2
        self.aggregated_seqs = agg1_seq
        self.aggregated_taxonomies = _agg_taxas
        for k,v in agg2_seq.items():
            self.aggregated_seqs[k] = v

        self.taxonomy = anchor.taxonomy

    def __str__(self):
        return 'AggregateASV\n\tid: {}\n\tidx: {}\n\tname: {}\n' \
            '\tAggregates: {}\n' \
            '\ttaxonomy:\n\t\tkingdom: {}\n\t\tphylum: {}\n' \
            '\t\tclass: {}\n\t\torder: {}\n\t\tfamily: {}\n' \
            '\t\tgenus: {}\n\t\tspecies: {}'.format(
            self.id, self.idx, self.name, self.aggregated_asvs,
            self.taxonomy['kingdom'], self.taxonomy['phylum'],
            self.taxonomy['class'], self.taxonomy['order'],
            self.taxonomy['family'], self.taxonomy['genus'],
            self.taxonomy['species'])


class Clusterable(Saveable):
    '''This is the base class for something to be clusterable (be used to cluster in
    pylab.cluster.Clustering). These are the functions that need to be implemented
    for it to be able to be clustered.

    `stritem`
    ---------
    This is the function that we use 
    '''
    def __len__(self):
        raise NotImplementedError('You must implement this function')

    def __getitem__(self, key):
        raise NotImplementedError('You must implement this function')

    def __iter__(self):
        raise NotImplementedError('You must implement this function')

    def __contains__(self, key):
        raise NotImplementedError('You must implement this function')

    def stritem(self, key):
        raise NotImplementedError('You must implement this function')


class ASVSet(Clusterable):
    '''Wraps a set of `ASV` objects. You can get the ASV object via the
    ASV id, ASV name.
    Provides functionality for aggregating sequeunces and getting subsets for lineages.

    Aggregating/Deaggregating
    -------------------------
    ASVs that are aggregated together to become OTUs are used because sequences are 
    very close together. This class provides functionality for aggregating asvs together
    (`mdsine2.ASVSet.aggregate_items`) and to deaggregate a specific name from an aggregation
    (`mdsine2.ASVSet.deaggregate_item`). If this object is within a `mdsine2.Study` object,
    MAKE SURE TO CALL THE AGGREGATION FUNCTIONS FROM THE `mdsine2.Study` OBJECT 
    (`mdsine2.Study.aggregate_items`, `mdsine2.Study.deaggregate_item`) so that the reads
    for the agglomerates and individual ASVs can be consistent with the ASVSet.

    `taxonomy_table`
    ----------------
    This is a dataframe that contains the taxonomic information for each ASV.
    The columns that must be included are:
        'name' : name of the asv
        'sequence' : sequence of the asv
    All of the taxonomy specifications are optional:
        'kingdom' : kingdom taxonomy
        'phylum' : phylum taxonomy
        'class' : class taxonomy
        'family' : family taxonomy
        'genus' : genus taxonomy
        'species' : species taxonomy

    Parameters
    ----------
    taxonomy_table : pandas.DataFrame, Optional
        DataFrame conttaxaining the required information (Taxonomy, sequence).
        If nothing is passed in, it will be an empty ASVSet
    '''

    def __init__(self, taxonomy_table=None):
        self.taxonomy_table = taxonomy_table
        self.ids = CustomOrderedDict()
        self.names = CustomOrderedDict()
        self.index = []
        self._len = 0

        # Add all of the ASVs from the dataframe if necessary
        if taxonomy_table is not None:
            taxonomy_table = taxonomy_table.rename(str.lower, axis='columns')
            if 'name' not in taxonomy_table.columns:
                if taxonomy_table.index.name == 'name':
                    taxonomy_table = taxonomy_table.reset_index()
                else:
                    raise ValueError('`"name"` ({}) not found as a column in `taxonomy_table`'.format(
                        taxonomy_table.columns))
            if SEQUENCE_COLUMN_LABEL not in taxonomy_table.columns:
                raise ValueError('`"{}"` ({}) not found as a column in `taxonomy_table`'.format(
                    SEQUENCE_COLUMN_LABEL, taxonomy_table.columns))

            for tax in _TAX_REV_IDXS[:-1]:
                if tax not in taxonomy_table.columns:
                    logging.info('Adding in `{}` column'.format(tax))
                    taxonomy_table = taxonomy_table.insert(-1, tax, 
                        [DEFAULT_TAXA_NAME for _ in range(len(taxonomy_table.index))])

            for i in taxonomy_table.index:
                seq = taxonomy_table[SEQUENCE_COLUMN_LABEL][i]
                name = taxonomy_table['name'][i]
                asv = ASV(name=name, sequence=seq, idx=self._len)
                asv.set_taxonomy(
                    tax_kingdom=taxonomy_table.loc[i]['kingdom'],
                    tax_phylum=taxonomy_table.loc[i]['phylum'],
                    tax_class=taxonomy_table.loc[i]['class'],
                    tax_order=taxonomy_table.loc[i]['order'],
                    tax_family=taxonomy_table.loc[i]['family'],
                    tax_genus=taxonomy_table.loc[i]['genus'],
                    tax_species=taxonomy_table.loc[i]['species'])

                self.ids[asv.id] = asv
                self.names[asv.name] = asv
                self.index.append(asv)  
                self._len += 1

            self.ids.update_order()
            self.names.update_order()

    def __contains__(self,key):
        try:
            self[key]
            return True
        except:
            return False

    def __getitem__(self,key):
        '''Get an ASV by either its sequence, name, index, or id

        Parameters
        ----------
        key : str, int
            Key to reference the ASV
        '''
        if isasvtype(key):
            return key
        if key in self.ids:
            return self.ids[key]
        elif plutil.isint(key):
            return self.index[key]
        elif key in self.names:
            return self.names[key]
        elif isasv(key):
            return key
        else:
            raise IndexError('`{}` ({}) was not found as a name, sequence, index, or id'.format(
                key, type(key)))

    def __iter__(self):
        '''Returns each ASV obejct in order
        '''
        for asv in self.index:
            yield asv

    def __len__(self):
        '''Return the number of ASVs in the ASVSet
        '''
        return self._len

    @property
    def n_asvs(self):
        '''Alias for __len__
        '''
        return self._len

    def add_asv(self, name, sequence=None):
        '''Adds an ASV to the set

        Parameters
        ----------
        name : str
            This is the name of the ASV
        sequence : str
            This is the sequence of the ASV
        '''
        asv = ASV(name=name, sequence=sequence, idx=self._len)
        self.ids[asv.id] = asv
        self.names[asv.name] = asv
        self.index.append(asv)

        # update the order of the ASVs
        self.ids.update_order()
        self.names.update_order()
        self._len += 1

        return self

    def del_asv(self, asv):
        '''Deletes the ASV from the set.

        Parameters
        ----------
        asv : str, int, ASV
            Can either be the name, sequence, or the ID of the ASV
        '''
        # Get the ID
        asv = self[asv]
        oidx = self.ids.index[asv.id]

        # Delete the ASV from everything
        # asv = self[asv]
        self.ids.pop(asv.id, None)
        self.names.pop(asv.name, None)
        self.index.pop(oidx)

        # update the order of the ASVs
        self.ids.update_order()
        self.names.update_order()

        # Update the indices of the asvs
        # Since everything points to the same object we only need to do it once
        for aidx, asv in enumerate(self.index):
            asv.idx = aidx

        self._len -= 1
        return self

    def taxonomic_similarity(self,oid1,oid2):
        '''Calculate the taxonomic similarity between ASV1 and ASV2
        Iterates through most broad to least broad taxonomic level and
        returns the fraction that are the same.

        Example:
            asv1.taxonomy = (A,B,C,D)
            asv2.taxonomy = (A,B,E,F)
            similarity = 0.5

            asv1.taxonomy = (A,B,C,D)
            asv2.taxonomy = (A,B,C,F)
            similarity = 0.75

            asv1.taxonomy = (A,B,C,D)
            asv2.taxonomy = (A,B,C,D)
            similarity = 1.0

            asv1.taxonomy = (X,Y,Z,M)
            asv2.taxonomy = (A,B,E,F)
            similarity = 0.0

        Parameters
        ----------
        oid1, oid2 : str, int
            The name, id, or sequence for the ASV
        '''
        if oid1 == oid2:
            return 1
        asv1 = self[oid1].get_lineage()
        asv2 = self[oid2].get_lineage()
        i = 0
        for a in asv1:
            if a == asv2[i]:
                i += 1
            else:
                break
        return i/8 # including ASV

    def aggregate_items(self, anchor, other):
        '''Create an aggreate asv with the anchor `anchor` and other asv  `other`.
        The aggregate takes the sequence and the taxonomy from the anchor.

        Parameters
        ----------
        anchor, other : str, int, mdsine2.ASV, mdsine2.AggregateASV
            These are the ASVs/Aggregates that you're joining together. The anchor is
            the one you are setting the sequeunce and taxonomy to

        Returns
        -------
        mdsine2.AggregateASV
            This is the new aggregated ASV containing anchor and other
        '''
        anchor = self[anchor]
        other = self[other]
        
        agg = AggregateASV(anchor=anchor, other=other)

        self.index[agg.idx] = agg
        self.index.pop(other.idx)

        self.ids = CustomOrderedDict()
        self.names = CustomOrderedDict()

        for idx, asv in enumerate(self.index):
            asv.idx = idx
            self.ids[asv.id] = asv
            self.names[asv.name] = asv
        
        # update the order of the ASVs
        self.ids.update_order()
        self.names.update_order()

        self._len = len(self.index)
        return agg

    def deaggregate_item(self, agg, other):
        '''Deaggregate the sequence `other` from AggregateASV `agg`.
        `other` is then appended to the end 

        Parameters
        ----------
        agg : AggregateASV, str
            This is an AggregateASV with multiple sequences contained. Must 
            have the name `other` in there
        other : str
            This is the name of the ASV that should be taken out of `agg`

        Returns
        -------
        mdsine2.ASV
            This is the deaggregated ASV
        '''
        agg = self[agg]
        if not isaggregatedasv(agg):
            raise TypeError('`agg` ({}) must be an AggregatedASV'.format(type(agg)))
        if not plutil.isstr(other):
            raise TypeError('`other` ({}) must be a str'.format(type(other)))
        if other not in agg.aggregated_asvs:
            raise ValueError('`other` ({}) is not contained in `agg` ({}) ({})'.format(
                other, agg.name, agg.aggregated_asvs))

        other = ASV(name=other, sequence=agg.aggregated_seqs[other], idx=self._len)
        other.taxonomy = agg.aggregated_taxonomies[other.name]
        agg.aggregated_seqs.pop(other.name, None)
        agg.aggregated_asvs.remove(other.name)
        agg.aggregated_taxonomies.pop(other.name, None)

        self.index.append(other)
        self.ids[other.id] = other
        self.names[other.name] = other

        self.ids.update_order()
        self.names.update_order()
        self._len += 1
        return other

    def rename(self, prefix, zero_based_index=False):
        '''Rename the contents based on their index:

        Example
        -------
        Names before in order:
        [ASV_22, ASV_9982, TUDD_8484]

        Calling asvs.rename(prefix='OTU')
        New names:
        [OTU_1, OTU_2, OTU_3]

        Calling asvs.rename(prefix='OTU', zero_based_index=True)
        New names:
        [OTU_0, OTU_1, OTU_2]

        Parameters
        ----------
        prefix : str
            This is the prefix of the new ASVs. The name of the ASVs will change
            to `'{}_{}'.format(prefix, index)`
        zero_based_index : bool
            If this is False, then we start the enumeration of the ASVs from 1
            instead of 0. If True, then the enumeration starts at 0
        '''
        if not plutil.isstr(prefix):
            raise TypeError('`prefix` ({}) must be a str'.format(type(prefix)))
        if not plutil.isbool(zero_based_index):
            raise TypeError('`zero_based_index` ({}) must be a bool'.format(
                type(zero_based_index)))

        offset = 0
        if not zero_based_index:
            offset = 1

        self.names = CustomOrderedDict()
        for asv in self.index:
            newname = prefix + '_{}'.format(int(asv.idx + offset))
            asv.name = newname
            self.names[asv.name] = asv


class qPCRdata:
    '''Single entry of qpcr data at a timepoint with maybe multiple technical replicates.
    Assumes that the dilution factor is constant between the replicate runs

    The normalized data is assumed to be:
        (cfus * dilution_factor / mass) * scaling_factor

    scaling_factor is a scale that we impose on the data so that the numbers don't get
    super large in the numerical calculations and we get errors, it does nothing to affect
    the empirical variance of the data.

    Parameters
    ----------
    cfus : np.ndarray
        These are the raw CFUs - it can be a single CFU measurement or a list of all
        the measurements
    mass : float
        This is the mass of the sample in grams
    dilution_factor : float
        This is the dilution factor of the samples
        Example:
            If the sample was diluted to 1/100 of its original concentration,
            the dilution factor is 100, NOT 1/100.

    '''
    def __init__(self, cfus, mass, dilution_factor):
        self._raw_data = np.asarray(cfus)
        self.mass = mass
        self.dilution_factor = dilution_factor
        self.scaling_factor = 1 # Initialize with no scaling factor
        self.recalculate_parameters()

    def recalculate_parameters(self):
        if len(self._raw_data) == 0:
            return

        self.data = (self._raw_data*self.dilution_factor/self.mass)*self.scaling_factor
        self.log_data = np.log(self.data)
        
        self.loc = np.mean(self.log_data)
        self.scale = np.std(self.log_data - self.loc)
        self.scale2 = self.scale ** 2


        self._mean_dist = np.exp(self.loc + (self.scale2/2) )
        self._var_dist = (np.exp(self.scale2) - 1) * np.exp(2*self.loc + self.scale2)
        self._std_dist = np.sqrt(self._var_dist)
        self._gmean = (np.prod(self.data))**(1/len(self.data))

    def __str__(self):
        s = 'cfus: {}\nmass: {}\ndilution_factor: {}\n scaling_factor: {}\n' \
            'data: {}\nlog_data: {}\nloc: {}\n scale: {}'.format( 
                self._raw_data, self.mass, self.dilution_factor, self.scaling_factor, 
                self.data, self.log_data, self.loc, self.scale)
        return s

    def add(self,raw_data):
        '''Add a single qPCR measurement to add to the set of observations

        Parameters
        ----------
        raw_data : float, array_like
            This is the measurement to add
        '''
        self._raw_data = np.append(self._raw_data,raw_data)
        self.recalculate_parameters()

    def set_to_nan(self):
        '''Set all attributes to `np.nan`
        '''
        self._raw_data *= np.nan
        self.data *= np.nan
        self.mass = np.nan
        self.dilution_factor = np.nan
        self._mean_dist = np.nan
        self._std_dist = np.nan
        self._var_dist = np.nan
        self._gmean = np.nan
        self.loc = np.nan
        self.scale = np.nan
        self.scale2 = np.nan
        self.scaling_factor = np.nan

    def set_scaling_factor(self, scaling_factor):
        '''Resets the scaling factor

        Parameters
        ----------
        scaling_factor : float, int
            This is the scaling factor to set everything to
        '''
        if scaling_factor <= 0:
            raise ValueError('The scaling factor must strictly be positive')
        self.scaling_factor = scaling_factor
        self.recalculate_parameters()

    def mean(self):
        '''Return the geometric mean
        '''
        return self.gmean()
    
    def var(self):
        return self._var_dist

    def std(self):
        return self._std_dist

    def gmean(self):
        return self._gmean


class CustomOrderedDict(dict):
    '''Order is an initialized version of self.keys() -> much more efficient
    index maps the key to the index in order

    order (list)
        - same as a numpy version of the keys in order
    index (dict)
        - Maps the key to the index that it was inserted in
    '''

    def __init__(self, *args, **kwargs):
        '''Extension of the OrderedDict

        Paramters
        ---------
        args, kwargs : Arguments
            These are extra arguments to initialize the baseline OrderedDict
        '''
        dict.__init__(self, *args, **kwargs)
        self.order = None
        self.index = None

    def update_order(self):
        '''This will update the reverse dictionary based on the index. It will 
        also redo the indexes if an ASV was deleted
        '''
        self.order = np.array(list(self.keys()))
        self.index = {}
        for i, asv in enumerate(self.order):
            self.index[asv] = i


class Subject(Saveable):
    '''Data for a single subject
    The ASVSet order is done with respect to the ordering in the `reads_table`
    Parameters
    ----------
    parent : Study
        This is the parent class (we have a reverse pointer)
    name : str
        This is the name of the subject
    '''
    def __init__(self, parent, name):
        self.name = name
        self.id = id(self)
        self.parent = parent
        self.qpcr = {}
        self.reads = {}
        self.times = np.asarray([])
        self._reads_individ = {} # for taking out aggregated asvs

    def add_time(self, timepoint):
        '''Add the timepoint `timepoint`. Set the reads and qpcr at that timepoint
        to None
        '''
        if timepoint in self.times:
            return
        self.times = np.sort(np.append(self.times, timepoint))
        self.reads[timepoint] = None
        self.qpcr[timepoint] = None

    def add_reads(self, timepoints, reads):
        '''Add the reads for timepoint `timepoint`

        Parameters
        ----------
        timepoint : numeric, array
            This is the time that the measurement occurs. If it is an array, then
            we are adding for multiple timepoints
        reads : np.ndarray(NASVS, N_TIMEPOINTS)
            These are the reads for the ASVs in order. Assumed to be in the 
            same order as the ASVSet. If it is a dataframe then we use the rows
            to index the ASV names. If timepoints is an array, then we are adding 
            for multiple timepoints. In this case we assume that the rows index the 
            ASV and the columns index the timepoint.
        '''
        if not plutil.isarray(timepoints):
            timepoints = [timepoints]
        for timepoint in timepoints:
            if not plutil.isnumeric(timepoint):
                raise TypeError('`timepoint` ({}) must be a numeric'.format(type(timepoint)))
        if not plutil.isarray(reads):
            raise TypeError('`reads` ({}) must be an array'.format(type(reads)))
        
        if reads.ndim == 1:
            reads = reads.reshape(-1,1)
        if reads.ndim != 2:
            raise ValueError('`reads` {} must be a matrix'.format(reads.shape))
        if reads.shape[0] != len(self.asvs) or reads.shape[1] != len(timepoints):
            raise ValueError('`reads` shape {} does not align with the number of asvs ({}) ' \
                'or timepoints ({})'.format(reads.shape, len(self.asvs), len(timepoints)))

        for tidx, timepoint in enumerate(timepoints):
            if timepoint in self.reads:
                logging.debug('There are already reads specified at time `{}` for subject `{}`, overwriting'.format(
                    timepoint, self.name))
                
            self.reads[timepoint] = reads[:,tidx]
            if timepoint not in self.times:
                self.times = np.sort(np.append(self.times, timepoint))
        return self

    def add_qpcr(self, timepoints, qpcr, masses=None, dilution_factors=None):
        '''Add qpcr measurements for timepoints `timepoints`

        Parameters
        ----------
        timepoint : numeric, array
            This is the time that the measurement occurs. If it is an array, then
            we are adding for multiple timepoints
        qpcr : np.ndarray(N_TIMEPOINTS, N_REPLICATES)
            These are the qPCR measurements in order of timepoints. Assumed to be in the 
            same order as timepoints.If timepoints is an array, then we are adding 
            for multiple timepoints. In this case we assume that the rows index the 
            timepoint and the columns index the replicates of the qpcr measurement.
        masses : numeric, np.ndarray
            These are the masses for each on of the qPCR measurements. If this is not 
            specified, then this assumes that the numbers in `qpcr` are already normalized
            by their sample weight.
        dilution_factors : numeric, np.ndarray
            These are the dilution factors for each of the qPCR measurements. If this is
            not specified, then this assumes that each one of the numbers in `qpcr` are
            already normalized by the dilution factor
        '''
        if not plutil.isarray(timepoints):
            timepoints = [timepoints]
        for timepoint in timepoints:
            if not plutil.isnumeric(timepoint):
                raise TypeError('`timepoint` ({}) must be a numeric'.format(type(timepoint)))
        if masses is not None:
            if plutil.isnumeric(masses):
                masses = [masses]
            for mass in masses:
                if not plutil.isnumeric(mass):
                    raise TypeError('Each mass in `masses` ({}) must be a numeric'.format(type(mass)))
                if mass <= 0:
                    raise ValueError('Each mass in `masses` ({}) must be > 0'.format(mass))
            if len(masses) != len(timepoints):
                raise ValueError('Number of timepoints ({}) and number of masses ({}) ' \
                    'must be equal'.format(len(timepoints), len(masses)))
        if dilution_factors is not None:
            if plutil.isnumeric(dilution_factors):
                dilution_factors = [dilution_factors]
            for dilution_factor in dilution_factors:
                if not plutil.isnumeric(dilution_factor):
                    raise TypeError('Each dilution_factor in `dilution_factors` ({}) ' \
                        'must be a numeric'.format(type(dilution_factor)))
                if dilution_factor <= 0:
                    raise ValueError('Each dilution_factor in `dilution_factors` ({}) ' \
                        'must be > 0'.format(dilution_factor))
            if len(dilution_factors) != len(timepoints):
                raise ValueError('Number of timepoints ({}) and number of dilution_factors ({}) ' \
                    'must be equal'.format(len(timepoints), len(dilution_factors)))
            
        if not plutil.isarray(qpcr):
            raise TypeError('`qpcr` ({}) must be an array'.format(type(qpcr)))
        if qpcr.ndim == 1:
            qpcr = qpcr.reshape(1,-1)
        if qpcr.ndim != 2:
            raise ValueError('`qpcr` {} must be a matrix'.format(qpcr.shape))
        if qpcr.shape[0] != len(timepoints):
            raise ValueError('`qpcr` shape {} does not align with the number of timepoints ({}) ' \
                ''.format(qpcr.shape, len(timepoints)))

        for tidx, timepoint in enumerate(timepoints):
            if timepoint in self.qpcr:
                logging.debug('There are already qpcr measurements specified at time `{}` for subject `{}`, overwriting'.format(
                    timepoint, self.name))
            if masses is not None:
                mass = masses[tidx]
            else:
                mass = 1
            if dilution_factors is not None:
                dil = dilution_factors[tidx]
            else:
                dil = 1

            self.qpcr[timepoint] = qPCRdata(cfus=qpcr[tidx,:], mass=mass, 
                dilution_factor=dil)
                
            if timepoint not in self.times:
                self.times = np.sort(np.append(self.times, timepoint))
        return self

    @property
    def perturbations(self):
        return self.parent.perturbations

    @property
    def asvs(self):
        return self.parent.asvs

    @property
    def index(self):
        '''Return the index of this subject in the Study file
        '''
        for iii, subj in enumerate(self.parent):
            if subj.name == self.name:
                return iii
        raise ValueError('Should not get here')

    def matrix(self):
        '''Make a numpy matrix out of our data - returns the raw reads,
        the relative abundance, and the absolute abundance.

        If there is no qPCR data, then the absolute abundance is set to None.
        '''

        shape = (len(self.asvs), len(self.times))
        raw = np.zeros(shape=shape, dtype=int)
        rel = np.zeros(shape=shape, dtype=float)
        abs = np.zeros(shape=shape, dtype=float)

        for i,t in enumerate(self.times):
            raw[:,i] = self.reads[t]
            rel[:,i] = raw[:,i]/np.sum(raw[:,i])
        
        try:
            for i,t in enumerate(self.times):
                abs[:,i] = rel[:,i] * self.qpcr[t].mean()
        except AttributeError as e:
            logging.info('Attribute Error ({}) for absolute abundance. This is likely ' \
                'because you did not set the qPCR abundances. Skipping `abs`'.format(e))
            abs = None

        return {'raw':raw, 'rel': rel, 'abs':abs}

    def df(self, **kwargs):
        '''Returns a dataframe of the data - same as matrix

        Parameters
        ----------
        These are the parameters for `matrix`
        '''
        d = self.matrix(**kwargs)
        index = self.asvs.names.order
        times = self.times
        for key in d:
            d[key] = pd.DataFrame(data=d[key], index=index, columns=times)
        return d

    def read_depth(self, t=None):
        '''Get the read depth at time `t`. If nothing is given then return all
        of them

        Parameters
        ----------
        t : int, float, Optional
            Get the read depth at this time. If nothing is provided, all of the read depths for this 
            subject are returned
        '''
        if t is None:
            return np.sum(self.matrix()['raw'], axis=0)
        if t not in self.reads:
            raise ValueError('`t` ({}) not recognized. Valid times: {}'.format(
                t, self.times))
        return np.sum(self.reads[t])

    def cluster_by_taxlevel(self, dtype, taxlevel, index_formatter=None, smart_unspec=True):
        '''Clusters the ASVs into the taxonomic level indicated in `taxlevel`.

        Smart Unspecified
        -----------------
        If True, returns the higher taxonomic classification while saying the desired taxonomic level
        is unspecified. Example: 'Order ABC, Family NA'. Note that this overrides the `index_formatter`.

        Parameters
        ----------
        subj : pylab.base.Subject
            This is the subject that we are getting the data from
        taxlevel : str, None
            This is the taxa level to aggregate the data at. If it is 
            None then we do not do any collapsing (this is the same as 'asv')
        dtype : str
            This is the type of data to cluster. Options are:
                'raw': These are the counts
                'rel': This is the relative abundances
                'abs': This is the absolute abundance (qPCR * rel)
        index_formatter : str
            How to make the index using `pylab.util.plutil.asvname_formatter`. Note that you cannot
            specify anything at a lower taxonomic level than what youre clustering at. For 
            example, you cannot cluster at the 'class' level and then specify '%(genus)s' 
            in the index formatter.
            If nothing is specified then only return the specified taxonomic level

        Returns
        -------
        pandas.DataFrame
            Dataframe of the data
        dict (str->str)
            Maps ASV name to the row it got allocated to
        '''
        # Type checking
        if not plutil.isstr(dtype):
            raise TypeError('`dtype` ({}) must be a str'.format(type(dtype)))
        if dtype not in ['raw', 'rel', 'abs']:
            raise ValueError('`dtype` ({}) not recognized'.format(dtype))
        if not plutil.isstr(taxlevel):
            raise TypeError('`taxlevel` ({}) must be a str'.format(type(taxlevel)))
        if taxlevel not in ['kingdom', 'phylum', 'class',  'order', 'family', 
            'genus', 'species', 'asv']:
            raise ValueError('`taxlevel` ({}) not recognized'.format(taxlevel))
        if index_formatter is None:
            index_formatter = taxlevel
        if index_formatter is not None:
            if not plutil.isstr(index_formatter):
                raise TypeError('`index_formatter` ({}) must be a str'.format(type(index_formatter)))
            
            for tx in TAX_IDXS:
                if tx in index_formatter and TAX_IDXS[tx] > TAX_IDXS[taxlevel]:
                    raise ValueError('You are clustering at the {} level but are specifying' \
                        ' {} in the `index_formatter`. This does not make sense. Either cluster' \
                        'at a lower tax level or specify the `index_formatter` to a higher tax ' \
                        'level'.format(taxlevel, tx))

        index_formatter = index_formatter.replace('%(asv)s', '%(name)s')

        # Everything is valid, get the data dataframe and the return dataframe
        asvname_map = {}
        df = self.df()[dtype]
        cols = list(df.columns)
        cols.append(taxlevel)
        dfnew = pd.DataFrame(columns = cols).set_index(taxlevel)

        # Get the level in the taxonomy, create a new entry if it is not there already
        taxas = {} # lineage -> label
        for i, asv in enumerate(self.asvs):
            row = df.index[i]
            tax = asv.get_lineage(level=taxlevel)
            tax = tuple(tax)
            tax = str(tax).replace("'", '')
            if tax in taxas:
                dfnew.loc[taxas[tax]] += df.loc[row]
            else:
                if not asv.tax_is_defined(taxlevel) and smart_unspec:
                    # Get the least common ancestor above the taxlevel
                    taxlevelidx = TAX_IDXS[taxlevel]
                    ttt = None
                    while taxlevelidx > -1:
                        if asv.tax_is_defined(_TAX_REV_IDXS[taxlevelidx]):
                            ttt = _TAX_REV_IDXS[taxlevelidx]
                            break
                        taxlevelidx -= 1
                    if ttt is None:
                        raise ValueError('Could not find a single taxlevel: {}'.format(str(asv)))
                    taxas[tax] = '{} {}, {} NA'.format(ttt.capitalize(), 
                        asv.taxonomy[ttt], taxlevel.capitalize())
                else:
                    taxas[tax] = plutil.asvname_formatter(format=index_formatter, asv=asv, asvs=self.asvs)
                toadd = pd.DataFrame(np.array(list(df.loc[row])).reshape(1,-1),
                    index=[taxas[tax]], columns=dfnew.columns)
                dfnew = dfnew.append(toadd)
            
            if taxas[tax] not in asvname_map:
                asvname_map[taxas[tax]] = []
            asvname_map[taxas[tax]].append(asv.name)
        
        return dfnew, asvname_map

    def _split_on_perturbations(self):
        '''If there are perturbations, then we take out the data on perturbations
        and we set the data in the different segments to different subjects

        Internal funciton, should not be used by the user
        '''
        if len(self.parent.perturbations) == 0:
            logging.info('No perturbations to split on, do nothing')
            return

        # Get the time intervals for each of the times that we are not on perturbations
        start_tidx = 0
        not_perts = []
        in_pert = False
        for i in range(len(self.times)):
            # check if the time is in a perturbation
            a = False
            for pert in self.parent.perturbations:
                start = pert.start
                end = pert.end
                # check if in the perturbation
                if self.times[i] > start and self.times[i] <= end:
                    a = True
                    break
            if a:
                # If the current time point is in a perturbation and we previously
                # have no been in a perturbation, this means we can add the previous
                # interval into the intervals that we want to keep
                if not in_pert:
                    not_perts.append((start_tidx, i))
                in_pert = True
            else:
                # If we are not currently in a perturbation but we previously were
                # then we restart to `start_tidx`
                if in_pert:
                    start_tidx = i
                    in_pert = False
        # If we have finished and we are out of a perturbation at the end, then
        # we can add the rest of the times at the end to a valid not in perturbation time
        if not in_pert:
            not_perts.append((start_tidx, len(self.times)))

        # For each of the time slices recorded, make a new subject
        if len(in_pert) == 0:
            raise ValueError('THere are perturbations ({}), this must not be zero.' \
                ' Something went wrong'.format(len(self.parent.perturbations)))
        ii = 0
        for start,end in not_perts:
            mid = self.name+'_{}'.format(ii)
            self.parent.add(name=mid)
            for i in range(start,end):
                t = self.times[i]
                self.parent[mid].qpcr[t] = self.qpcr[t]
                self.parent[mid].reads[t] = self.reads[t]
            self.parent[mid].times = self.times[start:end]

    def _deaggregate_item(self, agg, other):
        '''Deaggregate the sequence `other` from AggregateASV `agg`.
        `other` is then appended to the end. This is called from 
        `mdsine2.Study.deaggregate_item`.

        Parameters
        ----------
        agg : AggregateASV
            This is an AggregateASV with multiple sequences contained. Must 
            have the name `other` in there
        other : str
            This is the name of the ASV that should be taken out of `agg`
        '''
        # Append the reads of the deaggregated at the bottom and subtract them
        # from the aggregated index
        if other not in self._reads_individ:
            raise ValueError('`other` ({}) reads not found in archive. This probably ' \
                'happened because you called `aggregate_items` from the ASVSet object' \
                ' instead from this object. Study object not consistent. Failing.'.format(other))
        
        aggidx = agg.idx
        for t in self.times:
            try:
                new_reads = self._reads_individ[other][t]
            except:
                raise ValueError('Timepoint `{}` added into subject `{}` after ' \
                    'ASV `{}` was removed. Study object is not consistent. You ' \
                    'cannot add in other timepoints after you aggregate asvs. Failing.'.format(
                        t, self.name, other))
            self.reads[t][aggidx] = self.reads[t][aggidx] - new_reads
            self.reads[t] = np.append(self.reads[t], new_reads)
        self._reads_individ.pop(other)
        return

    def _aggregate_items(self, anchor, other):
        '''Aggregate the asv `other` into `anchor`. This is called from 
        `mdsine2.Study.aggregate_items`.

        Parameters
        ----------
        anchor, other : AggregateASV, ASV
            These are the ASVs to combine
        '''
        # If one of them are ASVs, then record their individual reads
        # if we want to dissociate them later
        for asv in [anchor, other]:
            if isasv(asv):
                if asv.name in self._reads_individ:
                    raise ValueError('ASV is already in this dict. This should not happen.')
                aidx = asv.idx
                self._reads_individ[asv.name] = {}
                for t in self.times:
                    self._reads_individ[asv.name][t] = self.reads[t][aidx]
        
        for t in self.times:
            self.reads[t][anchor.idx] += self.reads[t][other.idx]
            self.reads[t] = np.delete(self.reads[t], other.idx)
        return


class Study(Saveable):
    '''Holds data for all the subjects

    Paramters
    ---------
    asvs : ASVSet, Optional
        Contains all of the ASVs
    '''
    def __init__(self, asvs):
        self.id = id(self)
        self._subjects = {}
        self.perturbations = None
        self.qpcr_normalization_factor = None
        if not isasvset(asvs):
            raise ValueError('If `asvs` ({}) is specified, it must be an ASVSet' \
                ' type'.format(type(asvs)))
        self.asvs = asvs

        self._samples = {}
        

    def __getitem__(self, key):
        return self._subjects[key]

    def __len__(self):
        return len(self._subjects)

    def __iter__(self):
        for v in self._subjects.values():
            yield v

    def __contains__(self, key):
        return key in self._subjects

    def parse(self, metadata, reads=None, qpcr=None):
        '''Parse tables of samples and cast in Subject sets. Automatically creates
        the subject classes with the respective names.

        Parameters
        ----------
        metadata : pandas.DataFrame
            Contains the meta data for each one of the samples
            Columns:
                'sampleID' -> str : This is the name of the sample
                'subject' -> str : This is the name of the subject
                'time' -> float : This is the time the sample takes place
                'perturbation:`name`' -> int : This is a perturbation meta data where the
                    name of the perturbation is `name`
        reads : pandas.DataFrame, None
            Contains the reads for each one of the samples and asvs
                index (str) : indexes the ASV name
                columns (str) : indexes the sample ID
            If nothing is passed in, the reads are set to None
        qpcr : pandas.DataFrame, None
            Contains the qpcr measurements for each sample
                index (str) : indexes the sample ID
                columns (str) : Name is ignored. the values are set to the 
        '''
        if not plutil.isdataframe(metadata):
            raise TypeError('`metadata` ({}) must be a pandas.DataFrame'.format(type(metadata)))
        
        # Add the samples
        # ---------------
        if 'sampleID' in metadata.columns:
            metadata = metadata.set_index('sampleID')
        for sampleid in metadata.index:

            sid = str(metadata['subject'][sampleid])
            t = float(metadata['time'][sampleid])

            if sid not in self:
                self.add_subject(name=sid)
            if t not in self[sid].times:
                self[sid].add_time(timepoint=t)

            self._samples[str(sampleid)] = (sid,t)

        # Add the perturbations if there are any
        # --------------------------------------
        for col in metadata.columns:
            pert_name = col.replace('perturbation:', '')
            if 'perturbation:' in col:
                min_time = None
                max_time = None
                for sampleid in metadata.index:
                    if metadata[col][sampleid] == 1:
                        t = float(metadata['time'][sampleid])
                        if min_time is None:
                            min_time = t
                        else:
                            if t < min_time:
                                min_time = t
                        if max_time is None:
                            max_time = t
                        else:
                            if t > max_time:
                                max_time = t
                if max_time is None or min_time is None:
                    raise ValueError('Perturbation `{}` for column `{}` did not find any ' \
                        'times'.format(pert_name, col))

                self.add_perturbation(min_time, end=max_time, name=pert_name)

        # Add the reads if necessary
        # --------------------------
        if reads is not None:
            if not plutil.isdataframe(reads):
                raise TypeError('`reads` ({}) must be a pandas.DataFrame'.format(type(reads)))
            
            if 'name' in reads.columns:
                reads = reads.set_index('name')

            for sampleid in reads.columns:
                if sampleid == SEQUENCE_COLUMN_LABEL:
                    continue
                try:
                    sid, t = self._samples[sampleid]
                except:
                    raise ValueError('Sample ID `{}` not found in metadata ({}). Make sure ' \
                        'you set the sample ID as the columns in the `reads` dataframe'.format(
                            sampleid, list(self._samples.keys())))
                self[sid].add_reads(timepoints=t, reads=reads[sampleid].to_numpy())

        # Add the qPCR measurements if necessary
        # --------------------------------------
        if qpcr is not None:
            if not plutil.isdataframe(qpcr):
                raise TypeError('`qpcr` ({}) must be a pandas.DataFrame'.format(type(qpcr)))

            if 'sampleID' in qpcr.columns:
                qpcr = qpcr.set_index('sampleID')

            for sampleid in qpcr.index:
                try:
                    sid, t = self._samples[sampleid]
                except:
                    raise ValueError('Sample ID `{}` not found in metadata ({}). Make sure ' \
                        'you set the sample ID as the index in the `qpcr` dataframe'.format(
                            sampleid, list(self._samples.keys())))
                cfuspergram = qpcr.loc[sampleid].to_numpy()
                self[sid].add_qpcr(timepoints=t, qpcr=cfuspergram)
            
    def write_metadata_to_table(self, path, sep='\t'):
        raise NotImplementedError('Need to implement it')

    def write_reads_to_table(self, path, sep='\t'):
        raise NotImplementedError('Need to implement it')

    def write_qpcr_to_table(self, path, sep='\t'):
        raise NotImplementedError('Need to implement it')

    def names(self):
        '''List the names of the contained subjects
        '''
        return [subj.name for subj in self]

    def iloc(self, idx):
        '''Get the subject as an index

        Parameters
        ----------
        idx : int
            Index of the subject

        Returns
        -------
        pl.base.Subject
        '''
        for i,sid in enumerate(self._subjects):
            if i == idx:
                return self._subjects[sid]
        raise IndexError('Index ({}) not found'.format(idx))

    def add_subject(self, name):
        '''Create a subject with the name `name`

        Parameters
        ----------
        name : str
            This is the name of the new subject
        '''
        if name not in self._subjects:
            self._subjects[name] = Subject(name=name, parent=self)
        return self

    def pop_subject(self, sid):
        '''Remove the indicated subject id

        Parameters
        ----------
        sid : list(str), str, int
            This is the subject name/s or the index/es to pop out.
            Return a new Study with the specified subjects removed.
        '''
        if not plutil.isarray(sid):
            sids = [sid]
        else:
            sids = sid

        for i in range(len(sids)):
            if plutil.isint(sids[i]):
                sids[i] = list(self._subjects.keys())[sids[i]]
            elif not plutil.isstr(sids[i]):
                raise ValueError('`sid` ({}) must be a str'.format(type(sids[i])))
        ret = Study(asvs=self.asvs)
        ret.perturbations = self.perturbations
        ret.qpcr_normalization_factor = self.qpcr_normalization_factor

        for s in sids:
            if s in self._subjects:
                ret._subjects[s] =  self._subjects.pop(s, None)
            else:
                raise ValueError('`sid` ({}) not found'.format(sid))
        return ret

    def pop_asvs(self, oids):
        '''Delete the ASVs indicated in oidxs. Updates the reads table and
        the internal ASVSet

        Parameters
        ----------
        oids : str, int, list(str/int)
            These are the identifiers for each of the ASV/s to delete
        '''
        # get indices
        oidxs = []
        for oid in oids:
            oidxs.append(self.asvs[oid].idx)
        
        # Delete the ASVs from asvset
        for oid in oids:
            self.asvs.del_asv(oid)

        # Delete the reads
        for subj in self:
            for t in subj.reads:
                subj.reads[t] = np.delete(subj.reads[t], oidxs)
        return self

    def deaggregate_item(self, agg, other):
        '''Deaggregate the sequence `other` from AggregateASV `agg`.
        `other` is then appended to the end 

        Parameters
        ----------
        agg : AggregateASV, str
            This is an AggregateASV with multiple sequences contained. Must 
            have the name `other` in there
        other : str
            This is the name of the ASV that should be taken out of `agg`

        Returns
        -------
        mdsine2.ASV
            This is the deaggregated ASV
        '''
        agg = self.asvs[agg]
        if not isaggregatedasv(agg):
            raise TypeError('`agg` ({}) must be an AggregatedASV'.format(type(agg)))
        if not plutil.isstr(other):
            raise TypeError('`other` ({}) must be a str'.format(type(other)))
        if other not in agg.aggregated_asvs:
            raise ValueError('`other` ({}) is not contained in `agg` ({}) ({})'.format(
                other, agg.name, agg.aggregated_asvs))

        for subj in self:
            subj._deaggregate_item(agg=agg, other=other)
        return self.asvs.deaggregate_item(agg, other)

    def aggregate_items(self, asv1, asv2):
        '''Aggregates the abundances of `asv1` and `asv2`. Updates the reads table and
        internal ASVSet

        Parameters
        ----------
        asv1, asv2 : str, int, mdsine2.ASV, mdsine2.AggregateASV
            These are the ASVs you are agglomerating together

        Returns
        -------
        mdsine2.AggregateASV
            This is the new aggregated ASV containing anchor and other
        '''
        # Find the anchor - use the highest index
        aidx1 = self.asvs[asv1].idx
        aidx2 = self.asvs[asv2].idx

        if aidx1 == aidx2:
            raise ValueError('Cannot aggregate the same asv: {}'.format(self.asvs[asv1]))
        elif aidx1 < aidx2:
            anchor = self.asvs[asv1]
            other = self.asvs[asv2]
        else:
            anchor = self.asvs[asv2]
            other = self.asvs[asv1]

        for subj in self:
            subj._aggregate_items(anchor=anchor, other=other)
        return self.asvs.aggregate_items(anchor=anchor, other=other)

    def pop_times(self, times, sids='all'):
        '''Discard the times in `times` for the subjects listed in `sids`.
        If a timepoint is not found in a subject, no error is thrown.

        Parameters
        ----------
        times : numeric, list(numeric)
            Time/s to delete
        sids : str, int, list(int)
            The Subject ID or a list of subject IDs that you want to delete the timepoints
            from. If it is a str:
                'all' - delete from all subjects
        '''
        if plutil.isstr(sids):
            if sids == 'all':
                sids = list(self._subjects.keys())
            else:
                raise ValueError('`sids` ({}) not recognized'.format(sids))
        elif plutil.isint(sids):
            if sids not in self._subjects:
                raise IndexError('`sid` ({}) not found in subjects'.format(
                    list(self._subjects.keys())))
            sids = [sids]
        elif plutil.isarray(sids):
            for sid in sids:
                if not plutil.isint(sid):
                    raise TypeError('Each sid ({}) must be an int'.format(type(sid)))
                if sid not in self._subjects:
                    raise IndexError('Subject {} not found in subjects ({})'.format(
                        sid, list(self._subjects.keys())))
        else:
            raise TypeError('`sids` ({}) type not recognized'.format(type(sids)))
        if plutil.isnumeric(times):
            times = [times]
        elif plutil.isarray(times):
            for t in times:
                if not plutil.isnumeric(t):
                    raise TypeError('Each time ({}) must be a numeric'.format(type(t)))
        else:
            raise TypeError('`times` ({}) type not recognized'.format(type(times)))

        for t in times:
            for sid in sids:
                subj = self._subjects[sid]
                if t in subj.times:
                    subj.qpcr.pop(t, None)
                    subj.reads.pop(t,None)
                    subj.times = np.sort(list(subj.reads.keys()))

    def normalize_qpcr(self, max_value):
        '''Normalize the qPCR values such that the largest value is the max value
        over all the subjects

        Parameters
        ----------
        max_value : float, int
            This is the maximum qPCR value to

        Returns
        -------
        self
        '''
        if type(max_value) not in [int, float]:
            raise ValueError('max_value ({}) must either be an int or a float'.format(
                type(max_value)))

        if self.qpcr_normalization_factor is not None:
            logging.warning('qPCR is already rescaled. unscaling and rescaling')
            self.denormalize_qpcr()

        temp_max = -1
        for subj in self:
            for key in subj.qpcr:
                temp_max = np.max([temp_max, subj.qpcr[key].mean()])

        self.qpcr_normalization_factor = max_value/temp_max
        logging.info('max_value found: {}, scaling_factor: {}'.format(
            temp_max, self.qpcr_normalization_factor))

        for subj in self:
            for key in subj.qpcr:
                subj.qpcr[key].set_scaling_factor(scaling_factor=
                    self.qpcr_normalization_factor)
        return self

    def denormalize_qpcr(self):
        '''Denormalizes the qpcr values if necessary

        Returns
        -------
        self
        '''
        if self.qpcr_normalization_factor is None:
            logging.warning('qPCR is not normalized. Doing nothing')
            return
        for subj in self:
            for key in subj.qpcr:
                subj.qpcr[key].set_scaling_factor(scaling_factor=1)
        self.qpcr_normalization_factor = None
        return self

    def add_perturbation(self, a, end=None, name=None):
        '''Add a perturbation. 
        
        We can either do this by passing a perturbation object 
        (if we do this then we do not need to specify `end`) or we can 
        specify the start and stop times (if we do this them we need to
        specify `end`).

        Parameters
        ----------
        a : numeric, BasePerturbation
            If this is a numeric, then this corresponds to the start
            time of the perturbation. If this is a Pertubration object
            then we just add this.
        end : numeric
            Only necessary if `a` is a numeric
        name : str, None
            Only necessary if `a` is a numeric. Name of the perturbation
        
        Returns
        -------
        self
        '''
        if self.perturbations is None:
            self.perturbations = []
        if plutil.isnumeric(a):
            if not plutil.isnumeric(end):
                raise ValueError('If `a` is a numeric, then `end` ({}) ' \
                    'needs to be a numeric'.format(type(end)))
            self.perturbations.append(BasePerturbation(start=a, end=end, name=name))
        elif isperturbation(a):
            self.perturbations.append(a)
        else:
            raise ValueError('`a` ({}) must be a subclass of ' \
                'pl.base.BasePerturbation or a numeric'.format(type(a)))
        return self
        
    def split_on_perturbations(self):
        '''Make new subjects for the time points that are divided by perturbations. 
        Throw out all of the data  where the perturbations are active.

        Returns
        -------
        self
        '''
        for subj in self:
            subj._split_on_perturbations()
        return self

    def _matrix(self, dtype, agg, times):
        if dtype not in ['raw', 'rel', 'abs']:
            raise ValueError('`dtype` ({}) not recognized'.format(dtype))
        
        if agg == 'mean':
            aggfunc = np.nanmean
        elif agg == 'median':
            aggfunc = np.nanmedian
        elif agg == 'sum':
            aggfunc = np.nansum
        elif agg == 'max':
            aggfunc = np.nanmax
        elif agg == 'min':
            aggfunc = np.nanmin
        else:
            raise ValueError('`agg` ({}) not recognized'.format(agg))

        if plutil.isstr(times):
            all_times = []
            for subj in self:
                all_times = np.append(all_times, subj.times)
            all_times = np.sort(np.unique(all_times))
            if times == 'union':
                times = all_times

            elif times == 'intersection':
                times = []
                for t in all_times:
                    addin = True
                    for subj in self:
                        if t not in subj.times:
                            addin = False
                            break
                    if addin:
                        times = np.append(times, t)

            else:
                raise ValueError('`times` ({}) not recognized'.format(times))
        elif plutil.isarray(times):
            times = np.array(times)
        else:
            raise TypeError('`times` type ({}) not recognized'.format(type(times)))

        shape = (len(self.asvs), len(times))
        M = np.zeros(shape, dtype=float)
        for tidx, t in enumerate(times):
            temp = None
            for subj in self:
                if t not in subj.times:
                    continue
                if dtype == 'raw':
                    a = subj.reads[t]
                elif dtype == 'rel':
                    a = subj.reads[t]/np.sum(subj.reads[t])
                else:
                    rel = subj.reads[t]/np.sum(subj.reads[t])
                    a = rel * subj.qpcr[t].mean()
                if temp is None:
                    temp = (a.reshape(-1,1), )
                else:
                    temp = temp + (a.reshape(-1,1), )
            if temp is None:
                temp = np.zeros(len(self.asvs)) * np.nan
            else:
                temp = np.hstack(temp)
                temp = aggfunc(temp, axis=1)
            M[:, tidx] = temp

        return M, times

    def matrix(self, dtype, agg, times):
        '''Make a matrix of the aggregation of all the subjects in the subjectset

        Aggregation of subjects
        -----------------------
        What are the values for the ASVs? Set the aggregation type using the parameter `agg`. 
        These are the types of aggregations:
            'mean': Mean abundance of the ASV at a timepoint over all the subjects
            'median': Median abundance of the ASV at a timepoint over all the subjects
            'sum': Sum of all the abundances of the ASV at a timepoint over all the subjects
            'max': Maximum abundance of the ASV at a timepoint over all the subjects
            'min': Minimum abundance of the ASV at a timepoint over all the subjects

        Aggregation of times
        --------------------
        Which times to include? Set the times to include with the parameter `times`.
        These are the types of time aggregations:
            'union': Take  theunion of the times of the subjects
            'intersection': Take the intersection of the times of the subjects
        You can manually specify the times to include with a list of times. If times are not
        included in any of the subjects then we set them to NAN.

        Parameters
        ----------
        dtype : str
            What kind of data to return. Options:
                'raw': Count data
                'rel': Relative abundance
                'abs': Abundance data
        agg : str
            Type of aggregation of the values. Options specified above.
        times : str, array
            The times to include
        
        Returns
        -------
        np.ndarray(n_asvs, n_times)
        '''
        M, _ =  self._matrix(dtype=dtype, agg=agg, times=times)
        return M

    def df(self, *args, **kwargs):
        '''Returns a dataframe of the data in matrix. Rows are ASVs, columns are times.

        Returns
        -------
        pandas.DataFrame

        See Also
        --------
        mdsine2.matrix
        '''
        M, times = self._matrix(*args, **kwargs)
        index = self.asvs.names.order
        return pd.DataFrame(data=M, index=index, columns=times)

