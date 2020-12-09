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
import copy

from . import util as plutil
from .errors import NeedToImplementError
from . import diversity

# Constants
DEFAULT_TAXA_NAME = 'NA'
SEQUENCE_COLUMN_LABEL = 'sequence'
TAX_IDXS = {'kingdom': 0, 'phylum': 1, 'class': 2,  'order': 3, 'family': 4, 
    'genus': 5, 'species': 6, 'asv': 7}
TAX_LEVELS = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species', 'asv']


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

def istaxaset(x):
    '''Checks whether the input is a subclass of TaxaSet

    Parameters
    ----------
    x : any
        Input instance to check the type of TaxaSet
    
    Returns
    -------
    bool
        True if `x` is of type TaxaSet, else False
    '''
    return x is not None and issubclass(x.__class__, TaxaSet)

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

def istaxa(x):
    '''Checks whether the input is a subclass of Taxa

    Parameters
    ----------
    x : any
        Input instance to check the type of Taxa
    
    Returns
    -------
    bool
        True if `x` is of type Taxa, else False
    '''
    return x is not None and issubclass(x.__class__, Taxa)

def isotu(x):
    '''Checks whether the input is a subclass of OTU

    Parameters
    ----------
    x : any
        Input instance to check the type of Taxa
    
    Returns
    -------
    bool
        True if `x` is of type Taxa, else False
    '''
    return issubclass(x.__class__, OTU)

def istaxatype(x):
    '''Checks whether the input is a subclass of OTU or Taxa

    Parameters
    ----------
    x : any
        Input instance to check the type of OTU or Taxa
    
    Returns
    -------
    bool
        True if `x` is of type OTU or Taxa, else False
    '''
    return istaxa(x) or isotu(x)

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

def condense_matrix_with_taxonomy(M, taxas, fmt):
    '''Condense the specified matrix M thats on the asv level
    to a taxonomic label specified with `fmt`. If `M` 
    is a pandas.DataFrame then we assume the index are the Taxa
    names. If `M` is a numpy.ndarray, then we assume that the 
    order of the matrix mirrors the order of the taxas. `fmt` is
    passed through `pylab.util.taxaname_formatter` to get the label.

    Parameters
    ----------
    M : numpy.ndarray, pandas.DataFrame
        Matrix to condense
    taxas : pylab.base.TaxaSet
        Set of Taxas with the relevant taxonomic information
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
    if not istaxaset(taxas):
        raise TypeError('`taxas` ({}) must be a pylab.base.TaxaSet'.format(type(taxas)))
    if not plutil.isstr(fmt):
        raise TypeError('`fmt` ({}) must be a str'.format(type(fmt)))

    if type(M) == pd.DataFrame:
        for idx in M.index:
            if idx not in taxas:
                raise ValueError('row `{}` not found in taxas'.format(idx))
        names = M.index
    elif plutil.isarray(M):
        if M.shape[0] != len(taxas):
            raise ValueError('Number of rows in M ({}) not equal to number of taxas ({})'.format(
                M.shape[0], len(taxas)))
        names = taxas.names.order
    else:
        raise TypeError('`M` ({}) type not recognized'.format(type(M)))

    # Get the rows that correspond to each row
    d = {}
    for row, name in enumerate(names):
        taxa = taxas[name]
        tax = plutil.taxaname_formatter(format=fmt, taxa=taxa, taxas=taxas)
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

    Does not have to be applied to all subjects, and each subject can have a different start and
    end time to each other.

    Paramters
    ---------
    name : str, None
        This is the name of the perturabtion. If nothing is given then the name will be
        set to the perturbation index
    starts, ends : dict, None
        This is a map to the start and end times for the subject that have this perturbation
    '''
    def __init__(self, name, starts=None, ends=None):
        if not plutil.isstr(name):
            raise TypeError('`name` ({}) must be a str'.format(type(name)))
        if (starts is not None and ends is None) or (starts is None and ends is not None):
            raise ValueError('If `starts` or `ends` is specified, the other must be specified.')
        if starts is not None:
            if not plutil.isdict(starts):
                raise TypeError('`starts` ({}) must be a dict'.format(starts))
            if not plutil.isdict(ends):
                raise TypeError('`ends` ({}) must be a dict'.format(ends))

        self.starts = starts
        self.ends = ends
        self.name = name

    def __str__(self):
        s = 'Perturbation {}:\n'.format(self.name)
        if self.starts is not None:
            for subj in self.starts:
                s += '\tSubject {}: ({}, {})\n'.format(subj, self.starts[subj], 
                    self.ends[subj])
        return s

    def __contains__(self, a):
        '''Checks if subject name `a` is in this perturbation
        '''
        if issubject(a):
            a = a.name
        return a in self.starts

    def isactive(self, time, subj):
        '''Returns a `bool` if the perturbation is on at time `time`.

        Parameters
        ----------
        time : float, int
            Time to check
        subj : str
            Subject to check

        Returns
        -------
        bool

        Raises
        ------
        ValueError
            If there are no start and end times set
        '''
        if self.starts is None:
            raise ValueError('`start` is not set in {}'.format(self.name))
        try:
            start = self.starts[subj]
            end = self.ends[subj]
        except:
            raise KeyError('`subj` {} not specified for {}'.format(subj, self.name))

        return time > start and time <= end


class Perturbations:
    '''Aggregator for individual perturbation obejcts
    '''
    def __init__(self):
        self._d = {}
        self._rev_idx = []

    def __len__(self):
        return len(self._d)

    def __getitem__(self, a):
        if isperturbation(a):
            if a.name in self:
                return a
            else:
                raise KeyError('`a` ({}) not contained in this Set'.format(a))
        if plutil.isstr(a):
            return self._d[a]
        elif plutil.isint(a):
            return self._d[self._rev_idx[a]]
        else:
            raise KeyError('`a` {} ({}) not recognized'.format(a, type(a)))

    def __contains__(self, a):
        try:
            _ = self[a]
            return True
        except:
            False

    def __iter__(self):
        for a in self._d:
            yield self._d[a]

    def append(self, a):
        '''Add a perturbation

        a : mdsine2.BasePertubration
            Perturbation to add
        '''
        if not isperturbation(a):
            raise TypeError('`a` ({}) must be a perturbation'.format(type(a)))
        self._d[a.name] = a
        self._rev_idx.append(a.name)

    def remove(self, a):
        '''Remove the perturbation `a`

        Parameters
        ----------
        a : str, int, mdsine2.BasePerturbation
            Perturbation to remove
        
        Returns
        -------
        mdsine2.BasePerturbation
        '''
        a = self[a]
        self._d.pop(a.name, None)
        self._rev_idx = []
        for mer in self._d:
            self._rev_idx.append(mer.name)
        return a


class ClusterItem:
    '''These are single points that get clustered

    It must have the parameter
        'name'
    '''
    def __init__(self, name):
        self.name = name

    def cluster_str(self):
        return self.name


class Taxa(ClusterItem):
    '''Wrapper class for a single Taxa

    Parameters
    ----------
    name : str
        Name given to the Taxa
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
        '''Compares different Taxas between each other. Checks all of the attributes but the id

        Parameters
        ----------
        val : any
            This is what we are checking if they are equivalent
        '''
        if type(val) != Taxa:
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
        return 'Taxa\n\tid: {}\n\tidx: {}\n\tname: {}\n' \
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
        '''Whether or not the Taxa is defined at the specified taxonomic level

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


class OTU(Taxa):
    '''Aggregates of Taxa objects

    NOTE: For self consistency, let the class TaxaSet initialize this object.

    Parameters
    ----------
    anchor, other : mdsine2.Taxa, mdsine2.OTU
        These are the Taxas/Aggregates that you're joining together. The anchor is
        the one you are setting the sequeunce and taxonomy to
    '''
    def __init__(self, anchor, other):
        name = anchor.name + '_agg'
        Taxa.__init__(self, name=name, idx=anchor.idx, sequence=anchor.sequence)

        _agg_taxas = {}

        if isotu(anchor):
            if other.name in anchor.aggregated_taxas:
                raise ValueError('`other` ({}) already aggregated with anchor ' \
                    '({}) ({})'.format(other.name, anchor.name, anchor.aggregated_taxas))
            agg1 = anchor.aggregated_taxas
            agg1_seq = anchor.aggregated_seqs
            for k,v in anchor.aggregated_taxonomies.items():
                _agg_taxas[k] = v
        else:
            agg1 = [anchor.name]
            agg1_seq = {anchor.name: anchor.sequence}
            _agg_taxas[anchor.name] = anchor.taxonomy

        if isotu(other):
            if anchor.name in other.aggregated_taxas:
                raise ValueError('`anchor` ({}) already aggregated with other ' \
                    '({}) ({})'.format(anchor.name, other.name, other.aggregated_taxas))
            agg2 = other.aggregated_taxas
            agg2_seq = other.aggregated_seqs
            for k,v in other.aggregated_taxonomies.items():
                _agg_taxas[k] = v
        else:
            agg2 = [other.name]
            agg2_seq = {other.name: other.sequence}
            _agg_taxas[other.name] = other.taxonomy

        self.aggregated_taxas = agg1 + agg2
        self.aggregated_seqs = agg1_seq
        self.aggregated_taxonomies = _agg_taxas
        for k,v in agg2_seq.items():
            self.aggregated_seqs[k] = v

        self.taxonomy = anchor.taxonomy

    def __str__(self):
        return 'OTU\n\tid: {}\n\tidx: {}\n\tname: {}\n' \
            '\tAggregates: {}\n' \
            '\ttaxonomy:\n\t\tkingdom: {}\n\t\tphylum: {}\n' \
            '\t\tclass: {}\n\t\torder: {}\n\t\tfamily: {}\n' \
            '\t\tgenus: {}\n\t\tspecies: {}'.format(
            self.id, self.idx, self.name, self.aggregated_taxas,
            self.taxonomy['kingdom'], self.taxonomy['phylum'],
            self.taxonomy['class'], self.taxonomy['order'],
            self.taxonomy['family'], self.taxonomy['genus'],
            self.taxonomy['species'])

    def generate_consensus_seq(self, threshold=0.65, noconsensus_char='N'):
        '''Generate the consensus sequence for the OTU given the sequences
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
        '''
        if not plutil.isstr(noconsensus_char):
            raise TypeError('`noconsensus_char` ({}) must be a str'.format(
                type(noconsensus_char)))
        if not plutil.isnumeric(threshold):
            raise TypeError('`threshold` ({}) must be a numeric'.format(threshold))
        if threshold < 0 or threshold > 1:
            raise ValueError('`threshold` ({}) must be 0 <= thresold <= 1'.format(threshold))

        # Check if all of the sequences are the same length
        agg_seqs = [seq for seq in self.aggregated_seqs.values()]
        l = None
        for seq in agg_seqs:
            if l is None:
                l = len(seq)
            if len(seq) != l:
                raise NotImplementedError('Unaligned sequences not implemented yet')

        # Generate the consensus base for each base position
        consensus_seq = ''
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
                consensus_seq += list(found.keys())[0]
            else:
                # Get the maximum consensus
                consensus_percent = -1
                consensus_base = None
                for base in found:
                    consensus = 1 - (found[base]/len(agg_seqs))
                    if consensus > consensus_percent:
                        consensus_percent = consensus
                        consensus_base = base

                # Set the consensus base if it passes the threshold
                if consensus_percent >= threshold:
                    logging.debug('Consensus found for taxa {} in position {} as {}, found ' \
                        '{}'.format(self.name, i, consensus_base, found))
                    consensus_seq += consensus_base
                else:
                    logging.debug('No consensus for taxa {} in position {}. Consensus: {}' \
                        ', found {}'.format(self.name, i, consensus, found))
                    consensus_seq += noconsensus_char

        # Check for errors with consensus sequence
        for seq in agg_seqs:
            perc_dist = diversity.beta.hamming(seq, consensus_seq, 
                ignore_char=noconsensus_char)/l
            if perc_dist > 0.03:
                logging.warning('Taxa {} has a hamming distance > 3% ({}) to the generated ' \
                    'consensus sequence {} from individual sequence {}. Check that sequences ' \
                    'make sense'.format(self.name, perc_dist, consensus_seq, seq))

        # Set the consensus sequence as the OTU's sequence
        self.sequence = consensus_seq

    def generate_consensus_taxonomy(self, consensus_table=None):
        '''Set the taxonomy of the OTU to the consensus taxonomy of the.

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
        Input:
         kingdom          phylum                class        order             family  genus       species      asv
        Bacteria  Proteobacteria  Alphaproteobacteria  Rhizobiales  Bradyrhizobiaceae  Bosea  massiliensis  ASV_722
        Bacteria  Proteobacteria  Alphaproteobacteria  Rhizobiales  Bradyrhizobiaceae  Bosea            NA  ASV_991
        Output:
         kingdom          phylum                class        order             family  genus       species
        Bacteria  Proteobacteria  Alphaproteobacteria  Rhizobiales  Bradyrhizobiaceae  Bosea  massiliensis

        Input:
         kingdom          phylum           class              order              family            genus                 species      asv
        Bacteria  Actinobacteria  Actinobacteria  Bifidobacteriales  Bifidobacteriaceae  Bifidobacterium                      NA  ASV_283
        Bacteria  Actinobacteria  Actinobacteria  Bifidobacteriales  Bifidobacteriaceae  Bifidobacterium                      NA  ASV_302
        Bacteria  Actinobacteria  Actinobacteria  Bifidobacteriales  Bifidobacteriaceae  Bifidobacterium    adolescentis/faecale  ASV_340
        Bacteria  Actinobacteria  Actinobacteria  Bifidobacteriales  Bifidobacteriaceae  Bifidobacterium  choerinum/pseudolongum  ASV_668
        Ouput:
         kingdom          phylum           class              order              family            genus                                      species
        Bacteria  Actinobacteria  Actinobacteria  Bifidobacteriales  Bifidobacteriaceae  Bifidobacterium  adolescentis/faecale/choerinum/pseudolongum

        Input:
         kingdom         phylum class order family genus species       asv
        Bacteria  Bacteroidetes    NA    NA     NA    NA      NA   ASV_232
        Bacteria     Firmicutes    NA    NA     NA    NA      NA   ASV_762
        Output:

        '''
        # Check that all the taxonomies have the same lineage
        set_to_na = False
        set_from_table = False
        for tax in TAX_LEVELS:
            if set_to_na:
                self.taxonomy[tax] = DEFAULT_TAXA_NAME
                continue
            if set_from_table:
                if tax not in consensus_table.columns:
                    self.taxonomy[tax] = DEFAULT_TAXA_NAME
                else:
                    self.taxonomy[tax] = consensus_table[tax][self.name]
                continue
            if tax == 'asv':
                continue
            consensus = []
            for taxaname in self.aggregated_taxas:
                if tax == 'species':
                    aaa = self.aggregated_taxonomies[taxaname][tax].split('/')
                else:
                    aaa = [self.aggregated_taxonomies[taxaname][tax]]
                for bbb in aaa:
                    if bbb in consensus:
                        continue
                    else:
                        consensus.append(bbb)
            if DEFAULT_TAXA_NAME in consensus:
                consensus.remove(DEFAULT_TAXA_NAME)

            if len(consensus) == 0:
                # No taxonomy found at this level
                self.taxonomy[tax] = DEFAULT_TAXA_NAME
            elif len(consensus) == 1:
                # All taxonomies agree
                self.taxonomy[tax] = consensus[0]
            else:
                # All taxonomies do not agree
                if tax == 'species':
                    # Take the union of the species
                    self.taxonomy[tax] = '/'.join(consensus)
                else:
                    # This means that the taxonomy is different on a level different than
                    logging.critical('{} taxonomy does not agree'.format(self.name))
                    logging.critical(str(self))
                    for taxaname in self.aggregated_taxonomies:
                        logging.warning('{}'.format(list(self.aggregated_taxonomies[taxaname].values())))

                    if consensus_table is not None:
                        # Set from the table
                        self.taxonomy[tax] = consensus_table[tax][self.name]
                        set_from_table = True

                    else:
                        # Set this taxa level and everything below it to NA
                        self.taxonomy[tax] = DEFAULT_TAXA_NAME
                        set_to_na = True


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


class TaxaSet(Clusterable):
    '''Wraps a set of `` objects. You can get the  object via the
     id,  name.
    Provides functionality for aggregating sequeunces and getting subsets for lineages.

    Aggregating/Deaggregating
    -------------------------
    s that are aggregated together to become OTUs are used because sequences are 
    very close together. This class provides functionality for aggregating Taxas together
    (`mdsine2.TaxaSet.aggregate_items`) and to deaggregate a specific name from an aggregation
    (`mdsine2.TaxaSet.deaggregate_item`). If this object is within a `mdsine2.Study` object,
    MAKE SURE TO CALL THE AGGREGATION FUNCTIONS FROM THE `mdsine2.Study` OBJECT 
    (`mdsine2.Study.aggregate_items`, `mdsine2.Study.deaggregate_item`) so that the reads
    for the agglomerates and individual Taxas can be consistent with the TaxaSet.

    Parameters
    ----------
    taxonomy_table : pandas.DataFrame
        This is the table defining the set. If this is specified, then it is passed into
        TaxaSet.parse

    See also
    --------
    mdsine2.TaxaSet.parse
    '''

    def __init__(self, taxonomy_table=None):
        self.taxonomy_table = taxonomy_table
        self.ids = CustomOrderedDict()
        self.names = CustomOrderedDict()
        self.index = []
        self._len = 0

        # Add all of the Taxas from the dataframe if necessary
        if taxonomy_table is not None:
            self.parse(taxonomy_table=taxonomy_table)

    def __contains__(self,key):
        try:
            self[key]
            return True
        except:
            return False

    def __getitem__(self,key):
        '''Get an Taxa by either its sequence, name, index, or id

        Parameters
        ----------
        key : str, int
            Key to reference the Taxa
        '''
        if istaxatype(key):
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

    def __iter__(self):
        '''Returns each Taxa obejct in order
        '''
        for taxa in self.index:
            yield taxa

    def __len__(self):
        '''Return the number of Taxas in the TaxaSet
        '''
        return self._len

    @property
    def n_taxas(self):
        '''Alias for __len__
        '''
        return self._len

    def parse(self, taxonomy_table):
        '''Parse a taxonomy table

        `taxonomy_table`
        ----------------
        This is a dataframe that contains the taxonomic information for each Taxa.
        The columns that must be included are:
            'name' : name of the taxa
            'sequence' : sequence of the taxa
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
            DataFrame conttaxaining the required information (Taxonomy, sequence).
            If nothing is passed in, it will be an empty TaxaSet
        '''
        logging.info('TaxaSet parsng new taxonomy table. Resetting')
        self.taxonomy_table = taxonomy_table
        self.ids = CustomOrderedDict()
        self.names = CustomOrderedDict()
        self.index = []
        self._len = 0

        self.taxonomy_table = taxonomy_table
        taxonomy_table = taxonomy_table.rename(str.lower, axis='columns')
        if 'name' not in taxonomy_table.columns:
            logging.info('No `name` found - assuming index is the name')
        else:
            taxonomy_table = taxonomy_table.set_index('name')
        if SEQUENCE_COLUMN_LABEL not in taxonomy_table.columns:
            raise ValueError('`"{}"` ({}) not found as a column in `taxonomy_table`'.format(
                SEQUENCE_COLUMN_LABEL, taxonomy_table.columns))

        for tax in TAX_LEVELS[:-1]:
            if tax not in taxonomy_table.columns:
                logging.info('Adding in `{}` column'.format(tax))
                taxonomy_table = taxonomy_table.insert(-1, tax, 
                    [DEFAULT_TAXA_NAME for _ in range(len(taxonomy_table.index))])

        for i, name in enumerate(taxonomy_table.index):
            seq = taxonomy_table[SEQUENCE_COLUMN_LABEL][name]
            taxa = Taxa(name=name, sequence=seq, idx=self._len)
            taxa.set_taxonomy(
                tax_kingdom=taxonomy_table.loc[name]['kingdom'],
                tax_phylum=taxonomy_table.loc[name]['phylum'],
                tax_class=taxonomy_table.loc[name]['class'],
                tax_order=taxonomy_table.loc[name]['order'],
                tax_family=taxonomy_table.loc[name]['family'],
                tax_genus=taxonomy_table.loc[name]['genus'],
                tax_species=taxonomy_table.loc[name]['species'])

            self.ids[taxa.id] = taxa
            self.names[taxa.name] = taxa
            self.index.append(taxa)  
            self._len += 1

        self.ids.update_order()
        self.names.update_order()

    def add_taxa(self, name, sequence=None):
        '''Adds an Taxa to the set

        Parameters
        ----------
        name : str
            This is the name of the Taxa
        sequence : str
            This is the sequence of the Taxa
        '''
        taxa = Taxa(name=name, sequence=sequence, idx=self._len)
        self.ids[taxa.id] = taxa
        self.names[taxa.name] = taxa
        self.index.append(taxa)

        # update the order of the Taxas
        self.ids.update_order()
        self.names.update_order()
        self._len += 1

        return self

    def del_taxa(self, taxa):
        '''Deletes the Taxa from the set.

        Parameters
        ----------
        taxa : str, int, Taxa
            Can either be the name, sequence, or the ID of the Taxa
        '''
        # Get the ID
        taxa = self[taxa]
        oidx = self.ids.index[taxa.id]

        # Delete the Taxa from everything
        # taxa = self[taxa]
        self.ids.pop(taxa.id, None)
        self.names.pop(taxa.name, None)
        self.index.pop(oidx)

        # update the order of the Taxas
        self.ids.update_order()
        self.names.update_order()

        # Update the indices of the taxas
        # Since everything points to the same object we only need to do it once
        for aidx, taxa in enumerate(self.index):
            taxa.idx = aidx

        self._len -= 1
        return self

    def taxonomic_similarity(self,oid1,oid2):
        '''Calculate the taxonomic similarity between Taxa1 and Taxa2
        Iterates through most broad to least broad taxonomic level and
        returns the fraction that are the same.

        Example:
            taxa1.taxonomy = (A,B,C,D)
            taxa2.taxonomy = (A,B,E,F)
            similarity = 0.5

            taxa1.taxonomy = (A,B,C,D)
            taxa2.taxonomy = (A,B,C,F)
            similarity = 0.75

            taxa1.taxonomy = (A,B,C,D)
            taxa2.taxonomy = (A,B,C,D)
            similarity = 1.0

            taxa1.taxonomy = (X,Y,Z,M)
            taxa2.taxonomy = (A,B,E,F)
            similarity = 0.0

        Parameters
        ----------
        oid1, oid2 : str, int
            The name, id, or sequence for the Taxa
        '''
        if oid1 == oid2:
            return 1
        taxa1 = self[oid1].get_lineage()
        taxa2 = self[oid2].get_lineage()
        i = 0
        for a in taxa1:
            if a == taxa2[i]:
                i += 1
            else:
                break
        return i/8 # including Taxa

    def aggregate_items(self, anchor, other):
        '''Create an OTU with the anchor `anchor` and other taxa  `other`.
        The aggregate takes the sequence and the taxonomy from the anchor.

        Parameters
        ----------
        anchor, other : str, int, mdsine2.Taxa, mdsine2.OTU
            These are the Taxas/Aggregates that you're joining together. The anchor is
            the one you are setting the sequeunce and taxonomy to

        Returns
        -------
        mdsine2.OTU
            This is the new aggregated Taxa containing anchor and other
        '''
        anchor = self[anchor]
        other = self[other]
        
        agg = OTU(anchor=anchor, other=other)

        self.index[agg.idx] = agg
        self.index.pop(other.idx)

        self.ids = CustomOrderedDict()
        self.names = CustomOrderedDict()

        for idx, taxa in enumerate(self.index):
            taxa.idx = idx
            self.ids[taxa.id] = taxa
            self.names[taxa.name] = taxa
        
        # update the order of the Taxas
        self.ids.update_order()
        self.names.update_order()

        self._len = len(self.index)
        return agg

    def deaggregate_item(self, agg, other):
        '''Deaggregate the sequence `other` from OTU `agg`.
        `other` is then appended to the end 

        Parameters
        ----------
        agg : OTU, str
            This is an OTU with multiple sequences contained. Must 
            have the name `other` in there
        other : str
            This is the name of the Taxa that should be taken out of `agg`

        Returns
        -------
        mdsine2.Taxa
            This is the deaggregated Taxa
        '''
        agg = self[agg]
        if not isotu(agg):
            raise TypeError('`agg` ({}) must be an AggregatedTaxa'.format(type(agg)))
        if not plutil.isstr(other):
            raise TypeError('`other` ({}) must be a str'.format(type(other)))
        if other not in agg.aggregated_taxas:
            raise ValueError('`other` ({}) is not contained in `agg` ({}) ({})'.format(
                other, agg.name, agg.aggregated_taxas))

        other = Taxa(name=other, sequence=agg.aggregated_seqs[other], idx=self._len)
        other.taxonomy = agg.aggregated_taxonomies[other.name]
        agg.aggregated_seqs.pop(other.name, None)
        agg.aggregated_taxas.remove(other.name)
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
        [Taxa_22, Taxa_9982, TUDD_8484]

        Calling taxas.rename(prefix='OTU')
        New names:
        [OTU_1, OTU_2, OTU_3]

        Calling taxas.rename(prefix='OTU', zero_based_index=True)
        New names:
        [OTU_0, OTU_1, OTU_2]

        Parameters
        ----------
        prefix : str
            This is the prefix of the new Taxas. The name of the Taxas will change
            to `'{}_{}'.format(prefix, index)`
        zero_based_index : bool
            If this is False, then we start the enumeration of the Taxas from 1
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
        for taxa in self.index:
            newname = prefix + '_{}'.format(int(taxa.idx + offset))
            taxa.name = newname
            self.names[taxa.name] = taxa

    def generate_consensus_seqs(self, threshold=0.65, noconsensus_char='N'):
        '''Generate the consensus sequence for all of the Taxa given the sequences
        of all the contained ASVs of the respective OTUs

        Parameters
        ----------
        threshold : float
            This is the threshold for consensus (0 < threshold <= 1)
        noconsensus_char : str
            This is the character to replace
        '''
        for taxa in self:
            if isotu(taxa):
                taxa.generate_consensus_seq(
                    threshold=threshold, 
                    noconsensus_char=noconsensus_char)

    def generate_consensus_taxonomies(self, consensus_table=None):
        '''Generates the consensus taxonomies for all of the OTUs within the TaxaSet.
        For details on the algorithm - see `OTU.generate_consensus_taxonomy`

        See Also
        --------
        mdsine2.pylab.base.OTU.generate_consensus_taxonomy
        '''
        for taxa in self:
            if isotu(taxa):
                taxa.generate_consensus_taxonomy(consensus_table=consensus_table)


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
        also redo the indexes if an Taxa was deleted
        '''
        self.order = np.array(list(self.keys()))
        self.index = {}
        for i, taxa in enumerate(self.order):
            self.index[taxa] = i


class Subject(Saveable):
    '''Data for a single subject
    The TaxaSet order is done with respect to the ordering in the `reads_table`
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
        self._reads_individ = {} # for taking out aggregated taxas

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
        reads : np.ndarray(NTAXAS, N_TIMEPOINTS)
            These are the reads for the Taxas in order. Assumed to be in the 
            same order as the TaxaSet. If it is a dataframe then we use the rows
            to index the Taxa names. If timepoints is an array, then we are adding 
            for multiple timepoints. In this case we assume that the rows index  the 
            Taxa and the columns index the timepoint.
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
        if reads.shape[0] != len(self.taxas) or reads.shape[1] != len(timepoints):
            raise ValueError('`reads` shape {} does not align with the number of taxas ({}) ' \
                'or timepoints ({})'.format(reads.shape, len(self.taxas), len(timepoints)))

        for tidx, timepoint in enumerate(timepoints):
            if timepoint in self.reads:
                if self.reads[timepoint] is not None:
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
                if self.qpcr[timepoint] is not None:
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
    def taxas(self):
        return self.parent.taxas

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

        shape = (len(self.taxas), len(self.times))
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
        index = self.taxas.names.order
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
        '''Clusters the Taxas into the taxonomic level indicated in `taxlevel`.

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
            How to make the index using `pylab.util.plutil.taxaname_formatter`. Note that you cannot
            specify anything at a lower taxonomic level than what youre clustering at. For 
            example, you cannot cluster at the 'class' level and then specify '%(genus)s' 
            in the index formatter.
            If nothing is specified then only return the specified taxonomic level

        Returns
        -------
        pandas.DataFrame
            Dataframe of the data
        dict (str->str)
            Maps Taxa name to the row it got allocated to
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
        taxaname_map = {}
        df = self.df()[dtype]
        cols = list(df.columns)
        cols.append(taxlevel)
        dfnew = pd.DataFrame(columns = cols).set_index(taxlevel)

        # Get the level in the taxonomy, create a new entry if it is not there already
        taxas = {} # lineage -> label
        for i, taxa in enumerate(self.taxas):
            row = df.index[i]
            tax = taxa.get_lineage(level=taxlevel)
            tax = tuple(tax)
            tax = str(tax).replace("'", '')
            if tax in taxas:
                dfnew.loc[taxas[tax]] += df.loc[row]
            else:
                if not taxa.tax_is_defined(taxlevel) and smart_unspec:
                    # Get the least common ancestor above the taxlevel
                    taxlevelidx = TAX_IDXS[taxlevel]
                    ttt = None
                    while taxlevelidx > -1:
                        if taxa.tax_is_defined(TAX_LEVELS[taxlevelidx]):
                            ttt = TAX_LEVELS[taxlevelidx]
                            break
                        taxlevelidx -= 1
                    if ttt is None:
                        raise ValueError('Could not find a single taxlevel: {}'.format(str(taxa)))
                    taxas[tax] = '{} {}, {} NA'.format(ttt.capitalize(), 
                        taxa.taxonomy[ttt], taxlevel.capitalize())
                else:
                    taxas[tax] = plutil.taxaname_formatter(format=index_formatter, taxa=taxa, taxas=self.taxas)
                toadd = pd.DataFrame(np.array(list(df.loc[row])).reshape(1,-1),
                    index=[taxas[tax]], columns=dfnew.columns)
                dfnew = dfnew.append(toadd)
            
            if taxas[tax] not in taxaname_map:
                taxaname_map[taxas[tax]] = []
            taxaname_map[taxas[tax]].append(taxa.name)
        
        return dfnew, taxaname_map

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
                if self.name not in pert:
                    continue
                start = pert.starts[self.name]
                end = pert.ends[self.name]
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
        '''Deaggregate the sequence `other` from OTU `agg`.
        `other` is then appended to the end. This is called from 
        `mdsine2.Study.deaggregate_item`.

        Parameters
        ----------
        agg : OTU
            This is an OTU with multiple sequences contained. Must 
            have the name `other` in there
        other : str
            This is the name of the Taxa that should be taken out of `agg`
        '''
        # Append the reads of the deaggregated at the bottom and subtract them
        # from the aggregated index
        if other not in self._reads_individ:
            raise ValueError('`other` ({}) reads not found in archive. This probably ' \
                'happened because you called `aggregate_items` from the TaxaSet object' \
                ' instead from this object. Study object not consistent. Failing.'.format(other))
        
        aggidx = agg.idx
        for t in self.times:
            try:
                new_reads = self._reads_individ[other][t]
            except:
                raise ValueError('Timepoint `{}` added into subject `{}` after ' \
                    'Taxa `{}` was removed. Study object is not consistent. You ' \
                    'cannot add in other timepoints after you aggregate taxas. Failing.'.format(
                        t, self.name, other))
            self.reads[t][aggidx] = self.reads[t][aggidx] - new_reads
            self.reads[t] = np.append(self.reads[t], new_reads)
        self._reads_individ.pop(other)
        return

    def _aggregate_items(self, anchor, other):
        '''Aggregate the taxa `other` into `anchor`. This is called from 
        `mdsine2.Study.aggregate_items`.

        Parameters
        ----------
        anchor, other : OTU, Taxa
            These are the s to combine
        '''
        # If one of them are Taxas, then record their individual reads
        # if we want to dissociate them later
        for taxa in [anchor, other]:
            if istaxatype(taxa):
                if taxa.name in self._reads_individ:
                    raise ValueError('Taxa is already in this dict. This should not happen.')
                aidx = taxa.idx
                self._reads_individ[taxa.name] = {}
                for t in self.times:
                    self._reads_individ[taxa.name][t] = self.reads[t][aidx]
        
        for t in self.times:
            self.reads[t][anchor.idx] += self.reads[t][other.idx]
            self.reads[t] = np.delete(self.reads[t], other.idx)
        return


class Study(Saveable):
    '''Holds data for all the subjects

    Paramters
    ---------
    taxas : TaxaSet, Optional
        Contains all of the s
    '''
    def __init__(self, taxas, name='unnamed-study'):
        self.name = name
        self.id = id(self)
        self._subjects = {}
        self.perturbations = None
        self.qpcr_normalization_factor = None
        if not istaxaset(taxas):
            raise ValueError('If `taxas` ({}) is specified, it must be an TaxaSet' \
                ' type'.format(type(taxas)))
        self.taxas = taxas

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

    def parse(self, metadata, reads=None, qpcr=None, perturbations=None):
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
            Contains the reads for each one of the samples and taxas
                index (str) : indexes the Taxa name
                columns (str) : indexes the sample ID
            If nothing is passed in, the reads are set to None
        qpcr : pandas.DataFrame, None
            Contains the qpcr measurements for each sample
                index (str) : indexes the sample ID
                columns (str) : Name is ignored. the values are set to the measurements
        perturbations : pandas.DataFrame, None
            Contains the times and subjects for each perturbation
            columns:
                'name' -> str : Name of the perturbation
                'start' -> float : This is the start time for the perturbation
                'end' -> float : This is the end time for the perturbation
                'subject' -> str : This is the subject name the perturbation is applied to
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
        if perturbations is not None:
            logging.warning('Reseting perturbations')
            self.perturbations = Perturbations()
            if not plutil.isdataframe(perturbations):
                raise TypeError('`metadata` ({}) must be a pandas.DataFrame'.format(type(metadata)))
            try:
                for pidx in perturbations.index:
                    pname = perturbations['name'][pidx]
                    subj = str(perturbations['subject'][pidx])

                    if pname not in self.perturbations:
                        # Create a new one
                        pert = BasePerturbation(
                            name=pname, 
                            starts={subj: perturbations['start'][pidx]},
                            ends={subj: perturbations['end'][pidx]})
                        self.perturbations.append(pert)
                    else:
                        # Add this subject name to the pertubration
                        self.perturbations[pname].starts[subj] = perturbations['start'][pidx]
                        self.perturbations[pname].ends[subj] = perturbations['end'][pidx]
            except KeyError as e:
                logging.critical(e)
                raise KeyError('Make sure that `subject`, `start`, and `end` are columns')

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
        return self
            
    def write_metadata_to_csv(self, path, sep='\t'):
        '''Write the internal metadata to a table

        Parameters
        ----------
        path : str
            This is the location to save the metadata file
        sep : str
            This is the separator of the table
        '''
        columns = ['sampleID', 'subject', 'time']
        data = []
        for sampleid in self._samples:
            sid, t = self._samples[sampleid]
            data.append([sampleid, sid, t])

        df = pd.DataFrame(data, columns=columns)
        df.to_csv(path, sep=sep, index=False, header=True)

    def write_reads_to_csv(self, path, sep='\t'):
        '''Write the reads to a table

        Parameters
        ----------
        path : str
            This is the location to save the reads file
        sep : str
            This is the separator of the table
        '''
        data = [[taxa.name for taxa in self.taxas]]
        index = ['name']
        for sampleid in self._samples:
            index.append(sampleid)
            sid, t = self._samples[sampleid]

            reads = self[sid].reads[t]
            data.append(reads)

        df = pd.DataFrame(data, index=index).T
        df.to_csv(path, sep=sep, index=False, header=True)

    def write_qpcr_to_csv(self, path, sep='\t'):
        '''Write the qPCR measurements to a table

        Parameters
        ----------
        path : str
            This is the location to save the qPCR file
        sep : str
            This is the separator of the table
        '''
        max_n_measurements = -1
        data = []
        for sampleid in self._samples:
            sid, t = self._samples[sampleid]
            subj = self[sid]
            ms = subj.qpcr[t].data
            if len(ms) > max_n_measurements:
                max_n_measurements = len(ms)
            ms = [sampleid] + ms.tolist()
            data.append(ms)
        
        for i, ms in enumerate(data):
            if len(ms)-1 < max_n_measurements:
                data[i] = np.append(
                    ms, 
                    np.nan * np.ones(max_n_measurements - len(ms)))
        
        columns = ['sampleid'] + ['measurement{}'.format(i+1) for i in range(max_n_measurements)]

        df = pd.DataFrame(data, columns=columns)
        df.to_csv(path, sep=sep, index=False, header=True)

    def write_perturbations_to_csv(self, path, sep='\t'):
        '''Write the perturbations to a table

        Parameters
        ----------
        path : str
            This is the location to save the perturbations file
        sep : str
            This is the separator of the table
        '''
        columns = ['name', 'start', 'end', 'subject']
        data = []
        for perturbation in self.perturbations:
            for subjname in perturbation.starts:
                data.append([
                    perturbation.name, 
                    perturbation.starts[subjname],
                    perturbation.ends[subjname],
                    subjname])

        df = pd.DataFrame(data, columns=columns)
        df.to_csv(path, sep=sep, index=False, header=True)

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

    def pop_subject(self, sid, name='unnamed-study'):
        '''Remove the indicated subject id

        Parameters
        ----------
        sid : list(str), str, int
            This is the subject name/s or the index/es to pop out.
            Return a new Study with the specified subjects removed.
        name : str
            Name of the new study to return
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
        ret = Study(taxas=self.taxas, name=name)
        ret.qpcr_normalization_factor = self.qpcr_normalization_factor

        for s in sids:
            if s in self._subjects:
                ret._subjects[s] =  self._subjects.pop(s, None)
            else:
                raise ValueError('`sid` ({}) not found'.format(sid))

        ret.perturbations = copy.deepcopy(self.perturbations)

        # Remove the names of the subjects in the perturbations
        for study in [ret, self]:
            for perturbation in study.perturbations:
                names = list(perturbation.starts.keys())
                for subjname in names:
                    if subjname not in study:
                        perturbation.starts.pop(subjname, None)
                names = list(perturbation.ends.keys())
                for subjname in names:
                    if subjname not in study:
                        perturbation.ends.pop(subjname, None)

        return ret

    def pop_taxas_like(self, study):
        '''Remove s in the TaxaSet so that it matches the TaxaSet in `study`

        Parameters
        ----------
        study : mdsine2.study
            This is the study object we are mirroring in terms of Taxas
        '''
        to_delete = []
        for taxa in self.taxas:
            if taxa.name not in study.taxas:
                to_delete.append(taxa.name)
        self.pop_taxas(to_delete)

    def pop_taxas(self, oids):
        '''Delete the Taxas indicated in oidxs. Updates the reads table and
        the internal TaxaSet

        Parameters
        ----------
        oids : str, int, list(str/int)
            These are the identifiers for each of the Taxa/s to delete
        '''
        # get indices
        oidxs = []
        for oid in oids:
            oidxs.append(self.taxas[oid].idx)
        
        # Delete the s from taxaset
        for oid in oids:
            self.taxas.del_taxa(oid)

        # Delete the reads
        for subj in self:
            for t in subj.reads:
                subj.reads[t] = np.delete(subj.reads[t], oidxs)
        return self

    def deaggregate_item(self, agg, other):
        '''Deaggregate the sequence `other` from OTU `agg`.
        `other` is then appended to the end 

        Parameters
        ----------
        agg : Aggregate, str
            This is an OTU with multiple sequences contained. Must 
            have the name `other` in there
        other : str
            This is the name of the Taxa that should be taken out of `agg`

        Returns
        -------
        mdsine2.Taxa
            This is the deaggregated Taxa
        '''
        agg = self.taxas[agg]
        if not isotu(agg):
            raise TypeError('`agg` ({}) must be an AggregatedTaxa'.format(type(agg)))
        if not plutil.isstr(other):
            raise TypeError('`other` ({}) must be a str'.format(type(other)))
        if other not in agg.aggregated_taxas:
            raise ValueError('`other` ({}) is not contained in `agg` ({}) ({})'.format(
                other, agg.name, agg.aggregated_taxas))

        for subj in self:
            subj._deaggregate_item(agg=agg, other=other)
        return self.taxas.deaggregate_item(agg, other)

    def aggregate_items_like(self, study, prefix=None):
        '''Aggregate s like they are in study `study`

        Parameters
        ----------
        study : mdsine2.Study
            Data object we are mirroring
        prefix : str
            If provided, this is how you rename the Taxas after aggregation
        '''
        for taxa in study.taxas:
            if isotu(taxa):
                aname = taxa.aggregated_taxas[0]
                for bname in taxa.aggregated_taxas[1:]:
                    self.aggregate_items(aname, bname)
        if prefix is not None:
            self.taxas.rename(prefix=prefix)

    def aggregate_items(self, taxa1, taxa2):
        '''Aggregates the abundances of `taxa1` and `taxa2`. Updates the reads table and
        internal TaxaSet

        Parameters
        ----------
        taxa1, taxa2 : str, int, mdsine2.Taxa, mdsine2.OTU
            These are the Taxas you are agglomerating together

        Returns
        -------
        mdsine2.OTU
            This is the new aggregated Taxa containing anchor and other
        '''
        # Find the anchor - use the highest index
        aidx1 = self.taxas[taxa1].idx
        aidx2 = self.taxas[taxa2].idx

        if aidx1 == aidx2:
            raise ValueError('Cannot aggregate the same taxa: {}'.format(self.taxas[taxa1]))
        elif aidx1 < aidx2:
            anchor = self.taxas[taxa1]
            other = self.taxas[taxa2]
        else:
            anchor = self.taxas[taxa2]
            other = self.taxas[taxa1]

        for subj in self:
            subj._aggregate_items(anchor=anchor, other=other)
        return self.taxas.aggregate_items(anchor=anchor, other=other)

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

    def add_perturbation(self, a, ends=None, name=None, subjs=None):
        '''Add a perturbation. 
        
        We can either do this by passing a perturbation object 
        (if we do this then we do not need to specify `end`) or we can 
        specify the start and stop times (if we do this them we need to
        specify `end`).

        Parameters
        ----------
        a : dict, BasePerturbation
            If this is a dict, then this corresponds to the start
            times of the perturbation for each subject. If this is a Pertubration object
            then we just add this.
        ends : dict
            Only necessary if `a` is a dict
        name : str, None
            Only necessary if `a` is a dict. Name of the perturbation
        
        Returns
        -------
        self
        '''
        if self.perturbations is None:
            self.perturbations = Perturbations()
        if plutil.isdict(a):
            if not plutil.isdict(end):
                raise ValueError('If `a` is a dict, then `end` ({}) ' \
                    'needs to be a dict'.format(type(end)))
            if not plutil.isstr(name):
                raise ValueError('`name` ({}) must be defined as a str'.format(type(name)))
            self.perturbations.append(BasePerturbation(starts=a, ends=end, name=name))
        elif isperturbation(a):
            self.perturbations.append(a)
        else:
            raise ValueError('`a` ({}) must be a subclass of ' \
                'pl.base.BasePerturbation or a dict'.format(type(a)))
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

    def times(self, agg):
        '''Aggregate the times of all the contained subjects

        These are the types of time aggregations:
            'union': Take  theunion of the times of the subjects
            'intersection': Take the intersection of the times of the subjects
        You can manually specify the times to include with a list of times. If times are not
        included in any of the subjects then we set them to NAN.

        Parameters
        ----------
        agg : str
            Type of aggregation to do of the times. Options: 'union', 'intersection'
        '''
        if agg not in ['union', 'intersection']:
            raise ValueError('`agg` ({}) not recognized'.format(agg))

        all_times = []
        for subj in self:
            all_times = np.append(all_times, subj.times)
        all_times = np.sort(np.unique(all_times))
        if agg == 'union':
            times = all_times

        elif agg == 'intersection':
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
        return times

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
            times = self.times(agg=times)
        elif plutil.isarray(times):
            times = np.array(times)
        else:
            raise TypeError('`times` type ({}) not recognized'.format(type(times)))

        shape = (len(self.taxas), len(times))
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
                temp = np.zeros(len(self.taxas)) * np.nan
            else:
                temp = np.hstack(temp)
                temp = aggfunc(temp, axis=1)
            M[:, tidx] = temp

        return M, times

    def matrix(self, dtype, agg, times):
        '''Make a matrix of the aggregation of all the subjects in the subjectset

        Aggregation of subjects
        -----------------------
        What are the values for the Taxas? Set the aggregation type using the parameter `agg`. 
        These are the types of aggregations:
            'mean': Mean abundance of the Taxa at a timepoint over all the subjects
            'median': Median abundance of the Taxa at a timepoint over all the subjects
            'sum': Sum of all the abundances of the Taxa at a timepoint over all the subjects
            'max': Maximum abundance of the Taxa at a timepoint over all the subjects
            'min': Minimum abundance of the Taxa at a timepoint over all the subjects

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
        np.ndarray(n_taxas, n_times)
        '''
        M, _ =  self._matrix(dtype=dtype, agg=agg, times=times)
        return M

    def df(self, *args, **kwargs):
        '''Returns a dataframe of the data in matrix. Rows are Taxas, columns are times.

        Returns
        -------
        pandas.DataFrame

        See Also
        --------
        mdsine2.Study.matrix
        '''
        M, times = self._matrix(*args, **kwargs)
        index = [taxa.name for taxa in self.taxas]
        return pd.DataFrame(data=M, index=index, columns=times)

