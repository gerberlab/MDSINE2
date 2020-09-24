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

                   |-->|-->|-->
perturbation on    #############
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

from .util import isint, isnumeric, isarray, isstr, isbool, asvname_formatter, istree
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

def issubjectset(x):
    '''Checks whether the input is a subclass of SubjectSet

    Parameters
    ----------
    x : any
        Input instance to check the type of SubjectSet
    
    Returns
    -------
    bool
        True if `x` is of type SubjectSet, else False
    '''
    return x is not None and issubclass(x.__class__, SubjectSet)

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
    passed through `pylab.util.asvname_formatter` to get the label.

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
    if not isstr(fmt):
        raise TypeError('`fmt` ({}) must be a str'.format(type(fmt)))

    if type(M) == pd.DataFrame:
        for idx in M.index:
            if idx not in asvs:
                raise ValueError('row `{}` not found in asvs'.format(idx))
        names = M.index
    elif isarray(M):
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
        tax = asvname_formatter(format=fmt, asv=asv, asvs=asvs)
        if tax not in d:
            d[tax] = []
        d[tax].append(row)

    # Add all of the rows for each taxa
    Ms = ()
    index = []
    columns = None
    if not isarray(M):
        columns = M.columns
    for taxname, rows, in d.items():
        index.append(taxname)
        if isarray(M):
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
            with open(filename, 'wb') as output:  # Overwrites any existing file.
                pickle.dump(self, output, protocol=pickle.HIGHEST_PROTOCOL)
        except:
            os.system('rm {}'.format(filename))
            with open(filename, 'wb') as output:  # Overwrites any existing file.
                pickle.dump(self, output, protocol=pickle.HIGHEST_PROTOCOL)


    @classmethod
    def load(cls, filename):
        '''Unpickle the object

        Paramters
        ---------
        filename : str
            This is the location of the file to unpickle
        '''
        with open(filename, 'rb') as handle:
            b = pickle.load(handle)
        return b

    def set_save_location(self, filename):
        '''Set the save location for the object
        '''
        if not isstr(filename):
            raise TypeError('`filename` ({}) must be a str'.format(type(filename)))
        self._save_loc = filename


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

    def add_init_value(self):
        '''Saves the initialization value
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
        if not isnumeric(start):
            raise TypeError('`start` ({}) must be a numeric'.format(type(start)))
        if not isnumeric(end):
            raise TypeError('`end` ({}) must be a numeric'.format(type(end)))
        if end < start:
            raise ValueError('`end` ({}) must be >= `start` ({})'.format(end, start))
        if name is not None:
            if not isstr(name):
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
        if sequence is not None:
            self._sequence_as_array = np.array(list(sequence))
        else:
            self._sequence_as_array = None
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
        if tax_kingdom is not None:
            self.taxonomy['kingdom'] = tax_kingdom
        if tax_phylum is not None:
            self.taxonomy['phylum'] = tax_phylum
        if tax_class is not None:
            self.taxonomy['class'] = tax_class
        if tax_order is not None:
            self.taxonomy['order'] = tax_order
        if tax_family is not None:
            self.taxonomy['family'] = tax_family
        if tax_genus is not None:
            self.taxonomy['genus'] = tax_genus
        if tax_species is not None:
            self.taxonomy['species'] = tax_species

        return self

    def get_lineage(self, level=None, lca=False):
        '''Returns a tuple of the lineage in order from Kingdom to the level
        indicated. Default value for level is `asv`. If `lca` is True, then
        we return the lineage up to `level` where it is specified (no nans)

        Parameters
        ----------
        level : str, Optional
            The taxonomic level you want the lineage until
            If nothing is provided, it returns the entire taxonomic lineage
            Example:
                level = 'class'
                returns a tuple of (kingdom, phylum, class)
        lca : bool
            Least common ancestor
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

        if lca:
            i = len(a)-1
            while (type(a[i]) == float) or (a[i] == DEFAULT_TAXA_NAME):
                i -= 1
            a = a[:i+1]
        return a
    
    def get_taxonomy(self, level, lca=False):
        '''Get the taxonomy at the level specified

        Parameters
        ----------
        level : str
            This is the level to get
            Valid responses: 'kingdom', 'phylum', 'class', 'order', 'family', 'genus'
        lca : bool
            If True and the specified tax level is not specified, then supstitute it with
            the next highest taxonomy that's 
        '''
        return self.get_lineage(level=level, lca=lca)[-1]

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
        return (type(tax) != float) and (tax != DEFAULT_TAXA_NAME)


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
    ASV id, ASV name, or ASV sequence.
    Provides functionality for aggregating and getting subsets for lineages.

    Phylogenetic tree
    -----------------
    You can set the phylogenetic tree by calling `set_phylogenetic_tree`. 
    That package that we use the for phylogenetic tree is `ete3` 
    (https://github.com/etetoolkit/ete). You can pass in a file of the 
    saved newick tree or a `ete` tree object. Any object that is set as
    the phylogenetic tree will be immediately pruned to be only the asvs
    in the ASVSet using either the name or id, which you can set using the
    `identifier` keyword in `set_phylogenetic_tree`. If you delete an asv
    from the phylogenetic tree then we delete if from the tree as well.

    We need to cast the names as strings so that it is backwards compatible

    If you add OTUs to the set and there is a phylogenetic tree is there, 
    # then it will delete the phylogenetic Tree. NOTE: Is this necessary?
    '''

    def __init__(self, df=None, use_sequences=True):
        '''Load data from a dataframe

        Assumes the frame has the following columns:
            - sequence
            - name
            - taxonomy
                * kingdom, phylum, class, order, family, genus, species, asv

        Parameters
        ----------
        df - pandas.DataFrame, Optional
            DataFrame containing the required information (Taxonomy, sequence).
            If nothing is passed in, it will be an empty set.
        use_sequences : bool
            If True, Each ASV must have an associated sequence. Else 
        '''
        if not isbool(use_sequences):
            raise TypeError('`use_sequences` ({}) must be a bool'.format(
                type(use_sequences)))
        self.use_sequences = use_sequences

        self.ids = CustomOrderedDict()
        self.names = CustomOrderedDict()
        if self.use_sequences:
            self.seqs = CustomOrderedDict()
        else:
            self.seqs = None
        self.index = []
        self._len = 0
        self._phylogenetic_tree = None

        # Add all of the ASVs from the dataframe if necessary
        if df is not None:
            df = df.rename(str.lower, axis='columns')
            for name in df.index:
                if self.use_sequences and SEQUENCE_COLUMN_LABEL in df:
                    seq = df[SEQUENCE_COLUMN_LABEL][name]
                else:
                    seq = None
                self.add_asv(
                    name=name,
                    sequence=seq)
                self.names[name].set_taxonomy(
                    tax_kingdom=df.loc[name]['kingdom'],
                    tax_phylum=df.loc[name]['phylum'],
                    tax_class=df.loc[name]['class'],
                    tax_order=df.loc[name]['order'],
                    tax_family=df.loc[name]['family'],
                    tax_genus=df.loc[name]['genus'],
                    tax_species=df.loc[name]['species'])

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
        if isasv(key):
            return key
        if key in self.ids:
            return self.ids[key]
        elif isint(key):
            return self.index[key]
        elif key in self.names:
            return self.names[key]
        elif isasv(key):
            return key
        elif self.use_sequences:
            if key in self.seqs:
                return self.seqs[key]
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

    @property
    def phylogenetic_tree(self):
        return self._phylogenetic_tree

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
        if self.use_sequences:
            self.seqs[asv.sequence] = asv
        self.names[asv.name] = asv
        self.index.append(asv)

        # update the order of the ASVs
        self.ids.update_order()
        if self.use_sequences:
            self.seqs.update_order()
        self.names.update_order()
        self._len += 1

        self._phylogenetic_tree = None

        return self

    def del_asv(self, asv, preserve_branch_length=False):
        '''Deletes the ASV from the set. If there is a phylogenetic
        tree

        Parameters
        ----------
        asv : str, int, ASV
            Can either be the name, sequence, or the ID of the ASV
        preserve_branch_length : bool
            Only used if a phylogenetic tree is set
            Read the documentation for the function `ete3.Tree.prune`.
            Defualt is set to the default of the function.
        '''
        # Get the ID
        asv = self[asv]
        oidx = self.ids.index[asv.id]

        # Delete the ASV from everything
        # asv = self[asv]
        self.ids.pop(asv.id, None)
        if self.use_sequences:
            self.seqs.pop(asv.sequence, None)
        self.names.pop(asv.name, None)
        self.index.pop(oidx)

        # update the order of the ASVs
        self.ids.update_order()
        if self.use_sequences:
            self.seqs.update_order()
        self.names.update_order()

        # Update the indices of the asvs
        # Since everything points to the same object we only need to do it once
        for idx,asv in enumerate(self.ids.values()):
            asv.idx = idx

        self._len -= 1

        # Update the phylogenetic tree if necessary
        if self._phylogenetic_tree is not None:
            self._phylogenetic_tree.prune(self.names.order)

        return self

    def set_phylogenetic_tree(self, tree, identifier='name', preserve_branch_length=False):
        '''Set the phylogenetic tree for the ASV set

        Parameters
        ----------
        tree : str, ete3.Tree
            Location, newick specification of a tree, or a ete3 Tree 
            phylogenetic tree object
        identifier : str
            How the ASVs are specified in the tree.
            'name'
                Use the name to index
            'id'
                use the id of the asv
        preserve_branch_length : bool
            Read the documentation for the function `ete3.Tree.prune`.
            Defualt is set to the default of the function.
        '''
        if not isstr(identifier):
            raise TypeError('`identifier` ({}) must be a string'.format(type(identifier)))
        if identifier == 'id':
            raise NotImplementedError('This functionality is not yet implemented, mus tuse "name".')
        elif identifier != 'name':
            raise ValueError('`identifier` ({}) not recognized'.format(identifier))

        if isstr(tree):
            self._phylogenetic_tree = ete3.Tree(tree)
        elif istree(tree):
            self._phylogenetic_tree = tree
        else:
            raise TypeError('`tree` ({}) type not recognized'.format(type(tree)))
            
        logging.info('Tree accepted - pruning to only the asvs')
        self._phylogenetic_tree.prune(nodes=self.names.order, preserve_branch_length=preserve_branch_length)

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

    def phylogenetic_distance(self, a, b):
        '''Wrapper for ete3.Tree.get_distance

        You can pass in index, id, or name instead of just name

        Parameters
        ----------
        a,b : int, str  
            Identifier for ASV `a` and ASV `b` to calculate the distance between

        Returns
        -------
        float
        '''
        a = str(self[a].name)
        b = str(self[b].name)
        return self._phylogenetic_tree.get_distance(a,b)


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
        '''This will update the reverse dictionary
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
    parent : SubjectSet
        This is the parent class (we have a reverse pointer)
    name : str
        This is the name of the subject
    '''
    def __init__(self, parent, name):
        self.name = name
        self.id = id(self)
        self.parent = parent
        self.asvs = self.parent.asvs
        self.qpcr = {}
        self.reads = {}
        self.times = np.asarray([])

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
        if not isarray(timepoints):
            timepoints = [timepoints]
        for timepoint in timepoints:
            if not isnumeric(timepoint):
                raise TypeError('`timepoint` ({}) must be a numeric'.format(type(timepoint)))
        if not isarray(reads):
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
                logging.info('There are already reads specified at time `{}` for subject `{}`, overwriting'.format(
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
        if not isarray(timepoints):
            timepoints = [timepoints]
        for timepoint in timepoints:
            if not isnumeric(timepoint):
                raise TypeError('`timepoint` ({}) must be a numeric'.format(type(timepoint)))
        if masses is not None:
            if isnumeric(masses):
                masses = [masses]
            for mass in masses:
                if not isnumeric(mass):
                    raise TypeError('Each mass in `masses` ({}) must be a numeric'.format(type(mass)))
                if mass <= 0:
                    raise ValueError('Each mass in `masses` ({}) must be > 0'.format(mass))
            if len(masses) != len(timepoints):
                raise ValueError('Number of timepoints ({}) and number of masses ({}) ' \
                    'must be equal'.format(len(timepoints), len(masses)))
        if dilution_factors is not None:
            if isnumeric(dilution_factors):
                dilution_factors = [dilution_factors]
            for dilution_factor in dilution_factors:
                if not isnumeric(dilution_factor):
                    raise TypeError('Each dilution_factor in `dilution_factors` ({}) ' \
                        'must be a numeric'.format(type(dilution_factor)))
                if dilution_factor <= 0:
                    raise ValueError('Each dilution_factor in `dilution_factors` ({}) ' \
                        'must be > 0'.format(dilution_factor))
            if len(dilution_factors) != len(timepoints):
                raise ValueError('Number of timepoints ({}) and number of dilution_factors ({}) ' \
                    'must be equal'.format(len(timepoints), len(dilution_factors)))
            
        if not isarray(qpcr):
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
                logging.info('There are already qpcr measurements specified at time `{}` for subject `{}`, overwriting'.format(
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

    def set_from_tables(self, qpcr_table, reads_table):
        '''Set the qpcr and reads from pandas DataFrames

        Parameters
        ----------
        qpcr_table : pandas.DataFrame
            This is the qPCR table that holds all of the information for each of the samples
        reads_table : pandas.DataFrame
            This is the data for the reads table

        TODO add descritpion for how the table should be laid out
        '''
        raise NotImplementedError('This still needs to be tested/clarified if this is ' \
            'the format to have')
        qpcr = qpcr_table.values
        masses = qpcr[:,0].flatten()
        dilution_factor = qpcr[:,1].flatten()
        qpcr = qpcr[:,2:]
        times = np.asarray(qpcr_table.index)
        idxs = np.argsort(times)
        self.times = times[idxs]
        for idx in idxs:
            self.qpcr[times[idx]] = qPCRdata(
                cfus=qpcr[idx,:],
                mass=masses[idx],
                dilution_factor=dilution_factor[idx])

        # Reads - add in time ascending order (using times from qpcr)
        for t in self.times:
            self.reads[t] = np.asarray(reads_table[t])

    @property
    def perturbations(self):
        '''Returns the number of perturbation
        '''
        return self.parent.perturbations

    def matrix(self, min_rel_abund=None):
        '''Make a numpy matrix out of our data - returns the raw reads,
        the relative abundance, and the absolute abundance.

        If there is no qPCR data, then the absolute abundance is set to None.

        Parameters
        ----------
        min_rel_abund : float, int, Optional
            This is the minimum relative abundance to add to the 'rel' matrix.
            If nothing is specified then nothing gets added
        '''
        if np.issubdtype(type(min_rel_abund), np.bool_):
            if not min_rel_abund:
                min_rel_abund = None
            else:
                raise ValueError('Invalid `min_rel_abund` type ({})'.format(
                    type(min_rel_abund)))

        if min_rel_abund is not None:
            if type(min_rel_abund) != float or type(min_rel_abund) != int:
                raise ValueError('if `min_rel_abund` ({}) is specified, it must ' \
                    'be a float or an int'.format(type(min_rel_abund)))

        shape = (len(self.asvs), len(self.times))
        raw = np.zeros(shape=shape, dtype=int)
        rel = np.zeros(shape=shape, dtype=float)
        abs = np.zeros(shape=shape, dtype=float)

        if min_rel_abund is not None:
            rel += min_rel_abund

        for i,t in enumerate(self.times):
            raw[:,i] = self.reads[t]
            rel[:,i] = raw[:,i]/np.sum(raw[:,i])
        
        if len(self.qpcr) > 0:
            for i,t in enumerate(self.times):
                abs[:,i] = rel[:,i] * self.qpcr[t].mean()
        else:
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

    def cluster_by_taxlevel(self, dtype, lca, taxlevel, index_formatter=None, smart_unspec=True):
        '''Clusters the ASVs into the taxonomic level indicated in `taxlevel`.

        Smart Unspecified
        -----------------
        If True, returns the higher taxonomic classification while saying the desired taxonomic level
        is unspecified. Example: 'Order ABC, Family NA'. Note that this overrides the `index_formatter`.

        Parameters
        ----------
        subj : pylab.base.Subject
            This is the subject that we are getting the data from
        lca : bool
            If an ASV is unspecified at the taxonomic level and `lca` is True, then it will
            cluster at the higher taxonomic level
        taxlevel : str, None
            This is the taxa level to aggregate the data at. If it is 
            None then we do not do any collapsing (this is the same as 'asv')
        dtype : str
            This is the type of data to cluster. Options are:
                'raw': These are the counts
                'rel': This is the relative abundances
                'abs': This is the absolute abundance (qPCR * rel)
        index_formatter : str
            How to make the index using `pylab.util.asvname_formatter`. Note that you cannot
            specify anything at a lower taxonomic level than what youre clustering at. For 
            example, you cannot cluster at the 'class' level and then specify '%(genus)s' 
            in the index formatter.
            If nothing is specified then only return the specified taxonomic level
        '''
        # Type checking
        if not isstr(dtype):
            raise TypeError('`dtype` ({}) must be a str'.format(type(dtype)))
        if dtype not in ['raw', 'rel', 'abs']:
            raise ValueError('`dtype` ({}) not recognized'.format(dtype))
        if not isstr(taxlevel):
            raise TypeError('`taxlevel` ({}) must be a str'.format(type(taxlevel)))
        if taxlevel not in ['kingdom', 'phylum', 'class',  'order', 'family', 
            'genus', 'species', 'asv']:
            raise ValueError('`taxlevel` ({}) not recognized'.format(taxlevel))
        if index_formatter is None:
            index_formatter = taxlevel
        if index_formatter is not None:
            if not isstr(index_formatter):
                raise TypeError('`index_formatter` ({}) must be a str'.format(type(index_formatter)))
            
            for tx in TAX_IDXS:
                if tx in index_formatter and TAX_IDXS[tx] > TAX_IDXS[taxlevel]:
                    raise ValueError('You are clustering at the {} level but are specifying' \
                        ' {} in the `index_formatter`. This does not make sense. Either cluster' \
                        'at a lower tax level or specify the `index_formatter` to a higher tax ' \
                        'level'.format(taxlevel, tx))

        index_formatter = index_formatter.replace('%(asv)s', '%(name)s')

        # Everything is valid, get the data dataframe and the return dataframe
        df = self.df(min_rel_abund=None)[dtype]
        cols = list(df.columns)
        cols.append(taxlevel)
        dfnew = pd.DataFrame(columns = cols).set_index(taxlevel)

        # Get the level in the taxonomy, create a new entry if it is not there already
        taxas = {} # lineage -> label
        for i, asv in enumerate(self.asvs):
            row = df.index[i]
            tax = asv.get_lineage(level=taxlevel, lca=lca)
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
                    taxas[tax] = asvname_formatter(format=index_formatter, asv=asv, asvs=self.asvs, lca=lca)
                toadd = pd.DataFrame(np.array(list(df.loc[row])).reshape(1,-1),
                    index=[taxas[tax]], columns=dfnew.columns)
                dfnew = dfnew.append(toadd)
        
        return dfnew

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


class SubjectSet(Saveable):
    '''Holds data for all the subjects

    Paramters
    ---------
    sequences : pd.DataFrame, Optional
        index: ASV names
        column: sequences
        These are the sequences for each one of the ASVs
    taxonomy_table : pd.DataFrame, Optional
        index:
            - ASV names
            - Must be identical to the keys in `sequences` and the
              index in `reads_table`
        columns:
            - Must contain 'kingdom', 'phylum', 'class', 'order',
              'family', 'genus', 'species
        data:
            - These are the names
    asvs : ASVSet, Optional
        If you already have an ASVSetobject, you can just use that
    '''
    def __init__(self, sequences=None, taxonomy_table=None, asvs=None):
        self.id = id(self)
        self._subjects = {}
        self.perturbations = None
        self.qpcr_normalization_factor = None
        if asvs is not None:
            if not isasvset(asvs):
                raise ValueError('If `asvs` ({}) is specified, it must be an ASVSet' \
                    ' type'.format(type(asvs)))
            self.asvs = asvs
            return

        if type(sequences) != pd.DataFrame:
            raise ValueError('`sequences` ({}) must be a dict'.format(
                type(sequences)))
        if taxonomy_table is not None:
            if type(taxonomy_table) != pd.DataFrame:
                raise ValueError('`taxonomy_table` ({}) must be a pandas.DataFrame object'.format(
                    type(taxonomy_table)))

        valid_cols = ['sequences']
        seq_cols = list(sequences.columns)
        for col in seq_cols:
            if col not in valid_cols:
                raise ValueError('column `{}` not in valid columns: {}'.format(
                    col, valid_cols))
        for col in valid_cols:
            if col not in seq_cols:
                raise ValueError('column `{}` not in sequence columns: {}'.format(
                    col, seq_cols))

        seq_keys = sequences.index
        if taxonomy_table is not None:
            tax_index = taxonomy_table.index
            for key in seq_keys:
                if key not in tax_index:
                    raise ValueError("key '{}' not in taxonomy index ({})".format(
                        key, tax_index))
            for key in tax_index:
                if key not in seq_keys:
                    raise ValueError("key '{}' not in sequence keys ({})".format(
                        key, seq_keys))

        # Check the columns for taxonomy_table if necessary
        # Every column in taxonomy_table has to be in valid_cols
        # NOT every column in valid_cols has to be in taxonomy_table
        if taxonomy_table is not None:
            valid_cols = ['tax_kingdom', 'tax_phylum', 'tax_class', 'tax_order', 
                'tax_family', 'tax_genus', 'tax_species']
            tax_cols = list(taxonomy_table.columns)
            tax_cols = ['tax_{}'.format(str(col).lower()) for col in tax_cols]
            taxonomy_table.columns = tax_cols
            for col in tax_cols:
                if col not in valid_cols:
                    raise ValueError("col '{}' not in valid columns for taxonomy ({})".format(
                        col, valid_cols))

        # Everything is valid, make the asvset
        # ASVs
        self.asvs = ASVSet()
        seq_data = sequences.to_numpy().ravel()
        for oidx, asv_name in enumerate(sequences.index):
            self.asvs.add_asv(name=asv_name, sequence=seq_data[oidx])
            if taxonomy_table is not None:
                kwargs = {}
                for tax in tax_cols:
                    kwargs[tax] = taxonomy_table[tax][asv_name]
                self.asvs[asv_name].set_taxonomy(**kwargs)

    def __getitem__(self, key):
        return self._subjects[key]

    def __len__(self):
        return len(self._subjects)

    def __iter__(self):
        for v in self._subjects.values():
            yield v

    def __contains__(self, key):
        return key in self._subjects

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

    def add(self, name):
        '''Create a subject with the name `name`

        Parameters
        ----------
        name : str
            This is the name of the new subject
        '''
        if name not in self._subjects:
            self._subjects[name] = Subject(name=name, parent=self)
        return self

    def add_from_table(self, name, reads_table, qpcr_table):
        '''Adds a subject to the subject set

        Parameters
        ----------
        name : str
            This is the name of the subject
        qpcr_table, reads_table : pandas.DataFrame
            These specify the qpcr measurements
            index:
                - ['1.0','1.3', '2.2', ...] these are the time points in string
                  format
            columns:
                - ['mass', 'dilution factor', '1', '2', '3'] (for each of the 3 measurements)
            data:
                - these are the triplicate qpcr measurements
        reads_table : pandas.DataFrame
            index:
                - ASV names
                - Must be identical to the keys in `sequences` and the
                  index in `taxonomy_table`
            columns:
                - These are the time points in string format
                - These must be identical to the index in `qpcr_table`
            data:
                - Each one of the reads for each one of the ASVs
        '''
        # Type check
        if type(name) != str:
            raise ValueError('`name` ({}) must be a str'.format(type(name)))
        if type(qpcr_table) != pd.DataFrame:
            raise ValueError('`qpcr_table` ({}) must be a pandas.DataFrame object'.format(
                type(qpcr_table)))
        if type(reads_table) != pd.DataFrame:
            raise ValueError('`reads_table` ({}) must be a pandas.DataFrame object'.format(
                type(reads_table)))

        # Check ASV names are consistent
        read_index = reads_table.index
        for key in self.asvs.names:
            if key not in read_index:
                raise ValueError("key '{}' not in reads index ({})".format(
                    key, read_index))
        for key in read_index:
            if key not in self.asvs.names:
                raise ValueError("key '{}' not in sequence keys ({})".format(
                    key, self.asvs.names))


        # Check time labels are consistent
        # First convert both to strs of floats where necessary
        reads_table.columns = [float(t) for t in reads_table.columns]
        qpcr_table.index = [float(t) for t in qpcr_table.index]

        qpcr_times = qpcr_table.index
        reads_times = reads_table.columns
        for key in qpcr_times:
            if key not in reads_times:
                raise ValueError("key '{}' not in times for reads ({})".format(
                    key, reads_times))
        for key in reads_times:
            if key not in qpcr_times:
                raise ValueError("key '{}' not in times for qpcr ({})".format(
                    key, qpcr_table))

        # Check the columns for qpcr_table
        valid_cols = ['dilution factor', 'mass', '1', '2', '3']
        qpcr_cols = list(qpcr_table.columns)
        qpcr_cols = [str(col).lower() for col in qpcr_cols]
        for col in qpcr_cols:
            if col not in valid_cols:
                raise ValueError("col '{}' not in valid columns for qpcr ({})".format(
                    col, valid_cols))
        for col in valid_cols:
            if col not in qpcr_cols:
                raise ValueError("col '{}' not in qpcr columns ({})".format(
                    col, qpcr_cols))

        # Check ordering of reads index is consistent with the asvset order
        if len(reads_table.index) != len(self.asvs):
            raise ValueError('length of reads ({}) does not equal the right number ' \
                'of ASVs ({})'.format(len(reads_table.index), len(self.asvs)))
        for i,name in enumerate(self.asvs.names.order):
            if name != reads_table.index[i]:
                raise ValueError('The `{}`th row of the reads ({}) does not ' \
                    'correspond to the `{}`th ASV in ASVSet ({})'.format(
                        i,reads_table.index[i],i,name))
        self.add(name=name)
        self._subjects[name].set_from_tables(reads_table=reads_table,
            qpcr_table=qpcr_table)
        return self

    def pop_subject(self, sid):
        '''Remove the indicated subject id

        Parameters
        ----------
        sid : list(str), str, int
            This is the subject name/s or the index/es to pop out.
            Return a new SubjectSet with the specified subjects removed.
        '''
        if not isarray(sid):
            sids = [sid]
        else:
            sids = sid

        for i in range(len(sids)):
            if isint(sids[i]):
                sids[i] = list(self._subjects.keys())[sids[i]]
            elif not isstr(sids[i]):
                raise ValueError('`sid` ({}) must be a str'.format(type(sids[i])))
        ret = SubjectSet(asvs=self.asvs)
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
        if isint(oids):
            oids = [oids]
        if not isarray(oids):
            raise ValueError('`oids` ({}) must be an array'.format(type(oids)))

        # get indices
        oidxs = []
        for oid in oids:
            oidxs.append(self.asvs[oid].idx)

        # Get the IDs
        ids = []
        for oid in oids:
            ids.append(self.asvs[oid].id)

        # Delete the ASVs from asvset
        for oid in ids:
            if oid not in self.asvs:
                logging.warning('asv `{}` not contained in asvset. skipping'.format(oid))
            asv = self.asvs[oid]
            self.asvs.del_asv(asv.id)

        # Delete the reads
        for subj in self:
            for t in subj.reads:
                subj.reads[t] = np.delete(subj.reads[t], oidxs)
        return self

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
        if isstr(sids):
            if sids == 'all':
                sids = list(self._subjects.keys())
            else:
                raise ValueError('`sids` ({}) not recognized'.format(sids))
        elif isint(sids):
            if sids not in self._subjects:
                raise IndexError('`sid` ({}) not found in subjects'.format(
                    list(self._subjects.keys())))
            sids = [sids]
        elif isarray(sids):
            for sid in sids:
                if not isint(sid):
                    raise TypeError('Each sid ({}) must be an int'.format(type(sid)))
                if sid not in self._subjects:
                    raise IndexError('Subject {} not found in subjects ({})'.format(
                        sid, list(self._subjects.keys())))
        else:
            raise TypeError('`sids` ({}) type not recognized'.format(type(sids)))
        if isnumeric(times):
            times = [times]
        elif isarray(times):
            for t in times:
                if not isnumeric(t):
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
        '''
        if self.perturbations is None:
            self.perturbations = []
        if isnumeric(a):
            if not isnumeric(end):
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
        '''
        for subj in self:
            subj._split_on_perturbations()
        return self

    def _matrix(self, dtype, agg, times, min_rel_abund=None):
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

        if isstr(times):
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
        elif isarray(times):
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

    def matrix(self, dtype, agg, times, min_rel_abund=None):
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
        M, _ =  self._matrix(dtype=dtype, agg=agg, times=times, min_rel_abund=min_rel_abund)
        return M

    def df(self, *args, **kwargs):
        '''Returns a dataframe of the data in matrix. Rows are ASVs, columns are times
        '''
        M, times = self._matrix(*args, **kwargs)
        index = self.asvs.names.order
        return pd.DataFrame(data=M, index=index, columns=times)

