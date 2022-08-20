'''Defines classes used for defining a graphical model.

If no graph is specified when defining a node, the node gets
added to a default graph, which is defined at the bottom of the module.

The Node class is also defined here.
'''
from pathlib import Path
import pickle
import os
import numpy as np
import networkx as nx
from mdsine2.logger import logger
from mdsine2.pylab import util as plutil

# Typing
from typing import Any, Union, Iterator

from .random import seed as set_seed
from .errors import GraphIDError, UndefinedError

# Constants
DEFAULT_BASENODE_NAME_PREFIX = 'node_'
DEFAULT_PRIOR_SUFFIX = '_prior'

_graph_dict = {}
_PH_COUNT = 0

def get_default_graph() -> "Graph":
    '''Returns the default graph object

    Returns
    -------
    Graph
    '''
    return _graph_dict[_default_graph_id]

def clear_default_graph() -> "Graph":
    '''Delete the old default graph. Make a new one

    Returns
    -------
    dict(Graph)
    '''
    global _graph_dict
    global _default_graph_id
    _graph_dict.pop(_default_graph_id, None)
    a = Graph()
    _default_graph_id = a.id
    _graph_dict[a.id] = a
    return _graph_dict[_default_graph_id]

def _get_graph(id: int) -> "Graph":
    '''Gets the graph by the ID. Only internal use

    Returns
    -------
    Graph
    '''
    if id not in _graph_dict:
        raise GraphIDError('Graph ID `{}` not recognized. Available graph IDs are:' \
            ' {}'.format(id, list(_graph_dict.keys())))
    return _graph_dict[id]

def isgraph(x: Any) -> bool:
    '''Checks if the type (or subtype) of `x` is a Graph

    Parameters
    ----------
    x : any
        Instance to check
    
    Returns
    -------
    bool
        True if `x` is a subclass of a Graph, else False
    '''
    return x is not None and issubclass(x.__class__, Graph)

def isnode(x: Any) -> bool:
    '''Checks if the type (or subtype) of `x` is a Node

    Parameters
    ----------
    x : any
        Instance to check
    
    Returns
    -------
    bool
        True if `x` is a subclass of a Node, else False
    '''
    return x is not None and issubclass(x.__class__, BaseNode)

def hasprior(x: Any) -> bool:
    '''Checks whether `x` has a prior defined. It must be a subclass of 
    `pylab.graph.BaseNode` for it to be True.

    Parameters
    ----------
    x : any
        Instance to Check

    Returns
    -------
    bool
    '''
    if isnode(x):
        return x.prior is not None
    return False


class Saveable:
    '''Implements baseline saving classes with pickle for classes
    '''
    def save(self, filename: Union[str, Path]=None):
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
                raise TypeError('`filename` must be specified if you have not set the save location')
            filename = self._save_loc

        if isinstance(filename, Path):
            filename = Path(filename)

        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename: str):
        '''Unpickle the object

        Paramters
        ---------
        cls : type
            Type
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

    def set_save_location(self, filename: str):
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

    def get_save_location(self) -> str:
        try:
            return self._save_loc
        except:
            raise AttributeError('Save location is not set.')


class BaseNode(Saveable):
    '''This is the baseclass of a Node.

    Parameters
    ----------
    name : str, Optional
        - name of the node. If one is not provided, a unique one will be generated
    name_prefix : str, Optional
        - name prefix if `name` is not passed in
    G : Graph, Optional
        - Graph object to add the node to.
        - If not specified it adds it to the default graph
    '''

    def __init__(self, name: str = None, name_prefix: str = None, G: 'Graph' = None):
        global _PH_COUNT
        if name is None:
            if name_prefix is None:
                name_prefix = DEFAULT_BASENODE_NAME_PREFIX
            name = name_prefix + '{}'.format(_PH_COUNT)
            _PH_COUNT += 1

        if G is None:
            G = get_default_graph()
        self.G = G  # This is a pointer to the graph it is contained in
        self.name = name
        self.id = id(self)

        # Add itself into the graph
        self.G.nodes[self.id] = self
        self.G.name2id[self.name] = self.id
        self.prior = None

    def delete(self):
        '''Delete itself from the graph
        '''
        self.G.name2id.pop(self.name, None)
        self.G.nodes.pop(self.id)


class DataNode:
    '''Base class for an object that wants to be used as the data class in inference.
    This sets the `.data` pointer in `Graph`

    Parameters
    ----------
    name : str, Optional
        - name of the node. If one is not provided, a unique one will be generated
    name_prefix : str, Optional
        - name prefix if `name` is not passed in
    G : Graph, Optional
        - Graph object to add the node to.
        - If not specified it adds it to the default graph
    '''

    def __init__(self, name: str = None, name_prefix: str = None, G: 'Graph' = None):
        self.G = G
        self.name = name
        self.id = id(self)
        if self.G.data is not None:
            logger.info('Overriding old data object in graph {}'.format(self))
        self.G.data = self  # Set itself as the .data pointer

    def delete(self):
        '''Deletes this data object from the graph it belogns to
        '''
        self.G.data = None


class Node(BaseNode):
    '''Variable that can be in the graph

    Parameters
    ----------
    name : str
        - name of the node
    name_prefix : str
        - name prefix if `name` is not passed in
    G : Graph, int
        - Graph object or graph id to add the node to.
        - If not specified it adds it to the default graph
    '''

    def __init__(self, name: str = None, name_prefix: str = None, G: 'Graph' = None):
        BaseNode.__init__(self, name=name, name_prefix=name_prefix, G=G)
        self.parents = {}  # maps the ID of the node to the Node objects
        self.children = {}  # maps the ID of the node to the Node objects
        self.undirected = {}

        # Depreciated
        self._metropolis = None

    @property
    def metropolis(self):
        '''DEPRECIATED

        Get the metropolis object
        '''
        return self._metropolis

    def delete(self):
        '''Delete itself from the graph
        '''
        if len(self.parents) > 0:
            for pid in self.parents:
                self.G.nodes[pid].children.pop(self.id, None)
        if len(self.children) > 0:
            for cid in self.children:
                self.G.nodes[cid].parents.pop(self.id, None)
        if len(self.undirected) > 0:
            for uid in self.undirected:
                self.G.nodes[uid].undirected.pop(self.id, None)
        BaseNode.delete(self)

    @property
    def degree(self) -> int:
        '''Get the degree of the node

        Returns
        -------
        int
        '''
        # return the degree of the node
        return len(self.parents) + len(self.children) + len(self.undirected)

    def get_adjacent_keys(self) -> Iterator[int]:
        '''Get the adjacent nodes

        Returns
        -------
        list(int)
            A list of all the IDs of the adjacent nodes
        '''
        return list(self.parents.keys()) + list(self.children.keys())

    def add_parent(self, parent: "Node"):
        '''Adds `parent` as a parent to the node
        Also adds self as a child to `parent`

        Parameters
        ----------
        parent : Node
            - node we want to set as a parent
        '''
        if not isnode(parent):
            raise ValueError('parent ({}) must be a (subclass of) Node'.format(
                type(parent)))

        if self.G.id != parent.G.id:
            raise GraphIDError('Attempting to add a parent `{}` to `{}` ' \
                               'but they are not in the same graph'.format(self.name, parent.name))

        self.parents[parent.id] = parent
        parent.children[self.id] = self

    def add_child(self, child: "Node"):
        '''Adds `child` as a child to the node
        Also adds self as a parent to `child`

        Parameters
        ----------
        child : Node
            - node we want to set as a child
        '''
        if not isnode(child):
            raise ValueError('child ({}) must be a (subclass of) Node'.format(
                type(child)))

        if self.G.id != child.G.id:
            raise GraphIDError('Attempting to add a child `{}` to `{}` ' \
                               'but they are not in the same graph'.format(self.name, child.name))

        self.children[child.id] = child
        child.parents[self.id] = self

    def add_undirected(self, node: "Node"):
        '''Adds `node` as an undirected neighbor to the node
        Does the same for `node`

        Parameters
        ----------
        node : Node
            - node we want to set as an undirected node
        '''
        if not isnode(node):
            raise ValueError('node ({}) must be a (subclass of) Node'.format(
                type(node)))

        if self.G.id != node.G.id:
            raise GraphIDError('Attempting to add a node `{}` to `{}` ' \
                               'but they are not in the same graph'.format(self.name, node.name))
        self.undirected[node.id] = node
        node.undirected[self.id] = self

    def add_prior(self, prior: "Node"):
        '''Override the name of the passed in distribution `prior`.

        Parameters
        ----------
        prior : Node
            - node we want to set as a prior
        '''
        if not isnode(prior):
            raise ValueError('prior ({}) must be a (subclass of) Node'.format(
                type(prior)))

        self.add_parent(prior)
        self.prior = prior


class Data(DataNode):
    '''Simple class defining the covariates `X` and the observations `y`

    Parameters
    ----------
    X : array_like
        Defines the covariates
    y : array_like
        Defines the observations
    **kwargs
        See `pylab.graph.DataNode`
    '''

    def __init__(self, X: np.ndarray, y: np.ndarray, **kwargs):
        DataNode.__init__(self, **kwargs)
        self.X = X
        self.y = y


class TraceableNode(Node):
    '''
    Defines the functionality for a Node to interact with the Graph tracer object
    '''
    def __init__(self, dtype, **kwargs):
        super().__init__(**kwargs)
        self.dtype = dtype

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

    def get_trace_from_disk(self, section: str = 'posterior', slices: slice = None) -> np.ndarray:
        '''Returns the entire trace (after burnin) writen on the disk. NOTE: This may/may not
        include the samples in the local buffer trace and could be very large

        Parameters
        ----------
        section : str
            Which part of the trace to return - description above
        slices : list(slice), slice
            A list of slicing objects or a slice object.

            slice(start, stop, step)
            Example, single dimension:
                slice(None) == :
                slice(5) == :5
                slice(4, None, None) == 4:
                slice(9, 22,None) == 9:22
            Example, multiple dimensions:
                [slice(None), slice(4, None, None)] == :, 4:
                [slice(None), 4, 5] == :, 4, 5

        Returns
        -------
        np.ndarray
        '''
        return self.G.tracer.get_trace(name=self.name, section=section, slices=slices)

    def overwrite_entire_trace_on_disk(self, data: np.ndarray, **kwargs):
        '''Overwrites the entire trace of the variable with the given data.

        Parameters
        ----------
        data : np.ndarray
            Data you are overwriting the trace with.
        '''
        self.G.tracer.overwrite_entire_trace_on_disk(
            name=self.name, data=data, dtype=self.dtype, **kwargs
        )

    def get_iter(self) -> int:
        '''Get the number of iterations saved to the hdf5 file of the variable

        Returns
        -------
        int
        '''
        return self.G.tracer.get_iter(name=self.name)


def issavable(x: Any) -> bool:
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


def istraceable(x: Any) -> bool:
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
    return x is not None and issubclass(x.__class__, TraceableNode)


class Graph(Saveable):
    '''Graph class

    Parameters
    ----------
    name : str
        This is the name of the graph
    seed : int
        This is how we should seed the graph
    '''

    def __init__(self, name: str=None, seed: int=None):
        global _graph_dict
        global _PH_COUNT

        if name is None:
            name = 'graph_{}'.format(_PH_COUNT)
            _PH_COUNT += 1

        # Set seed if necessary
        if seed is not None:
            set_seed(x=seed)

        self.seed = seed
        self.name = name
        self.nodes = {} # dict id -> Node or Node subclass
        self.name2id = {} # dict name -> id
        self.perturbations = None # If this is set, it is set to a pylab.base.Perturbations object
        self.data = None # This is a pointer to the data object
        self.inference = None # This is a pointer to the pylab.inference.BaseMCMC object
        self.id = id(self)
        self.tracer = None # This is a pointer to the pylab.inference.Tracer

        # Add a persistent object to the graph so it knows 
        # what variables have persistent workers. This is optional
        # for the persistent Pool
        self._persistent_pntr = []

        # Add to the graph dict
        _graph_dict[self.id] = self

    def set_seed(self, seed: int):
        '''Sets the seed of the graph

        Parameters
        ----------
        seed : int
            Seed to set for the graph
        '''
        self.seed = seed
        set_seed(x=seed)

    @property
    def size(self) -> int:
        '''Alias for `__len__`

        Returns
        -------
        int
            How many nodes in the graph
        '''
        return len(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)

    def __getitem__(self, nid: Union[str, int]) -> "Node":
        '''Get a node either using the name or the id. First 
        check if it is an id, then a str. If neither work to 
        index the node then an error is thrown

        Parameters
        ----------
        id : int, str
            This is the identifier for the Node
        '''
        try:
            ret = self.nodes[nid]
        except:
            try:
                ret = self.nodes[self.name2id[nid]]
            except:
                raise UndefinedError('`id` ({}), "{}" not recognized in graph `{}`'.format( 
                    type(nid), nid, self.name))
        return ret

    def __contains__(self, nid: Union[str, int]) -> bool:
        return nid in self.name2id or nid in self.nodes

    def __str__(self) -> str:
        s = ''
        for node in self.nodes.values():
            s += node.name + ' ({})\n'.format(node.id)
            if len(node.children) > 0:
                s += '\tchildren:\n'
                for child in node.children.values():
                    s += '\t\t{} ({})\n'.format(child.name,child.id)
            if len(node.parents) > 0:
                s += '\tparents:\n'
                for parent in node.parents.values():
                    s += '\t\t{} ({})\n'.format(parent.name,parent.id)
        return s

    def __iter__(self) -> "Node":
        for _id in self.nodes:
            yield self.nodes[_id]

    def to_networkx(self) -> nx.DiGraph:
        '''Make a networkx graph

        Returns
        -------
        networkx.DiGraph
        '''

        G = nx.DiGraph()
        labels = {}

        # Add nodes
        for id,node in self.nodes.items():
            G.add_node(node.name)
            G.nodes[node.name]['id'] = id
            labels[node.name] = node.name
            G.nodes[node.name]['class'] = node.__class__.__name__

        # Add edges
        for id,node in self.nodes.items():
            try:
                for parent in node.parents.values():
                    G.add_edge(parent.name,node.name)
            except:
                pass

        return G

    def as_default(self) -> "Graph":
        '''Sets the current graph as the default graph

        Returns
        -------
        Graph
            self
        '''
        global _default_graph
        _default_graph = self
        return self

    def get_descendants(self, name: str) -> Iterator[int]:
        '''Returns a list of ids of variables that are dependent on
        the input name(recursively returns all of the child ids).
        The name can either be the name of the variable or the id. Right now
        it DOES NOT look at undirected edges

        Parameters
        ----------
        name : int, str
            - Identifier of the node to get the descendants of
        
        Returns
        -------
        list
            List of IDs of all of the descendants of this node
        '''
        id = None
        if name in self.nodes:
            id = name
        elif name in self.name2id:
            id = self.name2id[name]
        else:
            raise Exception('name `{}` not found in graph'.format(name))

        node = self.nodes[id]
        ret = []
        for id in node.children:
            ret.append(id)
            ret += self.get_descendants(id)
        return ret


# Creates a background graph that we can add to
_default_graph = Graph()
_default_graph_id = _default_graph.id
