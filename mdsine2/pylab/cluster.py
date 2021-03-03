import numpy as np
from mdsine2.logger import logger
import copy
import numba

# Typing
from typing import TypeVar, Generic, Any, Union, Dict, Iterator, Tuple, Type, List

from . import util
from .errors import NeedToImplementError
from .graph import Node
from .base import isclusterable, Traceable, TaxaSet
from .graph import Node
from .variables import Variable, summary

# Constants
DEFAULT_CLUSTERVALUE_DTYPE = float

def isclustering(x: Any) -> bool:
    '''Type check if `x` is a subclass of Clustering

    Parameters
    ----------
    x : any
        Returns True if `x` is a subclass of Clustering
    
    Returns
    -------
    bool
        True if `x` is the correct subtype
    '''
    return x is not None and issubclass(x.__class__, Clustering)

def isclusterproperty(x: Any) -> bool:
    '''Type check if `x` is a subclass of ClusterProperty

    Parameters
    ----------
    x : any
        Returns True if `x` is a subclass of ClusterProperty
    
    Returns
    -------
    bool
        True if `x` is the correct subtype
    '''
    return x is not None and issubclass(x.__class__, ClusterProperty)

def isclustervalue(x: Any) -> bool:
    '''Type check if `x` is a subclass of ClusterValue

    Parameters
    ----------
    x : any
        Returns True if `x` is a subclass of ClusterValue
    
    Returns
    -------
    bool
        True if `x` is the correct subtype
    '''
    return x is not None and issubclass(x.__class__, ClusterValue)


class Clustering(Node, Traceable):
    '''Base class for clustering. This will cluster items where the aggregate
    clas inherits the `pylab.base.Clusterable` class.
    
    Maps a unique cluster id (int) to a cluster object:
        members : set
            Which itemss are assigned to this cluster (using their 
            index specified in `items`)
        id : int
            Python ID of the cluster
        size : int
            How many members are in the cluster

    Accessing and moving items
    --------------------------
    It is strongly recommended that you only reassign an item from one cluster to
    another cluster with the inner functions and do not do it manually by
    directly accessing `self.clusters`. This is for 2 reasons: 
    
    (1) Calling the inner functions to change the cluster assignment also signals 
    the properties assigned to this clusters so everything is kept synchronized 
    with each other. 
    
    (2) there is no possibility that you can delete an item from the system if you 
    use the inner functions. If you were to manually manipulate `self.clusters` 
    and accidentally delete an item, the whole system would crash not know what to 
    do.

    Properties
    ----------
    There might be properties associated with this class. A property is signaled
    when either the cluster assignment of the items change or if a cluster 
    got deleted and/or added. THE PRIORITY OF A CLUSTER BEING DELETED SUPERCEDES
    THE PRIORITY OF AN ITEM BEING MOVED.
    Example:
    - clusters = [1,0,0,0,2,2,2]
    - If we moved oidx `0` to a different cluster using `move_item`:
    - clusters = [0,0,0,0,1,1,1]
    - Our signaling will call the function `clusters_changed`, even though
        item `0` "effectively" moved and there was a deletion of a cluster.

    Example:
    - clusters = [1,0,0,0,2,2,2]
    - If we moved oidx `1` to cluster [0] `move_item`: 
    - clusters = [1,1,0,0,2,2,2]
    - Our signaling will call the function `assignments_changed` because 
      there was no deletion and/or additions of clusters

    Tracing
    -------
    There are two variables that get traced. (1) `coclusters` (len(items) x len(items) matrix) 
    records which items were in the same cluster together at each iteration.
    (2) `n_clusters` (int) records the number of clusters that were at each iteration.
    
    Parameters
    ----------
    clusters : np.ndarray(n_taxa), None
        If None, do not set cluster assignments.
        The index of the array corresponds to the item index. The value of the
        index indicates the cluster for it to be assigned to
    items : pylab.base.Clusterable
        This is the object that stores all of the information of the items. The 
        ordering of the items in this object are assumed to be the global ordering
        and that they do not change when this Object is instantiated
    kwargs : dict
        These are the additional arguments for the Node class (name, Graph, etc.)
    '''
    def __init__(self, clusters: np.ndarray, items: TaxaSet, **kwargs):
        Node.__init__(self, **kwargs)
        if not isclusterable(items):
            raise TypeError('`items` ({}) must be a pylab.base.Clusterable object'.format( 
                type(items)))
        if clusters is None:
            clusters = np.arange(len(items))
        elif not util.isarray(clusters):
            raise TypeError('`clusters` ({}) must either be None or an array'.format( 
                type(clusters)))
        else:
            clusters = np.asarray(clusters)
            if np.any(clusters < 0):
                raise ValueError('All cluster indices must be > 0')
            for i in range(np.max(clusters)):
                if i not in clusters:
                    raise ValueError('Cluster {} not specified in `clusters`'.format(i))
            if len(clusters) != len(items):
                raise ValueError('`clusters` ({}) must be the same length as `items` ({})'.format(
                    len(clusters), len(items)))
        
        # Everything is good, make the cluster objects
        self._CIDX = 100100 # Start of the cluster index
        self.items = items # This is usually a TaxaSet object
        self.clusters = {} # dict: cluster id (int) -> _Cluster
        for cidx in np.arange(np.max(clusters)+1):
            idxs = np.where(clusters == cidx)[0]
            temp = _Cluster(members=idxs, parent=self, iden=self._CIDX)
            self.clusters[temp.id] = temp
            self._CIDX += 1

        self.order = list(self.clusters) # list of ids
        self.properties = _ClusterProperties() # properties of the clustering
        
        # Maps the item index to the cluster ID it is assigned to
        self.idx2cid = np.zeros(len(self.items), dtype=np.int64)
        for idx in range(len(self.items)):
            for cluster in self:
                if idx in cluster:
                    self.idx2cid[idx] = cluster.id

        # Maps the cluster ID to the cluster index
        self.cid2cidx = {}
        for cidx, cid in enumerate(self.order):
            self.cid2cidx[cid] = cidx

        # Make the tracing objects
        self.coclusters = Variable(
            name='{}_coclusters'.format(self.name),
            shape=(len(self.items), len(self.items)),
            dtype=bool, G=self.G, value=self.generate_coclusters())
        self.n_clusters = Variable(
            name='{}_n_clusters'.format(self.name), dtype=int, 
            G=self.G, value=len(clusters))

        # Make the inner lists for the properties
        self._cids_added = []
        self._cids_removed = []

    def __iter__(self) -> "_Cluster":
        '''Return each cluster
        '''
        for key in self.order:
            yield self.clusters[key]
    
    def __str__(self) -> str:
        s = self.name + ', n_clusters: {}'.format(len(self))
        for cluster in self:
            s += '\n{}'.format(str(cluster))
        return s

    def __len__(self) -> int:
        '''How many clusters

        Returns
        -------
        int
        '''
        return len(self.order)

    def __contains__(self, cid: int) -> bool:
        return (cid in self.clusters) or (cid < len(self.clusters))

    def __getitem__(self, cid):
        if cid in self.clusters:
            return self.clusters[cid]
        elif cid < len(self.clusters):
            return self.clusters[self.order[cid]]
        else:
            raise KeyError('`{}` not recognized as an ID or index'.format(cid))

    def keys(self) -> Iterator[int]:
        '''Alias for `self.order`

        Returns
        -------
        list(shape=(len(items)), dtype=int)
        '''
        return self.order

    def make_new_cluster_with(self, idx: int) -> int:
        '''Create a new cluster with the item index `idx`.
        Removes `idx` from the previous cluster.

        If you want a custom function to initialize the values and indicator variables
        for the new cluster, pass in the functions as parameters. If not, it will
        use the defualt that was used during initialization

        Parameters
        ----------
        idx : int
            This is the index of the item to make a new cluster with

        Returns
        -------
        int
            This is the ID of the new cluster that was created
        '''
        old_cid = self.idx2cid[idx]
        self.clusters[old_cid].remove(idx)
        if self.clusters[old_cid].size == 0:
            # Delete the cluster
            self.clusters.pop(old_cid, None)
            self._cids_removed.append(old_cid)
        
        temp = _Cluster(members=[idx], parent=self, iden=self._CIDX)
        self._CIDX += 1
        self.clusters[temp.id] = temp
        self.idx2cid[idx] = temp.id
        self._cids_added.append(temp.id)
        self.order = list(self.clusters)
        self.n_clusters.value = len(self.clusters)

        self.cid2cidx = {}
        for cidx, cid in enumerate(self.order):
            self.cid2cidx[cid] = cidx

        # Signal to the cluster properties
        for prop in self.properties.signal_when_clusters_change:
            prop.clusters_changed(
                cids_added=self._cids_added,
                cids_removed=self._cids_removed)
        self._cids_added = []
        self._cids_removed = []

        return temp.id

    def move_item(self, idx: int, cid: int) -> int:
        '''Move `idx` to cluster id `cid`. If `cid` does not exist, then we 
        will create a new cluster.

        Paramters
        ---------
        idx : int
            This is the index of the item to move clusters
        cid : int
            This is the Cluster ID to move `idx` to

        Returns
        -------
        int
            This is the cluster ID it was moved to
        '''
        if cid not in self:
            return self.make_new_cluster_with(idx)

        # get the id of the cluster (could be an index)
        cid = self[cid].id

        curr_cid = self.idx2cid[idx]
        if cid == curr_cid:
            # Do nothing
            return cid
        
        self.clusters[curr_cid].remove(idx)
        old_cluster_deleted = False
        if self.clusters[curr_cid].size == 0:
            old_cluster_deleted = True
            self.clusters.pop(curr_cid, None)
            self._cids_removed.append(curr_cid)

        self.clusters[cid].add(idx)
        self.idx2cid[idx] = cid
        self.order = list(self.clusters)
        self.cid2cidx = {}
        for cidx, cid in enumerate(self.order):
            self.cid2cidx[cid] = cidx
        self.n_clusters.value = len(self)

        if old_cluster_deleted:
            # Signal `clusters_changed`
            for prop in self.properties.signal_when_clusters_change:
                prop.clusters_changed(
                    cids_added=[],
                    cids_removed=self._cids_removed)
        else:
            # Signal `assignments_changed`
            for prop in self.properties.signal_when_item_assignment_changes:
                prop.assignments_changed()

        self._cids_removed = []
        self._cids_added = []
        return cid
        
    def merge_clusters(self, cid1, cid2):
        raise NotImplementedError('Not Implemented')

    def split_cluster(self, cid, members1, members2):
        raise NotImplementedError('Not Implemented')

    def generate_coclusters(self) -> np.ndarray:
        '''Make the cocluster matrix of the current cluster configuration
        '''
        return _generate_coclusters_fast(idx2cid=self.idx2cid)

    def fromlistoflists(self, clustering_arr: List[List[int]]) -> List:
        '''
        Takes a list of lists, representing clusterings (each constituent is an int), and repopulate a clustering.
        :return: The list of new Cluster IDs (cids).
        '''
        cids = []
        for new_clust in clustering_arr:
            cid = self.make_new_cluster_with(new_clust[0])
            for item in new_clust[1:]:
                self.move_item(item, cid)
            cids.append(cid)
        return cids

    def tolistoflists(self) -> List[List[int]]:
        '''Converts clusters into array format:
        clusters = [clus1, ..., clusN],
            clusters{i} = [idx1, ..., idxM]
        each clusters{i} is a list of indices that are in that cluster

        This is the same format was the input parameter for `__init__`
        
        Returns
        -------
        list
            This is the array of values with the correct order 
        '''
        ret = []
        for cluster in self:
            ret.append(list(cluster.members))
        return ret

    def toarray(self) -> np.ndarray:
        '''Converts clusters into array format:
        Each index is the index of an element that is being clustered. The value
        is the cluster index. This is the same format was the input parameter for 
        `__init__`.
        
        Returns
        -------
        np.ndarray
        '''
        ret = np.zeros(len(self.items),dtype=int)
        for cidx, cluster in enumerate(self):
            for idx in cluster.members:
                ret[idx] = cidx
        return ret

    def from_array(self, a: np.ndarray):
        '''Set the clustering from a numpy array - note that this resets
        all of the properties of these clusterings

        Parameters
        ----------
        a : np.ndarray
            The cluster configuration
        '''
        # Check
        if not util.isarray(a):
            raise TypeError('`a` ({}) must be an array'.format(type(a)))
        a = np.asarray(a)
        if np.any(a < 0):
            raise ValueError('All values in `a` must be >= 0')
        for i in range(np.max(a)):
            if i not in a:
                raise ValueError('Index `{}` skipped in a'.format(i))
        
        for cidx in range(np.max(a)+1):
            idxs = np.where(a == cidx)[0]
            cid = self.make_new_cluster_with(idxs[0])
            for idx in idxs[1:]:
                self.move_item(idx, cid=cid)

    def set_trace(self, *args, **kwargs):
        '''Set the trace of the cocluster and n_clusters
        '''
        self.coclusters.set_trace(*args, **kwargs)
        self.n_clusters.set_trace(*args, **kwargs)

    def add_trace(self):
        '''Add a trace of the cocluster and n_clusters
        '''
        self.coclusters.value = self.generate_coclusters()
        self.coclusters.add_trace()
        self.n_clusters.add_trace()


class _Cluster:
    '''Class for a single cluster.

    Parameters
    ----------
    members : array_like, set
        Individual item indices that are within the cluster
    parent : pylab.Clustering
        Pointer to parent 
    iden : int
        Identifier
    '''
    def __init__(self, members: Iterator[int], parent: Clustering, iden: int):
        self.members = set()
        for mem in members:
            self.members.add(mem)
        self.id = iden # Unique id for class
        self.size = len(members)
        self.parent = parent
                
    def __str__(self) -> str:
        return 'Cluster {}\n' \
            '\tmembers: {}\n' \
            '\tsize: {}'.format(
                self.id,
                [self.parent.items[idx].cluster_str() for idx in self.members],
                self.size)

    def __contains__(self, item: Any) -> bool:
        '''For the `in` operator
        '''
        return item in self.members

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> int:
        '''Let c = ClusterBase object.
        Let c[i] (c.clusters[i]) be this object.
        This method is useful for doing the command:
            `for idx in c[i]`
        That call iterates over every member that this cluster contains
        '''
        for item in self.members:
            yield item

    def add(self, item: int) -> bool:
        '''Add the item `item` to the cluster

        Paramters
        ---------
        item : int
            This is the index of the item that we want to add

        Returns
        -------
        bool
            True if successful
        '''
        if item in self.members:
            return True
        self.members.add(item)
        self.size += 1
        return True

    def remove(self, item: int) -> bool:
        '''Returns True if `item` was deleted from the cluster. Returns False
        if `item` was not in the cluster so it could not be deleted.

        Paramters
        ---------
        item : int
            This is the index of the item that we want to remove

        Returns
        -------
        bool
            True if the item is contained in the cluster
            False if the item is not contained
        '''
        if item in self.members:
            self.members.remove(item)
            self.size -= 1
            return True
        return False


class _ClusterProperties:
    '''Manages the properties associated with the clusters
    '''
    def __init__(self):
        self._d = {} # ID of object (int) -> ClusterProperty

        # A list of cluster properties to update when a cluster
        # is added or removed
        self.signal_when_clusters_change = [] 

        # A list of cluster properties to update when the assignment
        # of a cluster changes
        self.signal_when_item_assignment_changes = []

        # Local pointer of the keys
        self._keys = []

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, key: int) -> 'ClusterProperty':
        return self._d[key]

    def __iter__(self) -> 'ClusterProperty':
        for a in self._keys:
            yield self._d[a]

    def add(self, prop: 'ClusterProperty'):
        '''Add a property to the list of cluster properties. Additionally, add the property
        to any of the required lists

        Parameters
        ----------
        prop : ClusterProperty
            ClusterProperty to add
        '''
        self._d[prop.id] = prop
        if prop.signal_when_clusters_change:
            self.signal_when_clusters_change.append(prop)
        if prop.signal_when_item_assignment_changes:
            self.signal_when_item_assignment_changes.append(prop)
        self._keys = list(self._d.keys())

    def toarray(self) -> Iterator['ClusterProperty']:
        '''Make a list of the ClusterProperty's.
        '''
        return [self._d[key] for key in self._keys]

    def keys(self) -> Iterator[int]:
        return self._keys


class ClusterProperty:
    '''This is a class that stores the property of a set of clusters. The point
    of this class to provide methods of signaling to the cluster property when
    a cluster is added/removed or when the cluster assignment of an item changes. 
    You must inherit this class and implement the `cluster_removed` and 
    `cluster_added` manually.

    An example of this is the ClusterPerturbation class in pylab.contrib - 
    when a cluster is added/removed it must change the size of the indicator 
    object associated with it.

    Parameters
    ----------
    clustering : Clustering
        This is the clustering object
    signal_when_clusters_change : bool
        Set this to True if you want to signal this property when the number
        of clusters change/change cids
    signal_when_item_assignment_changes : bool
        Set to True if you want this property to be signaled when an item
        assignment changes but there is not necessarily a change in the number 
        of clusters/cluster ids
    '''
    def __init__(self, clustering: Clustering, signal_when_clusters_change: bool,
        signal_when_item_assignment_changes: bool):
        if not isclustering(clustering):
            raise ValueError('`clustering` ({}) must be a Clustering object'.format(
                type(clustering)))
        if not util.isbool(signal_when_clusters_change):
            raise ValueError('`signal_when_clusters_change` ({}) must be a bool'.format(
                type(signal_when_clusters_change)))
        if not util.isbool(signal_when_item_assignment_changes):
            raise ValueError('`signal_when_item_assignment_changes` ({}) must be a bool'.format(
                type(signal_when_item_assignment_changes)))
        self.clustering = clustering
        self.signal_when_clusters_change = signal_when_clusters_change
        self.signal_when_item_assignment_changes = signal_when_item_assignment_changes
        self.clustering.properties.add(self)

    def assignments_changed(self):
        '''Each object inheriting this class needs to implement this function
        '''
        raise NeedToImplementError('User needs to implement this function')

    def clusters_changed(self, cids_added, cids_removed):
        '''Each object inheriting this class needs to implement this function
        '''
        raise NeedToImplementError('User needs to implement this function')

    def set_signal_when_clusters_change(self, value: bool):
        '''Switch the signal `signal_when_clusters_change` to `value`

        Paramters
        ---------
        value : bool
            This is what to set the `signal_when_clusters_change` flag to
        '''
        if not util.isbool(value):
            raise ValueError('`value` ({}) must be a bool'.format(
                type(value)))
        # Only need to change if they are different
        if self.signal_when_clusters_change != value:
            if self.signal_when_clusters_change:
                # We need to take it out
                self.clustering.properties.signal_when_clusters_change.remove(self)
            else:
                self.clustering.properties.signal_when_clusters_change.append(self)
        self.signal_when_clusters_change = value
        self.reset()

    def set_signal_when_item_assignment_changes(self, value: bool):
        '''Switch the signal `signal_when_item_assignment_changes` to `value`

        Paramters
        ---------
        value : bool
            This is what to set the `signal_when_item_assignment_changes` flag to
        '''
        if not util.isbool(value):
            raise ValueError('`value` ({}) must be a bool'.format(
                type(value)))
        # Only need to chagne if they are different
        if self.signal_when_item_assignment_changes != value:
            if self.signal_when_item_assignment_changes:
                # We need to take it out
                self.clustering.properties.signal_when_item_assignment_changes.remove(self)
            else:
                self.clustering.properties.signal_when_item_assignment_changes.append(self)
        self.signal_when_item_assignment_changes = value
        self.reset()

    def reset(self):
        '''Call this function after you set the `signal_when_item_assignment_changes` or
        `signal_when_clusters_change`.
        '''
        raise NeedToImplementError('User needs to implement this function')


class ClusterValue(ClusterProperty, Node, Traceable):
    '''This is an object that has a value per cluster.
    The value is a dictionary, then there are functions to convert
    that dictionary into an item array or a cluster array.
    This records the data on an item-item basis.

    User nees to implement the signaling methods and reset if necessary

    Parameters
    ----------
    clustering : Clustering
        This is the clustering object you are adding it to
    signal_when_clusters_change : bool
        Flag for ClusterProperty
    signal_when_item_assignment_changes : bool
        Flag for ClusterProperty
    dtype : type
        This is the datatype to set the output to
    kwargs : dict
        These are the extra arguements for Node
    '''
    def __init__(self, clustering: Clustering, signal_when_clusters_change: bool,
        signal_when_item_assignment_changes: bool, dtype: Type=None, **kwargs):
        if dtype is None:
            dtype = DEFAULT_CLUSTERVALUE_DTYPE
        Node.__init__(self, **kwargs)
        self.value = {}
        self.dtype = dtype
        ClusterProperty.__init__(self, clustering=clustering,
            signal_when_clusters_change=signal_when_clusters_change,
            signal_when_item_assignment_changes=signal_when_item_assignment_changes)
        for cid in self.clustering.order:
            self.value[cid] = np.nan

    def item_array(self) -> np.ndarray:
        '''Converts these values per item

        Returns
        -------
        np.ndarray((n,), dtype=self.dtype)
            Array of the values expanded to the items in the overall item order
            specified in Clusterable
        '''
        ret = np.zeros(len(self.clustering.items), dtype=self.dtype)
        for cluster in self.clustering:
            idxs = list(cluster.members)
            ret[idxs] = self.value[cluster.id]
        return ret

    def cluster_array(self) -> np.ndarray:
        '''Converts the dictionary into a cluster array in the order of the clusters

        Returns
        -------
        np.ndarray((n,), dtype=self.dtype)
            Array of the values expanded for each cluster in the overall cluster order
        '''
        return np.asarray([self.value[cid] for cid in self.clustering.order], dtype=self.dtype)

    def set_values_from_array(self, values: Iterator[Any]):
        '''Set the values from an array of the same order as the clusters

        Paramters
        ---------
        values : array_like
            An array of the values
            Must be the same length as the number of clusters
        '''
        if not util.isarray(values):
            raise ValueError('`values` ({}) must be an array'.format(type(values)))
        if len(values) != len(self.clustering):
            raise ValueError('`values` ({}) must be the same length as the number ' \
                'of clusters ({})'.format(len(values), len(self.clustering)))
        self.value = {}
        for cidx, cid in enumerate(self.clustering.order):
            self.value[cid] = values[cidx]

    def set_trace(self):
        '''Set up the trace of the object
        '''
        tracer = self.G.tracer
        tracer.set_trace(
            self.name, 
            shape=(len(self.clustering.items), ), 
            dtype=self.dtype)
        self.ckpt_iter = 0
        self.sample_iter = 0
        self.trace = np.full(shape=(tracer.checkpoint, len(self.clustering.items)),
            dtype=self.dtype, fill_value=np.nan)
    
    def remove_local_trace(self):
        '''Delete the local trace
        '''
        self.trace = None

    def add_trace(self):
        '''Adds the current value to the trace on an item-basis. Writes to disk if
        local buffer is full
        '''
        value = self.item_array()
        self.trace[self.ckpt_iter] = value
        self.ckpt_iter += 1
        self.sample_iter += 1
        if self.ckpt_iter == len(self.trace):
            # We have gotten the largest we can in the local buffer, write to disk
            self.G.tracer.write_to_disk(name=self.name)
            self.trace = np.full(shape=(self.G.tracer.checkpoint, len(self.clustering.items)),
                dtype=self.dtype, fill_value=np.nan)
            self.ckpt_iter = 0


@numba.jit(nopython=True) #, fastmath=True, cache=True)
def _generate_coclusters_fast(idx2cid: np.ndarray) -> np.ndarray:
    '''Generates a cocluster matrix for the current cluster assignment.
    If two elements are in the same cluster, then the assignment is 1
    If two elements are in different clusters, then the assignment is 0

    A single element is in the same cluster as itself always, so the assignment
    for this is always 1 on the diagonal.

    The elemets are in the order of the elements in `idx2cid`

    Parameters
    ----------
    idx2cid : np.ndarray(shape=(len(items),), dtype=int)
        Each index (corresponds to an item index) maps to the cluster ID

    Returns
    -------
    np.ndarray(shape=(len(items), len(items)))
    '''
    n_items = len(idx2cid)
    ret = np.zeros(shape=(n_items, n_items), dtype=np.int64)
    for i in range(n_items):
        for j in range(i):
            if idx2cid[i] == idx2cid[j]:
                ret[i,j] = 1
                ret[j,i] = 1
        ret[i,i] = 1
    return ret

@numba.jit(nopython=True, cache=True)
def toarray_from_cocluster(coclusters: np.ndarray) -> np.ndarray:
    '''Generate the output that would be given from 
    `clustering.toarray` from the cocluster matrix.

    Numba is about 10X faster.

    Example:
        ```
        >>> coclusters = np.asarray(
            [[1,0,0,1],
             [0,1,0,0],
             [0,0,1,0],
             [1,0,0,1]])
        >>> toarray_from_coclusters(coclusters)
        [0, 1, 2, 0]
        ```

    Parameters
    ----------
    coclusters : 2-dim square np.ndarray
        Cocluster matrix
    
    Returns
    -------
    np.ndarray
    '''
    a = np.full(coclusters.shape[0], -1)
    i = 0
    for j in range(coclusters.shape[0]):
        if a[j] == -1:
            for k in range(j,coclusters.shape[0]):
                if coclusters[j,k] == 1:
                    a[k] = i
            i += 1
    return a