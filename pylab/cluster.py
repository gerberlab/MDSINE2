import numpy as np
import logging
import copy
import numba

from sklearn.cluster import AgglomerativeClustering
import scipy.stats

from . import util
from .errors import NeedToImplementError
from .graph import Node
from .base import isclusterable, Traceable
from .graph import Node
from .variables import Variable, summary

# Constants
DEFAULT_CLUSTERVALUE_DTYPE = float

def isclustering(x):
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

def isclusterproperty(x):
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

def isclustervalue(x):
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
        clusters = [[1,2,3], [0], [4,5]]
        If we moved oidx `0` to a different cluster using `move_item`:
        clusters = [[0,1,2,3], [4,5]]
        Our signaling will call the function `clusters_changed`, even though
        item `0` "effectively" moved and there was a deletion of a cluster.

    Example:
        clusters = [[1,2,3], [0], [4,5]]
        If we moved oidx `1` to cluster [0] `move_item`:
        clusters = [[2,3], [0,1], [4,5]]
        Our signaling will call the function `assignments_changed` because 
        there was no deletion and/or additions of clusters

    Tracing
    -------
    There are two variables that get traced. (1) `coclusters` (len(items) x len(items) matrix) 
    records which items were in the same cluster together at each iteration.
    (2) `n_clusters` (int) records the number of clusters that were at each iteration.
    
    Parameters
    ----------
    clusters : list(list(int)), None
        These are the cluster assignments of each item. Structure is as follows:
        The index of the top list indicates the index of the cluster. The elements
        in the second level lists are the indices of the items in that cluster.
        Example:
            >>> clusters = [[0,1,2], [5,3], [4]]
            Items 0,1,2 are in cluster 0
            Items 3,5 are in cluster 1
            Item 4 is in cluster 4
        If it is None, then assume that all of the items are in their own cluster.
        ALL MUST BE SPECIFIED ONLY ONCE.
    items : pylab.base.Clusterable
        This is the object that stores all of the information of the items. The 
        ordering of the items in this object are assumed to be the global ordering
        and that they do not change when this Object is instantiated
    kwargs : dict
        These are the additional arguments for the Node class (name, Graph, etc.)
    '''
    def __init__(self, clusters, items, **kwargs):
        Node.__init__(self, **kwargs)
        if not isclusterable(items):
            raise TypeError('`items` ({}) must be a pylab.base.Clusterable object'.format( 
                type(items)))
        if clusters is None:
            clusters = [[i] for i in range(len(items))]
        elif not util.isarray(clusters):
            raise TypeError('`clusters` ({}) must either be None or an array'.format( 
                type(clusters)))
        else:
            there = np.zeros(len(items), dtype=int)
            clusters = list(clusters)
            for ele in clusters:
                if not type(ele) == list:
                    raise TypeError('Each element in `clusters` ({}) must be a list'.format(
                        type(ele)))
                if not np.all(util.itercheck(ele, util.isint)):
                    raise TypeError('Each element in each cluster must be an int')
                for idx in ele:
                    if there[idx] > 0:
                        raise ValueError('Item index `{}` was specified more than once: {}'.format(
                            idx, clusters))
                    there[idx] = 1
        
        # Everything is good, make the cluster objects
        self.items = items
        self.clusters = {}
        for cluster in clusters:
            temp = _Cluster(members=cluster, parent=self)
            self.clusters[temp.id] = temp
        self.order = list(self.clusters.keys())
        self.properties = _ClusterProperties()
        
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

    def __iter__(self):
        '''Return each cluster
        '''
        for key in self.order:
            yield self.clusters[key]
    
    def __str__(self):
        s = self.name + ', n_clusters: {}'.format(len(self))
        for cluster in self:
            s += '\n{}'.format(str(cluster))
        return s

    def __len__(self):
        '''How many clusters

        Returns
        -------
        int
        '''
        return len(self.order)

    def __contains__(self, cid):
        return cid in self.clusters

    def __getitem__(self, cid):
        return self.clusters[cid]

    def keys(self):
        '''Alias for `self.order`

        Returns
        -------
        list(shape=(len(items)), dtype=int)
        '''
        return self.order

    def add_init_value(self):
        '''Set the initialization value. This is called by `pylab.inference.BaseMCMC.run`
        when first updating the variable. User should not use this function
        '''
        self._init_value = self.toarray()

    def make_new_cluster_with(self, idx):
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
        
        temp = _Cluster(members=[idx], parent=self)
        self.clusters[temp.id] = temp
        self.idx2cid[idx] = temp.id
        self._cids_added.append(temp.id)
        self.order = list(self.clusters.keys())
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

    def move_item(self, idx, cid):
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
        if cid not in self.clusters:
            return self.make_new_cluster_with(idx)
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
        self.order = list(self.clusters.keys())
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

    def generate_coclusters(self):
        return _generate_coclusters_fast(idx2cid=self.idx2cid)
    
    def toarray(self):
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

    def toarray_vec(self):
        '''Converts clusters into array format

        array = [cidx(idx1), cidx(idx2), ..., cidx(idxM)]
        This is the format for sklearn and scikit

        Returns
        -------
        np.ndarray
        '''
        ret = np.zeros(len(self.items), dtype=int)
        for idx in range(len(ret)):
            ret[idx] = self.cid2cidx[self.idx2cid[idx]]
        return ret

    def from_array(self, a):
        '''Set the clustering from a list of lists - note that this resets
        all of the properties of these clusterings

        Parameters
        ----------
        a : list(list(int))
            The cluster configuration
        '''
        # Check
        here = np.zeros(len(self.items))
        if not util.isarray(a):
            raise TypeError('`a` ({}) must be an array'.format(type(a)))
        for ele in a:
            if not util.isarray(ele):
                raise TypeError('Each element in `a` ({}) must be an array'.format(
                    type(ele)))
            for idx in ele:
                if not util.isint(idx):
                    raise TypeError('Each element to be set in clustering ' \
                        'must be an int ({}-{})'.format(type(ele),ele))
                if idx < 0:
                    raise ValueError('Each index ({}) must be >= 0'.format(idx))
                if here[idx] == 1:
                    raise ValueError('item index `{}` assigned twice ({})'.format(idx, a))
                here[idx] = 1
        if np.any(here == 0):
            raise ValueError('Not all elements specified ({})'.format(
                np.where(here == 0)[0]))

        # Set the clusters
        for arr in a:
            first = arr[0]
            rest = arr[1:]
            cid = self.make_new_cluster_with(first)
            for b in rest:
                self.move_item(b, cid=cid)

    def generate_cluster_assignments_posthoc(self, n_clusters='mode', linkage='average',
        set_as_value=False, section='posterior'):
        '''Once the inference is complete, compute the clusters posthoc using
        sklearn's AgglomerativeClustering function with distance matrix being
        1 - cocluster matrix (we subtrace the cocluster matrix from 1 because
        the cocluster matrix describes similarity, not distance).

        Parameters
        ----------
        n_clusters : str, int, callable, Optional
            This specifies the number of clusters that are used during
            Agglomerative clustering.
            If `n_clusters` is of type int, it will use that number as the number of
            clusters.
            If `n_clusters` is of type str, it calculates the number of clusters
            based on the trace for the number of clusters (self.n_clusters_trace).
            Possible calculation types are:
                * 'median', 'mode', and 'mean'.
            If `n_clusters` is callable, it will calculate n given the trace of n_clusters
            Default is 'mode'.
        linkage : str, Optional
            Which linkage criterion to use. Determines which distance to use
            between sets of observation. The AgglomerativeClustering algorithm
            will merge the pairs of cluster that minimize the linkage criterion.
            Possible types:
        set_as_value : bool
            If True then set the result as the value of the clustering object
        section : str
            What part of the chain to take the samples from
        
        Returns
        -------
        np.ndarray(size=(len(items), ), dtype=int)
            Each value is the cluster assignment for index i
        '''
        trace = self.n_clusters.get_trace_from_disk(section=section)
        if callable(n_clusters):
            n = n_clusters(trace)
        elif type(n_clusters) == int:
            n = n_clusters
        elif type(n_clusters) == str:
            if n_clusters == 'mode':
                n = scipy.stats.mode(trace)[0][0]
            elif n_clusters == 'mean':
                n = np.mean(trace)
            elif n_clusters == 'median':
                n = np.median(trace)
            else:
                raise ValueError('`n_clusters` ({}) not recognized. Valid inputs are ' \
                    '`mode`, `mean`, and `median`.'.format(n_clusters))
        else:
            raise ValueError('Type `n_clusters` ({}) not recognized. Must be of '\
                'type `str`, `int`, or callable.'.format(type(n_clusters)))
        if not util.isbool(set_as_value):
            raise TypeError('`set_as_value` ({}) must be a bool'.format(type(set_as_value)))

        A = summary(self.coclusters, section=section)['mean']
        A = 1 - A
        logging.info('Number of clusters: {}'.format(int(n)))
        c = AgglomerativeClustering(
            n_clusters=int(n),
            affinity='precomputed',
            linkage=linkage)
        ret = c.fit_predict(A)
        logging.info(ret)
        if set_as_value:
            ca = {}
            for idx, cidx in enumerate(ca):
                if cidx in ca:
                    ca[cidx].append(idx)
                else:
                    ca[cidx] = [idx]
            for cluster in ca:
                cid = self.make_new_cluster_with(idx=cluster[0])
                for oidx in cluster[1:]:
                    self.move_item(idx=oidx, cid=cid)
        return ret

    def set_trace(self, *args, **kwargs):
        self.coclusters.set_trace(*args, **kwargs)
        self.n_clusters.set_trace(*args, **kwargs)

    def add_trace(self):
        self.coclusters.value = self.generate_coclusters()
        self.coclusters.add_trace()
        self.n_clusters.add_trace()


class _Cluster:
    '''Class for a single cluster.

    Parameters
    ----------
    members : array_like, set
        - Individual item indices that are within the cluster
    parent : pylab.cluster._ClusterMap
        - pointer to parent _ClusterMap
    '''
    def __init__(self, members, parent):
        if type(members) == np.ndarray:
            members = np.squeeze(members).flatten().tolist()
        if type(members) == list:
            members = set(members)
        self.members = members
        self.id = id(self) # Unique id for class
        self.size = len(members)
        self.parent = parent
                
    def __str__(self):
        return 'Cluster {}\n' \
            '\tmembers: {}\n' \
            '\tsize: {}'.format(
                self.id,
                [self.parent.items[idx].cluster_str() for idx in self.members],
                self.size)

    def __contains__(self, item):
        '''For the `in` operator
        '''
        return item in self.members

    def __len__(self):
        return self.size

    def __iter__(self):
        '''Let c = ClusterBase object.
        Let c[i] (c.clusters[i]) be this object.
        This method is useful for doing the command:
            `for idx in c[i]`
        That call iterates over every member that this cluster contains
        '''
        for item in self.members:
            yield item

    def add(self, item):
        '''Add the item `item` to the cluster

        Paramters
        ---------
        item : int
            This is the index of the item that we want to add
        '''
        if item in self.members:
            return True
        self.members.add(item)
        self.size += 1
        return True

    def remove(self, item):
        '''Returns True if `item` was deleted from the cluster. Returns False
        if `item` was not in the cluster so it could not be deleted.

        Paramters
        ---------
        item : int
            This is the index of the item that we want to remove
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
        self._d = {}
        self.signal_when_clusters_change = []
        self.signal_when_item_assignment_changes = []
        self._keys = []

    def __len__(self):
        return len(self._keys)

    def __getitem__(self, key):
        return self._d[key]

    def __iter__(self):
        for a in self._keys:
            yield self._d[a]

    def add(self, prop):
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

    def toarray(self):
        '''Make a list of the ClusterProperty's.
        '''
        return [self._d[key] for key in self._keys]

    def keys(self):
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
    def __init__(self, clustering, signal_when_clusters_change,
        signal_when_item_assignment_changes):
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
        raise NeedToImplementError('User needs to implement this function')

    def clusters_changed(self, cids_added, cids_removed):
        raise NeedToImplementError('User needs to implement this function')

    def set_signal_when_clusters_change(self, value):
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

    def set_signal_when_item_assignment_changes(self, value):
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

    Paramters
    ---------
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
    def __init__(self, clustering, signal_when_clusters_change,
        signal_when_item_assignment_changes, dtype=None, **kwargs):
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

    def item_array(self):
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

    def cluster_array(self):
        '''Converts the dictionary into a cluster array in the order of the clusters

        Returns
        -------
        np.ndarray((n,), dtype=self.dtype)
            Array of the values expanded for each cluster in the overall cluster order
        '''
        return np.asarray([self.value[cid] for cid in self.clustering.order], dtype=self.dtype)

    def add_init_value(self):
        '''Set the initialization value. This is called by `pylab.inference.BaseMCMC.run`
        when first updating the variable. User should not use this function
        '''
        self._init_value = self.item_array()

    def set_values_from_array(self, values):
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
        tracer = self.G.tracer
        tracer.set_trace(
            self.name, 
            shape=(len(self.clustering.items), ), 
            dtype=self.dtype)
        self.ckpt_iter = 0
        self.sample_iter = 0
        self.trace = np.full(shape=(tracer.ckpt, len(self.clustering.items)),
            dtype=self.dtype, fill_value=np.nan)

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
            self.trace = np.full(shape=(self.G.tracer.ckpt, len(self.clustering.items)),
                dtype=self.dtype, fill_value=np.nan)
            self.ckpt_iter = 0


@numba.jit(nopython=True) #, fastmath=True, cache=True)
def _generate_coclusters_fast(idx2cid):
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
def toarray_from_cocluster(coclusters):
    '''Generate the output that would be given from 
    `clustering.toarray` from the cocluster matrix.

    Numba is about 10X faster.

    Example:
        coclusters = 
            [[1,0,0,1],
             [0,1,0,0],
             [0,0,1,0],
             [1,0,0,1]]
        >>> toarray_from_coclusters(coclusters)

        [[0,3], [1], [2]]

    Parameters
    ----------
    coclusters : 2-dim square np.ndarray
        Cocluster matrix
    
    Returns
    -------
    list(list(int))
        Returns a list of list of ints that correspond to the clusters
    '''
    a = np.full(coclusters.shape[0], -1)
    i = 0
    for j in range(coclusters.shape[0]):
        if a[j] == -1:
            for k in range(j,coclusters.shape[0]):
                if coclusters[j,k] == 1:
                    a[k] = i
            i += 1
    ret = [list(np.where(a == m)[0]) for m in range(np.max(a)+1)]
    return ret

def toarray_vec_from_coclusters(coclusters):
    '''Generate the output that would be given from 
    `clustering.toarray` from the cocluster matrix.

    Numba is about 10X faster.

    Example:
        coclusters = 
            [[1,0,0,1],
             [0,1,0,0],
             [0,0,1,0],
             [1,0,0,1]]
        >>> toarray_from_coclusters(coclusters)
        [0,1,2,0]

    Parameters
    ----------
    coclusters : 2-dim square np.ndarray
        Cocluster matrix
    
    Returns
    -------
    np.ndarray
        The index of the array is the item index, the value of the array is the cluster index
    '''

    '''Converts clusters into array format

    array = [cidx(idx1), cidx(idx2), ..., cidx(idxM)]
    This is the format for sklearn and scikit

        Returns
        -------
        np.ndarray
    '''
    ret = np.zeros(coclusters.shape[0], dtype=int)
    toarray = toarray_from_cocluster(coclusters)

    for cidx, cluster in toarray:
        for idx in cluster:
            ret[idx] = cidx
    return ret