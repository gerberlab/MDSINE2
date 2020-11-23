'''These are classes that should be apart of pylab but need to be built
from many different modules. To get dependency structure right, all of these
extra, bigger classes are built in here. No other modules can
depend on contrib
'''
import numpy as np
import numpy.random as npr
import sys
import logging
import scipy.special

from .base import BasePerturbation, Traceable
from . import variables
from .cluster import isclustervalue, ClusterValue, isclustering, \
    ClusterProperty
from .graph import Node
from . import util, base

# Constants
DEFAULT_SIGNAL_WHEN_CLUSTERS_CHANGE = False
DEFAULT_SIGNAL_WHEN_ITEM_ASSIGNMENT_CHANGES = False
DEFAULT_MAGNITUDE_SUFFIX = '_magnitude'
DEFAULT_PROBABILITY_SUFFIX = '_probability'
DEFAULT_INDICATOR_SUFFIX = '_indicator'

def isclusterperturbation(x):
    '''Checks whether the input is a subclass of ClusterPerturbation

    Parameters
    ----------
    x : any
        Input instance to check the type of ClusterPerturbation
    
    Returns
    -------
    bool
        True if `x` is of type ClusterPerturbation, else False
    '''
    return x is not None and issubclass(x.__class__, ClusterPerturbation)

def isclusterperturbationindicator(x):
    '''Checks whether the input is a subclass of ClusterPerturbationIndicator

    Parameters
    ----------
    x : any
        Input instance to check the type of ClusterPerturbationIndicator
    
    Returns
    -------
    bool
        True if `x` is of type ClusterPerturbationIndicator, else False
    '''
    return x is not None and issubclass(x.__class__, ClusterPerturbationIndicator)

def isinteractions(x):
    '''Type check if `x` is a subclass of Interactions

    Parameters
    ----------
    x : any
        Returns True if `x` is a subclass of Interactions
    
    Returns
    -------
    bool
        True if `x` is the correct subtype
    '''
    return x is not None and issubclass(x.__class__, Interactions)

class Perturbation(BasePerturbation, variables.Variable):
    '''This is an implementation of a perturbation where the 
    values *DO NOT* depend on clusters.

    If you want to compute the bayes factors for each item, you can 
    calculate ~np.isnan for the trace. This will give an indicator 
    array that you can then use to calculate the bayes factor.

    Parameters
    ----------
    starts, ends : dict
        Start and end of the perturbation for each subject
    asvs : pylab.base.ASVSet
        ASVSet of asvs
    magnitude : pylab.variables.Variable, int/float, array, Optional
        If a pylab.variables.Variable is passed in it will create one
        with the value indicated. Defualt value is None
    indicator : pylab.variables.Variable, array, Optional
        This is the indicator of the perturbation. Default value is False 
        for every asv
    probability : pylab.variables.Variable, float, Optional
        This is the probability that the perturbation affects an ASV, e.g.
          probability = 0.7, there's a 70% chance that the perturbation afffects 
          each asv
    kwargs : dict
        - Extra arguments for the Node class
    '''
    def __init__(self, asvs, starts, ends, magnitude=None, indicator=None, 
        probability=None, **kwargs):
        variables.Variable.__init__(self, **kwargs)
        if self.G.perturbations is None:
            self.G.perturbations = []
        BasePerturbation.__init__(self, starts=starts, ends=ends, name=self.name)
        
        if not base.isasvset(asvs):
            raise TypeError('`asvs` ({}) must be pylab.base.ASVSet'.format(type(asvs)))

        if self.G.perturbations is None:
            self.G.perturbations = []
        self.G.perturbations.append(self)
        self.asvs = asvs
        n_asvs = len(self.asvs)
        self.set_value_shape(shape=(n_asvs, ))

        # Set magnitude
        if magnitude is None:
            magnitude = np.full(n_asvs, 0)
        if util.isnumeric(magnitude):
            magnitude = np.full(n_asvs, magnitude)
        if util.isarray(magnitude):
            if len(magnitude) != n_asvs:
                raise ValueError('`magnitue` ({}) must have length {}'.format(
                    len(magnitude), n_asvs))
            magnitude = variables.Variable(
                G=self.G, dtype=float, name=self.name+DEFAULT_MAGNITUDE_SUFFIX, 
                value=magnitude)
            magnitude.set_value_shape((n_asvs,))
        elif variables.isVariable(magnitude):
            if len(magnitude) != n_asvs:
                raise ValueError('`magnitue` ({}) must have length {}'.format(magnitude, n_asvs))
        else:
            raise TypeError('`magnitude` ({}) type not recognized'.format(type(magnitude)))
              
        # Set probability
        if not variables.isVariable(probability):
            if not util.isfloat(probability) and probability is not None:
                raise ValueError('`probability` ({}) must be a pylab.variables.Variable' \
                    ', a float, or None'.format(type(probability)))
            if util.isfloat(probability):
                if probability < 0 or probability > 1:
                    raise ValueError('`probability` ({}) must be in [0,1]'.format(
                        probability))
            probability = variables.Variable(value=probability, G=self.G, dtype=float,
                name=self.name+DEFAULT_PROBABILITY_SUFFIX)

        # Set indicator
        if indicator is None:
            indicator = np.full(n_asvs, False, dtype=bool)
        if util.isbool(indicator):
            indicator = np.full(n_asvs, indicator, dtype=bool)
        if util.isarray(indicator):
            if len(indicator) != n_asvs:
                raise ValueError('`indicator` ({}) must have length {}'.format(
                    len(indicator), n_asvs))
            indicator = variables.Variable(
                G=self.G, dtype=bool, name=self.name+DEFAULT_INDICATOR_SUFFIX, 
                value=indicator)
            indicator.set_value_shape((n_asvs,))
        elif variables.isVariable(indicator):
            if len(indicator) != n_asvs:
                raise ValueError('`magnitue` ({}) must have length {}'.format(indicator, n_asvs))
        else:
            raise TypeError('`indicator` ({}) type not recognized'.format(type(indicator)))

        self.magnitude = magnitude
        self.indicator = indicator
        self.probability = probability

    def __str__(self):
        s = BasePerturbation.__str__(self)
        s += '\nMagnitude:\n'
        for oidx in range(len(self.asvs)):
            s += '\t{}: {}\n'.format(oidx, self.magnitude.value[oidx])
        s += 'Indicator:\n'
        for oidx in range(len(self.asvs)):
            s += '\t{}: {}\n'.format(oidx, self.indicator.value[oidx])
        s += 'Probability: {}'.format(self.probability.value)
        return s

    def add_trace(self):
        '''Set the negative indicators as np.nan
        '''
        self.value = np.full(len(self.asvs), np.nan)
        ind = self.indicator.value
        self.value[ind] = self.magnitude.value[ind]
        variables.Variable.add_trace(self)

    def remove_local_trace(self):
        '''Delete the local trace
        '''
        self.trace = None

    def array(self, only_pos_ind=False):
        '''Return the magnitudes with the indicatrs indexed out

        Parameters
        ----------
        only_pos_ind : bool
            If this is True, then it will return only for when the indicator is positive

        Returns
        -------
        np.ndarray((n_c,), dtype=float)
            Array of the cluster perturbation values for each cluster
        '''
        ind = self.indicator.value
        if only_pos_ind:
            val = self.magnitude.value[ind]
        else:
            val = np.zeros(len(self.asvs))
            val[ind] = self.magnitude.value[ind]
        return val


class ClusterPerturbationValue(ClusterValue):
    '''Extends `pylab.cluster.ClusterValue` object so it works for reset and cluster
    changed
    '''
    def reset(self):
        self.value = {}
        for cid in self.clustering.order:
            self.value[cid] = 0

    def clusters_changed(self, cids_added, cids_removed):
        '''Delete old clusters, sample from `prior` for the
        new clusters. We do not need to type check because it
        is self contained within pylab

        Parameters
        ----------
        cids_added (list(int))
            - These are a list of cluster ids to add
        cids_removed (list(int))
            - These are the cids that were removed
        '''
        for cid in cids_removed:
            self.value.pop(cid)
        for cid in cids_added:
            self.value[cid] = self.prior.sample()

    def remove_local_trace(self):
        '''Delete the local trace
        '''
        self.trace = None


class ClusterPerturbationIndicator(ClusterValue):
    '''Extends the `pylab.cluster.ClusterValue` object so that it works for being a 
    cluster perturbation indicator
    
    Implements the `clusters_changed` function required by a ClusterProperty
    and provides a direct pointer to the probability object

    Parameters
    ----------
    probability : pl.variables.Variable
        This is the variable object that holds the probability for a positive indicator
    kwargs : dict
        These are additional arguments for ClusterValue
    '''
    def __init__(self, probability, **kwargs):
        ClusterValue.__init__(self, dtype=bool, **kwargs)
        self.probability = probability

    def reset(self):
        self.value = {}
        for cid in self.clustering.order:
            self.value[cid] = False

    def remove_local_trace(self):
        '''Delete the local trace
        '''
        self.trace = None

    def clusters_changed(self, cids_added, cids_removed):
        '''Delete old clusters, sample from `probability` for the
        new clusters. We do not need to type check because it
        is self contained within pylab

        Parameters
        ----------
        cids_added (list(int))
            - These are a list of cluster ids to add
        cids_removed (list(int))
            - These are the cids that were removed
        '''
        for cid in cids_removed:
            self.value.pop(cid)
        for cid in cids_added:
            self.value[cid] = bool(npr.binomial(
                n=1,
                p=self.probability.value))

    def item_bool_array(self):
        '''Creates a boolean array expanded so that each item has the same 
        value that the cluster that contains it has. This is the same as 
        calling ClusterValue.item_array
        Example
            cluster1 = {0,2,4}
            cluster2 = {1,3}
            value[cluster1] = True
            value[cluster2] = False

            >>> np.ndarray([True, False, True, False, True])

        Returns
        -------
        np.ndarray((n,), dtype=bool)
            A numpy bool array for each item
        '''
        return ClusterValue.item_array(self)

    def cluster_bool_array(self):
        '''Creates a boolean array for each item in cluster order. This 
        is the same as calling ClusterValue.cluster_array()

        Example:
            cluster1 = {0,2,4}
            cluster2 = {1,3}
            value[cluster1] = True
            value[cluster2] = False

            >>> np.ndarray([True, False])
        
        Returns
        -------
        np.ndarray((n,), dtype=bool)
            A numpy bool array for each cluster
        '''
        return ClusterValue.cluster_array(self)

    def item_arg_array(self):
        '''Creates an ordered index array of items that are positive.
        Example:
            cluster1 = {0,2,4}
            cluster2 = {1,3}
            value[cluster1] = True
            value[cluster2] = False

            >>> np.ndarray([0,2,4])
        Returns
        -------
        np.ndarray((n_c,), dtype=int)
            A numpy index array for each item
        '''
        val = []
        for cluster in self.clustering:
            if self.value[cluster.id]:
                val += list(cluster.members)
        return np.asarray(val, dtype=int)

    def cluster_arg_array(self):
        '''Creates an ordered index array of clusters that are positive.
        Example:
            cluster1 = {0,2,4}
            cluster2 = {1,3}
            value[cluster1] = True
            value[cluster2] = False

            >>> np.ndarray([0])
        Returns
        -------
        np.ndarray((n_c,), dtype=int)
            A numpy index array for each cluster
        '''

        return np.asarray([idx for idx,cid in enumerate(self.clustering.order) \
            if self.value[cid]], dtype=int)

    def num_on_items(self):
        '''These are the number of on items for this perturbation
        Example:
            cluster1 = {0,2,4}
            cluster2 = {1,3}
            value[cluster1] = True
            value[cluster2] = False

            >>> 3

        Returns
        -------
        int
            This is the number of postiive items
        '''
        try:
            cumm = 0
            for cluster in self.clustering:
                if self.value[cluster.id]:
                    cumm += cluster.size
            return int(cumm)
        except:
            logging.critical('Inner cluster ids:\n{}'.format(list(self.value.keys())))
            logging.critical('Clustering cluster ids:\n{}'.format(self.clustering.order))
            raise

    def num_on_clusters(self):
        '''These are the number of on clusters for this perturbation
        Example:
            cluster1 = {0,2,4}
            cluster2 = {1,3}
            value[cluster1] = True
            value[cluster2] = False

            >>> 1

        Returns
        -------
        int
            This is the number of postiive clusters
        '''
        cumm = 0
        for cid in self.clustering.order:
            cumm += self.value[cid]
        return cumm

    def get_clusters_on(self):
        '''Return the cluster IDs that have a positive indicator for this
        perturbation.

        Returns
        -------
        list
        '''
        ret = [cid for cid in self.clustering.order if self.value[cid]]
        return ret

    def get_items_on(self):
        '''Get the item indecies that have a positive indicator for this
        perturbation.

        Returns
        -------
        list
        '''
        ret = []
        for cid in self.clustering.order:
            if self.value[cid]:
                ret += list(self.clustering.clusters[cid].members)
        return ret


class ClusterPerturbation(BasePerturbation, variables.Variable):
    '''This is an basic implementation for a perturbation where the 
    values **DO** depend on clusters. We trace the values at the item level.
    Effectively the same as `pylab.contrib.Perturbation` but it si extended
    to deal with clusters.

    If you want to compute the bayes factors for each item, you can 
    calculate ~np.isnan for the trace. This will give an indicator 
    array that you can then use to calculate the bayes factor.
    
    Parameters
    ----------
    starts, ends : int, float
        - Start and end of the perturbation
    clustering : pylab.cluster.Clustering
        - This is the clustering object it is being set with
    magnitude : pylab.variables.Variable, pylab.cluster.ClusterValue, int/float, array, Optional
        - If a pylab.variables.Variable is passed in it will create one
          with the value indicated. Defualt value is None
    indicator : pylab.cluster.ClusterValue, array, Optional
        - This is the indicator of the interaction (vector, an indicator
          for every cluster). Default value is False for every cluster
    probability : pylab.variables.Variable, float, Optional
        - This is the probability that the perturbation affects a cluster, e.g.
          probability = 0.7, there's a 70% chance that the perturbation afffects 
          each cluster
    kwargs : dict
        - Extra arguments for the Node class
    '''
    def __init__(self, clustering, starts, ends,
        magnitude=None, indicator=None, probability=None,
        signal_when_clusters_change=False,
        signal_when_item_assignment_changes=False, **kwargs):

        if signal_when_clusters_change is None:
            signal_when_clusters_change = DEFAULT_SIGNAL_WHEN_CLUSTERS_CHANGE
        if signal_when_item_assignment_changes is None:
            signal_when_item_assignment_changes = DEFAULT_SIGNAL_WHEN_ITEM_ASSIGNMENT_CHANGES
        if not isclustering(clustering):
            raise TypeError('`clustering` ({}) must be a pylab.cluster.Clustering object'.format(
                type(clustering)))
        
        variables.Variable.__init__(self, **kwargs)
        if self.G.perturbations is None:
            self.G.perturbations = []

        BasePerturbation.__init__(self, starts=starts, ends=ends, name=self.name)
        self.G.perturbations.append(self)
        self.clustering = clustering
        self.set_value_shape(shape=(len(self.clustering.items), ))
        if magnitude is not None:
            if util.isarray(magnitude):
                temp = ClusterPerturbationValue(clustering=clustering, 
                    G=self.G, dtype=float, name=self.name+DEFAULT_MAGNITUDE_SUFFIX,
                    signal_when_clusters_change=signal_when_clusters_change,
                    signal_when_item_assignment_changes=signal_when_item_assignment_changes)
                temp.set_values_from_array(magnitude)
                magnitude=temp
            elif not isclustervalue(magnitude):
                raise TypeError('`magnitude` ({})' \
                    ' must be an array or a pylab.cluster.ClusterValue'.format(type(magnitude)))
        else:
            magnitude = ClusterPerturbationValue(clustering=clustering, 
                G=self.G, dtype=float, name=self.name+DEFAULT_MAGNITUDE_SUFFIX,
                signal_when_clusters_change=signal_when_clusters_change,
                signal_when_item_assignment_changes=signal_when_item_assignment_changes)            

        if not variables.isVariable(probability):
            if not util.isfloat(probability) and probability is not None:
                raise ValueError('`probability` ({}) must be a pylab.variables.Variable' \
                    ', a float, or None'.format(type(probability)))
            if util.isfloat(probability):
                if probability < 0 or probability > 1:
                    raise ValueError('`probability` ({}) must be in [0,1]'.format(
                        probability))
            probability = variables.Variable(value=probability, G=self.G, dtype=float,
                name=self.name+DEFAULT_PROBABILITY_SUFFIX)
        if not isclustervalue(indicator):
            if not util.isarray(indicator) and indicator is not None:
                raise ValueError('`indicator` ({}) must be a pylab.cluster.ClusterValue,' \
                    ' array, or None'.format(type(indicator)))
            if util.isarray(indicator):
                if len(indicator) != len(self.clustering):
                    raise ValueError('If `indicator` ({}) is an array, it must have the ' \
                        'same number of elements as number of clusters ({})'.format(
                            len(indicator), len(self.clustering)))
            ind = ClusterPerturbationIndicator(
                G=self.G, 
                name=self.name+DEFAULT_INDICATOR_SUFFIX,
                clustering=clustering,
                probability=probability,
                signal_when_clusters_change=signal_when_clusters_change,
                signal_when_item_assignment_changes=signal_when_item_assignment_changes)
            if util.isarray(indicator):
                ind.set_values_from_array(indicator)
            else:
                for cid in ind.clustering.order:
                    ind.value[cid] = False
            indicator = ind

        self.magnitude = magnitude
        self.indicator = indicator
        self.probability = probability

    def __str__(self):
        s = BasePerturbation.__str__(self)
        s += '\nMagnitude:\n'
        for cid in self.clustering.order:
            s += '\t{}: {}\n'.format(cid, self.magnitude.value[cid])
        s += 'Indicator:\n'
        for cid in self.clustering.order:
            s += '\t{}: {}\n'.format(cid, self.indicator.value[cid])
        s += 'Probability: {}'.format(self.probability.value)
        return s

    def item_array(self, only_pos_ind=False):
        '''Expands the condensed form into a variable for each item
        in the data.

        Example
            ** Cluster assignments **
            cluster1 = {0,2,4}
            cluster2 = {1,3}

            ** Cluster indicators for perturbation **
            value[cluster1] = True
            value[cluster2] = False

            magnitude.value = -0.5

            >>> np.ndarray([-0.5, 0, -0.5, 0, -0.5])
        
        Parameters
        ----------
        only_pos_ind : bool
            If this is True, then it will return only for when the indicator is positive

        Returns
        -------
        np.ndarray((n_c,), dtype=float)
            Array of the cluster perturbation values for each cluster
        '''
        ind = self.indicator.item_bool_array()
        if only_pos_ind:
            val = self.magnitude.item_array()[ind]
        else:
            val = np.zeros(len(self.clustering.items))
            val[ind] = self.magnitude.item_array()[ind]
        return val

    def cluster_array(self, only_pos_ind=False):
        '''Make an array for each cluster with the magnitude

        Example
            ** Cluster assignments **
            cluster1 = {0,2,4}
            cluster2 = {1,3}

            ** Cluster indicators for perturbation **
            value[cluster1] = True
            value[cluster2] = False

            magnitude.value = -0.5

            >>> np.ndarray([-0.5, 0])
        
        Parameters
        ----------
        only_pos_ind : bool
            If this is True, then it will return only for when the indicator is positive

        Returns
        -------
        np.ndarray((n_c,), dtype=float)
            Array of the cluster perturbation values for each cluster
        '''
        ind = self.indicator.cluster_bool_array()
        if only_pos_ind:
            val = self.magnitude.cluster_array()[ind]
        else:
            val = np.zeros(len(self.clustering))
            val[ind] = self.magnitude.cluster_array()[ind]
        return val
    
    def add_trace(self):
        '''Set the negative indicators as np.nan
        '''
        self.value = np.full(len(self.clustering.items), np.nan)
        ind = self.indicator.item_arg_array()
        self.value[ind] = self.magnitude.item_array()[ind]
        variables.Variable.add_trace(self)

    def remove_local_trace(self):
        '''Delete the local trace
        '''
        self.trace = None

    def set_values_from_array(self, values, use_indicators=True):
        '''Sets the values from an array of the same order as the clusters.

        Paramters
        ---------
        values : array_like
            An array of the values
            Must be the same length as the number of clusters is `use_indicators` 
            is False. If `use_indicators` is True, then the length must correspond
            to how many positive indicators there are.
        use_indicators : bool
            If True, the values  only correspond to positive interactions. Else
            the values correspond to every cluster
        '''
        if not util.isarray(values):
            raise ValueError('`values` ({}) must be an array'.format(type(values)))
        if not util.isbool(use_indicators):
            raise TypeError('`use_indicators` ({}) must be a bool'.format(
                type(use_indicators)))
        if not use_indicators:
            # Checking is done within this function
            self.magnitude.set_values_from_array(values)
        else:
            if len(values) != self.indicator.num_on_clusters():
                raise ValueError('The length of the array ({}) does not correspond' \
                    ' to how many on indicators there are ({})'.format(
                        len(values), self.indicator.num_on_clusters()))
            i = 0
            for cid in self.clustering.order:
                if self.indicator.value[cid]:
                    self.magnitude.value[cid] = values[i]
                    i += 1


class Interactions(ClusterProperty, Node, Traceable):
    '''This is a basic class for interactions between clusters.

    This is a 2D dictionary. The first level of the dictionary indexes the target 
    cluster and the second level indexes the source cluster. You can make this 2 
    layer dictionary into a matrix or a vector with functions defined in this class. 
    The reason why the data is stored in a 2D dictionary is because the number of 
    clusters changes constantly, so inserting and deleting values in a dictionary is more
    efficient than using a matrix/pandas.DataFrame. Additionally, the order of the clusters
    are changing constantly. Having them as a 2D dictionary allows us to reference the 
    interactions in the same order as the clusters as they are defined in `clusters`.

    Tracing
    -------
    The interactions get traced on an item-item bases. In this class we assume there are no
    interactions within a cluster. If the indicator is False, we set the trace to np.nan.
    To get the indicators call `np.nan_to_num` on the trace. We do not trace the indicators 
    separately, but you can get the trace of the interactions by calling ~np.isnan(self.trace).

    Indicators
    ----------
    You can choose whether to use or not use indicators for the interactions. If you choose
    not to use the indicators, then we assume that every indicator is positive.

    Iterators
    ---------
    These interactions assume the following order during iterating:
        For target cluster in clusters:
            For source cluster in clusters:
                if they are the same cluster, skip
                else yield value[target][source]
    Clusters are ordered in the same way as clustering.

    Value and indicator initialization
    ----------------------------------
    The values and indicators for a new interaction that gets made need to be initialized to 
    a value so we use the parameters `value_initializer` and  `inidicator_initializer`. If they
    are not specified then we return `np.nan` During initialization of the inference these are 
    usually set to the priors of the variables.

    The initializer for the indicator is assumed to either return a `bool` or a float betwen 
    [0,1]. It will set it to true if the sampled value is >= 0.5. There is no checking for 
    this though.

    Parameters
    ----------
    clustering : Clustering
        Clustering object
    use_indicators : bool
        If True, use indicators. If False do not use indicators (automatically sets all indicators)
        to True
    value_initializer : callable, None
        This is the function that initializes the value of the `value` attribute for an interaction. 
        During MCMC you could set this to the sample method of the prior. 
        Defaults always returning np.nan.
    indicator_initializer : callable, None
        This is the function that initializes the value of the `value` attribute for an interaction. 
        During MCMC you could set this to the sample method of the prior.         
        Defaults to always returning True
    '''
    def __init__(self, clustering, use_indicators, 
        value_initializer=None, indicator_initializer=None, 
        signal_when_clusters_change=True, **kwargs):

        Node.__init__(self, **kwargs)
        ClusterProperty.__init__(self, clustering=clustering, 
            signal_when_clusters_change=signal_when_clusters_change, 
            signal_when_item_assignment_changes=False)

        if value_initializer is None:
            value_initializer = _always_return_nan
        if indicator_initializer is None:
            indicator_initializer = _always_return_nan
        if not np.all(util.itercheck([value_initializer, indicator_initializer], callable)):
            raise TypeError('`value_initializer` ({}) and `indicator_initializer` ({}) ' \
                'must be callable'.format(type(value_initializer), 
                type(indicator_initializer)))
        self.value_initializer = value_initializer
        self.indicator_initializer = indicator_initializer
        
        if not util.isbool(use_indicators):
            raise TypeError('`use_indicators` ({}) must be a bool'.format(type(use_indicators)))
        self.use_indicators = use_indicators
        if not self.use_indicators:
            self.indicator_initializer = _always_return_true

        order = self.clustering.order
        self.value = {}
        for tcid in order:
            self.value[tcid] = {}
            for scid in order:
                if tcid == scid:
                    continue
                self.value[tcid][scid] = _Interaction( 
                    source_cid=scid, target_cid=tcid,
                    value=self.value_initializer(),
                    indicator=self.indicator_initializer())

        self._shape = (len(self.clustering.items), len(self.clustering.items))
        self.dtype = float

    def __getitem__(self, key):
        return self.value[key]

    def __setitem__(self, key, val):
        self.value[key] = val

    def __iter__(self):
        '''Iterates over the interactions in order
        '''
        order = self.clustering.order
        for tcid in order:
            temp = self.value[tcid] # Faster pointer
            for scid in order:
                if tcid != scid:
                    yield temp[scid]

    def __str__(self):
        s=''
        for interaction in self:
            s += str(interaction) + '\n'
        return s
    
    @property
    def size(self):
        '''Return how many interactions there are possible according to the number of clusters.
        THIS IS NOT HOW MANY POSITIVE INTERACTIONS THERE ARE - USE `num_pos_interactions`

        Returns
        -------
        int
        '''
        n_clusters = len(self.clustering)
        return n_clusters * (n_clusters - 1)

    def iter_valid(self):
        '''Iterate only over the positive indicators
        '''
        order = self.clustering.order
        for tcid in order:
            temp = self.value[tcid] # Faster pointer
            for scid in order:
                if tcid != scid:
                    if temp[scid].indicator:
                        yield temp[scid]

    def iter_valid_pairs(self):
        '''Iterate only over the positive indicators
        '''
        order = self.clustering.order
        for tcid in order:
            temp = self.value[tcid] # Faster pointer
            for scid in order:
                if tcid != scid:
                    if temp[scid].indicator:
                        yield tcid, scid

    def iter_to_target(self, cid, only_valid=False):
        '''Iterates over interactions to the target cluster from all
        source clusters in the order specified by clusters

        Paramters
        ---------
        cid : int
            This is the target cluster id we are iterating from
        only_valid : bool
            If True, only returns the interactions with a positive indicator
        '''
        order = self.clustering.order
        temp = self.value[cid] # For quicker pointer
        if only_valid:
            for scid in order:
                if scid != cid:
                    if temp[scid].indicator:
                        yield temp[scid]
        else:
            for scid in order:
                if scid != cid:
                    yield temp[scid]

    def iter_from_source(self, cid, only_valid=False):
        '''Iterates over interactions from the source cluster to all
        target clusters in the order specified by clusters

        Paramters
        ---------
        cid (int)
            This is the source cluster id
        only_valid : bool
            If True, only returns the interactions with a positive indicator
        '''
        order = self.clustering.order
        if only_valid:
            for tcid in order:
                if tcid != cid:
                    if self.value[tcid][cid].indicator:
                        yield self.value[tcid][cid]
        else:
            for tcid in order:
                if tcid != cid:
                    yield self.value[tcid][cid]

    def reset(self):
        self.value = {}
        for tcid in self.clustering.order:
            self.value[tcid] = {}
            for scid in self.clustering.order:
                if tcid == scid:
                    continue
                self.value[tcid][scid] = _Interaction(
                    source_cid=scid, target_cid=tcid,
                    value=self.value_initializer(),
                    indicator=self.indicator_initializer()>=.5)

    def iloc(self, idx):
        '''Get the interaction as a function of the index that it occurs at.
        Reverse indexing is allowed.

        Parameters
        ----------
        idx : int
            This is the index that the interaction occurs at

        Returns
        -------
        pylab.contrib._Interaction
        '''
        if not util.isint(idx):
            raise TypeError('`idx` ({}) must be an int'.format(idx))
        if idx >= self.size:
            raise ValueError('`idx` ({}) cannot be >= the number of interactions ({})'.format(
                idx, self.size))
        if idx < 0:
            idx = self.size - idx
        tcidx = idx // (len(self.clustering) - 1)
        scidx = idx - tcidx * (len(self.clustering) - 1)
        if scidx >= tcidx:
            scidx += 1
        return self.value[
            self.clustering.order[tcidx]][
            self.clustering.order[scidx]]

    def clusters_changed(self, cids_added, cids_removed):
        '''Remove all of the interactions to and from the clusters
        in `cids_removed` and make interactions for the `cid_added`
        '''
        # Remove interactions from clusters deleted
        if len(cids_removed) > 0:
            for cid in cids_removed:
                self.value.pop(cid, None)
            for cid in self.value.keys():
                for cid_del in cids_removed:
                    self.value[cid].pop(cid_del, None)
        if len(cids_added) > 0:
            for cid in cids_added:
                self._add_single_cluster(cid)
        
    def _add_single_cluster(self, cid):
        other_cids = self.value.keys()
        # Add the interaction from clusters already there and 
        # the new cluster
        for ocid in other_cids:
            self.value[ocid][cid] = _Interaction(
                source_cid=cid, target_cid=ocid,
                value=self.value_initializer(),
                indicator=self.indicator_initializer() >= 0.5)
        self.value[cid] = {}
        for ocid in other_cids:
            self.value[cid][ocid] = _Interaction(
                source_cid=ocid, target_cid=cid,
                value=self.value_initializer(),
                indicator=self.indicator_initializer() >= 0.5)
    
    def key_pairs(self, only_valid=False):
        '''Returns (target,source) cluster ids in order

        Parameters
        ----------
        only_valid : bool
            If True, it will only return the key pairs that have a positive indicator.
            Else it will return all of the interactions regardless of the indicator.

        Returns 
        -------
        list((int,int))
            Return a list of the (target, source) cluster IDs for each interaciton
            in order.
        '''
        order = self.clustering.order
        l = []
        for tcid in order:
            for scid in order:
                if tcid != scid:
                    if only_valid:
                        if self.value[tcid][scid].indicator:
                            l.append((tcid, scid))
                    else:
                        l.append((tcid,scid))
        return l

    def num_neg_indicators(self, target_cid=None):
        '''Return the number of indicator variables that are 0

        If target_cid is not None, calculate them for only the interactions going into
        that cluster

        Paramters
        ---------
        target_cid : int, Optional
            If this is specified, get only the negative indicators going into the cluster 
            `target_cid`
        '''
        cumm = 0
        if target_cid is not None:
            for interaction in self.iter_to_target(target_cid):
                cumm += not interaction.indicator
        else:
            for interaction in self:
                cumm += not interaction.indicator
        return cumm

    def num_pos_indicators(self, target_cid=None):
        '''Return the number of indicator variables that are 1

        If target_cid is not None, calculate them for only the interactions going into
        that cluster

        Paramters
        ---------
        target_cid : int, Optional
            If this is specified, get only the positive indicators going into the cluster 
            `target_cid`
        '''
        cumm = 0
        if target_cid is not None:
            for interaction in self.iter_to_target(target_cid):
                cumm += interaction.indicator
        else:
            for interaction in self:
                cumm += interaction.indicator
        return cumm

    def get_arg_indicators(self, target_cid=None, source_cid=None):
        '''Get the positive indicators as indices, in order -> same convention
        as `get_indicators`.

        If `target_cid` is specfied, it will get all of the positive indicator indicies 
        going to the target cluster `target_cid` in order. If `source_cid` is specified,
        then it will get all of the positive indicator indices going to the source cluster 
        `source_cid` in order. If both `target_cid` and `source_cid` are specified,
        it will return an empty array if the indicator is False or it will return
        an array of size 1 if the indicator is True.

        Parameters
        ----------
        target_cid, source_cid : int, None
            These are the target cluster ID and source cluster ID, respectively.
            If None then nothing is specified.

        Returns
        -------
        list(int)
            Returns a list of the interactions that are positive in order
        '''
        ret = []
        try:
            if target_cid is not None:
                n_clusters = len(self.clustering)
                tcidx = self.clustering.cid2cidx[target_cid]

                if source_cid is not None:
                    if self.value[target_cid][source_cid].indicator:
                        scidx = self.clustering.cid2cidx[source_cid]
                        if tcidx < scidx:
                            scidx -= 1
                        iidx = tcidx * (n_clusters - 1) + scidx
                        ret.append(iidx)
                else:
                    base_idx = tcidx * (n_clusters - 1)
                    for offset, interaction in enumerate(self.iter_to_target(target_cid)):
                        if interaction.indicator:
                            ret.append(base_idx + offset)
            
            elif source_cid is not None:
                # We do not need to check if target_cid is not None because 
                # it would have been covered in the previous check
                scidx = self.clustering.cid2cidx[source_cid]
                for interaction in self.iter_from_source(source_cid):
                    if interaction.indicator:
                        tcidx = self.clustering.cid2cidx[interaction.target_cid]

                        iidx = tcidx * (n_clusters - 1) + scidx
                        if scidx > tcidx:
                            iidx -= 1
                        ret.append(iidx)

            else:
                for idx, interaction in enumerate(self):
                    if interaction.indicator:
                        ret.append(idx)
            return ret
        except:
            # Check to see if it is a key error, else it is something weird
            if target_cid is not None or source_cid is not None:
                if not util.isint(target_cid) or not util.isint(source_cid):
                    raise TypeError('`Either `target_cid` ({}) or `source_cid` ({})' \
                        ' must be an int'.format(type(target_cid), type(source_cid)))
                elif target_cid == source_cid:
                    raise ValueError('`target_cid` ({}) and `source_cid` ({}) cannot' \
                        ' be the same'.format(target_cid, source_cid))
            raise

    def get_indicators(self, target_cid=None, source_cid=None, return_idxs=False):
        '''Return a the indicator variables as a vector in the order specified
        by the clusters.

        if `target_cid` is specified then it will return all indicators going
        to that cluster. If `source_cid` is specified then it will return all
        indicators going from that cluster. If both are specifeid then it
        will return an array of size 1 of a bool

        Parameters
        ----------
        target_cid, source_cid : int, None
            These are the target cluster ID and source cluster ID, respectively.
            If None then nothing is specified.
        return_idxs : bool
            If True, we return the index of the positive indicators. If False we 
            return an array of the indicator flags for every interaction. Nothing
            is done if both `target_cid` and `source_cid` are specified

        Returns
        -------
        np.ndarray(n, dtype=bool)
            Returns a bool vector of the indicators in roder
        '''
        if target_cid is not None or source_cid is not None and return_idxs:
            mapping = {}
            order = self.clustering.order
            i = 0
            for tcid in order:
                for scid in order:
                    if tcid == scid:
                        continue
                    mapping[(tcid, scid)] = i
                    i += 1
            
        try:
            if target_cid is not None:
                if source_cid is not None:
                    ret = np.asarray([self.value[target_cid][source_cid].indicator])
                else:
                    if return_idxs:
                        ret = []
                    else:
                        l = len(self.clustering) - 1
                        ret = np.zeros(l, dtype=bool)
                    for iidx, interaction in enumerate(self.iter_to_target(target_cid)):
                        if interaction.indicator:
                            if return_idxs:
                                ret.append(mapping[(target_cid, interaction.source_cid)])
                            else:
                                ret[iidx] = True
            elif source_cid is not None:
                # We do not need to check if target_cid is not None because 
                # it would have been covered in the previous check
                if return_idxs:
                    ret = []
                else:
                    l = len(self.clustering) - 1
                    ret = np.zeros(l, dtype=bool)
                for iidx, interaction in enumerate(self.iter_from_source(source_cid)):
                    if interaction.indicator:
                        if return_idxs:
                            ret.append(mapping[(interaction.target_cid, source_cid)])
                        else:
                            ret[iidx] = True

            else:
                if return_idxs:
                    ret = []
                else:
                    ret = np.zeros(self.size, dtype=bool)
                for idx, interaction in enumerate(self):
                    if interaction.indicator:
                        if return_idxs:
                            ret.append(idx)
                        else:
                            ret[idx] = True
                return ret
            if return_idxs:
                ret = np.asarray(ret, dtpye=int)
            return ret

        except:
            # Check to see if it is a key error, else it is something weird
            if target_cid is not None or source_cid is not None:
                if not util.isint(target_cid) or not util.isint(source_cid):
                    raise TypeError('`Either `target_cid` ({}) or `source_cid` ({})' \
                        ' must be an int'.format(type(target_cid), type(source_cid)))
                elif target_cid == source_cid:
                    raise ValueError('`target_cid` ({}) and `source_cid` ({}) cannot' \
                        ' be the same'.format(target_cid, source_cid))
            raise

    def set_indicators(self, arr):
        '''Sets the values of the indicators of the interactions from a vector.

        If `include_self_interactions` is True, assumes that `arr` contains the
        values for the self interactions. If False, assume that the indices are
        skipped.

        Paramters
        ---------
        arr : np.ndarray(n, dtpye=bool)
            These are the indicator values to set, in order
        '''
        if len(arr) != self.size:
            raise ValueError('The number of elements in `arr` ({}) is not the ' \
                'same as the number of interactions ({})'.format(len(arr), self.size))
        for idx, interaction in enumerate(self):
            interaction.indicator = arr[idx]
            if interaction.indicator == 0:
                interaction.value = 0

    def set_values(self, arr, use_indicators=True):
        '''Sets the values of the interactions from a vector.

        If `use_indicators` is True, assumes that the values in the vector only contain
        values for interactions where the indicator variable is positive and the
        rest are skipped. If False, assumes the vector has values for where the
        indicator variables are positive and negative (all).

        Paramters
        ---------
        arr : np.ndarray(n, dtpye=float)
            These are the interaction values to set, in order
        use_indicators : bool, Optional
            If True, we only set the interactions with a positive indicator. Else we set every
            single interaction
        '''
        if not use_indicators:
            if len(arr) != self.size:
                raise ValueError('The number of elements in `arr` ({}) is not the ' \
                    'same as the number of interactions ({})'.format(len(arr), self.size))
            for idx, interaction in enumerate(self):
                interaction.value = arr[idx]
        else:
            # Dont check because it is too computationally intensive
            idx = 0
            for interaction in self:
                if not interaction.indicator:
                    continue
                interaction.value = arr[idx]
                idx += 1

    def get_values(self, use_indicators=True):
        '''Makes a vector of the interaction variables in the order of the
        clustering

        if use_indicators is True, it skips over the indices that have a negative
        indicator variable. if it is True, it goes over everthing

         Paramters
        ---------
        use_indicators : bool, Optional
            If True, we only return the interactions with a positive indicator. Else we get every
            single interaction

        Returns
        -------
        np.ndarray(n, dtype=float)
            Array of the interaction values, in order
        '''
        ret = np.zeros(self.size)
        idx = 0
        if use_indicators:
            for interaction in self.iter_valid():
                ret[idx] = interaction.value
                idx += 1
        else:
            for interaction in self:
                ret[idx] = interaction.value
                idx += 1
        # Trim if necessary
        return ret[:idx]

    def get_value_matrix(self, set_neg_indicators_to_nan=False):
        '''Get the interaction matrix at the clustert level (item-item). 
        The ordering of the clusters is the same as it is in clustering

        If `set_neg_indicators_to_nan` is True, interactions that have a negative
        indicator are set to np.nan. Else, they are set to 0.

        Parameters
        ----------
        set_neg_indicators_to_nan : bool
            If True, it will set the negative interaction indicator values to 
            np.nan. Else, it will set them to 0.

        Returns
        -------
        np.ndarray((n,n), dtype=float)
            This is the item-item interaction value matrix
        '''
        n_clusters = len(self.clustering)
        if set_neg_indicators_to_nan:
            fill = np.nan
        else:
            fill = 0
        ret = np.full(shape=(n_clusters, n_clusters), fill_value=fill, dtype=float)
        for interaction in self:
            if not interaction.indicator:
                continue
            tcidx = self.clustering.cid2cidx[interaction.target_cid]
            scidx = self.clustering.cid2cidx[interaction.source_cid]
            ret[tcidx, scidx] = interaction.value
        return ret

    def get_datalevel_value_matrix(self, set_neg_indicators_to_nan=False):
        '''Get the interaction matrix at the data level (item-item), not
        at the cluster level. The ordering of the items is the same as 
        it is in the items in clustering (self.clustering.times).

        If `set_neg_indicators_to_nan` is True, interactions that have a negative
        indicator are set to np.nan. Else, they are set to 0.

        Parameters
        ----------
        set_neg_indicators_to_nan : bool
            If True, it will set the negative interaction indicator values to 
            np.nan. Else, it will set them to 0.

        Returns
        -------
        np.ndarray((n,n), dtype=float)
            This is the item-item interaction value matrix
        '''
        n_items = len(self.clustering.items)
        if set_neg_indicators_to_nan:
            fill = np.nan
        else:
            fill = 0
        ret = np.full(shape=(n_items, n_items), fill_value=fill, dtype=float)
        for interaction in self:
            if not interaction.indicator:
                continue
            val = interaction.value
            for tidx in self.clustering.clusters[interaction.target_cid].members:
                for sidx in self.clustering.clusters[interaction.source_cid].members:
                    ret[tidx, sidx] = val
        return ret
    
    def get_datalevel_indicator_matrix(self):
        '''Get the item-item indicator matrix.

        The ordering of the items are the same as the order in
        self.clusters.items.ids.order

        Returns
        -------
        np.ndarray((n,n), dtype=float)
            This is the item-item interaction value matrix
        '''
        n_items = len(self.clustering.items)
        ret = np.zeros(shape=(n_items, n_items), dtype=bool)
        for interaction in self:
            if not interaction.indicator:
                continue
            for tidx in self.clustering.clusters[interaction.target_cid].members:
                for sidx in self.clustering.clusters[interaction.source_cid].members:
                    ret[tidx, sidx] = True
        return ret

    def generate_in_out_degree_posthoc(self, section='posterior'):
        '''Returns a dictionary of arrays
        "in"
            For each index in the array, corresponding to the index of the items, returns
            the number of incoming interactions for each iteration of the item
        "out"
            For each index in the array, corresponding to the index of the items, returns
            the number of outgoing interactions for each iteration of the item

        Parameters
        ----------
        section : str
            Which section of the inference you want to choose. 
            Options:
                'posterior'
                    Only look at the posterior
                'burnin'
                    Returns the samples that were in the burnin
                'entire'
                    Returns all the samples
        '''
        trace = self.get_trace_from_disk(section=section)
        trace = ~np.isnan(trace)
        return {'in': np.sum(trace, axis=2), 'out':np.sum(trace, axis=1)}

    def set_trace(self):
        '''Initialize the trace arrays for the variable in the Tracer object. 

        It will initialize a buffer the size of the checkpoint size in Tracer
        '''
        tracer = self.G.tracer
        tracer.set_trace(self.name, shape=self._shape, dtype=self.dtype)

        self.ckpt_iter = 0
        self.sample_iter = 0
        shape = (tracer.ckpt, ) + self._shape
        self.trace = np.full(shape=shape, fill_value=np.nan, dtype=self.dtype)

    def remove_local_trace(self):
        '''Delete the local trace
        '''
        self.trace = None

    def add_trace(self):
        '''Adds the current value to the trace. If the buffer is full
        it will end it to disk
        '''
        self.trace[self.ckpt_iter] = self.get_datalevel_value_matrix(set_neg_indicators_to_nan=True)
        self.ckpt_iter += 1
        self.sample_iter += 1
        if self.ckpt_iter == len(self.trace):
            # We have gotten the largest we can in the local buffer, write to disk
            self.G.tracer.write_to_disk(name=self.name)
            shape = (self.G.tracer.ckpt, ) + self._shape
            self.trace = np.full(shape=shape, fill_value=np.nan, dtype=self.dtype)
            self.ckpt_iter = 0

    def get_adjacent(self, cid, incoming, outgoing, use_indicators=True):
        '''Get all of the cluster IDs that have a positive interaction going into
        or from the cluster `cid`.

        Parameters
        ----------
        cid : int
            This is the Cluster ID you want to get the adjacent clusters of
        incoming : bool 
            Get the cids of the incoming edges
        outgoing : bool
            Get the cids of the outgoing edges
        use_indicators : bool
            If this is True then if the indicator is False then we do not include.
            If this is False then we always include the interaction

        Returns
        -------
        list
            List of cids
        '''
        if cid not in self.clustering.order:
            raise ValueError('`cid` ({}) not found'.format(cid))
        cids = []
        if incoming:
            for interaction in self.iter_from_source(cid):
                if interaction.indicator or not use_indicators:
                    cids.append(interaction.target_cid)
        if outgoing:
            for interaction in self.iter_to_target(cid):
                if interaction.indicator or not use_indicators:
                    cids.append(interaction.source_cid)
        return cids
        

class _Interaction:
    '''Defines an interaction from cluster `source` to cluster `target`.

    Parameters
    ----------
    source_cid : int
        Unique id of the source cluster
    target_cid : int
        Unique id of the target cluster
    value : numeric
        The value of the interaction
    indicator : bool
        Indicator variable of the interaction
    id : int
        Unique identifier of this interaction object
    '''
    def __init__(self, source_cid, target_cid, value, indicator):
        self.source_cid = source_cid
        self.target_cid = target_cid
        self.value = value
        self.indicator = indicator
        self.id = id(self)

    def __str__(self):
        return 'Interaction {}\n' \
            '\tTarget cluster: {}\n' \
            '\tSource cluster: {}\n' \
            '\tValue: {}\n' \
            '\tIndicator: {}\n'.format(
                self.id,
                self.target_cid,
                self.source_cid,
                self.value,
                self.indicator)


def _always_return_true(*args, **kwargs):
    return True

def _always_return_nan(*args, **kwargs):
    return np.nan

