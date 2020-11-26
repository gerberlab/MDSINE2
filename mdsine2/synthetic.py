'''Generating synthetic data 

References
----------
[1] Robust and Scalable Models of Microbiome Dynamics, TE Gibson, GK Gerber (2018)
'''
import numpy as np
import logging
import time
import copy

import random
import numpy.random as npr
import scipy.stats
from sklearn.cluster import AgglomerativeClustering

from . import pylab as pl
from . import model
from . import diversity

class SyntheticData(pl.Saveable):
    '''This class is used to generate synthetic data using fixed
    or sampled topologies. It also gives us the ability to convert
    the internal representation into a `pylab.base.Study` so
    we can use it in inference.

    We add the parameters for the dynamical system to self.dynamics,
    which we assume to be a Discretized Generalized Lotka-Voltera 
    Dynamics with clustered interactions and perturbations. The 
    dynamical class is specified in the module `model`.

    NOTE: We cannot specify the dynamical class until we have our `taxas` 
    object so we wait to define it then, and it is done automatically 
    once we call the function `set_taxas`

    Parameters
    ----------
    n_days : numeric
        Total length of days
    perturbations_additive : bool
        If True, model the dynamics as additive. Else model it as multiplicative
    '''
    def __init__(self, n_days, perturbations_additive):
        self.G = pl.graph.Graph(name='synthetic') # make local graph
        self.data = []
        self.times = []
        self.dt = None
        self.dynamics = None
        self.n_replicates = 0
        self.perturbations_additive = perturbations_additive

        self.dt = None
        self.n_time_steps = None
        self.n_days = n_days

    @property
    def perturbations(self):
        try:
            return self.dynamics.perturbations
        except:
            raise pl.UndefinedError('You need to define `dynamics` ({}) first.'.format(
                type(self.dynamics)))

    def get_full_interaction_matrix(self):
        '''Make the interaction matrix. If `with_interactions` is False,
        you're effectively only getting the self-interaction terms.
        '''
        A = self.dynamics.interactions.get_datalevel_value_matrix(set_neg_indicators_to_nan=False)
        for i in range(A.shape[0]):
            A[i,i] = -self.dynamics.self_interactions[i]
        return A

    def set_taxas(self, n_taxas=None, sequences=None, filename=None, taxas=None):
        '''Sets the set. If you have a list of sequences you want to read in,
        pass in the list of `sequences` and it will create a new Taxa for every sequence. 
        If you  just want to specify how many Taxas you want and you dont care about 
        sequences, then specify the number of Taxas with `n_taxas`. If there is an
        TaxaSet saved on file, you can load it with the keyword `filename`.

        Defines the dynamics once done.

        Example:
            >>> self.set_taxas(n_taxas=5)
            5 Taxas with random sequences

            >>> seq = ['AAAA', 'TTTT', 'GGGG', 'CCCC']
            >>> self.set_taxas(sequences=seq)
            3 Taxas with the sequences specified above
            
            >>> self.set_taxas(filename='pickles/test.pkl')
            Reads in the TaxaSet saved at 'pickles/test.pkl'

        Parameters
        ----------
        n_taxas : int, Optional
            How many Taxas you want. This is unnecessary if you specify a list of 
            sequences.
        sequences : array(str), Optional
            The sequence for each Taxa. If you do not want any, dont specify anything
            and specify the number of Taxas. If nothing is provided it will create
            a random sequence of sequences for each Taxa
        filename : str, Optional
            The filename to lead the Taxa
        taxas : pylab.base.TaxaSet
            This is an TaxaSet object

        Returns
        -------
        pl.TaxaSet
            This is the TaxaSet that gets created
        '''
        a = n_taxas is not None
        b = sequences is not None
        c = filename is not None
        d = taxas is not None

        if a + b + c + d != 1:
            raise TypeError('Only one of `n_taxas` ({}), `sequences` ({}), '\
                '`filename` ({}), or taxas ({})  can be specified'.format(
                type(n_taxas), type(sequences), type(filename), type(taxas)))
        
        if n_taxas is not None:
            if not pl.isint(n_taxas):
                raise TypeError('`n_taxas` ({}) must be an int'.format(type(n_taxas)))
            self.taxas = pl.TaxaSet()
            for i in range(n_taxas):
                name = 'Taxa_{}'.format(i)
                seq = ''.join(random.choices(['A','T','G','C','U'], k=50))
                self.taxas.add_taxa(name=name, sequence=seq)
        elif sequences is not None:
            if not pl.isarray(sequences):
                raise TypeError('`sequences` ({}) must be an array'.format(type(sequences)))
            if not np.all(pl.itercheck(sequences, pl.isstr)):
                raise TypeError('All elements in `sequences` must be strs: {}'.format(
                    pl.itercheck(sequences, pl.isstr)))
            for i, seq in enumerate(sequences):
                name = 'Taxa_{}'.format(i)
                self.taxas.add_taxa(name=name, sequence=seq)
        elif filename is not None:
            if not pl.isstr(filename):
                raise TypeError('`filename` ({}) must be a str'.format(type(filename)))
            self.taxas = pl.TaxaSet.load(filename)
        else:
            self.taxas = taxas

        self.dynamics = model.gLVDynamicsSingleClustering(taxas=self.taxas, 
            perturbations_additive=self.perturbations_additive)
        
    def set_cluster_assignments(self, clusters=None, n_clusters=None, evenness=None):
        '''Create clusters for the interactions and perturbations. If you have the cluster
        assignments already, set them with `clusters`. else we can randomly generate them 
        with the parameters `n_clusters` and `evenness`.

        Parameters
        ----------
        clusters : list(list(int))
            These are the cluster assignments of the Taxas
        n_clusters : int
            Number of clusters to create
        evenness : str, 1 or 2-dim array
            How to initialize the clusters. This can be generated automatically or by 
            reading in a similarity matrix.
            If it is a str:
                'even': Have each cluster have as close to even number of clusters as possible
                'heavy-tail': TODO : NOT IMPLEMENTED
                'sequence' : TODO : NOT IMPLEMENTED (given the sequences, make an adjacency matrix)
            If it is an array:
                If 1-dim
                    This is how you assign the clusters by index to each cluster. This is the 
                    initialization format for pylab.cluster.Clustering
                If it is 2-dim
                    This is a 2 dimensional DISTANCE matrix. It will build the cluster 
                    assignments given this distance matrix

        Returns
        -------
        pl.cluster.Clustering
            This is the clustering object that gets created

        See also
        --------
        pylab.cluster.Clustering.__init__
        '''
        if self.taxas is None:
            raise ValueError('Must specify the TaxaSet before by calling `self.set_taxas`')

        if clusters is None:
            if not pl.isint(n_clusters):
                raise TypeError('`n_clusters` ({}) must be an int'.format(n_clusters))

            clusters = []
            start = 0
            if pl.isstr(evenness):
                if evenness == 'even':
                    size = int(len(self.taxas)/n_clusters)
                    for _ in range(n_clusters-1):
                        clusters.append(np.arange(start,start+size, dtype=int))
                        start += size
                    clusters.append(np.arange(start, len(self.taxas), dtype=int))
                elif evenness == 'sequence':
                    logging.info('Making affinity matrix from sequences')
                    evenness = np.diag(np.ones(len(self.taxas), dtype=float))

                    for i in range(len(self.taxas)):
                        for j in range(len(self.taxas)):
                            if j <= i:
                                continue
                            # Subtract because we want to make a similarity matrix
                            dist = 1-diversity.beta.hamming(
                                list(self.taxas[i].sequence), list(self.taxas[j].sequence))
                            evenness[i,j] = dist
                            evenness[j,i] = dist

                    # print(evenness)

                elif evenness == 'heavy-tail':
                    raise NotImplementedError('`heavy-tail` not implemented yet')
                else:
                    raise ValueError('cluster evenness ({}) not recognized'.format(evenness))
            if pl.isarray(evenness):
                evenness = np.asarray(evenness)
                if evenness.ndim == 1:
                    clusters = evenness.tolist()
                elif evenness.ndim == 2:
                    if evenness.shape[0] != evenness.shape[1]:
                        raise ValueError('Must be a square matrix')
                    if evenness.shape[0] != len(self.taxas):
                        raise ValueError('Length of the side ({}) must be the same as the number of s ({})'.format(
                            evenness.shape[0], len(self.taxas)))
                    
                    c = AgglomerativeClustering(
                        n_clusters=n_clusters,
                        affinity='precomputed',
                        linkage='average')
                    assignments = c.fit_predict(evenness)
                    clusters = {}
                    for oidx,cidx in enumerate(assignments):
                        if cidx not in clusters:
                            clusters[cidx] = []
                        clusters[cidx].append(oidx)
                    clusters = [val for val in clusters.values()]
                else:
                    raise ValueError('`evenness` ({}) must be a 1 or 2-dimensional array'.format(
                        evenness.ndim))
            else:
                raise TypeError('`evenness` ({}) must be either a string or an array'.format(
                    type(evenness)))

        # Initialize clustering object
        logging.info('cluster assignments: {}'.format(clusters))
        self.dynamics.clustering = pl.Clustering(clusters = clusters, items=self.taxas, G=self.G)
        return self

    def shuffle_cluster_assignments(self, p):
        '''Shuffle the cluster assignments that were specified in `set_cluster_assignments`.

        `p` indicates what proportion of the Taxas to be reassigned. Example: `p=.1` means
        that you want to shuffle 10% of the Taxas.

        NOTE: THIS SHOULD BE CALLED BEFORE YOU CALL THE FUNCTION `sample_dynamics`.

        Parameters
        ----------
        p : float
            Proportion of the Taxas to be shuffled
        '''
        if self.dynamics.clustering is None:
            raise ValueError('Must specfiy the clusters before you call this function')
        if not pl.isnumeric(p):
            raise TypeError('`p` ({}) should be a numeric'.format(type(p)))
        if p < 0 or p > 1:
            raise ValueError('`p` ({}) should be 0 =< p =< 1'.format(p))

        n = int(p*len(self.taxas))
        oidxs = np.random.randint(len(self.taxas), size=n)

        logging.info('Taxa indices to shuffle: {}'.format(oidxs))

        for oidx in oidxs:
            curr_cid = self.dynamics.clustering.idx2cid[oidx]
            assigned_cid = curr_cid
            while assigned_cid == curr_cid:
                assigned_cid = random.choice(self.dynamics.clustering.order)

            self.dynamics.clustering.move_item(idx=oidx, cid=assigned_cid)

        logging.info('new cluster assignments: {}'.format(self.dynamics.clustering.toarray()))

    def sample_single_perturbation(self, start, end, prob_pos, prob_affect, prob_strength, 
        mean_strength, std_strength):
        '''Sample a perturbation to add to the system.

        If there are no clusters that are selected, we sample again until we get at least 1.

        Defaults for strength parameters
        --------------------------------
        `prob_strength = [0.2, 0.4, 0.4]`
        `mean_strength   = [0.5, 1.0, 2.0]`
        `std_strength  = 0.1`

        Parameters
        ----------
        start : float
            - Time to start the perturbation.
        end : float
            - Time to end the perturbation
        pob_pos : float, [0,1]
            - This is the probability that the perturbation is going to be positive
            - Sampled from a Bernoulli distribution
        mean_strength : array
            These are the means of the magnitudes of the perturbations to sample around
        prob_strength : array
            These are the probabilities to sample a mean magnitude of the perturbation
        std_strength : float
            This is the standard deviation to sample the magnitude of the perturbation
        prob_affect : float ([0,1]), str
            - This is the probability it will affect a cluster (positive indicator)
            - Sampled from a Bernoulli distribution
            - If str, then only that number of clusters is set.
              Example:
                '1' means only one cluster is set

        Returns
        -------
        pylab.contrib.ClusterPerturbation
            This is the cluster perturbation that was sampled
        '''
        if self.dynamics.clustering is None:
            raise TypeError('Clustering module is None, this is likely because you did ' \
                'not set the dynamics')

        # Check variables
        if not pl.isnumeric(start):
            raise TypeError('`start` ({}) must be a numeric'.format(type(start)))
        if not pl.isnumeric(end):
            raise TypeError('`end` ({}) must be a numeric'.format(type(end)))
        if end <= start:
            raise ValueError('`start` ({}) must be strictly smaller than `end` ({})'.format(
                start,end))
        if not pl.isfloat(prob_pos):
            raise TypeError('`prob_pos` ({}) must be a numeric'.format(type(prob_pos)))
        elif prob_pos < 0 or prob_pos > 1:
            raise ValueError('`prob_pos` ({}) must be [0,1]'.format(prob_pos))
        if pl.isstr(prob_affect):
            set_num = True
            try:
                pa = int(prob_affect)
            except:
                logging.critical('Cannot cast `prob_affect` ({}) as an int'.format(prob_affect))
                raise
            if pa < 0:
                raise ValueError('`prob_affect` ({}) must be > 0'.format(prob_affect))
            if pa > len(self.dynamics.clustering.clusters):
                raise ValueError('`prob_affect` ({}) must be less than n_clusters'.format(prob_affect))
        else:
            set_num = False
            pa = None
            if not pl.isfloat(prob_affect):
                raise TypeError('`prob_affect` ({}) must be a numeric'.format(type(prob_affect)))
            elif prob_affect < 0 or prob_affect > 1:
                raise ValueError('`prob_affect` ({}) must be [0,1]'.format(prob_affect))
        # check the strengths
        if not pl.isarray(prob_strength):
            raise TypeError('`prob_strength` ({}) be an array'.format(type(prob_strength)))
        for ele in prob_strength:
            if not pl.isfloat(ele):
                raise TypeError('`every element in `prob_strength` ({}) must be a float'.format( 
                    type(ele)))
            if ele <= 0:
                raise ValueError('`every probability in `prob_strength` ({}) must be > 0'.format(ele))
        if not pl.isarray(mean_strength):
            raise TypeError('`mean_strength` ({}) must be an array'.format(type(mean_strength)))
        for ele in mean_strength:
            if not pl.isnumeric(ele):
                raise TypeError('Every element in `mean_strength` ({}) must be a numeric'.format(ele))
            if ele < 0:
                raise ValueError('Every element in `mean_strength` ({}) must be positive.' \
                    ' If you want a perturbation to have a negative magnitude then set the ' \
                    '`prob_pos` parameter to a very low nuber'.format(ele))
        if len(mean_strength) != len(prob_strength):
            raise ValueError('`mean_strength` ({}) and `prob_strength` ({}) must have the same number' \
                ' of elements'.format(len(mean_strength), len(prob_strength)))
        if not pl.isnumeric(std_strength):
            raise TypeError('`std_strength` ({}) must be a numeric'.format(type(std_strength)))
        if std_strength <= 0:
            raise ValueError('`std_stregnth` ({}) must be > 0'.format(std_strength))

        # Set indicator at the Taxa level
        if set_num:
            # Pick the number of clusters on
            order = copy.deepcopy(self.dynamics.clustering.order)
            order = list(order)
            indicator = np.zeros(len(order), dtype=bool)

            # pick the cids to set to true
            while pa > 0:
                # pick a cluster
                idx = npr.randint(0, len(order))
                indicator[self.dynamics.clustering.cid2cidx[order[idx]]] = True
                order.pop(idx)
                pa -= 1
        else:
            i = 0
            while True:
                indicator = np.zeros(len(order), dtype=bool)
                for cidx in range(len(self.dynamics.clustering.clusters)):
                    indicator[cidx] = bool(pl.random.bernoulli.sample(prob_affect))
                if np.sum(indicator) > 0:
                    break
                if i == 1000:
                    raise ValueError('Assigning cluster ids failed 1000 times. set `prob_affect` {}' \
                        ' to a larger number'.format(prob_affect))
                i += 1

        magnitude = np.zeros(len(self.dynamics.clustering), dtype=float)
        for i,ind in enumerate(indicator):
            if ind:
                # sample the magnitude
                sign = pl.random.bernoulli.sample(prob_pos) * 2 - 1
                mean = np.random.choice(mean_strength, p=prob_strength)
                magnitude[i] = pl.random.normal.sample(loc=mean*sign, 
                    scale=std_strength)

        a = pl.contrib.ClusterPerturbationEffect(start=start, end=end, magnitude=magnitude,
            clustering=self.dynamics.clustering, indicator=indicator, G=self.G)
        
        logging.info('Perturbation:\n\tstart,end ({},{})\n\t' \
            'magnitude {}\n\tindicator {}'.format(start,end,a.cluster_array(),
            indicator))

        if self.dynamics.perturbations is None:
            self.dynamics.perturbations = [a]
        else:
            self.dynamics.perturbations.append(a)
        return a

    def set_single_perturbation(self, start, end, magnitude, indicator):
        '''Sets a single perturbation - no sampling. This assumes that you have
        initialized the system - (`clustering` is not None)

        Parameters
        ----------
        start, end (float)
            - Time to start/end the perturbation
        magnitude (float)
            - This is the strength of the perturbation
        indicator (int, np.ndarray)
            - If it is an int, this is the cluster id that it is positive for
            - If it is an array, it must either be the length of the
              number of clusters, and must be either a boolean or 1,0s
        '''
        if self.dynamics.clustering is None:
            raise ValueError('Must sample the system before you call this function')
        a = pl.contrib.ClusterPerturbationEffect(start=start, end=end, magnitude=magnitude,
            indicator=indicator, clustering=self.dynamics.clustering, G=self.G)
        if self.dynamics.perturbations is None:
            self.dynamics.perturbations = [a]
        else:
            self.dynamics.perturbations.append(a)
        return a

    def icml_topology(self, n_taxas=13, max_abundance=None):
        '''Recreate the dynamical system used in the ICML paper [1]. If you use the 
        default parameters you will get the same interaction matrix that was used in [1].

        We rescale the self-interactions and the interactions (and potentially growth) by 150
        so that the time-scales of the trajectories happen over days instead of minutes

        Parameters
        ----------
        n_taxas : int
            These are how many Taxas to include in the system. We will always have 3 clusters and
            the proportion of Taxas in each cluster is as follows:
                cluster 1 - 5/13
                cluster 2 - 6/13
                cluster 3 - 2/13
            We assign each Taxa to each cluster the best we can, any extra Taxas we put into cluster 3
        max_abundance : numeric, None
            This is the abundance to set the maximum. All of the other 
            parameters change proportionally. If `None` then we assume no change.
        '''
        if not pl.isint(n_taxas):
            raise TypeError('`n_taxas` ({}) must be an int'.format(type(n_taxas)))
        if n_taxas < 3:
            raise ValueError('`n_taxas` ({}) must be >= 3'.format(n_taxas))
        if max_abundance is not None:
            if not pl.isnumeric(max_abundance):
                raise TypeError('`max_abundance` ({}) must be a numeric'.format(type(max_abundance)))
            if max_abundance <= 0:
                raise ValueError('`max_abundance` ({}) must be > 0'.format(max_abundance))
        
        # Generate s
        self.set_taxas(n_taxas)

        # Make cluster assignments with the approximate proportions
        c0size = int(5*n_taxas/13)
        c1size = int(6*n_taxas/13)
        c2size = int(n_taxas - c0size - c1size)

        frac = 150*n_taxas/13

        clusters = [
            np.arange(0, c0size, dtype=int).tolist(), 
            np.arange(c0size, c0size+c1size, dtype=int).tolist(),
            np.arange(c0size+c1size, c0size+c1size+c2size, dtype=int).tolist()]

        self.dynamics.clustering = pl.Clustering(clusters=clusters, items=self.taxas, G=self.G)
        self.dynamics.interactions = pl.Interactions(clustering=self.dynamics.clustering, 
            use_indicators=True, G=self.G)

        c0 = self.dynamics.clustering.order[0]
        c1 = self.dynamics.clustering.order[1]
        c2 = self.dynamics.clustering.order[2]
        for interaction in self.dynamics.interactions:
            if interaction.target_cid == c0 and interaction.source_cid == c1:
                interaction.value = 3/frac
                interaction.indicator = True
            elif interaction.target_cid == c0 and interaction.source_cid == c2:
                interaction.value = -1/frac
                interaction.indicator = True
            elif interaction.target_cid == c2 and interaction.source_cid == c0:
                interaction.value = 2/frac
                interaction.indicator = True
            elif interaction.target_cid == c2 and interaction.source_cid == c1:
                interaction.value = -4/frac
                interaction.indicator = True
            else:
                interaction.value = 0
                interaction.indicator = False
        
        self.dynamics.growth = pl.random.uniform.sample(low=18, high=22, size=n_taxas)/150
        # self.dynamics.growth = pl.random.uniform.sample(.1, 0.6, size=n_taxas)
        # self.dynamics.growth = pl.random.uniform.sample(low=.1 + 0.2, high=math.log(10) - 0.2, size=n_taxas)
        # self.dynamics.growth = pl.random.uniform.sample(low=0.12, high=2, size=n_taxas)
        self.dynamics.self_interactions = np.ones(n_taxas, dtype=float)*(5)/150
        # self.dynamics.self_interactions = pl.random.uniform.sample(-.05/150, -50/150, size=n_taxas)

        if max_abundance is not None:
            # self.dynamics.growth = pl.random.uniform.sample(low=.1, high=math.log(10), size=n_taxas)
            # self.dynamics.growth = pl.random.uniform.sample(low=.21, high=.5, size=n_taxas)/2
            # rescale the abundance such that hte max the max is ~max_abundance
            frac = 25/max_abundance
            self.dynamics.self_interactions *= frac
            for interaction in self.dynamics.interactions:
                if interaction.indicator:
                    interaction.value *= frac
    
    def icml_topology_real(self, n_taxas=13, max_abundance=None, scale_interaction=None):
        '''Recreate the dynamical system used in the ICML paper [1]. If you use the 
        default parameters you will get the same interaction matrix that was used in [1].

        We rescale the self-interactions and the interactions (and potentially growth) by 150
        so that the time-scales of the trajectories happen over days instead of minutes

        Parameters
        ----------
        n_taxas : int
            These are how many Taxas to include in the system. We will always have 3 clusters and
            the proportion of Taxas in each cluster is as follows:
                cluster 1 - 5/13
                cluster 2 - 6/13
                cluster 3 - 2/13
            We assign each Taxa to each cluster the best we can, any extra Taxas we put into cluster 3
        max_abundance : numeric, None
            This is the abundance to set the maximum. All of the other 
            parameters change proportionally. If `None` then we assume no change.
        '''
        if not pl.isint(n_taxas):
            raise TypeError('`n_taxas` ({}) must be an int'.format(type(n_taxas)))
        if n_taxas < 3:
            raise ValueError('`n_taxas` ({}) must be >= 3'.format(n_taxas))
        if max_abundance is not None:
            if not pl.isnumeric(max_abundance):
                raise TypeError('`max_abundance` ({}) must be a numeric'.format(type(max_abundance)))
            if max_abundance <= 0:
                raise ValueError('`max_abundance` ({}) must be > 0'.format(max_abundance))
        if scale_interaction is None:
            scale_interaction = 1
        # Generate Taxas
        self.set_taxas(n_taxas)

        # Make cluster assignments with the approximate proportions
        c0size = int(5*n_taxas/13)
        c1size = int(6*n_taxas/13)
        c2size = int(n_taxas - c0size - c1size)

        frac = 150*n_taxas/13

        clusters = [
            np.arange(0, c0size, dtype=int).tolist(), 
            np.arange(c0size, c0size+c1size, dtype=int).tolist(),
            np.arange(c0size+c1size, c0size+c1size+c2size, dtype=int).tolist()]

        self.dynamics.clustering = pl.Clustering(clusters=clusters, items=self.taxas, G=self.G)
        self.dynamics.interactions = pl.Interactions(clustering=self.dynamics.clustering, 
            use_indicators=True, G=self.G)

        c0 = self.dynamics.clustering.order[0]
        c1 = self.dynamics.clustering.order[1]
        c2 = self.dynamics.clustering.order[2]
        for interaction in self.dynamics.interactions:
            if interaction.target_cid == c0 and interaction.source_cid == c1:
                interaction.value = scale_interaction*3/frac
                interaction.indicator = True
            elif interaction.target_cid == c0 and interaction.source_cid == c2:
                interaction.value = scale_interaction*(-1)/frac
                interaction.indicator = True
            elif interaction.target_cid == c2 and interaction.source_cid == c0:
                interaction.value = scale_interaction*2/frac
                interaction.indicator = True
            elif interaction.target_cid == c2 and interaction.source_cid == c1:
                interaction.value = scale_interaction*(-4)/frac
                interaction.indicator = True
            else:
                interaction.value = 0
                interaction.indicator = False
        
        # self.dynamics.growth = pl.random.uniform.sample(low=18, high=22, size=n_taxas)/150
        self.dynamics.growth = pl.random.uniform.sample(0.5, 1.5, size=n_taxas)
        # self.dynamics.growth = pl.random.uniform.sample(low=.1 + 0.2, high=math.log(10) - 0.2, size=n_taxas)
        self.dynamics.self_interactions = np.ones(n_taxas, dtype=float)*(5)/150
        # self.dynamics.self_interactions = pl.random.uniform.sample(-.05/150, -50/150, size=n_taxas)

        if max_abundance is not None:
            # self.dynamics.growth = pl.random.uniform.sample(low=.1, high=math.log(10), size=n_taxas)
            # self.dynamics.growth = pl.random.uniform.sample(low=.21, high=.5, size=n_taxas)/2
            # rescale the abundance such that hte max the max is ~max_abundance
            frac = 25/max_abundance
            self.dynamics.self_interactions *= frac
            for interaction in self.dynamics.interactions:
                if interaction.indicator:
                    interaction.value *= frac

    def icml_perturbations(self, starts, ends):
        '''Set the perturbations. Made to be informative. The `starts` and
        `end` are starts and ends of 3 perturbations
        '''
        if self.dynamics.clustering is None:
            raise TypeError('Clustering module is None, this is likely because you did ' \
                'not set the dynamics')

        if not pl.isarray(starts):
            raise TypeError('`starts` ({}) must be an array'.format(type(starts)))
        if len(starts) != 3:
            raise ValueError('`starts` ({}) must be 3 elements long'.format(len(starts)))
        for ele in starts:
            if not pl.isnumeric(ele):
                raise TypeError('Each element in `starts` ({}) must be an array ({})'.format(
                    ele, starts))
        if not pl.isarray(ends):
            raise TypeError('`ends` ({}) must be an array'.format(type(ends)))
        if len(ends) != 3:
            raise ValueError('`ends` ({}) must be 3 elements long'.format(len(ends)))
        for ele in ends:
            if not pl.isnumeric(ele):
                raise TypeError('Each element in `ends` ({}) must be an array ({})'.format(
                    ele, ends))
        for idx in range(len(starts)):
            start = starts[idx]
            end = ends[idx]

            if end <= start:
                raise ValueError('`start` ({}) must be smaller than `end` ({})'.format(
                    start,end))

        # Set perturbations
        c0 = self.dynamics.clustering.order[0]
        c1 = self.dynamics.clustering.order[1]
        c2 = self.dynamics.clustering.order[2]

        # Set perturbation 0
        if self.perturbations_additive:
            magnitude0 = np.asarray([
                    # pl.random.normal.sample(loc=-1, scale=0.1),
                    0, pl.random.normal.sample(loc=0.5, scale=0.1), 0])
        else:
            magnitude0 = np.asarray([
                    # pl.random.normal.sample(loc=-1, scale=0.1),
                    0, pl.random.normal.sample(loc=1, scale=0.1), 0])
        indicator0 = np.array([False, True, False], dtype=bool)
        p0 = pl.contrib.ClusterPerturbationEffect(start=starts[0], end=ends[0], 
            magnitude=magnitude0, indicator=indicator0, G=self.G, 
            clustering=self.dynamics.clustering)

        # Set perturbation 1
        if self.perturbations_additive:
            magnitude1 = np.asarray([
                    pl.random.normal.sample(loc=1, scale=0.1), 0,
                    pl.random.normal.sample(loc=-2, scale=0.1)])
        else:
            magnitude1 = np.asarray([
                    pl.random.normal.sample(loc=1, scale=0.1), 0,
                    pl.random.normal.sample(loc=-2, scale=0.1)])
        indicator1 = np.array([True, False, True], dtype=bool)
        p1 = pl.contrib.ClusterPerturbationEffect(start=starts[1], end=ends[1], 
            magnitude=magnitude1, indicator=indicator1, G=self.G, 
            clustering=self.dynamics.clustering)

        # Set perturbation 2
        if self.perturbations_additive:
            magnitude2 = np.asarray([
                    0, pl.random.normal.sample(mean=-0.5, std=0.1),
                    pl.random.normal.sample(mean=1, std=0.1)])
        else:
            magnitude2 = np.asarray([
                    0, pl.random.normal.sample(mean=-0.5, std=0.1),
                    pl.random.normal.sample(mean=1, std=0.1)])
        indicator2 = np.array([False, True, True], dtype=bool)
        p2 = pl.contrib.ClusterPerturbationEffect(start=starts[2], end=ends[2], 
            magnitude=magnitude2, indicator=indicator2, G=self.G, 
            clustering=self.dynamics.clustering)

        self.dynamics.perturbations = [p0,p1,p2]

    def set_timepoints(self, times):
        '''Times to set the timepoints

        Parameters
        ----------
        times : np.ndarray
        '''
        if not pl.isarray(times):
            raise TypeError('`times` ({}) must be an array'.format(times))
        times = np.sort(np.array(times))
        self.master_times = times
        return self

    def set_times_without_timeseries(self, N, D='auto', initial_growth=4, pretransition=1, 
        posttransition=2, transition_density=2, uniform_sampling=False):
        '''Set the time points for the replicates. 
        
        It is highly recommended that you call this function to set the timepoints
        of the synthetic trajectories instead of directly calling `set_timepoints_without_timeseries` 
        or setting them yourself.

        Types of time setting
        ---------------------
        There are three ways we can set the spacing of the timepoints: (1) We can
        set them with a uniform spacing by setting the parameter `uniform_spacing=True`.
        (2) We can space the timepoints non-uniformly using the algorithm 
        `synthetic.set_timepoints_without_timeseries`. (3) We can manually set the times with the parameter 
        `N` if `N` is an array. In terms of precidence of the parameters:
            (1) If `N` is an array, then we set them according to times
            (2) If `uniform_sampling = True`, then we ignore the parameters for the
                non-uniform
            (3) If `uniform_sampling = False`, then we do non-uniform sampling
 
        Non-uniform sampling
        --------------------
        We set the times according to the denisty intervals sepcified
        in D. If D is auto, we set the density to be `transition_density` 
        more intense than regular time points in the following scenarios:
            - For `initial_growth` days from the start of the trajectory
            - For `pretransition` days before a transition from off- and
              on- a perturbation (starting or ending of a perturbation)
            - For `posttransition` days after a transition from off- and
              on- a perturbation (starting or ending of a perturbation)

        Parameters
        ----------
        N : int, array
            If an int, it represents how many timepoints to allocate. Look at 
            `set_timepoints_without_timeseries` for more information.
            If an array, these are the times to set.
        D : str, 3-tuple, list(3-tuple)
            If this is a string, then we set it according to the densities
            specified above. Otherwise, these are a list of intervals that
            we want specific densities for. If all of the densities are not
            adjacent then we add in intervals in between with a density of 1.
            None of the densities listed in D can overlap. For options on 
        initial_growth : numeric, None
            How many days to double sample during the first `initial_growth` days.
            If None then we do not set the initial growth
        pretransition, posttransition : numeric
            How many days before and after, respectively, of a perturbation 
            transition to have double density. If None then we do not set it.
        transition_density : numeric
            How dense to make the higher densities (initial growth and transition
            times). Must be >= 1

        See also
        --------
        synthetic.set_timepoints_without_timeseries
        '''
        if pl.isarray(N):
            # Set times manually
            N = np.sort(np.unique(N))
            if np.any(N < 0):
                raise ValueError('Every value in `times` ({}) must be positive or 0'.format(N))
            if np.any(N > self.n_days):
                raise ValueError('Values in `N` ({}) out of range'.format(N))
            self.master_times = N
            return
            
        if not pl.isbool(uniform_sampling):
            raise TypeError('`uniform_sampling` ({}) must be a bool'.format(type(uniform_sampling)))
        
        if uniform_sampling:
            if not pl.isint(N):
                raise TypeError('`N` ({}) must be an int'.format(type(N)))
            if N <= 0:
                raise ValueError('`N` ({}) must be > 0'.format(N))
            ts = np.arange(0,self.n_days, step=self.n_days/N)
            for i in range(len(ts)):
                ts[i] = round(ts[i], 2)
            self.master_times = ts
            return

        # Else do non-uniform sampling
        if D == 'auto':
            if not pl.isnumeric(transition_density):
                raise TypeError('`transition_density` ({}) must be a numeric'.format(
                    type(transition_density)))
            if transition_density < 1:
                raise TypeError('`transition_density` ({}) must be >= 1'.format(
                    transition_density))
            if initial_growth is not None:
                if not pl.isnumeric(initial_growth):
                    raise TypeError('`initial_growth` ({}) should be a numeric'.format( 
                        type(initial_growth)))
                if initial_growth < 0:
                    raise ValueError('`initial_growth` ({}) must be positive'.format(
                        initial_growth))
            if pretransition is not None:
                if not pl.isnumeric(pretransition):
                    raise TypeError('`pretransition` ({}) should be a numeric'.format( 
                        type(pretransition)))
                if pretransition < 0:
                    raise ValueError('`pretransition` ({}) must be positive'.format(
                        pretransition))
            if posttransition is not None:
                if not pl.isnumeric(posttransition):
                    raise TypeError('`posttransition` ({}) should be a numeric'.format( 
                        type(posttransition)))
                if posttransition < 0:
                    raise ValueError('`posttransition` ({}) must be positive'.format(
                        posttransition))

            # Sort perturbations if there are
            if self.dynamics.perturbations is not None:
                perts = []
                pert_starts = []
                for perturbation in self.dynamics.perturbations:
                    perts.append((perturbation.start, perturbation.end))
                    pert_starts.append(perturbation.start)
                
                idxs = np.argsort(pert_starts)
                temp = []
                for idx in idxs:
                    temp.append(perts[idx])
                perts = temp

                # fail if the start or end of a perturbation is greater than n_days
                for s,e in perts:
                    if s >= self.n_days or e > self.n_days:
                        raise ValueError('Perturbation start and end (`{}`,`{}`) ' \
                            ' is out of range for the number of days `{}`'.format( 
                                s,e,self.n_days))
            else:
                perts = None

            # Add the densities in order
            D = []
            
            # Set initial growth
            l = np.min([initial_growth, self.n_days])
            D.append((0, l, transition_density))

            # Set for each perturbation:
            if perts is not None:
                for s, e in perts:
                    D.append((s-pretransition, s+posttransition, transition_density))
                    l = np.min([e+posttransition,self.n_days])
                    D.append((e-pretransition, l, transition_density))
        else:
            # check D
            if pl.istuple(D):
                D = [D]
            if not pl.isarray(D):
                raise TypeError('`D` ({}) must be an array'.format(type(D)))
            for ele in D:
                if not pl.istuple(ele):
                    raise ValueError('Each element in D ({}) must be a tuple'.format(
                        type(ele)))
                if len(ele) != 3:
                    raise ValueError('Each element in D must have 3 elements ({})' \
                        ''.format(len(ele)))
                
                s,e,d = ele
                if not np.all(pl.itercheck([s,e,d], pl.isnumeric)):
                    raise TypeError('All values in ({},{},{}) must be numerics'.format( 
                        type(s), type(e), type(d)))
                if s < 0 or s >= self.n_days or s >= e:
                    raise ValueError('`start` ({}) out of range'.format(s))
                if e > self.n_days:
                    raise ValueError('`end` ({}) out of range'.format(e))
                if d < 1:
                    raise ValueError('`density` ({}) must be >= 1'.format(d))
            
            # Order by the start
            starts = [s for (s,e,d) in D]
            idxs = np.argsort(starts)
            temp = []
            for idx in idxs:
                temp.append(D[idx])
            D = temp

            # Check if any of the intervals overlap
            # If they overlap and they have different densities then 
            # throw an error
            for i in range(len(D)-1):
                si, ei, di = D[i]
                sj, ej, dj = D[i+1]

                if si == sj:
                    raise ValueError('Two intervals ({} and {}) cannot have the ' \
                        'same start point'.format(D[i], D[i+1]))
                if ei > sj:
                    if di != dj:
                        raise ValueError('Intervals {} and {} overlap and have ' \
                            'different densities'.format(D[i], D[i+1]))

        # Merge any overlapping time periods. If the time periods are not 
        # overlapping then add in extra intervals with the background density
        # 1
        D_new = [D[0]]
        i = 1
        while True:
            if i == len(D):
                break

            # Compare the last element in D_new to the ith element of D
            si, ei, di = D_new[-1]
            sj, ej, dj = D[i]

            if ei > sj and ei < ej:
                # they overlap, merge
                if di != dj:
                    raise ValueError('Something went wrong. D_new: {}, D: {}, ' \
                        'i: {}'.format(D_new, D, i))
                D[-1] = (si, ej, di)
            else:
                if ei < sj:
                    # These do not overlap, add an intermediate
                    D_new.append((ei, sj, 1))
                elif ei != sj:
                    raise ValueError('Something went wrong. D_new: {}, D: {}, ' \
                        'i: {}'.format(D_new, D, i))
                D_new.append(D[i])
            i += 1
        D = D_new

        # check if the last timepoint goes up to the last time
        s,e,d = D[-1]
        if e < self.n_days:
            D.append((e,self.n_days,1))
        if e > self.n_days:
            raise ValueError('Last interval {} ends after the set number of days {}' \
                ''.format(D[-1], self.n_days))

        # Set the days that the perturbations start and end as essential
        essential_timepoints = None
        if self.dynamics.perturbations is not None:
            essential_timepoints = []
            for perturbation in self.dynamics.perturbations:
                essential_timepoints += [perturbation.start, perturbation.end]

        # Set the timepoints for each replicate
        self.master_times = set_timepoints_without_timeseries(N=N, T_start=0, T_end=self.n_days, D=D,
            move_timepoints_if_fail=True, essential_timepoints=essential_timepoints)

    def generate_trajectories(self, dt, init_dist=None, processvar=None):
        '''Generate gLV dynamics given the dynamics sampled or set

        Parameters
        ----------
        init_dist : pylab.variables.RandomVariable
            This is the distribution that we are sampling the initial 
            condition from
        n_replicates : int
            How many replicates to make
        dt : float
            This is the sampling rate to generate the trajectories
        - processvar : pl.dynamics.BaseProcessVariance
            Must be a subset of process variance class
        '''
        if processvar is not None:
            if not pl.isprocessvariance(processvar):
                raise TypeError('`processvar` ({}) not recognized'.format(
                    type(processvar)))
        if init_dist is not None:
            self.init_dist = init_dist
        self.processvar = processvar
        self.n_replicates += 1
        
        self.dt = dt
        n_valid = 1

        while n_valid > 0:
            init_abundance = self.init_dist.sample(size=len(self.taxas)).reshape(-1,1)
            d = self.dynamics.integrate(
                initial_conditions=init_abundance,
                dt=self.dt, processvar=processvar, subsample=True, 
                times=self.master_times, n_days=self.n_days)
            if not np.any(np.isnan(d['X'])):
                # valid, we dont have to resample
                self.data.append(d['X'])
                self.times.append(d['times'])
                n_valid -= 1
            else:
                logging.info('resampling')
                print(d['X'])

        return self
    
    def stability(self):
        return self.dynamics.stability()

    def simulateRealRegressionDataDMD(self, subjset, alpha, qpcr_scale=None):
        '''This function converts the synthetic trajectories into "real" data
        by simulating read counts and qPCR measurements. We base the sampling
        on parameters that we learn from the real data (MouseSet `subjset`). We assume
        that the real data has already been filtered.

        This uses a dirichlet multinomial to simulate the reads

        Simulating qPCR measurements
        ----------------------------
        We simulate the qPCR measurements with a lognormal distribution that has
        the standard deviation `qpcr_scale`. If `qpcr_scale` is not given, then we fit
        fit the qPCR measurements in `subjset` and use that value.

        Simulating count data
        ---------------------
        We first simulate the read depth for each day from a negative binomial
        that was fitted using the read depth of our data from `subjset` (using function 
        minimization). To generate the indivual counts for each Taxa, we then 'sample' 
        from a dirichlet multinomial using the sample read depth as our counts and the 
        `alpha` parameter as the multiplicative concentration for our relative abundances, where
            $\mathbf{\alpha}_i = \alpha * r_i$,
            $\mathbf{\alpha}_i$ is the concentration parameter for Taxa $i$ to sample from a dirichlet
            distribution,
            $r_i$ is the relative abundance for Taxa $i$,
            $\alpha$ is the input concentration
        To emulate a dirichlet multinomial distribution, we first sample the relative abundances 
        of each of the s from a dirichlet distribution using our alpha term as sepcified above, 
        and then sample from a multinomial distribution given our sampled a read depth and our 
        probabilities sampled concentrations for each Taxa from the dirichlet distribution.

        Parameters
        ----------
        subjset : pl.Study
            This is the real data that we are fitting to
        alpha : numeric, Optional
            This is a parameter for how much noise you have in sampling the individual reads.
            The larger the number, the less noise there is in the system. The smaller the number,
            the more noise there is in sampling the relative abundances
        qpcr_scale : numeric, None, Opional
            This is the scale of the lognormal distribution used to generate the qPCR measurements
            of the simulated data

        Returns
        -------
        pl.Study
            This is the subject set that contains the data in terms of counts and absolute abundance
        '''
        logging.info('Fitting real data')
        # load Study and filter
        if type(subjset) == str:
            subjset = pl.Study.load(subjset)
        elif not pl.isstudy(subjset):
            raise ValueError('`subjset` ({}) must eb a pl.Study object'.format(
                type(subjset)))
        if not pl.isnumeric(alpha):
            raise ValueError('`alpha` ({}) must either be a float or an int'.format(type(alpha)))
        elif alpha < 0:
            raise ValueError('`alpha` ({}) must be greater than 0'.format(alpha))
        if qpcr_scale is not None:
            if not pl.isnumeric(qpcr_scale):
                raise ValueError('`qpcr_scale` ({}) must either be a float or an int'.format(
                    type(qpcr_scale)))
            elif qpcr_scale < 0:
                raise ValueError('`qpcr_scale` ({}) must be greater than 0'.format(
                    qpcr_scale))
        
        # Fit qPCR with lognormal if necessary
        if qpcr_scale is not None:
            data = []
            for subj in subjset:
                for t, qpcr in subj.qpcr.items():
                    d = np.log(qpcr.data)
                    data = np.append(data, d - np.mean(d))
            qpcr_scale = np.std(data)
            logging.info('qpcr fitted scale: {}'.format(qpcr_scale))

        # Fit read depth with negative binomial
        read_depths = np.asarray([])
        for subj in subjset:
            read_depths = np.append(read_depths, subj.read_depth())
        n_pred, p_pred = _fit_nbinom(read_depths)
        logging.info('read depth negbin n: {}, p: {}'.format(n_pred, p_pred))
        readdepth_negbin = scipy.stats.nbinom(n_pred, p_pred)
        
        # Make data, record the data with noise
        ret_subjset = pl.Study(taxas=self.taxas)
        for ridx in range(self.n_replicates):
            mid = str(ridx)
            ret_subjset.add(name=mid)

            for tidx in range(len(self.times[ridx])):
                # make time id
                t = self.times[ridx][tidx]

                # Sample counts
                sum_abund = np.sum(self.data[ridx][:,tidx])
                rel = self.data[ridx][:,tidx] / sum_abund
                probs = (scipy.stats.dirichlet.rvs(rel*alpha)).flatten()
                read_depth = readdepth_negbin.rvs()
                ret_subjset[mid].reads[t] = scipy.stats.multinomial.rvs(read_depth, probs).ravel()

                # Sample qPCR
                triplicates = np.exp(np.log(sum_abund) + qpcr_scale * npr.normal(size=3))
                ret_subjset[mid].qpcr[t] = pl.qPCRdata(cfus=triplicates, mass=1., 
                    dilution_factor=1.)

            ret_subjset[mid].times = self.times[ridx]
        
        # Add in the perturbations
        if self.dynamics.perturbations is not None:
            for perturbation in self.dynamics.perturbations:
                ret_subjset.add_perturbation(perturbation.start, perturbation.end)
        return ret_subjset

    def simulate_reads(self, a0, a1, read_depth):
        '''This function converts the synthetic trajectories into "real" data
        by simulating read counts and qPCR measurements. We base the sampling
        on parameters that we learn from the real data (MouseSet `subjset`). We assume
        that the real data has already been filtered.

        Simulating count data
        ---------------------
        We first fit the read depth for each day (`r_k`) from a negative binomial from the 
        read depth of our data (`subjset`) using function minimization. We then use `r_k`, 
        `a_0`, and `a_1` with the relative abundances to sample from a negative binomial 
        distirbution. We then use the relative abundances from this sample as the concentrations
        for a multinomial distribution with read depth `r_k`.

        Parameters
        ----------
        subjset : pl.Study
            This is the real data that we are fitting to
        a0, a1 : numeric
            These are the negative binomial dispersion parameters that we are using to 
            simulate the data
        '''
        if not np.all(pl.itercheck([a0, a1], pl.isnumeric)):
            raise TypeError('`a0` ({}) and `a1` ({}) must be numerics'.format(
                type(a0), type(a1)))
        if a0 < 0 or a1 < 0:
            raise ValueError('`a0` ({}) and `a1` ({}) must be > 0'.format(a0, a1))
        
        # Make data, record the data with noise
        n_time_points = 0
        for times in self.times:
            n_time_points += len(times)

        ret_subjset = pl.Study(taxas=self.taxas)
        data = self.data
        times = self.times
        for ridx in np.arange(self.n_replicates):
            mid = str(ridx)
            ret_subjset.add(name=mid)
            # self.data_w_noise.append(np.zeros(shape=data[ridx].shape,dtype=float))

            for tidx in range(len(times[ridx])):
                # make time id
                t = times[ridx][tidx]

                # Sample counts
                sum_abund = np.sum(data[ridx][:,tidx])
                rel = data[ridx][:,tidx] / sum_abund

                phi = read_depth * rel
                eps = a0 / rel + a1

                reads = pl.random.negative_binomial.sample(phi, eps)
                # concentration = reads/np.sum(reads)
                # ret_subjset[mid].reads[t] = scipy.stats.multinomial.rvs(
                #     r_k, concentration).ravel()
                ret_subjset[mid].reads[t] = reads

            ret_subjset[mid].times = times[ridx]
        
        # Add in the perturbations
        if self.dynamics.perturbations is not None:
            for perturbation in self.dynamics.perturbations:
                ret_subjset.add_perturbation(perturbation.start, perturbation.end)
        self.subjset_with_noise = ret_subjset

    def simulate_qpcr(self, qpcr_noise_scale):
        '''This function converts the synthetic trajectories into "real" data
        by simulating read counts and qPCR measurements. We base the sampling
        on parameters that we learn from the real data (MouseSet `subjset`). We assume
        that the real data has already been filtered.

        We assume that the function `simulate_reads`

        Simulating qPCR measurements
        ----------------------------
        We simulate the qPCR measurements with a lognormal distribution that was
        fitted using the data from `subjset`. We use this parameterization to sample
        a qPCR measurement with mean from the total biomass of the simulated data.

        Parameters
        ----------
        qpcr_noise_scale : numeric
            This is the parameter to scale the `s` parameter learned by the lognormal
            distribution.
        '''
        if not pl.isnumeric(qpcr_noise_scale):
            raise TypeError('`qpcr_noise_scale` ({}) must either be a float or an int'.format(
                type(qpcr_noise_scale)))
        elif qpcr_noise_scale < 0:
            raise ValueError('`qpcr_noise_scale` ({}) must be greater than 0'.format(
                qpcr_noise_scale))

        for ridx in np.arange(self.n_replicates):
            mid = str(ridx)
            subj = self.subjset_with_noise[mid]

            for tidx, t in enumerate(self.times[ridx]):
                
                # Sample qPCR
                sum_abund = np.sum(self.data[ridx][:,tidx])
                triplicates = np.exp(np.log(sum_abund) + qpcr_noise_scale * pl.random.normal.sample(size=3))
                subj.qpcr[t] = pl.qPCRdata(
                    cfus=triplicates, mass=1., dilution_factor=1.)

    def get_subjset(self):
        '''Return the subjectset with the noise

        Returns
        -------
        pylab.base.Study
        '''
        return self.subjset_with_noise

    def simulateRealRegressionDataNegBinMD(self, a0, a1, qpcr_noise_scale, subjset, replicates='all', read_depth=None):
        '''This function converts the synthetic trajectories into "real" data
        by simulating read counts and qPCR measurements. We base the sampling
        on parameters that we learn from the real data (MouseSet `subjset`). We assume
        that the real data has already been filtered.

        This uses a negative binomial distribution to simiulate the reads.

        Simulating qPCR measurements
        ----------------------------
        We simulate the qPCR measurements with a lognormal distribution that was
        fitted using the data from `subjset`. We use this parameterization to sample
        a qPCR measurement with mean from the total biomass of the simulated data.

        Simulating count data
        ---------------------
        We first fit the read depth for each day (`r_k`) from a negative binomial from the 
        read depth of our data (`subjset`) using function minimization. We then use `r_k`, 
        `a_0`, and `a_1` with the relative abundances to sample from a negative binomial 
        distirbution. We then use the relative abundances from this sample as the concentrations
        for a multinomial distribution with read depth `r_k`.

        Parameters
        ----------
        subjset : pl.Study
            This is the real data that we are fitting to
        a0, a1 : numeric
            These are the negative binomial dispersion parameters that we are using to 
            simulate the data
        qpcr_noise_scale : numeric
            This is the parameter to scale the `s` parameter learned by the lognormal
            distribution.
        replicates : str, array, int
            Which replicates to make the data for. If `replicates='all'`, then we make
            it for all of the replicates. If it is an array, we assume it is an array of
            replicate indices of the replicates that you want.

        Returns
        -------
        pl.Study
            This is the subject set that contains the data in terms of counts and absolute abundance
        '''
        logging.info('Fitting real data')
        if pl.isstr(subjset):
            subjset = pl.base.Study.load(subjset)
        elif not pl.isstudy(subjset):
            raise TypeError('`subjset` ({}) must be a pylab.base.SubjsetSet'.format( 
                type(subjset)))
        if not pl.isnumeric(qpcr_noise_scale):
            raise TypeError('`qpcr_noise_scale` ({}) must either be a float or an int'.format(
                type(qpcr_noise_scale)))
        elif qpcr_noise_scale < 0:
            raise ValueError('`qpcr_noise_scale` ({}) must be greater than 0'.format(
                qpcr_noise_scale))
        if pl.isstr(replicates):
            if replicates == 'all':
                replicates = np.arange(len(self.data))
            else:
                raise ValueError('`replicates` ({}) not recognized'.format(replicates))
        elif pl.isint(replicates):
            replicates = [replicates]
        if pl.isarray(replicates):
            for i in replicates:
                if not pl.isint(i):
                    raise TypeError('`replicates` ({}) must be ints'.format(type(i)))
                if i >= len(self.data):
                    raise IndexError('`replicates` ({}) out of range ({})'.format(
                        i, len(self.data)))
        else:
            raise TypeError('`replicates` ({}) type not recognized'.format(type(replicates)))


        # # Fit qPCR with lognormal
        # data = []
        # for subj in subjset:
        #     for t, qpcr in subj.qpcr.items():
        #         if np.any(np.isnan(qpcr.data)):
        #             continue
        #         data.append(qpcr.data)
        # std_biomass = _fit_qpcr(data) * self.qpcr_noise_scale
        std_biomass = qpcr_noise_scale
        logging.info('lognormal s: {}'.format(std_biomass))

        # Fit read depth with negative binomial
        if read_depth is None:
            read_depths = np.asarray([])
            for subj in subjset:
                read_depths = np.append(read_depths, subj.read_depth())
            n_pred, p_pred = _fit_nbinom(read_depths)
            logging.info('negbin n: {}, p: {}'.format(n_pred, p_pred))
            negbin_read_depth = scipy.stats.nbinom(n_pred, p_pred)

        # Make data, record the data with noise
        n_time_points = 0
        for times in self.times:
            n_time_points += len(times)

        ret_subjset = pl.Study(taxas=self.taxas)
        data = self.data
        times = self.times
        for ridx in replicates:
            mid = str(ridx)
            ret_subjset.add(name=mid)
            # self.data_w_noise.append(np.zeros(shape=data[ridx].shape,dtype=float))

            for tidx in range(len(times[ridx])):
                # make time id
                t = times[ridx][tidx]

                # Sample counts
                sum_abund = np.sum(data[ridx][:,tidx])
                rel = data[ridx][:,tidx] / sum_abund
                if read_depth is None:
                    r_k = negbin_read_depth.rvs()
                else:
                    r_k = read_depth
                read_depth = 75000

                phi = r_k * rel
                eps = a0 / rel + a1

                reads = pl.random.negative_binomial.sample(phi, eps)
                # concentration = reads/np.sum(reads)
                # ret_subjset[mid].reads[t] = scipy.stats.multinomial.rvs(
                #     r_k, concentration).ravel()
                ret_subjset[mid].reads[t] = reads

                # Sample qPCR
                triplicates = np.exp(np.log(sum_abund) + std_biomass * npr.normal(size=3))
                ret_subjset[mid].qpcr[t] = pl.qPCRdata(
                    cfus=triplicates, mass=1., dilution_factor=1.)

            ret_subjset[mid].times = times[ridx]
        
        # Add in the perturbations
        if self.dynamics.perturbations is not None:
            for perturbation in self.dynamics.perturbations:
                ret_subjset.add_perturbation(perturbation.start, perturbation.end)
        return ret_subjset

    def simulateExactSubjset(self):
        '''This function is effectively the same as `simulateRealRegressionData*` but 
        we add an :math:`\epsilon` amount of measurement noise.

        Returns
        -------
        pylab.Base.Study
        '''
        subjset = pl.Study(taxas=self.taxas)
        data = self.data
        times = self.times

        for ridx in range(len(data)):
            mid = str(ridx)
            subjset.add(name=mid)

            for tidx in range(len(times[ridx])):
                t = times[ridx][tidx]

                # Make "exact" counts we read depth at ~1000000
                total_abund = np.sum(data[ridx][:,tidx])
                rel = data[ridx][:,tidx]/total_abund
                counts = np.asarray(rel * 1000000, dtype=int)
                subjset[mid].reads[t] = counts

                # make "exact" qpcr by having very little qpcr noise
                subjset[mid].qpcr[t] = pl.qPCRdata( 
                    cfus=pl.random.normal.sample(loc=total_abund, scale=1e-10, size=3), 
                    mass=1., dilution_factor=1.)
            subjset[mid].times = times[ridx]

        # Add in the perturbations
        if self.dynamics.perturbations is not None:
            for perturbation in self.dynamics.perturbations:
                subjset.add_perturbation(perturbation.start, perturbation.end, 
                    name=perturbation.name)
        
        return subjset


def _fit_nbinom(X, initial_params=None):
    from scipy.special import gammaln
    from scipy.special import psi
    from scipy.special import factorial
    from scipy.optimize import fmin_l_bfgs_b as optim
    '''Fit Total read depth with a negative binomial distribution
    Source: https://github.com/gokceneraslan/fit_nbinom/blob/master/fit_nbinom.py

    Parameters
    ----------
    X : np.ndarray
        - Vector of observations

    Returns
    -------
    n, p
        - Fitted parameters (for scipy parameterization)
    '''
    infinitesimal = np.finfo(np.float).eps

    def log_likelihood(params, *args):
        r, p = params
        X = args[0]
        N = X.size

        #MLE estimate based on the formula on Wikipedia:
        # http://en.wikipedia.org/wiki/Negative_binomial_distribution#Maximum_likelihood_estimation
        result = np.sum(gammaln(X + r)) \
            - np.sum(np.log(factorial(X))) \
            - N*(gammaln(r)) \
            + N*r*np.log(p) \
            + np.sum(X*np.log(1-(p if p < 1 else 1-infinitesimal)))

        return -result

    def log_likelihood_deriv(params, *args):
        r, p = params
        X = args[0]
        N = X.size

        pderiv = (N*r)/p - np.sum(X)/(1-(p if p < 1 else 1-infinitesimal))
        rderiv = np.sum(psi(X + r)) \
            - N*psi(r) \
            + N*np.log(p)

        return np.array([-rderiv, -pderiv])

    if initial_params is None:
        #reasonable initial values (from fitdistr function in R)
        m = np.mean(X)
        v = np.var(X)
        size = (m**2)/(v-m) if v > m else 10

        #convert mu/size parameterization to prob/size
        p0 = size / ((size+m) if size+m != 0 else 1)
        r0 = size
        initial_params = np.array([r0, p0])

    bounds = [(infinitesimal, None), (infinitesimal, 1)]
    optimres = optim(log_likelihood,
                     x0=initial_params,
                     #fprime=log_likelihood_deriv,
                     args=(X,),
                     approx_grad=1,
                     bounds=bounds)

    params = optimres[0]
    # Return n and p
    return params[0], params[1]

def _fit_qpcr(X):
    '''Take log of the data, subtract the mean of the triplicates (in log space),
    fit residual to a normal distribution, return the fitted standard deviation.

    Since the shape may not be totally uniform (not 3 samples for each qpcr data
    point, do it element by element)

    Parameters
    ----------
    X : 2-dim np.ndarray
        - dim 0: references the triplicate index
        - dim 2: references the triplicate

    Returns
    -------
    std (float)
        - This is the fitted shape parameter of the lognormal distribution
    '''
    if not pl.isarray(X):
        raise ValueError('`X` ({}) must be an array'.format(type(X)))

    # Take the log of the data
    for i in range(len(X)):
        X[i] = np.log(X[i])

    # Center for each triplicate in log space
    for i in range(len(X)):
        X[i] = X[i] - np.mean(X[i])

    # Fit with a normal distribution
    data = np.array([])
    for i in range(len(X)):
        data = np.append(data, X[i])
    _, s = scipy.stats.norm.fit(data)

    return s

def set_timepoints_without_timeseries(N, T_start, T_end, D, move_timepoints_if_fail=True, 
    essential_timepoints=None):
    '''Set time points for trajectory.

    This is a direct implementation of the algorithm specified in the method
    supplement.

    We deterministically allocate timepoints such that the density over certain
    time periods is proportionally higher than other time periods. The default
    denisty is 1. Additionally, we choose timepoints such that:
        t1 = set_timepoints_without_timeseries(N, ...)
        t2 = set_timepoints_without_timeseries(M, ...)

        We have two time-series that have all the same parameters but
        M >= N, then t1 is a subset of t2.

    Note that every subsection of the time series must have at least 1 point.

    Essential Timepoints
    --------------------
    Essential timepoints are timepoints that must be included.

    Moving timepoints if necessary
    ------------------------------
    Each interval needs to have at least one timepoint allocated to it (more if there
    are additional essential timepoints). If by the end of the allocation process there is 
    no time points in one of the intervals, this will fail. To avoid failing - we can set 
    the flag `move_timepoints_if_fail` to True. This moves a timepoint allocated from
    the most populated interval to the one/s with 0. If there are multiple timepoints 
    that have the same number of timepoints, then we use the earliest interval.
    Example:
        D_size = [5,6,2,1,2,3,0] >>> [5,5,2,1,2,3,1]
        D_size = [5,6,2,0,2,3,0] >>> [4,5,2,1,2,3,1]

    Allocating timepoints proportionally
    ------------------------------------
    We add the timepoints proportionally to each interval. If there are extra timepoints
    at the end of this step then we sequentially add them to the interval based on their
    priority. From a high level, the highest density has the highest priority. If multiple
    intervals have the same density, then we add the timepoints sequentially from left
    to right to only the intervals that have the high priority. Exceptions are: If an
    interval has no timepoints, it has the highest priority.

    Parameters
    ----------
    N : int
        Total number of timepoints to allocate
    T_start, T_end : float
        These are the start and end times of the time-series, respectivelly.
        `T_start` is inclusive and `T_end` is not inclusive.
    D : list(3-tuple), 3-tuple, None
        These are the densities which we allocate to
    move_timepoints_if_fail : bool
        If False, we raise an exception if there is an interval that does not have
        a timepoint by the end of allocation. If True, we move around the timepoints
        such that None of the intervals have no timepoints. We do this using the 
        algorithm described above.
    essential_timepoints : list, None
        If not None, these are a list of numeric timepoints that we must include.
    '''
    # Type and value checking
    if not pl.isbool(move_timepoints_if_fail):
        raise TypeError('`move_timepoints_if_fail` ({}) must be a bool'.format( 
            type(move_timepoints_if_fail)))
    if not pl.isint(N):
        raise TypeError('`N` ({}) must be an int'.format(type(N)))
    if N <= 0:
        raise ValueError('`N` ({}) must be > 0'.format(N))
    if pl.istuple(D):
        D = [D]
    for ele in D:
        if not pl.istuple(ele):
            raise TypeError('Every element in `D` ({}) must be a tuple'.format(
                type(ele)))
        if len(ele) != 3:
            raise ValueError('`Every element in `D` must be a tuple of length 3 ({})'.format( 
                len(ele)))
        start, stop, density = ele
        if not pl.isnumeric(start):
            raise TypeError('`start` ({}) must be a numeric'.format(type(start)))
        if start < T_start or start > T_end:
            raise ValueError('`start` ({}) out of range'.format(start))
        if not pl.isnumeric(stop):
            raise TypeError('`stop` ({}) must be a numeric'.format(type(stop)))
        if stop <= T_start or stop > T_end:
            raise ValueError('`stop` ({}) out of range'.format(stop))
        if stop <= start:
            raise ValueError('`stop` ({}) must be > `start` ({})'.format(stop, start))
        if not pl.isnumeric(density):
            raise TypeError('`density` ({}) must be a numeric'.format(density))
        if density < 1:
            raise ValueError('`density` ({}) must be >= 1'.format(density))
    
    # Make sure that all of the densities are disjoint to each other and that they
    # span the entire time-series
    start_there = False
    end_there = False
    for (s,e,d) in D:
        if s == T_start:
            start_there = True
        if e == T_end:
            end_there = True
    if not (start_there and end_there):
        raise ValueError('`D` ({}) does not include the start and end points'.format(D))
    for i, (si,ei,di) in enumerate(D):
        si_there = si == T_start
        ei_there = ei == T_end
        for j, (sj, ej, dj) in enumerate(D):
            if i == j:
                continue
            if (si >= sj and si < ej) or (ei > sj and ei <= ej):
                raise ValueError('`{}`th interval ({}) is contained in the `{}`th interval ({})'.format( 
                    i, (si,ei,di), j, (sj,ej,dj)))
            if not si_there:
                si_there = si == ej
            if not ei_there:
                ei_there = ei == sj
        if (not si_there) or (not ei_there):
            raise ValueError('In {}th interval, either `{}` or `{}` was not adjacent to any other ' \
                'interval ({})'.format(i, si, ei, D))
    total_length = 0
    for s,e,d in D:
        total_length += (e-s)
    if total_length != (T_end - T_start):
        raise ValueError('lengths ({}) and ({}) are not the same'.format(total_length, (T_end-T_start)))

    if essential_timepoints is not None:
        if pl.isnumeric(essential_timepoints):
            essential_timepoints = [essential_timepoints]
        if not pl.isarray(essential_timepoints):
            raise TypeError('`essential_timepoints` ({}) must be an array'.format(
                type(essential_timepoints)))
        for ele in essential_timepoints:
            if not pl.isnumeric(ele):
                raise TypeError('Each element in `essential_timepoints` ({} must be a numeric'.format(
                    type(ele)))
            if ele < T_start or ele > T_end:
                raise ValueError('Timepoint ({}) in `essential_timepoints` out of range ({}, {})'.format(
                    ele, T_start, T_end))
        essential_timepoints = np.unique(essential_timepoints)
        if len(essential_timepoints) > N:
            raise ValueError('There cannot be more essential timepoints ({}) than times to allocate ({})'.format(
                len(essential_timepoints), N))

    # Order D (by the start)
    D_new = []
    while len(D) > 0:
        # Get the smallest start
        min_s = None
        min_i = None
        for i, (s,_,_) in enumerate(D):
            if min_s is None:
                min_s = s
                min_i = 0
                continue
            if s < min_s:
                min_s = s
                min_i = i
        D_new.append(D.pop(min_i))
    D = D_new

    # Perform algorithm
    D_size = []
    for s,e,d in D:
        D_size.append(d * (e - s))
    D_total = np.sum(D_size)
    D_size = [int(s * N / D_total) for s in D_size] # casting to an int rounds down
    D_size = np.asarray(D_size, dtype=int)

    # Set the essential points
    min_D_size = []
    for s,e,_ in D:
        cnt = 1 # Must have at least 1 point
        if essential_timepoints is not None:
            for essential_point in essential_timepoints:
                if essential_point >= s and essential_point < e:
                    cnt += 1
        min_D_size.append(cnt)
    min_D_size = np.asarray(min_D_size, dtype=int)

    # Add in extra points if necessary
    if np.sum(D_size) != N:
        Nextra = N - np.sum(D_size)

        # First we check if there are any intervals that have less than necessary times
        while np.any(D_size < min_D_size):
            if Nextra == 0:
                break
            # Get the first index
            j = np.where(D_size < min_D_size)[0][0]
            D_size[j] += 1
            Nextra -= 1

        if Nextra > 0:
            dmax = np.max([d for (_,_,d) in D])
            I = [] # interval indexes that have the highest density
            for i, (s,e,d) in enumerate(D):
                if d == dmax:
                    I.append(i)
            if len(I) == 1:
                D_size[I[0]] += Nextra
            else:
                j = 0
                while Nextra > 0:
                    i = I[j]
                    D_size[i] += 1
                    Nextra -= 1
                    j += 1
                    if j == len(I):
                        j = 0
    
    # check if every interval has the minimum
    if np.any(D_size < min_D_size):
        if move_timepoints_if_fail:
            while np.any(D_size < min_D_size):
                logging.info('An interval was flagged to not have a time point allocated. Moved')
                # find the first instance of the max
                max_size = np.max(D_size)
                index_to_take = np.where(D_size == max_size)[0][0]
                
                # Find the first interval with less than the minimum
                index_to_give = np.where(D_size < min_D_size)[0][0]

                D_size[index_to_give] += 1
                D_size[index_to_take] -= 1

                if D_size[index_to_take] < min_D_size[index_to_take]:
                    print('D_size', D_size)
                    print('min_D_size', min_D_size)
                    print('index_to_take', index_to_take)
                    raise ValueError('This CAN happen but will only happen when there are a lot of essential' \
                        ' timepoints and not a lot of timepoints in total. Increase N.')

        else:
            raise ValueError('An interval has less than the minimum required timepoints. '\
                'Must increase N ({}) or set the flag ' \
                '`move_timepoints_if_fail` to True. D: {}, D_size: {}, min_D_size: {}'.format(
                    N, D, D_size, min_D_size))

    # Set essential timepoints
    essential_D = [[s] for s,_,_ in D]
    if essential_timepoints is not None:
        for essential_point in essential_timepoints:
            # Find where this essential timepoint goes
            for j, (s,e,_) in enumerate(D):
                if essential_point >= s and essential_point < e:
                    # Check to see if essential_timepoint is == to the start. skip if it is
                    if essential_point not in essential_D[j]:
                        essential_D[j].append(essential_point)


    T_N = []
    for i, (si, ei, di) in enumerate(D):
        n = D_size[i]
        Ti = essential_D[i]

        if n > 1:
            done = False
            while not done:
                Ttemp = []
                for k in range(len(Ti)-1):
                    if D_size[i] == (len(Ti) + len(Ttemp)):
                        break
                    newval = (Ti[k] + Ti[k+1])/2
                    Ttemp = np.append(Ttemp, newval)

                if D_size[i] == (len(Ti) + len(Ttemp)):
                    done = True
                else:
                    newval = (Ti[-1] + ei) / 2
                    Ttemp = np.append(Ttemp, newval)
                Ti = np.sort(np.append(Ti, Ttemp))
        T_N = np.append(T_N, Ti)

    return np.sort(T_N)

def issynthetic(x):
    '''Checks whether the input is a subclass of SyntheticData

    Parameters
    ----------
    x : any
        Input instance to check the type of SyntheticData
    
    Returns
    -------
    bool
        True if `x` is of type SyntheticData, else False
    '''
    return x is not None and issubclass(x.__class__, SyntheticData)

def make_semisynthetic(chain, min_bayes_factor, init_dist_start, init_dist_end,
    set_times=True, hdf5_filename=None):
    '''Make a semi synthetic system. We take the system learned in the chain and
    we set the modeling parameters of `SyntheticData` to the learned system. We assume
    that the chain that we pass in was run with a fixed topology.

    Notation
    --------
    In the following procedure, variable names are capitalized if they are constants or
    they are the variables/parameters that are taken from the variable `chain`.
    
    How the synthetic system is set
    -------------------------------
    n_taxas: Set to the number in chain.
    clustering: The clusters assignments are set to the value of the Clustering class
    interactions: Set to the expected value of the posterior. We only include interactions
        whose bayes factor is greater than `min_bayes_factor`.
    perturbations: The number of perturbations is set to be the same as what is in the
        chain. The topology and values of the perturbations are set to the expected value
        of the posterior. We only include perturbation effects whose bayes factor is
        greater than `min_bayes_factor`
    growth and self-interactions: These are set to the learned values for each of the 
        taxas.
    init_dist: The distirbution of the initial timepoints are set by fitting a log normal
        ditribution to the `init_dist_timepoint`th timepoint

    Parameters
    ----------
    chain : str, pylab.inference.BaseMCMC
        This is chain or the file location of the chain
    min_bayes_factor : numeric
        This is the minimum bayes factor needed for a perturbation/interaction
        to be used in the synthetic dataset
    set_times : bool
        If True, we set the times of the subject to be be a union of all of the subjects
        in chain
    init_dist_timepoint : numeric
        Which timepoint to set the initial distribution to. If nothing is provided it will
        set it to the first timepoint (0)
    hdf5_filename : str
        Location of the HDF5 object

    Returns
    -------
    synthetic.SyntheticData
    '''
    from names import STRNAMES

    if pl.isstr(chain):
        chain = pl.inference.BaseMCMC.load(chain)
    if not pl.isMCMC(chain):
        raise TypeError('`chain` ({}) is not a pylab.inference.BaseMCMC object'.format(
            type(chain)))
    if hdf5_filename is not None:
        chain.tracer.filename = hdf5_filename

    if not pl.isnumeric(min_bayes_factor):
        raise TypeError('`min_bayes_factor` ({}) nmust be a numeric'.format(
            type(min_bayes_factor)))
    if min_bayes_factor < 0:
        raise ValueError('`min_bayes_factor` ({}) must be >= 0'.format(min_bayes_factor))
    if not pl.isbool(set_times):
        raise TypeError('`set_times` ({}) must be a bool'.format(type(set_times)))
    # if init_dist_timepoint is None:
    #     init_dist_timepoint = 0
    # if not pl.isnumeric(init_dist_timepoint):
    #     raise TypeError('`init_dist_timepoint` ({}) must be a numeric'.format(
    #         type(init_dist_timepoint)))

    GRAPH = chain.graph
    DATA = GRAPH.data
    SUBJSET = DATA.subjects
    TAXAS = DATA.taxas
    logging.info('Number of Taxas ({})'.format(len(TAXAS)))

    n_days = -1
    for subj in SUBJSET:
        maxday = np.max(subj.times)
        if maxday > n_days:
            n_days = maxday
    logging.info('Number of days: {}'.format(n_days))

    GROWTH = GRAPH[STRNAMES.GROWTH_VALUE]
    SELF_INTERACTIONS = GRAPH[STRNAMES.SELF_INTERACTION_VALUE]
    INTERACTIONS = GRAPH[STRNAMES.INTERACTIONS_OBJ]
    CLUSTERING = GRAPH[STRNAMES.CLUSTERING_OBJ]
    PERTURBATIONS = GRAPH.perturbations
    perturbations_additive = GROWTH.perturbations_additive

    synth = SyntheticData(n_days=n_days,
        perturbations_additive=perturbations_additive)

    synth.set_taxas(taxas=DATA.taxas)
    synth.set_cluster_assignments(clusters=CLUSTERING.toarray())

    # Set the interactions
    # --------------------
    synth.dynamics.interactions = pl.Interactions(clustering=synth.dynamics.clustering, 
        use_indicators=True, G=synth.G)
    logging.info('Generating bayes factors')
    bayes_factors_taxas = INTERACTIONS.generate_bayes_factors_posthoc(
        prior=GRAPH[STRNAMES.CLUSTER_INTERACTION_INDICATOR].prior,
        section='posterior')
    logging.info('Getting values')
    cluster_interactions_taxas = pl.variables.summary(INTERACTIONS, set_nan_to_0=False,
        section='posterior', only=['mean'])['mean']

    logging.info('Set the interactions')
    for interaction in synth.dynamics.interactions:

        # Get the target and source cluster index of the interaction
        tcidx = synth.dynamics.clustering.cid2cidx[interaction.target_cid]
        scidx = synth.dynamics.clustering.cid2cidx[interaction.source_cid]

        # Get a set of target and source taxa indices of the interaction (it can
        # be any of them because we assume the chain was run with a fixed topology)
        taidx = list(CLUSTERING.clusters[CLUSTERING.order[tcidx]].members)[0]
        saidx = list(CLUSTERING.clusters[CLUSTERING.order[scidx]].members)[0]

        # If the bayes factor of the interaction is greater than the minimum bayes
        # factor, then we set the interaction. If it is less, then we set the 
        # indicator to false
        if bayes_factors_taxas[taidx, saidx] > min_bayes_factor:
            interaction.value = cluster_interactions_taxas[taidx, saidx]
            interaction.indicator = True
        else:
            interaction.value = 0
            interaction.indicator = False

    # Set the perturbations
    # ---------------------
    _ps = []
    for PERTURBATION in PERTURBATIONS:
        perturbation = pl.contrib.ClusterPerturbationEffect(
            start=PERTURBATION.start, end=PERTURBATION.end,
            G=synth.G, clustering=synth.dynamics.clustering)

        # values and bayes factors are on an taxa level
        values = pl.variables.summary(PERTURBATION, section='posterior', 
            only=['mean'])['mean']
        indicator_trace = ~np.isnan(PERTURBATION.get_trace_from_disk(section='posterior'))
        bayes_factor = pl.variables.summary(indicator_trace, only=['mean'])['mean']
        bayes_factor = bayes_factor/(1. - bayes_factor)
        bayes_factor = bayes_factor * (PERTURBATION.probability.prior.b.value + 1) / \
            (PERTURBATION.probability.prior.a.value + 1)

        cluster_order = synth.dynamics.clustering.order
        for cidx in range(len(CLUSTERING)):
            
            aidx = list(CLUSTERING.clusters[CLUSTERING.order[cidx]].members)[0]
            if bayes_factor[aidx] < min_bayes_factor:
                perturbation.magnitude.value[cluster_order[cidx]] = 0
                perturbation.indicator.value[cluster_order[cidx]] = False
            else:
                perturbation.magnitude.value[cluster_order[cidx]] = \
                    values[aidx]
                perturbation.indicator.value[cluster_order[cidx]] = True

        _ps.append(perturbation)
    synth.dynamics.perturbations = _ps

    # Set the growth and self-interactions
    # ------------------------------------
    synth.dynamics.growth = pl.variables.summary(GROWTH, section='posterior', 
        only=['mean'])['mean']
    synth.dynamics.self_interactions = pl.variables.summary(
        SELF_INTERACTIONS, section='posterior', only=['mean'])['mean']

    # Set the timepoints if possible
    # ------------------------------
    if set_times:
        times = []
        for subj in SUBJSET:
            times = np.append(times, subj.times)
        times = np.sort(np.unique(times))
        synth.master_times = times

    # Set the initial distribution
    # ----------------------------
    # values = []
    # for subj in SUBJSET:
    #     if init_dist_timepoint in subj.times:

    #         idx = np.searchsorted(subj.times, init_dist_timepoint)

    #         matrix = subj.matrix()['abs']
    #         values = np.append(values, matrix[:,idx].ravel())
    #     else:
    #         logging.warning('Timepoint `{}` not in subject `{}` ({})'.format(
    #             init_dist_timepoint, subj.name, subj.times))
    # values = np.asarray(values)
    # values = values[values > 0]
    # logvalues = np.log(values)
    synth.init_dist = pl.variables.Uniform(low=init_dist_start, high=init_dist_end)

    return synth
    
def subsample_timepoints(times, N, required=None):
    '''Subsample the timepoints `times` so that it has `N` timepoints.

    If required is not None, it is a list of timepoints that must be
    included (start/ends of perturbations, etc.) in the return times.

    Parameters
    ----------
    times : np.ndarray
        An array of timepoints
    N : int
        Total number of timepoints to be remaining
    required : None, np.ndarray
        An array of timepoints that need to be included in the return array
    '''
    if not pl.isarray(times):
        raise TypeError('`times` ({}) must be an array'.format(type(times)))
    times = np.sort(np.array(times))

    if not pl.isint(N):
        raise TypeError('`N` ({}) must be an int'.format(type(N)))
    if N <= 1:
        raise ValueError('`N` ({}) must be > 1'.format(N))
    if required is not None:
        if not pl.isarray(required):
            raise TypeError('`required` ({}) must be an array'.format(type(required)))
        for ele in required:
            if ele not in times:
                raise ValueError('({}) in `required` is not in times: ({})'.format(ele,times))

        if N - len(required) <= 0:
            raise ValueError('The number of required points ({}) is more than the total number ' \
                'of points that need to remain ({})'.format(len(required), N))

        add_at_end = []
        for ele in required:
            # get index at 
            idx = np.searchsorted(times, ele)
            add_at_end.append(ele)
            times = np.delete(times, [idx])
        N -= len(add_at_end)

    l = len(times)
    if N > l/2:
        # These are the indices to take away
        idxs = np.arange(0, l, step=l/(l-N), dtype=int)
        a = np.delete(times, idxs)

    else:
        # These are the indicates to keep
        idxs = np.arange(0, l, step=l/N, dtype=int)
        a = times[idxs]

    if required is not None:
        a = np.sort(np.append(a, add_at_end))
    return a

    