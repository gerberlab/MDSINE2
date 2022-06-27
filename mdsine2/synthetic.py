'''Generates synthetic systems
'''

import numpy as np
from mdsine2.logger import logger
import os
import random

from typing import Union, Dict, Iterator, Tuple, List, Any, IO, Callable

from . import model as plmodel
from . import pylab as pl
from .names import STRNAMES
from .pylab import variables, Study, BaseMCMC

class Synthetic(pl.Saveable):
    '''Generate synthetic and semi synthetic datasets for testing MDSINE2.

    Parameters
    ----------
    name : str
        Name of the dataset and the name of the graph
    seed : int
        Seed to initialize the system
    '''

    def __init__(self, name: str, seed: int):
        self.G = pl.Graph(name=name, seed=seed)
        self.model = plmodel.gLVDynamicsSingleClustering(growth=None, interactions=None)
        self.taxa = None # mdsine2.pylab.base.TaxaSet
        self.subjs = [] # list[str]
        self._data = {} # list[np.ndarray]
        self.seed = seed
        pl.random.seed(self.seed)

    @property
    def perturbations(self):
        return self.G.perturbations

    def icml_dynamics(self, n_taxa: int=13):
        '''Recreate the dynamical system used in the ICML paper [1]. If you use the
        default parameters you will get the same interaction matrix that was used in [1].

        We rescale the self-interactions and the interactions (and potentially growth) by 150
        so that the time-scales of the trajectories happen over days instead of minutes

        Parameters
        ----------
        n_taxa : int
            - These are how many taxa to include in the system. We will always have 3 clusters and
              the proportion of taxa in each cluster is as follows:
                - cluster 1 - 5/13
                - cluster 2 - 6/13
                - cluster 3 - 2/13
            - We assign each taxoon to each cluster the best we can, any extra taxa we put into cluster 3

        References
        ----------
        [1] Robust and Scalable Models of Microbiome Dynamics, TE Gibson, GK Gerber (2018)
        '''
        if not pl.isint(n_taxa):
            raise TypeError('`n_taxa` ({}) must be an int'.format(type(n_taxa)))
        if n_taxa < 3:
            raise ValueError('`n_taxa` ({}) must be >= 3'.format(n_taxa))

        # Make the TaxaSet
        taxa = pl.TaxaSet()
        for aidx in range(n_taxa):
            seq = ''.join(random.choice(['A', 'T', 'G', 'C']) for _ in range(250))
            taxa.add_taxon(name='TAXA_{}'.format(aidx+1), sequence=seq)
        self.taxa = taxa

        # Generate the cluster assignments
        c0size = int(5*n_taxa/13)
        c1size = int(6*n_taxa/13)
        c2size = int(n_taxa - c0size - c1size)

        clusters = np.asarray(([0]*c0size) + ([1]*c1size) + ([2]*c2size), dtype=int)
        clustering = pl.Clustering(items=taxa, clusters=clusters, name=STRNAMES.CLUSTERING_OBJ, G=self.G)

        # Generate the interactions
        interactions = pl.Interactions(clustering=clustering, use_indicators=True, G=self.G,
            name=STRNAMES.INTERACTIONS_OBJ)
        c0 = clustering.order[0]
        c1 = clustering.order[1]
        c2 = clustering.order[2]

        frac = 150*n_taxa/13
        for interaction in interactions:
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
        self_interactions = np.ones(n_taxa, dtype=float) * 5 / 150
        self_interactions = pl.Variable(value=self_interactions, shape=(n_taxa, ),
            name=STRNAMES.SELF_INTERACTION_VALUE, G=self.G)

        A = interactions.get_datalevel_value_matrix()
        for i in range(A.shape[0]):
            A[i,i] = -self_interactions.value[i]

        self.model.interactions = A #

        # Generate growth
        self.model.growth = pl.random.uniform.sample(low=0.1, high=0.12, size=n_taxa)

    def set_timepoints(self, times: np.ndarray):
        '''Set the timepoints of the trajectory

        Parameters
        ----------
        times : np.ndarray
        '''
        if not pl.isarray(times):
            raise TypeError('`times` ({}) must be an array'.format(times))
        times = np.sort(np.array(times))
        self.times = times

    def set_subjects(self, subjs: List[str]):
        '''Set the names of the subjects

        Parameters
        ----------
        subjs : list(str)
            A list of all the names of the subjects
        '''
        self.subjs = subjs

    def add_subject(self, name: str):
        '''Add another subject

        Parameters
        ----------
        name : str
            Subject to add
        '''
        if name in self.subjs:
            logger.warning('Subject {} already in synthetic. Skipping'.format(name))
            return
        if not pl.isstr(name):
            raise TypeError('`name` ({}) must be a str'.format(type(name)))
        self.subjs.append(name)

    def generate_trajectories(self, dt: float, init_dist: variables.Variable,
        processvar: plmodel.MultiplicativeGlobal=None):
        '''Forward simulate trajectories given the dynamics

        Parameters
        ----------
        dt : float
            Step size of the forward simulation
        init_dist : mdsine2.variables.RandomVariable
            This is the random distribution to intiialize the trajectories at
        processvar : mdsine2.model.MultiplicativeGlobal
            This is the process variance to simulate with.
            If this is not given then there will be no processvariance during simulation
        seed : int
            This is the seed to initialize at. If this is not given then the seed is no reset.

        :return: The raw simulated trajectory (for debugging purposes.)
        '''
        raw_trajs = {}

        for subj in self.subjs:
            if subj in self._data:
                continue
            logger.info('Forward simulating {}'.format(subj))
            init_abund = init_dist.sample(size=len(self.taxa)).reshape(-1,1)

            pert_start = None
            pert_end = None
            pert_eff = None

            if self.perturbations is not None:
                # Set it to the first subj in the list
                pert_start = []
                pert_end = []
                pert_eff = []
                for perturbation in self.perturbations:
                    sss = list(perturbation.starts.keys())[0]
                    start = perturbation.starts[sss]
                    end = perturbation.ends[sss]

                    # add name to the perturbations
                    perturbation.starts[subj] = start
                    perturbation.ends[subj] = end

                    pert_start.append(start)
                    pert_end.append(end)
                    pert_eff.append(perturbation.item_array())



            self.model.perturbation_ends = pert_end
            self.model.perturbation_starts = pert_start
            self.model.perturbations = pert_eff
            n_days = self.times[-1] + dt

            d = pl.integrate(dynamics=self.model, initial_conditions=init_abund,
                dt=dt, n_days=n_days, processvar=processvar,
                subsample=False, times=self.times)

            n_timepoints_to_integrate = np.ceil(n_days / dt)
            steps_per_day = int(n_timepoints_to_integrate / n_days)
            idxs = []
            for t in self.times:
                idxs.append(int(steps_per_day * t))
            X = d['X']

            self._data[subj] = X[:, idxs]
            raw_trajs[subj] = d
        return raw_trajs

    def simulateMeasurementNoise(self, a0: float, a1: float, qpcr_noise_scale: float,
        approx_read_depth: int, name: str='unnamed-study') -> Study:
        '''This function converts the synthetic trajectories into "real" data
        by simulating read counts and qPCR measurements.

        Simulating qPCR measurements
        ----------------------------
        We simulate the qPCR measurements with a lognormal distribution that was
        fitted using the data from `subjset`. We use this parameterization to sample
        a qPCR measurement with mean from the total biomass of the simulated data.

        Simulating count data
        ---------------------
        First, we get the approximate read depth `r_k` with `approx_read_depth`. We then use `r_k`,
        `a_0`, and `a_1` with the relative abundances to sample from a negative binomial
        distirbution. We then use the relative abundances from this sample as the concentrations
        for a multinomial distribution with read depth `r_k`.

        Parameters
        ----------
        a0, a1 : numeric
            These are the negative binomial dispersion parameters that we are using to
            simulate the data
        qpcr_noise_scale : numeric
            This is the parameter to scale the `s` parameter learned by the lognormal
            distribution.
        approx_read_depth : int
            This is the read depth to simulate to
        name : str
            Name of the study

        Returns
        -------
        mdsine2.Study
        '''
        if self.times is None or self._data is None or self.subjs is None:
            raise ValueError('Need to fully initialize the system before')

        logger.info('Fitting real data')

        # Make the study object
        study = pl.Study(taxa=self.taxa, name=name)
        for subjname in self.subjs:
            study.add_subject(name=subjname)

        # Add times for each subject
        for subj in study:
            subj.times = self.times

        # Make the qPCR measurements
        for subj in study:
            total_abund = np.sum(self._data[subj.name], axis=0)

            for tidx, t in enumerate(self.times):
                # Get the total abundance

                triplicates = np.exp(np.log(total_abund[tidx]) + \
                    qpcr_noise_scale * pl.random.normal.sample(size=3))
                subj.qpcr[t] = pl.qPCRdata(cfus=triplicates, mass=1., dilution_factor=1.)

        # Make the reads
        for subj in study:
            for tidx, t in enumerate(self.times):

                total_mass = np.sum(self._data[subj.name][:, tidx])
                rel_abund = (self._data[subj.name][:, tidx] + 1e-20) / total_mass

                phi = approx_read_depth * rel_abund
                eps = a0/rel_abund + a1

                logger.debug(f'negbin mean = {phi}')
                logger.debug(f'negbin dispersion = {eps}')

                reads = pl.random.negative_binomial.sample(mean=phi, dispersion=eps)
                subj.reads[t] = reads

        # Make the perturbations:
        if self.perturbations is not None and len(self.perturbations) > 0:
            study.perturbations = self.G.perturbations

        return study

def make_semisynthetic(chain: BaseMCMC, seed: int, min_bayes_factor: Union[float, int], name: str=None,
    set_times: bool=True) -> Synthetic:
    '''Make a semi synthetic system. We assume that the chain that we pass in was
    run with FIXED CLUSTERING. We assume this because we need to set the cluster-cluster
    interactions and the cluster perturbations.

    How the synthetic system is set
    -------------------------------
    - n_taxa: Set to the number in chain.
    - clustering: The clusters assignments are set to the value of the Clustering class
    - interactions: Set to the expected value of the posterior. We only include interactions
      whose bayes factor is greater than `min_bayes_factor`.
    - perturbations: The number of perturbations is set to be the same as what is in the
      chain. The topology and values of the perturbations are set to the expected value
      of the posterior. We only include perturbation effects whose bayes factor is
      greater than `min_bayes_factor`
    - growth and self-interactions: These are set to the learned values for each of the taxa.
    - init_dist: The distirbution of the initial timepoints are set by fitting a log normal
      ditribution to the `init_dist_timepoint`th timepoint

    Parameters
    ----------
    chain : str, pylab.inference.BaseMCMC
        This is chain or the file location of the chain
    min_bayes_factor : numeric
        This is the minimum bayes factor needed for a perturbation/interaction
        to be used in the synthetic dataset
    name : str
        Name of the synthetic object
    set_times : bool
        If True, we set the times of the subject to be be a union of all of the subjects
        in chain

    Returns
    -------
    synthetic.Synthetic
    '''
    from .util import condense_fixed_clustering_interaction_matrix
    from .util import generate_interation_bayes_factors_posthoc
    from .util import condense_fixed_clustering_perturbation
    from .util import generate_perturbation_bayes_factors_posthoc

    if pl.isstr(chain):
        chain = pl.BaseMCMC.load(chain)
    if not pl.isMCMC(chain):
        raise TypeError('`chain` ({}) is not a mdsine2.BaseMCMC object'.format(type(chain)))

    if not pl.isnumeric(min_bayes_factor):
        raise TypeError('`min_bayes_factor` ({}) nmust be a numeric'.format(
            type(min_bayes_factor)))
    if min_bayes_factor < 0:
        raise ValueError('`min_bayes_factor` ({}) must be >= 0'.format(min_bayes_factor))
    if not pl.isbool(set_times):
        raise TypeError('`set_times` ({}) must be a bool'.format(type(set_times)))

    if name is None:
        name = chain.graph.data.subjects.name + '-synthetic'
    syn = Synthetic(name=name, seed=seed)
    syn.set_subjects(subjs=[subj.name for subj in chain.graph.data.subjects])
    syn.taxa = chain.graph.data.taxa

    if set_times:
        syn.times = chain.graph.data.subjects.times('union')
    else:
        syn.times = None

    # Set the clustering
    # ------------------
    cluster_assignments = chain.graph[STRNAMES.CLUSTERING_OBJ].toarray()
    clustering = pl.Clustering(cluster_assignments, G=syn.G, items=syn.taxa,
        name=STRNAMES.CLUSTERING_OBJ)

    # Set the interactions
    # --------------------
    self_interactions = pl.summary(chain.graph[STRNAMES.SELF_INTERACTION_VALUE])['mean']
    A = pl.summary(chain.graph[STRNAMES.INTERACTIONS_OBJ], set_nan_to_0=True)['mean']
    A_cluster = condense_fixed_clustering_interaction_matrix(A,
        clustering=chain.graph[STRNAMES.CLUSTERING_OBJ])

    bf = generate_interation_bayes_factors_posthoc(mcmc=chain)
    bf_cluster = condense_fixed_clustering_interaction_matrix(bf,
        clustering=chain.graph[STRNAMES.CLUSTERING_OBJ])

    interactions = pl.Interactions(clustering=clustering, use_indicators=True,
        name=STRNAMES.INTERACTIONS_OBJ, G=syn.G)

    for interaction in interactions:
        target_cid = interaction.target_cid
        source_cid = interaction.source_cid

        tcidx = clustering.cid2cidx[target_cid]
        scidx = clustering.cid2cidx[source_cid]

        if bf_cluster[tcidx, scidx] >= min_bayes_factor:
            interaction.value = A_cluster[tcidx, scidx]
            interaction.indicator = True
        else:
            interaction.value = 0
            interaction.indicator = False

    A = interactions.get_datalevel_value_matrix()
    for i in range(A.shape[0]):
        A[i,i] = -self_interactions[i]

    syn.model.interactions = A

    # Set perturbations (if necessary)
    # --------------------------------
    if chain.graph.perturbations is not None:
        for perturbation_master in chain.graph.perturbations:
            perturbation = pl.ClusterPerturbationEffect(
                starts=perturbation_master.starts,
                ends=perturbation_master.ends,
                name=perturbation_master.name,
                G=syn.G, clustering=clustering)

            # Get the values and the bayes factors
            bf = generate_perturbation_bayes_factors_posthoc(chain,
                perturbation=perturbation_master)
            bf_cluster = condense_fixed_clustering_perturbation(bf,
                clustering=chain.graph[STRNAMES.CLUSTERING_OBJ])

            M = pl.summary(perturbation_master, set_nan_to_0=True)['mean']
            M_cluster = condense_fixed_clustering_perturbation(M,
                clustering=chain.graph[STRNAMES.CLUSTERING_OBJ])

            for cidx, cluster in enumerate(clustering):
                if bf_cluster[cidx] >= min_bayes_factor:
                    perturbation.magnitude.value[cluster.id] = M_cluster[cidx]
                    perturbation.indicator.value[cluster.id] = True
                else:
                    perturbation.magnitude.value[cluster.id] = 0
                    perturbation.indicator.value[cluster.id] = False

    # Set the growth rates
    # --------------------
    syn.model.growth = pl.summary(chain.graph[STRNAMES.GROWTH_VALUE])['mean']
    return syn

def subsample_timepoints(times: np.ndarray, N: int, required: np.ndarray=None) -> np.ndarray:
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
    if N >= l/2:
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
