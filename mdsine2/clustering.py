'''Clustering parameters for the posterior
'''

import logging
import time
import itertools
import psutil

import numpy as np
import numba
import scipy.sparse
import numpy.random as npr
import scipy.stats
import scipy
import math

from .util import build_prior_covariance, build_prior_mean, sample_categorical_log, \
    log_det, pinv, generate_cluster_assignments_posthoc
from .names import STRNAMES, REPRNAMES
from . import pylab as pl

from . import visualization
import matplotlib.pyplot as plt


class Concentration(pl.variables.Gamma):
    '''Defines the posterior for the concentration parameter that is used
    in learning the cluster assignments.
    The posterior is implemented as it is describes in 'Bayesian Inference
    for Density Estimation' by M. D. Escobar and M. West, 1995.
    '''
    def __init__(self, prior, value=None, n_iter=None, **kwargs):
        '''Parameters

        value (float, int)
            - Initial value of the concentration
            - Default value is the mean of the prior
        '''
        kwargs['name'] = STRNAMES.CONCENTRATION
        # initialize shape and scale as the same as the priors
        # we will be updating this later
        pl.variables.Gamma.__init__(self, shape=prior.shape.value, scale=prior.scale.value,
            dtype=float, **kwargs)
        self.add_prior(prior)

    def initialize(self, value_option, hyperparam_option, n_iter=None, value=None,
        shape=None, scale=None, delay=0):
        '''Initialize the hyperparameters of the beta prior

        Parameters
        ----------
        value_option (str)
            - Options to initialize the value
            - 'manual'
                - Set the value manually, `value` must also be specified
            - 'auto', 'prior-mean'
                - Set to the mean of the prior
        hyperparam_option (str)
            - Options ot initialize the hyperparameters
            - Options
                - 'manual'
                    - Set the values manually. `shape` and `scale` must also be specified
                - 'auto', 'diffuse'
                    - shape = 1e-5, scale= 1e5
        shape, scale (int, float)
            - User specified values
            - Only necessary if `hyperparam_option` == 'manual'
        '''
        if not pl.isint(delay):
            raise TypeError('`delay` ({}) must be an int'.format(type(delay)))
        if delay < 0:
            raise ValueError('`delay` ({}) must be >= 0'.format(delay))
        self.delay = delay

        if hyperparam_option == 'manual':
            if pl.isnumeric(shape) and pl.isnumeric(scale):
                self.prior.shape.override_value(shape)
                self.prior.scale.override_value(scale)
            else:
                raise ValueError('shape ({}) and scale ({}) must be numeric' \
                    ' (float, int)'.format(shape.__class__, scale.__class__))

            if not pl.isint(n_iter):
                raise ValueError('`n_iter` ({}) needs ot be an int'.format(n_iter.__class__))
            self.n_iter = n_iter

        elif hyperparam_option == 'strong-few':
            self.prior.shape.override_value(1)
            self.prior.scale.override_value(1)
            self.n_iter = 20

        elif hyperparam_option in ['diffuse', 'auto']:
            self.prior.shape.override_value(1e-5)
            self.prior.scale.override_value(1e5)
            self.n_iter = 20
        else:
            raise ValueError('hyperparam_option `{}` not recognized'.format(hyperparam_option))

        if value_option == 'manual':
            if pl.isnumeric(value):
                self.value = value
            else:
                raise ValueError('`value` ({}) must be numeric (float, int)'.format(
                    value.__class__))
        elif value_option in ['auto', 'prior-mean']:
            self.value = self.prior.mean()
        else:
            raise ValueError('value_option `{}` not recognized'.format(value_option))

        self.shape.value = self.prior.shape.value
        self.scale.value = self.prior.scale.value
        logging.info('Cluster Concentration initialization results:\n' \
            '\tprior shape: {}\n' \
            '\tprior scale: {}\n' \
            '\tvalue: {}'.format(
                self.prior.shape.value, self.prior.scale.value, self.value))

    def update(self):
        '''Sample the posterior of the concentration parameter
        '''
        if self.sample_iter < self.delay:
            return

        clustering = self.G[REPRNAMES.CLUSTER_INTERACTION_VALUE].clustering
        k = len(clustering)
        n = self.G.data.n_asvs
        # If there are 1 or 0 clusters, do not update
        # if k <= 1:
        #     return
        for i in range(self.n_iter):
            #first sample eta from a beta distribution
            eta = npr.beta(self.value+1,n)
            #sample alpha from a mixture of gammas
            pi_eta = [0.0, n]
            pi_eta[0] = (self.prior.shape.value+k-1.)/(1/(self.prior.scale.value)-np.log(eta))
            self.scale.value =  1/(1/self.prior.scale.value - np.log(eta))
            self.shape.value = self.prior.shape.value + k
            if np.random.choice([0,1], p=pi_eta/np.sum(pi_eta)) != 0:
                self.shape.value -= 1
            self.sample()
            # print('pi_eta[0]',pi_eta[0])

    def visualize_posterior(self, path, f, section='posterior'):
        '''Render the traces in the folder `basepath` and write the 
        learned values to the file `f`.

        Parameters
        ----------
        path : str
            This is the path to write the files to
        f : _io.TextIOWrapper
            File that we are writing the values to
        section : str
            Section of the trace to compute on. Options:
                'posterior' : posterior samples
                'burnin' : burn-in samples
                'entire' : both burn-in and posterior samples

        Returns
        -------
        _io.TextIOWrapper
        '''
        f.write('\n\n###################################\n')
        f.write(self.name)
        f.write('\n###################################\n')
        if not self.G.inference.is_being_traced(self):
            f.write('`{}` not learned\n\tValue: {}\n'.format(self.name, self.value))
            return f

        summ = pl.summary(self, section=section)
        for k,v in summ.items():
            f.write('\t{}: {}\n'.format(k,v))

        ax1, _ = visualization.render_trace(var=self, plt_type='both', 
            section=section, include_burnin=True, log_scale=True, rasterized=True)

        l,h = ax1.get_xlim()
        xs = np.arange(l,h,step=(h-l)/1000) 
        ys = []
        for x in xs:
            ys.append(self.prior.pdf(value=x))
        ax1.plot(xs, ys, label='prior', alpha=0.5, color='red')
        ax1.legend()

        fig = plt.gcf()
        fig.suptitle('Concentration')
        plt.savefig(path)
        plt.close()
        f.close()


class ClusterAssignments(pl.graph.Node):
    '''This is the posterior of the cluster assignments for each ASV.

    To calculate the loglikelihood of an ASV being in a cluster, we have to
    marginalize out the cluster that the ASV belongs to - this means marginalizing
    out the interactions going in and out of the cluster in question, along with all
    the perturbations associated with it.
    '''
    def __init__(self, clustering, concentration, m=1, mp=None, **kwargs):
        '''Parameters

        clustering (pylab.cluster.Clustering)
            - Defines the clusters
        concentration (pylab.Variable or subclass of pylab.Variable)
            - Defines the concentration parameter for the base distribution
        m (int, Optional)
            - Number of auxiliary variables defined in the model
            - Default is 1
        mp : str, None
            This is the type of multiprocessing it is going to be. Options:
                None
                    No multiprocessing
                'full-#'
                    Send out to the different processors, where '#' is the number of
                    processors to make
                'debug'
                    Send out to the different classes but stay on a single core. This
                    is necessary for benchmarking and easier debugging.
        '''
        self.clustering = clustering
        self.concentration = concentration
        self.m = m
        self.mp = mp
        self._strtime = -1
        # self.compute_device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.mp is not None:
            self.update = self.update_mp
        else:
            self.update = self.update_slow_fast

        kwargs['name'] = STRNAMES.CLUSTERING
        pl.graph.Node.__init__(self, **kwargs)

    def __str__(self):
        return str(self.clustering) + '\nTotal time: {}'.format(self._strtime)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('pool', None)
        state.pop('actors', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.pool = []
        self.actors = None

    @property
    def value(self):
        '''This is so that the cluster assignments get printed in inference.MCMC
        '''
        return self.clustering

    @property
    def sample_iter(self):
        return self.clustering.n_clusters.sample_iter

    def initialize(self, value_option, hyperparam_option=None, value=None, n_clusters=None,
        delay=0, run_every_n_iterations=1):
        '''Initialize the cluster assingments - there are no hyperparamters to
        initialize because the concentration is initialized somewhere else

        Note - if `n_clusters` is not specified and the cluster initialization
        method requires it - it will be set to the expected number of clusters
        which = log(n_asvs)/log(2)

        Parameters
        ----------
        value_option : str
            The different methods to initialize the clusters
            Options
                'manual'
                    Manually set the cluster assignments
                'no-clusters'
                    Every ASV in their own cluster
                'random'
                    Every ASV is randomly assigned to the number of clusters. `n_clusters` required
                'taxonomy'
                    Cluster ASVs based on their taxonomic similarity. `n_clusters` required
                'sequence'
                    Cluster ASVs based on their sequence similarity. `n_clusters` required
                'phylogeny'
                    Cluster ASVs based on their phylogenetic similarity. `n_clusters` required
                'spearman', 'auto'
                    Creates a distance matrix based on the spearman rank similarity
                    between two trajectories. We use the raw data. `n_clusters` required
                'fixed-topology'
                    Sets the clustering assignment to the most likely clustering configuration
                    specified in the graph at the location `value` (`value` is a str).
                    We take the mean coclusterings and do agglomerative clustering on that matrix
                    with the `mode` number of clusters.
        hyperparam_option : None
            Not used in this function - only here for API consistency
        value : list of list
            Cluster assingments for each of the ASVs
            Only necessary if `value_option` == 'manual'
        n_clusters : int, str
            Necessary if `value_option` is not 'manual' or 'no-clusters'
            If str, options:
                'expected', 'auto': log_2(n_asvs)
        run_every_n_iterations : int
            Only run the update every `run_every_n_iterations` iterations
        '''
        from sklearn.cluster import AgglomerativeClustering
        asvs = self.G.data.asvs

        if not pl.isint(run_every_n_iterations):
            raise TypeError('`run_every_n_iterations` ({}) must be an int'.format(
                type(run_every_n_iterations)))
        if run_every_n_iterations <= 0:
            raise ValueError('`run_every_n_iterations` ({}) must be >= 0'.format(
                run_every_n_iterations))

        self.run_every_n_iterations = run_every_n_iterations
        self.delay = delay

        if not pl.isstr(value_option):
            raise TypeError('`value_option` ({}) must be a str'.format(type(value_option)))
        if value_option not in ['manual', 'no-clusters', 'fixed-topology']:
            if pl.isstr(n_clusters):
                if n_clusters in ['expected', 'auto']:
                    n_clusters = int(round(np.log(len(asvs))/np.log(2)))
                else:
                    raise ValueError('`n_clusters` ({}) not recognized'.format(n_clusters))
            if not pl.isint(n_clusters):
                raise TypeError('`n_clusters` ({}) must be a str or an int'.format(type(n_clusters)))
            if n_clusters <= 0:
                raise ValueError('`n_clusters` ({}) must be > 0'.format(n_clusters))
            if n_clusters > self.G.data.n_asvs:
                raise ValueError('`n_clusters` ({}) must be <= than the number of ASVs ({})'.format(
                    n_clusters, self.G.data.n_asvs))

        if value_option == 'manual':
            # Check that all of the ASVs are in the init and that it is in the right
            # structure
            if not pl.isarray(value):
                raise ValueError('if `value_option` is "manual", value ({}) must ' \
                    'be of type array'.format(value.__class__))
            clusters = list(value)

            idxs_to_delete = []
            for idx, cluster in enumerate(clusters):
                if not pl.isarray(cluster):
                    raise ValueError('cluster at index `{}` ({}) is not an array'.format(
                        idx, cluster))
                cluster = list(cluster)
                if len(cluster) == 0:
                    logging.warning('Cluster index {} has 0 elements, deleting'.format(
                        idx))
                    idxs_to_delete.append(idx)
            if len(idxs_to_delete) > 0:
                clusters = np.delete(clusters, idxs_to_delete).tolist()

            all_oidxs = set()
            for cluster in clusters:
                for oidx in cluster:
                    if not pl.isint(oidx):
                        raise ValueError('`oidx` ({}) must be an int'.format(oidx.__class__))
                    if oidx >= len(asvs):
                        raise ValueError('oidx `{}` not in our ASVSet'.format(oidx))
                    all_oidxs.add(oidx)

            for oidx in range(len(asvs)):
                if oidx not in all_oidxs:
                    raise ValueError('oidx `{}` in ASVSet not in `value` ({})'.format(
                        oidx, value))
            # Now everything is checked and valid

        elif value_option == 'fixed-topology':
            logging.info('Fixed topology initialization')
            if not pl.isstr(value):
                raise TypeError('`value` ({}) must be a str'.format(value))

            CHAIN2 = pl.inference.BaseMCMC.load(value)
            CLUSTERING2 = CHAIN2.graph[STRNAMES.CLUSTERING_OBJ]
            ASVS2 = CHAIN2.graph.data.asvs
            asvs_curr = self.G.data.asvs
            for asv in ASVS2:
                if asv.name not in asvs_curr:
                    raise ValueError('Cannot perform fixed topology because the ASV {} in ' \
                        'the passed in clustering is not in this clustering: {}'.format(
                            asv.name, asvs_curr.names.order))
            for asv in asvs_curr:
                if asv.name not in ASVS2:
                    raise ValueError('Cannot perform fixed topology because the ASV {} in ' \
                        'the current clustering is not in the passed in clustering: {}'.format(
                            asv.name, ASVS2.names.order))

            # Get the most likely cluster configuration and set as the value for the passed in cluster
            ret = generate_cluster_assignments_posthoc(CLUSTERING2, n_clusters='mode', set_as_value=True)
            ca = {}
            for aidx, cidx in enumerate(ret):
                if cidx not in ca:
                    ca[cidx] = []
                ca[cidx].append(aidx)
            ret = []
            for v in ca.values():
                ret.append(v)
            CLUSTERING2.from_array(ret)
            logging.info('Clustering set to:\n{}'.format(str(CLUSTERING2)))

            # Set the passed in cluster assignment as the current cluster assignment
            # Need to be careful because the indices of the ASVs might not line up
            clusters = []
            for cluster in CLUSTERING2:
                anames = [asvs_curr[ASVS2.names.order[aidx]].name for aidx in cluster.members]
                aidxs = [asvs_curr[aname].idx for aname in anames]
                clusters.append(aidxs)

        elif value_option == 'no-clusters':
            clusters = []
            for oidx in range(len(asvs)):
                clusters.append([oidx])

        elif value_option == 'random':
            clusters = {}
            for oidx in range(len(asvs)):
                idx = npr.choice(n_clusters)
                if idx in clusters:
                    clusters[idx].append(oidx)
                else:
                    clusters[idx] = [oidx]
            c = []
            for cid in clusters.keys():
                c.append(clusters[cid])
            clusters = c

        elif value_option == 'taxonomy':
            # Create an affinity matrix, we can precompute the self-similarity to 1
            M = np.diag(np.ones(len(asvs), dtype=float))
            for i, oid1 in enumerate(asvs.ids.order):
                for j, oid2 in enumerate(asvs.ids.order):
                    if i == j:
                        continue
                    M[i,j] = asvs.taxonomic_similarity(oid1=oid1, oid2=oid2)

            c = AgglomerativeClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                linkage='complete')
            assignments = c.fit_predict(1-M)

            # Convert assignments into clusters
            clusters = {}
            for oidx,cidx in enumerate(assignments):
                if cidx not in clusters:
                    clusters[cidx] = []
                clusters[cidx].append(oidx)
            clusters = [val for val in clusters.values()]

        elif value_option == 'sequence':
            import diversity

            logging.info('Making affinity matrix from sequences')
            evenness = np.diag(np.ones(len(self.G.data.asvs), dtype=float))

            for i in range(len(self.G.data.asvs)):
                for j in range(len(self.G.data.asvs)):
                    if j <= i:
                        continue
                    # Subtract because we want to make a similarity matrix
                    dist = 1-diversity.beta.hamming(
                        list(self.G.data.asvs[i].sequence),
                        list(self.G.data.asvs[j].sequence))
                    evenness[i,j] = dist
                    evenness[j,i] = dist

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

        elif value_option == 'spearman':
            # Use spearman correlation to create a distance matrix
            # Use agglomerative clustering to make the clusters based
            # on distance matrix (distance = 1 - pearson(x,y))
            dm = np.zeros(shape=(len(asvs), len(asvs)))
            data = []
            for ridx in range(self.G.data.n_replicates):
                data.append(self.G.data.abs_data[ridx])
            data = np.hstack(data)
            for i in range(len(asvs)):
                for j in range(i+1):
                    distance = (1 - scipy.stats.spearmanr(data[i, :], data[j, :])[0])/2
                    dm[i,j] = distance
                    dm[j,i] = distance

            c = AgglomerativeClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                linkage='complete')
            assignments = c.fit_predict(dm)

            # convert into clusters
            clusters = {}
            for oidx, cidx in enumerate(assignments):
                if cidx not in clusters:
                    clusters[cidx] = []
                clusters[cidx].append(oidx)
            clusters = [val for val in clusters.values()]

        elif value_option == 'phylogeny':
            raise NotImplementedError('`phylogeny` not implemented yet')

        else:
            raise ValueError('`value_option` "{}" not recognized'.format(value_option))

        # Move all the ASVs into their assigned clusters
        for cluster in clusters:
            cid = None
            for oidx in cluster:
                if cid is None:
                    # make new cluster
                    cid = self.clustering.make_new_cluster_with(idx=oidx)
                else:
                    self.clustering.move_item(idx=oidx, cid=cid)
        logging.info('Cluster Assingments initialization results:\n{}'.format(
            str(self.clustering)))
        self._there_are_perturbations = self.G.perturbations is not None

        # Initialize the multiprocessors if necessary
        if self.mp is not None:
            if not pl.isstr(self.mp):
                raise TypeError('`mp` ({}) must be a str'.format(type(self.mp)))
            if 'full' in self.mp:
                n_cpus = self.mp.split('-')[1]
                if n_cpus == 'auto':
                    self.n_cpus = psutil.cpu_count(logical=False)
                else:
                    try:
                        self.n_cpus = int(n_cpus)
                    except:
                        raise ValueError('`mp` ({}) not recognized'.format(self.mp))
                self.pool = pl.multiprocessing.PersistentPool(ptype='dasw', G=self.G)
            elif self.mp == 'debug':
                self.pool = None
            else:
                raise ValueError('`mp` ({}) not recognized'.format(self.mp))
        else:
            self.pool = None

        self.ndts_bias = []
        self.n_asvs = len(self.G.data.asvs)
        self.n_replicates = self.G.data.n_replicates
        self.n_dts_for_replicate = self.G.data.n_dts_for_replicate
        self.total_dts = np.sum(self.n_dts_for_replicate)
        for ridx in range(self.G.data.n_replicates):
            self.ndts_bias.append(
                np.arange(0, self.G.data.n_dts_for_replicate[ridx] * self.n_asvs, self.n_asvs))
        self.replicate_bias = np.zeros(self.n_replicates, dtype=int)
        for ridx in range(1, self.n_replicates):
            self.replicate_bias[ridx] = self.replicate_bias[ridx-1] + \
                self.n_asvs * self.n_dts_for_replicate[ridx - 1]

    def visualize_posterior(self, basepath, f, section='posterior', asv_formatter='%(name)s',
        yticklabels='%(name)s %(index)s', xticklabels='%(index)s'):
        '''Render the traces in the folder `basepath` and write the 
        learned values to the file `f`.

        Parameters
        ----------
        basepath : str
            This is the loction to write the files to
        f : _io.TextIOWrapper
            File that we are writing the values to
        section : str
            Section of the trace to compute on. Options:
                'posterior' : posterior samples
                'burnin' : burn-in samples
                'entire' : both burn-in and posterior samples

        Returns
        -------
        _io.TextIOWrapper
        '''
        asvs = self.G.data.asvs
        f.write('\n\n###################################\n')
        f.write(self.name)
        f.write('\n###################################\n')
        if not self.G.inference.is_in_inference_order(self):
            f.write('`{}` not learned. These were the fixed cluster assignments\n'.format(self.name))
            for cidx, cluster in enumerate(self.clustering):
                f.write('Cluster {}:\n'.format(cidx+1))
                for aidx in cluster.members:
                    label = pl.asvname_formatter(format=asv_formatter, asv=asvs[aidx], asvs=asvs)
                    f.write('\t- {}\n'.format(label))

            return f

        # Coclusters
        cocluster_trace = self.clustering.coclusters.get_trace_from_disk(section=section)
        coclusters = pl.variables.summary(cocluster_trace, section=section)['mean']
        for i in range(coclusters.shape[0]):
            coclusters[i,i] = np.nan

        visualization.render_cocluster_proportions(
            coclusters=coclusters, asvs=self.G.data.asvs, clustering=self.clustering,
            yticklabels=yticklabels, include_tick_marks=False, xticklabels=xticklabels,
            title='Cluster Assignments')
        fig = plt.gcf()
        fig.tight_layout()
        plt.savefig(basepath + 'coclusters.pdf')
        plt.close()

        # N clusters
        visualization.render_trace(var=self.clustering.n_clusters, plt_type='both', 
            section=section, include_burnin=True, rasterized=True)
        fig = plt.gcf()
        fig.suptitle('Number of Clusters')
        plt.savefig(basepath + 'n_clusters.pdf')
        plt.close()

        ca = generate_cluster_assignments_posthoc(clustering=self.clustering, n_clusters='mode', 
            section=section)
        cluster_assignments = {}
        for idx, assignment in enumerate(ca):
            if assignment in cluster_assignments:
                cluster_assignments[assignment].append(idx)
            else:
                cluster_assignments[assignment] = [idx]

        f.write('Mode number of clusters: {}\n'.format(len(self.clustering)))
        for idx,lst in enumerate(cluster_assignments.values()):
            f.write('Cluster {} - Size {}\n'.format(idx+1, len(lst)))
            for oidx in lst:
                # Get rid of index because that does not really make sense here
                label = pl.asvname_formatter(format=asv_formatter, asv=asvs[oidx], asvs=asvs)
                f.write('\t- {}\n'.format(label))
        
        return f

    def set_trace(self):
        self.clustering.set_trace()

    def add_trace(self):
        self.clustering.add_trace()

    def add_init_value(self):
        self.clustering.add_init_value()

    def kill(self):
        if pl.ispersistentpool(self.pool):
            # For pylab multiprocessing, explicitly kill them
            self.pool.kill()
        return

    # Update super safe - meant to be used during debugging
    # =====================================================
    def update_slow(self):
        ''' This is updating the new cluster. Depending on the iteration you do
        either split-merge Metropolis-Hasting update or a regular Gibbs update. To
        get highest mixing we alternate between each.
        '''
        if self.clustering.n_clusters.sample_iter < self.delay:
            return

        if self.clustering.n_clusters.sample_iter % self.run_every_n_iterations != 0:
           return

        # print('in clustering')
        start_time = time.time()
        oidxs = npr.permutation(np.arange(len(self.G.data.asvs)))

        for oidx in oidxs:
            self.gibbs_update_single_asv_slow(oidx=oidx)
        self._strtime = time.time() - start_time

    def gibbs_update_single_asv_slow(self, oidx):
        '''The update function is based off of Algorithm 8 in 'Markov Chain
        Sampling Methods for Dirichlet Process Mixture Models' by Radford M.
        Neal, 2000.

        Calculate the marginal likelihood of the asv in every cluster
        and a new cluster then sample from `self.sample_categorical_log`
        to get the cluster assignment.

        Parameters
        ----------
        oidx : int
            ASV index that we are updating the cluster assignment of
        '''
        curr_cluster = self.clustering.idx2cid[oidx]
        concentration = self.concentration.value

        # start as a dictionary then send values to `sample_categorical_log`
        LOG_P = []
        LOG_KEYS = []

        # Calculate current cluster
        # =========================
        # If the element is already in its own cluster, use the new cluster case
        if self.clustering.clusters[curr_cluster].size == 1:
            a = np.log(concentration/self.m)
        else:
            a = np.log(self.clustering.clusters[curr_cluster].size - 1)
        LOG_P.append(a + self.calculate_marginal_loglikelihood_slow()['ret'])
        LOG_KEYS.append(curr_cluster)

        # Calculate going to every other cluster
        # ======================================
        for cid in self.clustering.order:
            if curr_cluster == cid:
                continue

            # Move ASV and recompute the matrices
            self.clustering.move_item(idx=oidx,cid=cid)
            self.G[REPRNAMES.CLUSTER_INTERACTION_INDICATOR].update_cnt_indicators()
            self.G.data.design_matrices[REPRNAMES.CLUSTER_INTERACTION_VALUE].M.build()
            if self._there_are_perturbations:
                self.G.data.design_matrices[REPRNAMES.PERT_VALUE].M.build()

            LOG_P.append(np.log(self.clustering.clusters[cid].size - 1) + \
                self.calculate_marginal_loglikelihood_slow()['ret'])
            LOG_KEYS.append(cid)


        # Calculate new cluster
        # =====================
        cid=self.clustering.make_new_cluster_with(idx=oidx)
        self.G[REPRNAMES.CLUSTER_INTERACTION_INDICATOR].update_cnt_indicators()
        self.G.data.design_matrices[REPRNAMES.CLUSTER_INTERACTION_VALUE].M.build()
        if self._there_are_perturbations:
            self.G.data.design_matrices[REPRNAMES.PERT_VALUE].M.build()
        
        LOG_KEYS.append(cid)
        LOG_P.append(np.log(concentration/self.m) + \
            self.calculate_marginal_loglikelihood_slow()['ret'])

        # Sample the assignment
        # =====================
        idx = sample_categorical_log(LOG_P)
        assigned_cid = LOG_KEYS[idx]
        curr_clus = self.clustering.idx2cid[oidx]

        if assigned_cid != curr_clus:
            self.clustering.move_item(idx=oidx,cid=assigned_cid)
            self.G[REPRNAMES.CLUSTER_INTERACTION_INDICATOR].update_cnt_indicators()

            # Change the mixing matrix for the interactions and (potentially) perturbations
            self.G.data.design_matrices[REPRNAMES.CLUSTER_INTERACTION_VALUE].M.build()
            if self._there_are_perturbations:
                self.G.data.design_matrices[REPRNAMES.PERT_VALUE].M.build()

    def calculate_marginal_loglikelihood_slow(self):
        '''Marginalizes out the interactions and the perturbations
        '''
        # Build the parameters
        # ====================
        self.G[REPRNAMES.CLUSTER_INTERACTION_INDICATOR].update_cnt_indicators()
        lhs = [REPRNAMES.GROWTH_VALUE, REPRNAMES.SELF_INTERACTION_VALUE]
        if self._there_are_perturbations:
            rhs = [REPRNAMES.PERT_VALUE, REPRNAMES.CLUSTER_INTERACTION_VALUE]
        else:
            rhs = [REPRNAMES.CLUSTER_INTERACTION_VALUE]

        # reconstruct the X matrices
        for v in rhs:
            self.G.data.design_matrices[v].M.build()
        
        y = self.G.data.construct_lhs(lhs, 
            kwargs_dict={REPRNAMES.GROWTH_VALUE:{'with_perturbations': False}})
        X = self.G.data.construct_rhs(keys=rhs, toarray=True)
        process_prec = self.G[REPRNAMES.PROCESSVAR].build_matrix(cov=False, sparse=False)
        prior_prec = build_prior_covariance(G=self.G, cov=False, order=rhs, sparse=False)
        prior_var = build_prior_covariance(G=self.G, cov=True, order=rhs, sparse=False)
        prior_mean = build_prior_mean(G=self.G, order=rhs, shape=(-1,1))

        # If nothing is on, return 0
        if X.shape[1] == 0:
            return {
                'a': 0,
                'beta_prec': 0,
                'process_prec': prior_prec,
                'ret': 0,
                'beta_logdet': 0,
                'priorvar_logdet': 0,
                'bEb': 0,
                'bEbprior': 0}

        # Calculate the marginalization
        # =============================
        beta_prec = X.T @ process_prec @ X + prior_prec
        beta_cov = pinv(beta_prec, self)
        beta_mean = beta_cov @ ( X.T @ process_prec @ y + prior_prec @ prior_mean )
        beta_mean = np.asarray(beta_mean).reshape(-1,1)

        try:
            beta_logdet = log_det(beta_cov, self)
        except:
            logging.critical('Crashed in log_det')
            logging.critical('beta_cov:\n{}'.format(beta_cov))
            logging.critical('prior_prec\n{}'.format(prior_prec))
            raise
        priorvar_logdet = log_det(prior_var, self)
        ll2 = 0.5 * (beta_logdet - priorvar_logdet)

        bEbprior = np.asarray(prior_mean.T @ prior_prec @ prior_mean)[0,0]
        bEb = np.asarray(beta_mean.T @ beta_prec @ beta_mean)[0,0]
        ll3 = 0.5 * (bEb  - bEbprior)

        # print('prior_prec truth:\n', prior_prec)

        return {
            'a': X.T @ process_prec,
            'beta_prec': beta_prec,
            'process_prec': prior_prec,
            'ret': ll2+ll3,
            'beta_logdet': beta_logdet,
            'priorvar_logdet': priorvar_logdet,
            'bEb': bEb,
            'bEbprior': bEbprior}

    # Update regular - meant to be used during inference
    # ==================================================
    # @profile
    def update_slow_fast(self):
        '''Much faster than `update_slow`
        '''

        if self.clustering.n_clusters.sample_iter < self.delay:
            return

        if self.clustering.n_clusters.sample_iter % self.run_every_n_iterations != 0:
           return

        start_time = time.time()

        self.process_prec = self.G[REPRNAMES.PROCESSVAR].prec.ravel() #.build_matrix(cov=False, sparse=False)
        self.process_prec_matrix = self.G[REPRNAMES.PROCESSVAR].build_matrix(sparse=True, cov=False)
        lhs = [REPRNAMES.GROWTH_VALUE, REPRNAMES.SELF_INTERACTION_VALUE]
        self.y = self.G.data.construct_lhs(lhs, 
            kwargs_dict={REPRNAMES.GROWTH_VALUE:{'with_perturbations': False}})

        oidxs = npr.permutation(np.arange(len(self.G.data.asvs)))
        iii = 0
        for oidx in oidxs:
            logging.info('{}/{}: {}'.format(iii, len(oidxs), oidx))
            self.gibbs_update_single_asv_slow_fast(oidx=oidx)
            iii += 1
        self._strtime = time.time() - start_time
    
    # @profile
    def gibbs_update_single_asv_slow_fast(self, oidx):
        '''The update function is based off of Algorithm 8 in 'Markov Chain
        Sampling Methods for Dirichlet Process Mixture Models' by Radford M.
        Neal, 2000.

        Calculate the marginal likelihood of the asv in every cluster
        and a new cluster then sample from `self.sample_categorical_log`
        to get the cluster assignment.

        Parameters
        ----------
        oidx : int
            ASV index that we are updating the cluster assignment of
        '''
        curr_cluster = self.clustering.idx2cid[oidx]
        concentration = self.concentration.value

        # start as a dictionary then send values to `sample_categorical_log`
        LOG_P = []
        LOG_KEYS = []

        # Calculate current cluster
        # =========================
        # If the element is already in its own cluster, use the new cluster case
        if self.clustering.clusters[curr_cluster].size == 1:
            a = np.log(concentration/self.m)
        else:
            a = np.log(self.clustering.clusters[curr_cluster].size - 1)
        LOG_P.append(a + self.calculate_marginal_loglikelihood_slow_fast_sparse())
        LOG_KEYS.append(curr_cluster)

        # Calculate going to every other cluster
        # ======================================
        for cid in self.clustering.order:
            if curr_cluster == cid:
                continue

            # Move ASV and recompute the matrices
            self.clustering.move_item(idx=oidx,cid=cid)
            self.G[REPRNAMES.CLUSTER_INTERACTION_INDICATOR].update_cnt_indicators()
            self.G.data.design_matrices[REPRNAMES.CLUSTER_INTERACTION_VALUE].M.build()
            if self._there_are_perturbations:
                self.G.data.design_matrices[REPRNAMES.PERT_VALUE].M.build()

            LOG_P.append(np.log(self.clustering.clusters[cid].size - 1) + \
                self.calculate_marginal_loglikelihood_slow_fast_sparse())
            LOG_KEYS.append(cid)


        # Calculate new cluster
        # =====================
        cid=self.clustering.make_new_cluster_with(idx=oidx)
        self.G[REPRNAMES.CLUSTER_INTERACTION_INDICATOR].update_cnt_indicators()
        self.G.data.design_matrices[REPRNAMES.CLUSTER_INTERACTION_VALUE].M.build()
        if self._there_are_perturbations:
            self.G.data.design_matrices[REPRNAMES.PERT_VALUE].M.build()
        
        LOG_KEYS.append(cid)
        LOG_P.append(np.log(concentration/self.m) + \
            self.calculate_marginal_loglikelihood_slow_fast_sparse())

        # Sample the assignment
        # =====================
        idx = sample_categorical_log(LOG_P)
        assigned_cid = LOG_KEYS[idx]
        curr_clus = self.clustering.idx2cid[oidx]

        if assigned_cid != curr_clus:
            self.clustering.move_item(idx=oidx,cid=assigned_cid)
            self.G[REPRNAMES.CLUSTER_INTERACTION_INDICATOR].update_cnt_indicators()

            # Change the mixing matrix for the interactions and (potentially) perturbations
            self.G.data.design_matrices[REPRNAMES.CLUSTER_INTERACTION_VALUE].M.build()
            if self._there_are_perturbations:
                self.G.data.design_matrices[REPRNAMES.PERT_VALUE].M.build()

    # @profile
    def calculate_marginal_loglikelihood_slow_fast(self):
        '''Marginalizes out the interactions and the perturbations
        '''
        # Build the parameters
        # ====================
        if self._there_are_perturbations:
            rhs = [REPRNAMES.PERT_VALUE, REPRNAMES.CLUSTER_INTERACTION_VALUE]
        else:
            rhs = [REPRNAMES.CLUSTER_INTERACTION_VALUE]
        
        
        y = self.y
        X = self.G.data.construct_rhs(keys=rhs, toarray=True)

        process_prec = self.process_prec
        prior_prec = build_prior_covariance(G=self.G, cov=False, order=rhs, sparse=False)
        prior_prec_diag = np.diag(prior_prec)
        prior_var = build_prior_covariance(G=self.G, cov=True, order=rhs, sparse=False)
        prior_mean = build_prior_mean(G=self.G, order=rhs, shape=(-1,1))

        # Calculate the marginalization
        # =============================
        a = X.T * process_prec

        beta_prec = a @ X + prior_prec
        beta_cov = pinv(beta_prec, self)
        beta_mean = beta_cov @ ( a @ y + prior_prec @ prior_mean )
        beta_mean = np.asarray(beta_mean).reshape(-1,1)

        try:
            beta_logdet = log_det(beta_cov, self)
        except:
            logging.critical('Crashed in log_det')
            logging.critical('beta_cov:\n{}'.format(beta_cov))
            logging.critical('prior_prec\n{}'.format(prior_prec))
            raise
        priorvar_logdet = log_det(prior_var, self)
        ll2 = 0.5 * (beta_logdet - priorvar_logdet)

        a = np.sum((prior_mean.ravel() ** 2) *prior_prec_diag)
        # np.asarray(prior_mean.T @ prior_prec @ prior_mean)[0,0]
        b = np.asarray(beta_mean.T @ beta_prec @ beta_mean)[0,0]
        ll3 = -0.5 * (a  - b)

        return ll2+ll3

    # @profile
    def calculate_marginal_loglikelihood_slow_fast_sparse(self):
        '''Marginalizes out the interactions and the perturbations
        '''
        # Build the parameters
        # ====================
        if self._there_are_perturbations:
            rhs = [REPRNAMES.PERT_VALUE, REPRNAMES.CLUSTER_INTERACTION_VALUE]
        else:
            rhs = [REPRNAMES.CLUSTER_INTERACTION_VALUE]
        
        
        y = self.y
        X = self.G.data.construct_rhs(keys=rhs, toarray=False)

        process_prec = self.process_prec_matrix
        prior_prec = build_prior_covariance(G=self.G, cov=False, order=rhs, sparse=True)  
        prior_prec_diag = build_prior_covariance(G=self.G, cov=False, order=rhs, diag=True)        
        prior_var = build_prior_covariance(G=self.G, cov=True, order=rhs, sparse=True)
        prior_mean = build_prior_mean(G=self.G, order=rhs, shape=(-1,1))

        # Calculate the marginalization
        # =============================

        # print('X')
        # print(type(X))
        # print(X.shape)

        # print('process_prec')
        # print(type(process_prec))
        # print(process_prec.shape)

        # print('prior_prec')
        # print(type(prior_prec))
        # print(prior_prec.shape)

        # print('prior mean')
        # print(type(prior_mean))
        # print(prior_mean.shape)

        a = X.T.dot(process_prec)
        beta_prec = a.dot(X) + prior_prec
        beta_cov = pinv(beta_prec, self)
        beta_mean = beta_cov @ ( a.dot(y) + prior_prec.dot(prior_mean))
        beta_mean = np.asarray(beta_mean).reshape(-1,1)

        try:
            beta_logdet = log_det(beta_cov, self)
        except:
            logging.critical('Crashed in log_det')
            logging.critical('beta_cov:\n{}'.format(beta_cov))
            logging.critical('prior_prec\n{}'.format(prior_prec))
            raise
        priorvar_logdet = log_det(prior_var, self)
        ll2 = 0.5 * (beta_logdet - priorvar_logdet)

        a = np.sum((prior_mean.ravel() ** 2) *prior_prec_diag)
        # np.asarray(prior_mean.T @ prior_prec @ prior_mean)[0,0]

        # print('beta_prec.shape', beta_prec.shape)
        # print('beta_mean.shape', beta_mean.shape)

        b = np.asarray(beta_mean.T @ beta_prec.dot(beta_mean))[0,0]
        ll3 = -0.5 * (a  - b)

        return ll2+ll3

    # Update MP - meant to be used during inference
    # =============================================
    def update_mp(self):
        '''Implements `update_slow` but parallelizes calculating the likelihood
        of being in a cluster. NOTE that this does not parallelize on the ASV level.

        On the first gibb step with initialize the workers that we implement with DASW (
        different arguments, single worker). For more information what this means look
        at pylab.multiprocessing documentation.

        If we initialized our pool as a pylab.multiprocessing.PersistentPool, then we 
        multiprocess the likelihood calculations for each asv. If we didnt then this 
        implementation has the same performance as `ClusterAssignments.update_slow_fast`.
        '''
        if self.G.data.zero_inflation_transition_policy is not None:
            raise NotImplementedError('Multiprocessing for zero inflation data is not implemented yet.' \
                ' Use `mp=None`')
        DMI = self.G.data.design_matrices[REPRNAMES.CLUSTER_INTERACTION_VALUE]
        DMP = self.G.data.design_matrices[REPRNAMES.PERT_VALUE]
        if self.clustering.n_clusters.sample_iter == 0 or self.pool == []:
            kwargs = {
                'n_asvs': len(self.G.data.asvs),
                'total_n_dts_per_asv': self.G.data.total_n_dts_per_asv,
                'n_replicates': self.G.data.n_replicates,
                'n_dts_for_replicate': self.G.data.n_dts_for_replicate,
                'there_are_perturbations': self._there_are_perturbations,
                'keypair2col_interactions': DMI.M.keypair2col,
                'keypair2col_perturbations': DMP.M.keypair2col,
                'n_perturbations': len(self.G.perturbations) if self._there_are_perturbations else None,
                'base_Xrows': DMI.base.rows,
                'base_Xcols': DMI.base.cols,
                'base_Xshape': DMI.base.shape,
                'base_Xpertrows': DMP.base.rows,
                'base_Xpertcols': DMP.base.cols,
                'base_Xpertshape': DMP.base.shape,
                'n_rowsM': DMI.M.n_rows,
                'n_rowsMpert': DMP.M.n_rows}

            if pl.ispersistentpool(self.pool):
                for _ in range(self.n_cpus):
                    self.pool.add_worker(SingleClusterFullParallelization(**kwargs))
            else:
                self.pool = SingleClusterFullParallelization(**kwargs)

        if self.clustering.n_clusters.sample_iter < self.delay:
            return

        if self.clustering.n_clusters.sample_iter % self.run_every_n_iterations != 0:
            return

        # Send in arguments for the start of the gibbs step
        start_time = time.time()
        base_Xdata = DMI.base.data
        self.concentration = self.G[REPRNAMES.CONCENTRATION].value
        y = self.G.data.construct_lhs(keys=[REPRNAMES.GROWTH_VALUE, REPRNAMES.SELF_INTERACTION_VALUE],
            kwargs_dict={REPRNAMES.GROWTH_VALUE: {'with_perturbations':False}})
        prior_var_interactions = self.G[REPRNAMES.PRIOR_VAR_INTERACTIONS].value
        prior_mean_interactions = self.G[REPRNAMES.PRIOR_MEAN_INTERACTIONS].value
        process_prec_diag = self.G[REPRNAMES.PROCESSVAR].prec

        if self._there_are_perturbations:
            prior_var_pert = self.G[REPRNAMES.PRIOR_VAR_PERT].get_single_value_of_perts()
            prior_mean_pert = self.G[REPRNAMES.PRIOR_MEAN_PERT].get_single_value_of_perts()
            base_Xpertdata = DMP.base.data
        else:
            prior_var_pert = None
            prior_mean_pert = None
            base_Xpertdata = None

        kwargs = {
            'base_Xdata': base_Xdata,
            'base_Xpertdata': base_Xpertdata,
            'concentration': self.concentration,
            'm': self.m,
            'y': y,
            'process_prec_diag': process_prec_diag,
            'prior_var_interactions': prior_var_interactions,
            'prior_var_pert': prior_var_pert,
            'prior_mean_interactions': prior_mean_interactions,
            'prior_mean_pert': prior_mean_pert}
        
        if pl.ispersistentpool(self.pool):
            self.pool.map('initialize_gibbs', [kwargs]*self.pool.num_workers)
        else:
            self.pool.initialize_gibbs(**kwargs)

        oidxs = npr.permutation(np.arange(len(self.G.data.asvs)))
        for iii, oidx in enumerate(oidxs):
            logging.info('{}/{} - {}'.format(iii, len(self.G.data.asvs), oidx))
            self.oidx = oidx
            self.gibbs_update_single_asv_parallel()

        self._strtime = time.time() - start_time

    def gibbs_update_single_asv_parallel(self):
        '''Update for a single asvs
        '''
        self.original_cluster = self.clustering.idx2cid[self.oidx]
        self.curr_cluster = self.original_cluster

        interactions = self.G[REPRNAMES.INTERACTIONS_OBJ]
        interaction_on_idxs = interactions.get_indicators(return_idxs=True)
        if self._there_are_perturbations:
            perturbation_on_idxs = [p.indicator.cluster_arg_array() for p in self.G.perturbations]
        else:
            perturbation_on_idxs = None

        # # Send topology parameters if the topology wont change
        # TODO: this works on windows but not on the cluster when dispatching?
        # if self.clustering.clusters[self.original_cluster].size > 1:
        #     use_saved_params = True
        #     kwargs = {
        #         'interaction_on_idxs': interaction_on_idxs,
        #         'perturbation_on_idxs': perturbation_on_idxs}
        #     if pl.ispersistentpool(self.pool):
        #         self.pool.map('initialize_oidx', [kwargs]*self.pool.num_workers)
        #     else:
        #         use_saved_params = False
        # else:
        #     use_saved_params = False
        use_saved_params = False

        if pl.ispersistentpool(self.pool):
            self.pool.staged_map_start('run')
        else:
            notpool_ret = []

        # Get the likelihood of the current configuration
        if self.clustering.clusters[self.original_cluster].size == 1:
            log_mult_factor = math.log(self.concentration/self.m)
        else:
            log_mult_factor = math.log(self.clustering.clusters[self.original_cluster].size - 1)

        if use_saved_params and pl.ispersistentpool(self.pool):
            interaction_on_idxs = None
            perturbation_on_idxs = None

        cluster_config = np.asarray([self.clustering.cid2cidx[self.clustering.idx2cid[i]] \
            for i in range(len(self.G.data.asvs))])

        kwargs = {
            'interaction_on_idxs': interaction_on_idxs,
            'perturbation_on_idxs': perturbation_on_idxs,
            'cluster_config': cluster_config,
            'log_mult_factor': log_mult_factor,
            'cid': self.original_cluster,
            'use_saved_params': use_saved_params}

        if pl.ispersistentpool(self.pool):
            self.pool.staged_map_put(kwargs)
        else:
            notpool_ret.append(self.pool.run(**kwargs))

        # Check every cluster
        for cid in self.clustering.order:
            if cid == self.original_cluster:
                continue
            self.clustering.move_item(idx=self.oidx, cid=cid)
            self.curr_cluster = cid

            if not use_saved_params:
                interaction_on_idxs = interactions.get_indicators(return_idxs=True)
                if self._there_are_perturbations:
                    perturbation_on_idxs = [p.indicator.cluster_arg_array() for p in self.G.perturbations]
                else:
                    perturbation_on_idxs = None

            cluster_config = np.asarray([self.clustering.cid2cidx[self.clustering.idx2cid[i]] \
                for i in range(len(self.G.data.asvs))])
            log_mult_factor = np.log(self.clustering.clusters[self.curr_cluster].size - 1)

            kwargs = {
                'interaction_on_idxs': interaction_on_idxs,
                'perturbation_on_idxs': perturbation_on_idxs,
                'cluster_config': cluster_config,
                'log_mult_factor': log_mult_factor,
                'cid': self.curr_cluster,
                'use_saved_params': use_saved_params}

            if pl.ispersistentpool(self.pool):
                self.pool.staged_map_put(kwargs)
            else:
                notpool_ret.append(self.pool.run(**kwargs))

        # Make a new cluster
        self.curr_cluster = self.clustering.make_new_cluster_with(idx=self.oidx)
        cluster_config = np.asarray([self.clustering.cid2cidx[self.clustering.idx2cid[i]] \
            for i in range(len(self.G.data.asvs))])
        interaction_on_idxs = interactions.get_indicators(return_idxs=True)
        if self._there_are_perturbations:
            perturbation_on_idxs = [p.indicator.cluster_arg_array() for p in self.G.perturbations]
        else:
            perturbation_on_idxs = None
        log_mult_factor = np.log(self.concentration/self.m)

        kwargs = {
            'interaction_on_idxs': interaction_on_idxs,
            'perturbation_on_idxs': perturbation_on_idxs,
            'cluster_config': cluster_config,
            'log_mult_factor': log_mult_factor,
            'cid': self.curr_cluster,
            'use_saved_params': False}

        # Put the values and get if necessary
        KEYS = []
        LOG_P = []
        if pl.ispersistentpool(self.pool):
            self.pool.staged_map_put(kwargs)
            ret = self.pool.staged_map_get()
        else:
            notpool_ret.append(self.pool.run(**kwargs))
            ret = notpool_ret
        for c, p in ret:
            KEYS.append(c)
            LOG_P.append(p)

        idx = sample_categorical_log(LOG_P)
        assigned_cid = KEYS[idx]

        if assigned_cid != self.original_cluster:
            logging.info('cluster changed')

        self.clustering.move_item(idx=self.oidx, cid=assigned_cid)

        self.G[REPRNAMES.CLUSTER_INTERACTION_INDICATOR].update_cnt_indicators()
        self.G.data.design_matrices[REPRNAMES.CLUSTER_INTERACTION_VALUE].M.build()
        if self._there_are_perturbations:
            self.G.data.design_matrices[REPRNAMES.PERT_VALUE].M.build()

        
class SingleClusterFullParallelization(pl.multiprocessing.PersistentWorker):
    '''Make the full parallelization
        - Mixture matricies for interactions and perturbations
        - calculating the marginalization for the sent in cluster

    Parameters
    ----------
    n_asvs : int
        Total number of OTUs
    total_n_dts_per_asv : int
        Total number of time changes for each OTU
    n_replicates : int
        Total number of replicates
    n_dts_for_replicate : np.ndarray
        Total number of time changes for each replicate
    there_are_perturbations : bool
        If True, there are perturbations
    keypair2col_interactions : np.ndarray
        These map the OTU indices of the pairs of OTUs to their column index
        in `big_X`
    keypair2col_perturbations : np.ndarray, None
        These map the OTU indices nad perturbation index to the column in
        `big_Xpert`. If there are no perturbations then this is None
    n_perturbations : int, None
        Number of perturbations. None if there are no perturbations
    base_Xrows, base_Xcols, base_Xpertrows, base_Xpertcols : np.ndarray
        These are the rows and columns necessary to build the interaction and perturbation 
        matrices, respectively. Whats passed in is the data vector and then we build
        it using sparse matrices
    n_rowsM, n_rowsMpert : int
        These are the number of rows for the mixing matrix for the interactions and
        perturbations respectively.
    '''
    def __init__(self, n_asvs, total_n_dts_per_asv, n_replicates, n_dts_for_replicate,
        there_are_perturbations, keypair2col_interactions, keypair2col_perturbations,
        n_perturbations, base_Xrows, base_Xcols, base_Xshape, base_Xpertrows, base_Xpertcols,
        base_Xpertshape, n_rowsM, n_rowsMpert):
        self.n_asvs = n_asvs
        self.total_n_dts_per_asv = total_n_dts_per_asv
        self.n_replicates = n_replicates
        self.n_dts_for_replicate = n_dts_for_replicate
        self.there_are_perturbations = there_are_perturbations
        self.keypair2col_interactions = keypair2col_interactions
        if self.there_are_perturbations:
            self.keypair2col_perturbations = keypair2col_perturbations
            self.n_perturbations = n_perturbations

        self.base_Xrows = base_Xrows
        self.base_Xcols = base_Xcols
        self.base_Xshape = base_Xshape
        self.base_Xpertrows = base_Xpertrows
        self.base_Xpertcols = base_Xpertcols
        self.base_Xpertshape = base_Xpertshape

        self.n_rowsM = n_rowsM
        self.n_rowsMpert = n_rowsMpert

    def initialize_gibbs(self, base_Xdata, base_Xpertdata, concentration, m, y,
        process_prec_diag, prior_var_interactions, prior_var_pert, 
        prior_mean_interactions, prior_mean_pert):
        '''Pass in the information that changes every Gibbs step

        Parameters
        ----------
        base_X : scipy.sparse.csc_matrix
            Sparse matrix for the interaction terms
        base_Xpert : scipy.sparse.csc_matrix, None
            Sparse matrix for the perturbation terms
            If None, there are no perturbations
        concentration : float
            This is the concentration of the system
        m : int
            This is the auxiliary variable for the marginalization
        y : np.ndarray
            This is the observation array
        process_prec_diag : np.ndarray
            This is the process precision diagonal
        '''
        self.base_X = scipy.sparse.coo_matrix(
            (base_Xdata,(self.base_Xrows,self.base_Xcols)),
            shape=self.base_Xshape).tocsc()
        
        self.concentration = concentration
        self.m = m
        self.y = y.reshape(-1,1)
        self.n_rows = len(y)
        self.n_cols_X = self.base_X.shape[1]
        self.prior_var_interactions = prior_var_interactions
        self.prior_prec_interactions = 1/prior_var_interactions
        self.prior_mean_interactions = prior_mean_interactions

        if self.there_are_perturbations:
            self.base_Xpert = scipy.sparse.coo_matrix(
                (base_Xpertdata,(self.base_Xpertrows,self.base_Xpertcols)),
                shape=self.base_Xpertshape).tocsc()
            self.prior_var_pert = prior_var_pert
            self.prior_prec_pert = 1/prior_var_pert
            self.prior_mean_pert = prior_mean_pert

        self.process_prec_matrix = scipy.sparse.dia_matrix(
            (process_prec_diag,[0]), shape=(len(process_prec_diag),len(process_prec_diag))).tocsc()

    def initialize_oidx(self, interaction_on_idxs, perturbation_on_idxs):
        '''Pass in the parameters that change for every OTU - potentially

        Parameters
        ----------
        
        '''
        self.saved_interaction_on_idxs = interaction_on_idxs
        self.saved_perturbation_on_idxs = perturbation_on_idxs

    # @profile
    def run(self, interaction_on_idxs, perturbation_on_idxs, cluster_config, log_mult_factor, cid, 
        use_saved_params):
        '''Pass in the parameters for the specific cluster assignment for
        the OTU and run the marginalization


        Parameters
        ----------
        interaction_on_idxs : np.array(int)
            An array of indices for the interactions that are on. Assumes that the
            clustering is in the order specified in `cluster_config`
        perturbation_on_idxs : list(np.ndarray(int)), None
            If there are perturbations, then we set the perturbation idxs on
            Each element in the list are the indices of that perturbation that are on
        cluster_config : list(list(int))
            This is the cluster configuration and in cluster order.
        log_mult_factor : float
            This is the log multiplication factor that we add onto the marginalization
        use_saved_params : bool
            If True, passed in `interaction_on_idxs` and `perturbation_on_idxs` are None
            and we can use `saved_interaction_on_idxs` and `saved_perturbation_on_idxs`
        '''
        if use_saved_params:
            interaction_on_idxs = self.saved_interaction_on_idxs
            perturbation_on_idxs = self.saved_perturbation_on_idxs

        # We need to make the arrays for interactions and perturbations
        self.set_clustering(cluster_config=cluster_config)
        Xinteractions = self.build_interactions_matrix(on_columns=interaction_on_idxs)

        if self.there_are_perturbations:
            Xperturbations = self.build_perturbations_matrix(on_columns=perturbation_on_idxs)
            X = scipy.sparse.hstack([Xperturbations, Xinteractions])
        else:
            X = Xinteractions
        self.X = X
        self.prior_mean = self.build_prior_mean(on_interactions=interaction_on_idxs,
            on_perturbations=perturbation_on_idxs)
        self.prior_cov, self.prior_prec, self.prior_prec_diag = self.build_prior_cov_and_prec_and_diag(
            on_interactions=interaction_on_idxs, on_perturbations=perturbation_on_idxs)
        
        return cid, self.calculate_marginal_loglikelihood_slow_fast_sparse()

    def set_clustering(self, cluster_config):
        self.clustering = CondensedClustering(oidx2cidx=cluster_config)
        self.iidx2cidxpair = np.zeros(shape=(len(self.clustering)*(len(self.clustering)-1), 2), 
            dtype=int)
        self.iidx2cidxpair = SingleClusterFullParallelization.make_iidx2cidxpair(
            ret=self.iidx2cidxpair,
            n_clusters=len(self.clustering))

    def build_interactions_matrix(self, on_columns):
        '''Build the interaction matrix

        First we make the rows and columns for the mixing matrix,
        then we multiple the base matrix and the mixing matrix.
        '''
        rows = []
        cols = []

        # c2ciidx = Cluster-to-Cluster Interaction InDeX
        c2ciidx = 0
        for ccc in on_columns:
            tcidx = self.iidx2cidxpair[ccc, 0]
            scidx = self.iidx2cidxpair[ccc, 1]
            
            smems = self.clustering.clusters[scidx]
            tmems = self.clustering.clusters[tcidx]
            
            a = np.zeros(len(smems)*len(tmems), dtype=int)
            rows.append(SingleClusterFullParallelization.get_indices(a,
                self.keypair2col_interactions, tmems, smems))
            cols.append(np.full(len(tmems)*len(smems), fill_value=c2ciidx))
            c2ciidx += 1

        rows = np.asarray(list(itertools.chain.from_iterable(rows)))
        cols = np.asarray(list(itertools.chain.from_iterable(cols)))
        data = np.ones(len(rows), dtype=int)

        # print('rows', rows)
        # print(cols)
        # print(data)
        # print((self.n_rowsM, c2ciidx))

        M = scipy.sparse.coo_matrix((data,(rows,cols)),
            shape=(self.n_rowsM, c2ciidx)).tocsc()
        ret = self.base_X @ M
        return ret

    def build_perturbations_matrix(self, on_columns):
        if not self.there_are_perturbations:
            raise ValueError('You should not be here')
        
        keypair2col = self.keypair2col_perturbations
        rows = []
        cols = []

        col = 0
        for pidx, pert_ind_idxs in enumerate(on_columns):
            for cidx in pert_ind_idxs:
                for oidx in self.clustering.clusters[cidx]:
                    rows.append(keypair2col[oidx, pidx])
                    cols.append(col)
                col += 1
        
        data = np.ones(len(rows), dtype=np.float64)
        M = scipy.sparse.coo_matrix((data,(rows,cols)),
            shape=(self.n_rowsMpert, col)).tocsc()
        ret = self.base_Xpert @ M
        return ret

    def build_prior_mean(self, on_interactions, on_perturbations):
        '''Build the prior mean array

        Perturbations go first and then interactions
        '''
        ret = []
        for pidx, pert in enumerate(on_perturbations):
            ret = np.append(ret, 
                np.full(len(pert), fill_value=self.prior_mean_pert[pidx]))
        ret = np.append(ret, np.full(len(on_interactions), fill_value=self.prior_mean_interactions))
        return ret.reshape(-1,1)

    def build_prior_cov_and_prec_and_diag(self, on_interactions, on_perturbations):
        '''Build the prior covariance matrices and others
        '''
        ret = []
        for pidx, pert in enumerate(on_perturbations):
            ret = np.append(ret, 
                np.full(len(pert), fill_value=self.prior_var_pert[pidx]))
        prior_var_diag = np.append(ret, np.full(len(on_interactions), fill_value=self.prior_var_interactions))
        prior_prec_diag = 1/prior_var_diag

        prior_var = scipy.sparse.dia_matrix((prior_var_diag,[0]), 
            shape=(len(prior_var_diag),len(prior_var_diag))).tocsc()
        prior_prec = scipy.sparse.dia_matrix((prior_prec_diag,[0]), 
            shape=(len(prior_prec_diag),len(prior_prec_diag))).tocsc()

        return prior_var, prior_prec, prior_prec_diag

    # @profile
    def calculate_marginal_loglikelihood_slow_fast_sparse(self):
        y = self.y
        X = self.X
        process_prec = self.process_prec_matrix
        prior_mean = self.prior_mean
        prior_cov = self.prior_cov
        prior_prec = self.prior_prec
        prior_prec_diag = self.prior_prec_diag

        a = X.T.dot(process_prec)
        beta_prec = a.dot(X) + prior_prec
        beta_cov = pinv(beta_prec, self)
        beta_mean = beta_cov @ (a.dot(y) + prior_prec.dot(prior_mean))
        beta_mean = np.asarray(beta_mean).reshape(-1,1)

        try:
            beta_logdet = log_det(beta_cov, self)
        except:
            logging.critical('Crashed in log_det')
            logging.critical('beta_cov:\n{}'.format(beta_cov))
            logging.critical('prior_prec\n{}'.format(prior_prec))
            raise
        
        priorvar_logdet = log_det(prior_cov, self)
        ll2 = 0.5 * (beta_logdet - priorvar_logdet)

        bEbprior = np.asarray(prior_mean.T @ prior_prec.dot(prior_mean))[0,0]
        bEb = np.asarray(beta_mean.T @ beta_prec.dot(beta_mean) )[0,0]
        ll3 = 0.5 * (bEb - bEbprior)

        self.a = a
        self.beta_prec = beta_prec

        return ll2 + ll3

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def get_indices(a, keypair2col, tmems, smems):
        '''Use Just in Time compilation to reduce the 'getting' time
        by about 95%

        Parameters
        ----------
        keypair2col : np.ndarray
            Maps (target_oidx, source_oidx) pair to the place the interaction
            index would be on a full interactio design matrix on the OTU level
        tmems, smems : np.ndarray
            These are the OTU indices in the target cluster and the source cluster
            respectively
        '''
        i = 0
        for tidx in tmems:
            for sidx in smems:
                a[i] = keypair2col[tidx, sidx]
                i += 1
        return a

    @staticmethod
    # @numba.jit(nopython=True, cache=True)
    def make_iidx2cidxpair(ret, n_clusters):
        '''Map the index of a cluster interaction to (dst,src) of clusters

        Parameters
        ----------
        n_clusters : int
            Number of clusters

        Returns
        -------
        np.ndarray(n_interactions,2)
            First column is the destination cluster index, second column is the source
            cluster index
        '''
        i = 0
        for dst_cidx in range(n_clusters):
            for src_cidx in range(n_clusters):
                if dst_cidx == src_cidx:
                    continue
                ret[i,0] = dst_cidx
                ret[i,1] = src_cidx
                i += 1
        
        return ret


class CondensedClustering:
    '''Condensed clustering object that is not associated with the graph

    Parameters
    ----------
    oidx2cidx : np.ndarray
        Maps the cluster assignment to each asv.
        index -> ASV index
        output -> cluster index

    '''
    def __init__(self, oidx2cidx):

        self.clusters = []
        self.oidx2cidx = oidx2cidx
        a = {}
        for oidx, cidx in enumerate(self.oidx2cidx):
            if cidx not in a:
                a[cidx] = [oidx]
            else:
                a[cidx].append(oidx)
        
        cidx = 0
        while cidx in a:
            self.clusters.append(np.asarray(a[cidx], dtype=int))
            cidx += 1

    def __len__(self):
        return len(self.clusters)
