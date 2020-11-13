'''Perturbation parameters for the posterior
'''
import logging
import time
import itertools
import numpy as np
import numpy.random as npr

from .util import expected_n_clusters, build_prior_covariance, build_prior_mean, \
    sample_categorical_log, log_det, pinv
from .names import STRNAMES, REPRNAMES

from . import pylab as pl

class PerturbationMagnitudes(pl.variables.Normal):
    '''These update the perturbation values jointly.
    '''
    def __init__(self, **kwargs):
        '''Parameters
        '''

        kwargs['name'] = STRNAMES.PERT_VALUE
        pl.variables.Normal.__init__(self, mean=None, var=None, dtype=float, **kwargs)
        self.perturbations = self.G.perturbations

    def __str__(self):
        s = 'Perturbation Magnitudes (multiplicative)'
        for perturbation in self.perturbations:
            s += '\n\t perturbation {}: {}'.format(
                perturbation.name, perturbation.cluster_array(only_pos_ind=True))
        return s

    def __len__(self):
        '''Return the number of on indicators
        '''
        n = 0
        for perturbation in self.perturbations:
            n += perturbation.indicator.num_on_clusters()
        return n

    def set_values(self, arr, use_indicators=True):
        '''Set the values of the perturbation of them stacked one on top of each other

        Parameters
        ----------
        arr : np.ndarray
            Values for all of the perturbations in order
        use_indicators : bool
            If True, the values only refer to the on indicators
        '''
        i = 0
        for perturbation in self.perturbations:
            l = perturbation.indicator.num_on_clusters()
            perturbation.set_values_from_array(values=arr[i:i+l],
                use_indicators=use_indicators)
            i += l

    def update_str(self):
        return

    @property
    def sample_iter(self):
        return self.perturbations[0].sample_iter

    def initialize(self, value_option, value=None, mean=None, var=None, delay=0):
        '''Initialize the prior and the value of the perturbation. We assume that
        each perturbation has the same hyperparameters for the prior

        Parameters
        ----------
        value_option : str
            How to initialize the values. Options:
                'manual'
                    Set the value manually, `value` must also be specified
                'zero'
                    Set all the values to zero.
                'auto', 'prior-mean'
                    Initialize to the same value as the prior mean
        delay : int, None
            How many MCMC iterations to delay the update of the values.
        mean, var, value : int, float, array
            - Only necessary if any of the options are 'manual'
        '''
        if delay is None:
            delay = 0
        if not pl.isint(delay):
            raise TypeError('`delay` ({}) must be an int'.format(type(delay)))
        if delay < 0:
            raise ValueError('`delay` ({}) must be >= 0'.format(delay))
        self.delay = delay
        for perturbation in self.perturbations:
            perturbation.magnitude.set_signal_when_clusters_change(True)

        # Set the value of the perturbations
        if not pl.isstr(value_option):
            raise TypeError('`value_option` ({}) must be a str'.format(type(value_option)))
        if value_option == 'manual':
            for pidx, perturbation in enumerate(self.perturbations):
                v = value[pidx]
                for cidx, val in enumerate(v):
                    cid = perturbation.clustering.order[cidx]
                    perturbation.indicator.value[cid] = not np.isnan(val)
                    perturbation.magnitude.value[cid] = val if not np.isnan(val) else 0

        elif value_option == 'zero':
            for perturbation in self.perturbations:
                for cid in perturbation.clustering.order:
                    perturbation.magnitude.value[cid] = 0
        elif value_option in ['auto', 'prior-mean']:
            for perturbation in self.perturbations:
                mean = perturbation.magnitude.prior.mean.value
                for cid in perturbation.clustering.order:
                    perturbation.magnitude.value[cid] = mean
        else:
            raise ValueError('`value_option` ({}) not recognized'.format(value_option))


        s = 'Perturbation magnitude initialization results:\n'
        for perturbation in self.perturbations:
            if perturbation.name is not None:
                a = perturbation.name
            s += '\tPerturbation {}:\n' \
                '\t\tvalue: {}\n'.format(a, perturbation.magnitude.cluster_array())
        logging.info(s)

    def update(self):
        '''Update with a gibbs step jointly
        '''
        if self.sample_iter < self.delay:
            return

        n_on = [perturbation.indicator.num_on_clusters() for perturbation in \
            self.perturbations]
        
        if n_on == 0:
            return

        rhs = [REPRNAMES.PERT_VALUE]
        lhs = [
            REPRNAMES.GROWTH_VALUE,
            REPRNAMES.SELF_INTERACTION_VALUE,
            REPRNAMES.CLUSTER_INTERACTION_VALUE]
        X = self.G.data.construct_rhs(keys=rhs, toarray=True)
        y = self.G.data.construct_lhs(keys=lhs,
            kwargs_dict={REPRNAMES.GROWTH_VALUE:{
                'with_perturbations':False}})

        process_prec = self.G[REPRNAMES.PROCESSVAR].prec
        prior_prec = build_prior_covariance(G=self.G, cov=False, order=rhs, sparse=False)

        prior_mean = build_prior_mean(G=self.G, order=rhs).reshape(-1,1)

        a = X.T * process_prec
        prec = a @ X + prior_prec
        cov = pinv(prec, self)
        mean = np.asarray(cov @ (a @ y + prior_prec @ prior_mean)).ravel()

        # print('\n\ny\n',np.hstack((y, self.G.data.lhs.vector.reshape(-1,1))))
        # print(self.G[REPRNAMES.CLUSTER_INTERACTION_VALUE].value)
        # print(self.G[REPRNAMES.GROWTH_VALUE].value)
        # print(self.G[REPRNAMES.SELF_INTERACTION_VALUE].value)

        self.mean.value = mean
        self.var.value = np.diag(cov)
        value = self.sample()

        if np.any(np.isnan(value)):
            logging.critical('mean: {}'.format(self.mean.value))
            logging.critical('var: {}'.format(self.var.value))
            logging.critical('value: {}'.format(self.value))
            logging.critical('prior mean: {}'.format(prior_mean.ravel()))
            raise ValueError('`Values in {} are nan: {}'.format(self.name, self.value))

        i = 0
        for pidx, perturbation in enumerate(self.perturbations):
            perturbation.set_values_from_array(value[i:i+n_on[pidx]], use_indicators=True)
            i += n_on[pidx]

        # Rebuild the design matrix
        self.G.data.design_matrices[REPRNAMES.GROWTH_VALUE].build_with_perturbations()

    def set_trace(self, *args, **kwargs):
        for perturbation in self.perturbations:
            perturbation.set_trace(*args, **kwargs)

    def add_trace(self, *args, **kwargs):
        for perturbation in self.perturbations:
            perturbation.add_trace(*args, **kwargs)

    def add_init_value(self):
        '''Set the initialization value. This is called by `pylab.inference.BaseMCMC.run`
        when first updating the variable. User should not use this function
        '''
        for perturbation in self.perturbations:
            perturbation.add_init_value()

    def asarray(self):
        '''Get an array of the perturbation magnitudes
        '''
        a = []
        for perturbation in self.perturbations:
            a.append(perturbation.cluster_array(only_pos_ind=True))
        return np.asarray(list(itertools.chain.from_iterable(a)))

    def toarray(self):
        return self.asarray()


class PerturbationProbabilities(pl.Node):
    '''This is the probability for a positive interaction for a perturbation
    '''
    def __init__(self, **kwargs):
        '''Parameters

        prior (pl.variables.Beta)
            - prior probability
        pert_n (int)
            - This is the perturbation number that it corresponds to
        **kwargs
            - Other options like graph, value
        '''
        kwargs['name'] = STRNAMES.PERT_INDICATOR_PROB
        pl.Node.__init__(self, **kwargs)
        self.perturbations = self.G.perturbations

    def __str__(self):
        s = 'Perturbation Indicator probabilities'
        for perturbation in self.perturbations:
            s += '\n\tperturbation {}: {}'.format(
                perturbation.name,
                perturbation.probability.value)
        return s

    @property
    def sample_iter(self):
        return self.perturbations[0].probability.sample_iter

    def initialize(self, value_option, hyperparam_option, a=None, b=None, value=None,
        N='auto', delay=0):
        '''Initialize the hyperparameters of the prior and the value. Each
        perturbation has the same prior.

        Parameters
        ----------
        value_option : str
            How to initialize the values. Options:
                'manual'
                    Set the value manually, `value` must also be specified
                'auto', 'prior-mean'
                    Initialize the value as the prior mean
        hyperparam_option : str
            How to initialize `a` and `b`. Options:
                'manual'
                    Set the value manually. `a` and `b` must also be specified
                'weak-agnostic' or 'auto'
                    a=b=0.5
                'strong-dense'
                    a = N, N are the expected number of clusters
                    b = 0.5
                'strong-sparse'
                    a = 0.5
                    b = N, N are the expected number of clusters
                'very-strong-sparse'
                    a = 0.5
                    b = N, N are the expected number of ASVs
        N : str, int
            This is the number of clusters to set the hyperparam options to 
            (if they are dependent on the number of cluster). If 'auto', set to the expected number
            of clusters from a dirichlet process. Else use this number (must be an int).
        delay : int
            How many MCMC iterations to delay starting the update of the variable
        value, a, b : int, float
            User specified values
            Only necessary if `hyperparam_option` == 'manual'
        '''
        if not pl.isint(delay):
            raise TypeError('`delay` ({}) must be an int'.format(type(delay)))
        if delay < 0:
            raise ValueError('`delay` ({}) must be >= 0'.format(delay))
        self.delay = delay

        # Set the hyper-parameters
        if not pl.isstr(hyperparam_option):
            raise ValueError('`hyperparam_option` ({}) must be a str'.format(type(hyperparam_option)))
        if hyperparam_option == 'manual':
            if (not pl.isnumeric(a)) or (not pl.isnumeric(b)):
                raise TypeError('If `hyperparam_option` is "manual" then `a` ({})' \
                    ' and `b` ({}) must be numerics'.format(type(a), type(b)))
        elif hyperparam_option in ['auto', 'weak-agnostic']:
            a = 0.5
            b = 0.5
        elif hyperparam_option == 'strong-dense':
            if pl.isstr(N):
                if N == 'auto':
                    N = expected_n_clusters(G=self.G)
                else:
                    raise ValueError('`N` ({}) nto recognized'.format(N))
            elif pl.isint(N):
                if N < 0:
                    raise ValueError('`N` ({}) must be positive'.format(N))
            else:
                raise TypeError('`N` ({}) type not recognized'.format(type(N)))
            a = N
            b = 0.5
        elif hyperparam_option == 'strong-sparse':
            if pl.isstr(N):
                if N == 'auto':
                    N = expected_n_clusters(G=self.G)
                else:
                    raise ValueError('`N` ({}) nto recognized'.format(N))
            elif pl.isint(N):
                if N < 0:
                    raise ValueError('`N` ({}) must be positive'.format(N))
            else:
                raise TypeError('`N` ({}) type not recognized'.format(type(N)))
            a = 0.5
            b = N
        elif hyperparam_option == 'very-strong-sparse':
            N = self.G.data.n_asvs
            a = 0.5
            b = N
        else:
            raise ValueError('`hyperparam_option` ({}) not recognized'.format(hyperparam_option))
        for perturbation in self.perturbations:
            perturbation.probability.prior.a.override_value(a)
            perturbation.probability.prior.b.override_value(b)

        # Set the value
        if not pl.isstr(value_option):
            raise TypeError('`value_option` ({}) must be a str'.format(type(value_option)))
        if value_option == 'manual':
            if not pl.isnumeric(value):
                raise TypeError('If `value_option` is "manual" then `value` ({})' \
                    ' must be a numeric'.format(type(value)))
            for perturbation in self.perturbations:
                perturbation.probability.value = value
        elif value_option in ['auto', 'prior-mean']:
            for perturbation in self.perturbations:
                perturbation.probability.value = perturbation.probability.prior.mean()
        else:
            raise ValueError('`value_option` ({}) not recognized'.format(value_option))

        s = 'Perturbation indicator probability initialization results:\n'
        for i, perturbation in enumerate(self.perturbations):
            s += '\tPerturbation {}:\n' \
                '\t\tprior a: {}\n' \
                '\t\tprior b: {}\n' \
                '\t\tvalue: {}\n'.format(i,
                    perturbation.probability.prior.a.value,
                    perturbation.probability.prior.b.value,
                    perturbation.probability.value)
        logging.info(s)

    def update(self):
        '''Update according to how many positive and negative indicators there
        are
        '''
        if self.sample_iter < self.delay:
            return
        for perturbation in self.perturbations:
            num_pos = perturbation.indicator.num_on_clusters()
            num_neg = len(perturbation.clustering.clusters) - num_pos
            perturbation.probability.a.value = perturbation.probability.prior.a.value + num_pos
            perturbation.probability.b.value = perturbation.probability.prior.b.value + num_neg
            perturbation.probability.sample()

    def set_trace(self, *args, **kwargs):
        for perturbation in self.perturbations:
            perturbation.probability.set_trace(*args, **kwargs)

    def add_trace(self, *args, **kwargs):
        for perturbation in self.perturbations:
            perturbation.probability.add_trace(*args, **kwargs)

    def add_init_value(self):
        '''Set the initialization value. This is called by `pylab.inference.BaseMCMC.run`
        when first updating the variable. User should not use this function
        '''
        for perturbation in self.perturbations:
            perturbation.add_init_value()


class PerturbationIndicators(pl.Node):
    '''This is the indicator for a perturbation

    We only need to trace once for the perturbations. Our default is to only
    trace from the magnitudes. Thus, we only trace the indicators (here) if
    we are learning here and not learning the magnitudes.
    '''
    def __init__(self, need_to_trace, relative, **kwargs):
        '''Parameters
        '''
        kwargs['name'] = STRNAMES.PERT_INDICATOR
        pl.Node.__init__(self, **kwargs)
        self.need_to_trace = need_to_trace
        self.perturbations = self.G.perturbations
        self.clustering = None
        self._time_taken = None
        if relative:
            self.update = self.update_relative
        else:
            self.update = self.update_slow

    def __str__(self):
        s = 'Perturbation Indicators - time: {}s'.format(self._time_taken)
        for perturbation in self.perturbations:
            arr = perturbation.indicator.cluster_bool_array()
            s += '\nperturbation {} ({}/{}): {}'.format(perturbation.name,
                np.sum(arr), len(arr), arr)
        return s

    @property
    def sample_iter(self):
        return self.perturbations[0].sample_iter

    def add_trace(self):
        '''Only trace if perturbation indicators are being learned and the
        perturbation value is not being learned
        '''
        if self.need_to_trace:
            for perturbation in self.perturbations:
                perturbation.add_trace()

    def set_trace(self, *args, **kwargs):
        '''Only trace if perturbation indicators are being learned and the
        perturbation value is not being learned
        '''
        if self.need_to_trace:
            for perturbation in self.perturbations:
                perturbation.set_trace(*args, **kwargs)

    def add_init_value(self):
        '''Set the initialization value. This is called by `pylab.inference.BaseMCMC.run`
        when first updating the variable. User should not use this function
        '''
        if self.need_to_trace:
            for perturbation in self.perturbations:
                perturbation.add_init_value()

    def initialize(self, value_option, p=None, delay=0):
        '''Initialize the based on the passed in option.

        Parameters
        ----------
        value_option (str)
            Different ways to initialize the values. Options:
                'auto', 'all-off'
                    Turn all of the indicators off
                'all-on'
                    Turn all the indicators on
                'random'
                    Randomly assign the indicator with probability `p`
        p : float
            Only required if `value_option` == 'random'
        delay : int
            How many Gibbs steps to delay updating the values
        '''
        # print('in pert ind')
        if not pl.isint(delay):
            raise TypeError('`delay` ({}) must be an int'.format(type(delay)))
        if delay < 0:
            raise ValueError('`delay` ({}) must be >= 0'.format(delay))
        self.delay = delay

        for perturbation in self.perturbations:
            perturbation.indicator.set_signal_when_clusters_change(True)
        self.clustering = self.G.perturbations[0].indicator.clustering

        # Set the value
        if not pl.isstr(value_option):
            raise ValueError('`value_option` ({}) must be a str'.format(type(value_option)))
        if value_option in ['all-off', 'auto']:
            value = False
        elif value_option == 'all-on':
            value = True
        elif value_option == 'random':
            if not pl.isfloat(p):
                raise TypeError('`p` ({}) must be a float'.format(type(p)))
            if p < 0 or p > 1:
                raise ValueError('`p` ({}) must be [0,1]'.format(p))
        else:
            raise ValueError('`value_option` ({}) not recognized'.format(value_option))
        for perturbation in self.perturbations:
            for cid in perturbation.clustering.clusters:
                if value_option == 'random':
                    perturbation.indicator.value[cid] = bool(pl.random.bernoulli.sample(p))
                else:
                    perturbation.indicator.value[cid] = value

        # These are for the function `self._make_idx_for_clusters`
        self.ndts_bias = []
        self.n_replicates = self.G.data.n_replicates
        self.n_perturbations = len(self.G.perturbations)
        self.n_dts_for_replicate = self.G.data.n_dts_for_replicate
        self.total_dts = np.sum(self.n_dts_for_replicate)
        self.replicate_bias = np.zeros(self.n_replicates, dtype=int)
        self.n_asvs = len(self.G.data.asvs)
        for ridx in range(1, self.n_replicates):
            self.replicate_bias[ridx] = self.replicate_bias[ridx-1] + \
                self.n_asvs * self.n_dts_for_replicate[ridx - 1]
        for ridx in range(self.G.data.n_replicates):
            self.ndts_bias.append(
                np.arange(0, self.G.data.n_dts_for_replicate[ridx] * self.n_asvs, self.n_asvs))

        s = 'Perturbation indicator initialization results:\n'
        for i, perturbation in enumerate(self.perturbations):
            s += '\tPerturbation {}:\n' \
                '\t\tindicator: {}\n'.format(i, perturbation.indicator.cluster_bool_array())
        logging.info(s)

    def _make_idx_for_clusters(self):
        '''Creates a dictionary that maps the cluster id to the
        rows that correspond to each ASV in the cluster.

        We cannot cast this with numba because it does not support Fortran style
        raveling :(.

        Returns
        -------
        dict: int -> np.ndarray
            Maps the cluster ID to the row indices corresponding to it
        '''
        clusters = [np.asarray(oidxs, dtype=int).reshape(-1,1) \
            for oidxs in self.clustering.toarray()]
        n_dts=self.G.data.n_dts_for_replicate

        d = {}
        cids = self.clustering.order

        for cidx,cid in enumerate(cids):
            a = np.zeros(len(clusters[cidx]) * self.total_dts, dtype=int)
            i = 0
            for ridx in range(self.n_replicates):
                idxs = np.zeros(
                    (len(clusters[cidx]),
                    self.n_dts_for_replicate[ridx]), int)
                idxs = idxs + clusters[cidx]
                idxs = idxs + self.ndts_bias[ridx]
                idxs = idxs + self.replicate_bias[ridx]
                idxs = idxs.ravel('F')
                l = len(idxs)
                a[i:i+l] = idxs
                i += l

            d[cid] = a
        
        if self.G.data.zero_inflation_transition_policy is not None:
            # We need to convert the indices that are meant from no zero inflation to 
            # ones that take into account zero inflation - use the array from 
            # `data.Data._setrows_to_include_zero_inflation`. If the index should be
            # included, then we subtract the number of indexes that are previously off
            # before that index. If it should not be included then we exclude it
            prevoff_arr = self.G.data.off_previously_arr_zero_inflation
            rows_to_include = self.G.data.zero_inflation_transition_policy
            for cid in d:
                arr = d[cid]
                new_arr = np.zeros(len(arr), dtype=int)
                n = 0
                for i, idx in enumerate(arr):
                    if rows_to_include[idx]:
                        new_arr[n] = idx - prevoff_arr[i]
                        n += 1
                new_arr = new_arr[:n]
        return d

    # @profile
    def make_rel_params(self):
        '''We make the parameters needed to update the relative log-likelihod.
        This function is called once at the beginning of the update.

        THIS ASSUMES THAT EACH PERTURBATION CLUSTERS ARE DEFINED BY THE SAME CLUSTERS
            - To make this separate, make a higher level list for each perturbation index
              for each individual perturbation

        Parameters that we create with this function
        --------------------------------------------
        ys : dict (int -> np.ndarray)
            Maps the target cluster id to the observation matrix that it
            corresponds to (only the ASVs in the target cluster). This 
            array already has the growth and self-interactions subtracted
            out:
                $ \frac{log(x_{k+1}) - log(x_{k})}{dt} - a_{1,k} - a_{2,k}x_{k} $
        process_precs : dict (int -> np.ndarray)
            Maps the target cluster id to the vector of the process precision
            that corresponds to the target cluster (only the ASVs in the target
            cluster). This is a 1D array that corresponds to the diagonal of what
            would be the precision matrix.
        interactionXs : dict (int -> np.ndarray)
            Maps the target cluster id to the design matrix for the interactions
            going into that cluster. We pre-index it with the rows and columns
        prior_prec_interaction : dict (int -> np.ndarray)
            Maps the target cluster id to to the diagonal of the prior precision 
            for the interaction values.
        prior_mean_interaction : dict (int -> np.ndarray)
            Maps the target cluster id to to the diagonal of the prior mean 
            for the interaction values.
        prior_ll_ons : np.ndarray
            Prior log likelihood of a positive indicator. These are separate for each
            perturbation.
        prior_ll_offs : np.ndarray
            Prior log likelihood of the negative indicator. These are separate for each
            perturbation.
        priorvar_logdet_diffs : np.ndarray
            This is the prior variance log determinant that we add when the indicator
            is positive. This is different for each perturbation.
        perturbationsXs : dict (int -> np.ndarray)
            Maps the target cluster id to the design matrix that corresponds to 
            the on perturbations of the target clusters. This is preindexed by the 
            rows but not the columns - the columns assume that all of the perturbations
            are on and we index the ones that we want.
        prior_prec_perturbations : np.ndarray
            This is the prior precision of the magnitude for each of the perturbations. Use
            the perturbation index to get the value
        prior_mean_perturbations : np.ndarray
            This is the prior mean of the magnitude for each one of the perturbations. Use
            the perturbation index to get the value
        '''
        row_idxs = self._make_idx_for_clusters()

        # Create ys
        self.ys = {}
        y = self.G.data.construct_lhs(keys=[
            REPRNAMES.SELF_INTERACTION_VALUE, REPRNAMES.GROWTH_VALUE],
            kwargs_dict={REPRNAMES.GROWTH_VALUE:{'with_perturbations': False}})
        for tcid in self.clustering.order:
            self.ys[tcid] = y[row_idxs[tcid], :]

        # Create process_precs
        self.process_precs = {}
        process_prec_diag = self.G[REPRNAMES.PROCESSVAR].prec
        for tcid in self.clustering.order:
            self.process_precs[tcid] = process_prec_diag[row_idxs[tcid]]

        # Make interactionXs
        self.interactionXs = {}
        interactions = self.G[REPRNAMES.INTERACTIONS_OBJ]
        XM_master = self.G.data.design_matrices[REPRNAMES.CLUSTER_INTERACTION_VALUE].toarray()
        for tcid in self.clustering.order:
            cols = []
            for i, interaction in enumerate(interactions.iter_valid()):
                if interaction.target_cid == tcid:
                    if interaction.indicator:
                        cols.append(i)
            cols = np.asarray(cols, dtype=int)
            self.interactionXs[tcid] = pl.util.fast_index(M=XM_master, 
                rows=row_idxs[tcid], cols=cols)

        # Make prior parameters for interactions
        self.prior_prec_interaction = 1/self.G[REPRNAMES.PRIOR_VAR_INTERACTIONS].value
        self.prior_mean_interaction = self.G[REPRNAMES.PRIOR_MEAN_INTERACTIONS].value

        # Make the perturbation parameters
        self.prior_ll_ons = []
        self.prior_ll_offs = []
        self.priorvar_logdet_diffs = []
        self.prior_prec_perturbations = []
        self.prior_mean_perturbations = []

        for perturbation in self.G.perturbations:
            prob_on = perturbation.probability.value
            self.prior_ll_ons.append(np.log(prob_on))
            self.prior_ll_offs.append(np.log(1 - prob_on))
            
            self.priorvar_logdet_diffs.append(
                np.log(perturbation.magnitude.prior.var.value))

            self.prior_prec_perturbations.append( 
                1/perturbation.magnitude.prior.var.value)

            self.prior_mean_perturbations.append(
                perturbation.magnitude.prior.mean.value)

        # Make perturbation matrices
        self.perturbationsXs = {}
        self.G.data.design_matrices[REPRNAMES.PERT_VALUE].M.build(build=True, 
            build_for_neg_ind=True)
        Xpert_master = self.G.data.design_matrices[REPRNAMES.PERT_VALUE].toarray()
        for tcid in self.clustering.order:
            self.perturbationsXs[tcid] = Xpert_master[row_idxs[tcid], :]

        self.n_clusters = len(self.clustering.order)
        self.clustering_order = self.clustering.order

        self.col2pidxcidx = []
        for pidx in range(len(self.perturbations)):
            for cidx in range(len(self.clustering.order)):
                self.col2pidxcidx.append((pidx, cidx))

        self.arr = []
        for perturbation in self.perturbations:
            self.arr = np.append(
                self.arr, 
                perturbation.indicator.cluster_bool_array())
        self.arr = np.asarray(self.arr, dtype=bool)

    # @profile
    def update_relative(self):
        '''Update each perturbation indicator for the given cluster by
        calculating the realtive loglikelihoods of it being on/off as
        supposed to as is. Because this is a relative loglikelihood, we
        only need to take into account the following parameters of the
        model:
            - Only the ASVs in the cluster in question
            - Only the perturbations for that cluster
            - Only the interactions going into the cluster

        Because these matrices are considerably smaller and considered 'dense', we
        do the operations in numpy instead of scipy sparse.

        We permute the order that the indices are updated for more robust mixing
        '''
        if self.sample_iter < self.delay:
            return
        start_time = time.time()

        self.make_rel_params()

        # Iterate over each perturbation indicator variable
        iidxs = npr.permutation(len(self.arr))
        for iidx in iidxs:
            self.update_single_idx_fast(idx=iidx)

        # Set the perturbation indicators from arr
        i = 0
        for perturbation in self.perturbations:
            for cid in self.clustering.order:
                perturbation.indicator.value[cid] = self.arr[i]
                i += 1

        # rebuild the growth design matrix
        self.G.data.design_matrices[REPRNAMES.PERT_VALUE].M.build()
        self.G.data.design_matrices[REPRNAMES.GROWTH_VALUE].build_with_perturbations()
        self._time_taken = time.time() - start_time

    # @profile
    def update_single_idx_fast(self, idx):
        '''Do a Gibbs step for a single cluster
        '''
        pidx, cidx = self.col2pidxcidx[idx]

        prior_ll_on = self.prior_ll_ons[pidx]
        prior_ll_off = self.prior_ll_offs[pidx]

        d_on = self.calculate_relative_marginal_loglikelihood(idx=idx, val=True)
        d_off = self.calculate_relative_marginal_loglikelihood(idx=idx, val=False)

        ll_on = d_on + prior_ll_on
        ll_off = d_off + prior_ll_off
        dd = [ll_off, ll_on]

        # print('\nindicator', idx)
        # print('fast\n\ttotal: {}\n\tbeta_logdet_diff: {}\n\t' \
        #     'priorvar_logdet_diff: {}\n\tbEb_diff: {}\n\t' \
        #     'bEbprior_diff: {}'.format(
        #         ll_on - ll_off,
        #         d_on['beta_logdet'] - d_off['beta_logdet'],
        #         d_on['priorvar_logdet'] - d_off['priorvar_logdet'],
        #         d_on['bEb'] - d_off['bEb'],
        #         d_on['bEbprior'] - d_off['bEbprior']))
        # self.update_single_idx_slow(idx)

        res = bool(sample_categorical_log(dd))
        self.arr[idx] = res

    # @profile
    def calculate_relative_marginal_loglikelihood(self, idx, val):
        '''Calculate the relative marginal loglikelihood of switching the `idx`'th index
        of the perturbation matrix to `val`

        Parameters
        ----------
        idx : int
            This is the index of the indicator we are sampling
        val : bool
            This is the value we are testing it at.

        Returns
        -------
        float
        '''
        # Create and get the data
        self.arr[idx] = val
        pidx, cidx = self.col2pidxcidx[idx]
        tcid = self.clustering_order[cidx]

        y = self.ys[tcid]
        process_prec = self.process_precs[tcid]
        X = self.interactionXs[tcid]

        prior_mean = []
        prior_prec_diag = []
        cols = []
        for temp_pidx in range(len(self.perturbations)):
            col = int(cidx + temp_pidx * self.n_clusters)
            if self.arr[col]:
                cols.append(col)
                prior_mean.append(self.prior_mean_perturbations[temp_pidx])
                prior_prec_diag.append(self.prior_prec_perturbations[temp_pidx])
        Xpert = self.perturbationsXs[tcid][:, cols]

        if Xpert.shape[1] + X.shape[1] == 0:
            # return {
            #     'ret': 0,
            #     'beta_logdet': 0,
            #     'priorvar_logdet': 0,
            #     'bEb': 0,
            #     'bEbprior': 0}
            return 0

        prior_mean = np.append(
            prior_mean,
            np.full(X.shape[1], self.prior_mean_interaction))
        prior_prec_diag = np.append(
            prior_prec_diag,
            np.full(X.shape[1], self.prior_prec_interaction))
        X = np.hstack((Xpert, X))
        prior_prec = np.diag(prior_prec_diag)
        pm = (prior_prec_diag * prior_mean).reshape(-1,1)

        # Do the marginalization
        a = X.T * process_prec
        beta_prec = (a @ X) + prior_prec
        beta_cov = pinv(beta_prec, self)
        beta_mean = beta_cov @ ((a @ y) + pm )

        bEb = (beta_mean.T @ beta_prec @ beta_mean)[0,0]
        try:
            beta_logdet = log_det(beta_cov, self)
        except:
            logging.critical('Crashed in log_det')
            logging.critical('beta_cov:\n{}'.format(beta_cov))
            logging.critical('prior_prec\n{}'.format(prior_prec))
            raise
        
        if val:
            bEbprior = (self.prior_mean_perturbations[pidx]**2) * \
                self.prior_prec_perturbations[pidx]
            priorvar_logdet = self.priorvar_logdet_diffs[pidx]
        else:
            bEbprior = 0
            priorvar_logdet = 0

        ll2 = 0.5 * (beta_logdet - priorvar_logdet)
        ll3 = 0.5 * (bEb - bEbprior)

        # return {
        #     'ret': ll2+ll3,
        #     'beta_logdet': beta_logdet,
        #     'priorvar_logdet': priorvar_logdet,
        #     'bEb': bEb,
        #     'bEbprior': bEbprior}
        return ll2 + ll3        

    # @profile
    def update_slow(self):
        '''Update each cluster indicator variable for the perturbation
        '''
        start_time = time.time()

        if self.sample_iter < self.delay:
            return

        n_clusters = len(self.clustering.order)
        n_perturbations = len(self.perturbations)
        idxs = npr.permutation(int(n_clusters*n_perturbations))
        for idx in idxs:
            self.update_single_idx_slow(idx=idx)

        # rebuild the growth design matrix
        self.G.data.design_matrices[REPRNAMES.PERT_VALUE].M.build()
        self.G.data.design_matrices[REPRNAMES.GROWTH_VALUE].build_with_perturbations()
        self._time_taken = time.time() - start_time

    # @profile
    def update_single_idx_slow(self, idx):
        '''Do a Gibbs step for a single cluster and perturbation

        Parameters
        ----------
        This is the index of the indicator in vectorized form
        '''
        cidx = idx % self.G.data.n_asvs
        cid = self.clustering.order[cidx]
        
        pidx = idx // self.G.data.n_asvs
        perturbation = self.perturbations[pidx]

        d_on = self.calculate_marginal_loglikelihood(cid=cid, val=True,
            perturbation=perturbation)
        d_off = self.calculate_marginal_loglikelihood(cid=cid, val=False,
            perturbation=perturbation)

        prior_ll_on = np.log(perturbation.probability.value)
        prior_ll_off = np.log(1 - perturbation.probability.value)

        ll_on = d_on['ret'] + prior_ll_on
        ll_off = d_off['ret'] + prior_ll_off

        # print('slow\n\ttotal: {}\n\tbeta_logdet_diff: {}\n\t' \
        #     'priorvar_logdet_diff: {}\n\tbEb_diff: {}\n\t' \
        #     'bEbprior_diff: {}'.format(
        #         ll_on - ll_off,
        #         d_on['beta_logdet'] - d_off['beta_logdet'],
        #         d_on['priorvar_logdet'] - d_off['priorvar_logdet'],
        #         d_on['bEb'] - d_off['bEb'],
        #         d_on['bEbprior'] - d_off['bEbprior']))

        dd = [ll_off, ll_on]

        res = bool(sample_categorical_log(dd))
        if perturbation.indicator.value[cid] != res:
            perturbation.indicator.value[cid] = res
            self.G.data.design_matrices[REPRNAMES.PERT_VALUE].build()

    # @profile
    def calculate_marginal_loglikelihood(self, cid, val, perturbation):
        '''Calculate the log marginal likelihood with the perturbations integrated
        out
        '''
        # Set parameters
        perturbation.indicator.value[cid] = val
        self.G.data.design_matrices[REPRNAMES.PERT_VALUE].M.build()

        # Make matrices
        rhs = [REPRNAMES.PERT_VALUE, REPRNAMES.CLUSTER_INTERACTION_VALUE]
        lhs = [REPRNAMES.GROWTH_VALUE, REPRNAMES.SELF_INTERACTION_VALUE]
        X = self.G.data.construct_rhs(keys=rhs)
        y = self.G.data.construct_lhs(keys=lhs, 
            kwargs_dict={REPRNAMES.GROWTH_VALUE:{'with_perturbations': False}})
        
        if X.shape[1] == 0:
            return {
            'ret': 0,
            'beta_logdet': 0,
            'priorvar_logdet': 0,
            'bEb': 0,
            'bEbprior': 0}

        process_prec = self.G[REPRNAMES.PROCESSVAR].build_matrix(cov=False, sparse=False)
        prior_prec = build_prior_covariance(G=self.G, cov=False, order=rhs, sparse=False)
        prior_var = build_prior_covariance(G=self.G, cov=True, order=rhs, sparse=False)
        prior_mean = build_prior_mean(G=self.G, order=rhs, shape=(-1,1))

        # Calculate the posterior
        beta_prec = X.T @ process_prec @ X + prior_prec
        beta_cov = pinv(beta_prec, self)
        beta_mean = beta_cov @ ( X.T @ process_prec @ y + prior_prec @ prior_mean )
        beta_mean = np.asarray(beta_mean).reshape(-1,1)

        # Perform the marginalization
        try:
            beta_logdet = log_det(beta_cov, self)
        except:
            logging.critical('Crashed in log_det')
            logging.critical('beta_cov:\n{}'.format(beta_cov))
            logging.critical('prior_prec\n{}'.format(prior_prec))
            raise
        priorvar_logdet = log_det(prior_var, self)
        ll2 = 0.5 * (beta_logdet - priorvar_logdet)

        a = np.asarray(prior_mean.T @ prior_prec @ prior_mean)[0,0]
        b = np.asarray(beta_mean.T @ beta_prec @ beta_mean)[0,0]
        ll3 = -0.5 * (a  - b)

        return {
            'ret': ll2+ll3,
            'beta_logdet': beta_logdet,
            'priorvar_logdet': priorvar_logdet,
            'bEb': b,
            'bEbprior': a}

    def total_on(self):
        n = 0
        for perturbation in self.perturbations:
            n += perturbation.indicator.num_on_clusters()
        return n


class PriorVarPerturbations(pl.Variable):
    '''Agglomerates the prior variances of the magnitudes for the perturbations.

    All perturbations get the same hyperparameters
    '''
    def __init__(self, **kwargs):
        kwargs['name'] = STRNAMES.PRIOR_VAR_PERT
        pl.Variable.__init__(self, **kwargs)
        self.perturbations = self.G.perturbations

        if self.perturbations is None:
            raise TypeError('Only instantiate this object if there are perturbations')

    def __str__(self):
        s = 'Perturbation Magnitude Prior Variances'
        for perturbation in self.perturbations:
            s += '\n\tperturbation {}: {}'.format(
                perturbation.name,
                perturbation.magnitude.prior.var.value)
        return s

    @property
    def sample_iter(self):
        return self.perturbations[0].magnitude.prior.var.sample_iter

    def initialize(self, **kwargs):
        '''Every prior variance on the perturbations gets the same hyperparameters
        '''
        for perturbation in self.perturbations:
            perturbation.magnitude.prior.var.initialize(**kwargs)

    def update(self):
        for perturbation in self.perturbations:
            perturbation.magnitude.prior.var.update()

    def set_trace(self, *args, **kwargs):
        for perturbation in self.perturbations:
            perturbation.magnitude.prior.var.set_trace(*args, **kwargs)

    def add_trace(self, *args, **kwargs):
        for perturbation in self.perturbations:
            perturbation.magnitude.prior.var.add_trace(*args, **kwargs)

    def add_init_value(self):
        '''Set the initialization value. This is called by `pylab.inference.BaseMCMC.run`
        when first updating the variable. User should not use this function
        '''
        for perturbation in self.perturbations:
            perturbation.add_init_value()

    def get_single_value_of_perts(self):
        '''Get the variance for each perturbation
        '''
        return np.asarray([p.magnitude.prior.var.value for p in self.perturbations])

    def diag(self, only_pos_ind=True):
        '''Return the diagonal of the prior variances stacked up in order

        Parameters
        ----------
        only_pos_ind : bool
            If True, only put in the values for the positively indicated clusters
            for each perturbation

        Returns
        -------
        np.ndarray
        '''
        ret = []
        for perturbation in self.perturbations:
            if only_pos_ind:
                n = perturbation.indicator.num_on_clusters()
            else:
                n = len(perturbation.clustering)
            ret = np.append(
                ret,
                np.ones(n, dtype=float)*perturbation.magnitude.prior.var.value)
        return ret


class PriorVarPerturbationSingle(pl.variables.SICS):
    '''This is the posterior of the prior variance of regression coefficients
    for the interaction (off diagonal) variables
    '''
    def __init__(self, prior, perturbation, value=None, **kwargs):

        kwargs['name'] = STRNAMES.PRIOR_VAR_PERT + '_' + perturbation.name
        pl.variables.SICS.__init__(self, value=value, dtype=float, **kwargs)
        self.add_prior(prior)
        self.perturbation = perturbation

    def initialize(self, value_option, dof_option, scale_option, value=None,
        dof=None, scale=None, delay=0):
        '''Initialize the hyperparameters of the perturbation prior variance based on the
        passed in option

        Parameters
        ----------
        value_option : str
            - Initialize the value based on the specified option
            - Options
                'manual'
                    Set the value manually, `value` must also be specified
                'auto', 'prior-mean'
                    Set the value to the mean of the prior
                'tight'
                    value = 10^2
                'diffuse'
                    value = 10^4
        scale_option : str
            Initialize the scale of the prior
            Options
                'manual'
                    Set the value manually, `scale` must also be specified
                'auto', 'diffuse'
                    Set so that the mean of the distribution is 10^4
                'tight'
                    Set so that the mean of the distribution is 10^2
        dof_option : str
            Initialize the dof of the parameter
            Options:
                'manual': Set the value with the parameter `dof`
                'diffuse': Set the value to 2.5
                'strong': Set the value to the expected number of interactions
                'auto': Set to diffuse
        dof, scale : int, float
            User specified values
            Only necessary if  any of the options are 'manual'
        '''
        if not pl.isint(delay):
            raise TypeError('`delay` ({}) must be an int'.format(type(delay)))
        if delay < 0:
            raise ValueError('`delay` ({}) must be >= 0'.format(delay))
        self.delay = delay

        if not pl.isstr(dof_option):
            raise TypeError('`dof_option` ({}) must be a str'.format(type(dof_option)))
        if dof_option == 'manual':
            if not pl.isnumeric(dof):
                raise TypeError('`dof` ({}) must be a numeric'.format(type(dof)))
            if dof < 0:
                raise ValueError('`dof` ({}) must be > 0 for it to be a valid prior'.format(shape))
        elif dof_option in ['diffuse', 'auto']:
            dof = 2.5
        elif dof_option == 'strong':
            dof = expected_n_clusters(G=self.G)
        else:
            raise ValueError('`dof_option` ({}) not recognized'.format(dof_option))
        self.prior.dof.override_value(dof)

        if not pl.isstr(scale_option):
            raise TypeError('`scale_option` ({}) must be a str'.format(type(scale_option)))
        if scale_option == 'manual':
            if not pl.isnumeric(scale):
                raise TypeError('`scale` ({}) must be a numeric'.format(type(scale)))
            if scale < 0:
                raise ValueError('`scale` ({}) must be > 0 for it to be a valid prior'.format(scale))
        elif scale_option in ['auto', 'diffuse']:
            # Calculate the mean to be 10
            scale = 1e4 * (self.prior.dof.value - 2) / self.prior.dof.value
        elif scale_option == 'tight':
            scale = 100 * (self.prior.dof.value - 2) / self.prior.dof.value
        else:
            raise ValueError('`scale_option` ({}) not recognized'.format(scale_option))
        self.prior.scale.override_value(scale)

        if not pl.isstr(value_option):
            raise TypeError('`value_option` ({}) must be a str'.format(type(value_option)))
        if value_option == 'manual':
            if not pl.isnumeric(value):
                raise ValueError('`value` ({}) must be numeric (float,int)'.format(value.__class__))
            self.value = value
        elif value_option in ['auto', 'prior-mean']:
            self.value = self.prior.mean()
        elif value_option == 'diffuse':
            self.value = 1e4
        elif value_option == 'tight':
            self.value = 1e2
        else:
            raise ValueError('`value_option` ({}) not recognized'.format(value_option))

        logging.info('Prior Variance Interactions initialization results:\n' \
            '\tprior dof: {}\n' \
            '\tprior scale: {}\n' \
            '\tvalue: {}'.format(
                self.prior.dof.value, self.prior.scale.value, self.value))

    # @profile
    def update(self):
        '''Calculate the posterior of the prior variance
        '''
        if self.sample_iter < self.delay:
            return

        x = self.perturbation.cluster_array(only_pos_ind=True)
        mu = self.perturbation.magnitude.prior.mean.value

        se = np.sum(np.square(x - mu))
        n = len(x)

        self.dof.value = self.prior.dof.value + n
        self.scale.value = ((self.prior.scale.value * self.prior.dof.value) + \
           se)/self.dof.value
        self.sample()


class PriorMeanPerturbations(pl.Variable):
    '''Agglomerates the prior variances of the magnitudes for the perturbations.

    All perturbations get the same hyperparameters
    '''
    def __init__(self, **kwargs):
        kwargs['name'] = STRNAMES.PRIOR_MEAN_PERT
        pl.Variable.__init__(self, **kwargs)
        self.perturbations = self.G.perturbations

        if self.perturbations is None:
            raise TypeError('Only instantiate this object if there are perturbations')

    def __str__(self):
        s = 'Perturbation Magnitude Prior Means'
        for perturbation in self.perturbations:
            s += '\n\tperturbation {}: {}'.format(
                perturbation.name,
                perturbation.magnitude.prior.mean.value)
        return s

    @property
    def sample_iter(self):
        return self.perturbations[0].magnitude.prior.mean.sample_iter

    def initialize(self, **kwargs):
        '''Every prior variance on the perturbations gets the same hyperparameters
        '''
        for perturbation in self.perturbations:
            perturbation.magnitude.prior.mean.initialize(**kwargs)

    def update(self):
        for perturbation in self.perturbations:
            perturbation.magnitude.prior.mean.update()

    def set_trace(self, *args, **kwargs):
        for perturbation in self.perturbations:
            perturbation.magnitude.prior.mean.set_trace(*args, **kwargs)

    def add_trace(self, *args, **kwargs):
        for perturbation in self.perturbations:
            perturbation.magnitude.prior.mean.add_trace(*args, **kwargs)

    def add_init_value(self):
        '''Set the initialization value. This is called by `pylab.inference.BaseMCMC.run`
        when first updating the variable. User should not use this function
        '''
        for perturbation in self.perturbations:
            perturbation.add_init_value()

    def get_single_value_of_perts(self):
        '''Get the variance for each perturbation
        '''
        return np.asarray([p.magnitude.prior.mean.value for p in self.perturbations])

    def toarray(self, only_pos_ind=True):
        '''Return the diagonal of the prior variances stacked up in order

        Parameters
        ----------
        only_pos_ind : bool
            If True, only put in the values for the positively indicated clusters
            for each perturbation

        Returns
        -------
        np.ndarray
        '''
        ret = []
        for perturbation in self.perturbations:
            if only_pos_ind:
                n = perturbation.indicator.num_on_clusters()
            else:
                n = len(perturbation.clustering)
            ret = np.append(
                ret,
                np.ones(n, dtype=float)*perturbation.magnitude.prior.var.value)
        return ret


class PriorMeanPerturbationSingle(pl.variables.Normal):
    
    def __init__(self, prior, perturbation, **kwargs):

        kwargs['name'] = STRNAMES.PRIOR_MEAN_PERT + '_' + perturbation.name
        pl.variables.Normal.__init__(self, mean=None, var=None, dtype=float, **kwargs)
        self.add_prior(prior)
        self.perturbation = perturbation

    def initialize(self, value_option, mean_option, var_option, value=None,
        mean=None, var=None, delay=0):
        '''Initialize the hyperparameters

        Parameters
        ----------
        value_option : str
            How to set the value. Options:
                'zero'
                    Set to zero
                'prior-mean', 'auto'
                    Set to the mean of the prior
                'manual'
                    Specify with the `value` parameter
        mean_option : str
            How to set the mean of the prior
                'zero', 'auto'
                    Set to zero
                'manual'
                    Set with the `mean` parameter
        var_option : str
            'diffuse', 'auto'
                Variance is set to 10e4
            'tight'
                Variance is set to 1e2
            'manual'
                Set with the `var` parameter
        value, mean, var : float
            These are only necessary if we specify manual for any of the other 
            options
        delay : int
            How much to delay the start of the update during inference
        '''
        if not pl.isint(delay):
            raise TypeError('`delay` ({}) must be an int'.format(type(delay)))
        if delay < 0:
            raise ValueError('`delay` ({}) must be >= 0'.format(delay))
        self.delay = delay

        # Set the mean
        if not pl.isstr(mean_option):
            raise TypeError('`mean_option` ({}) must be a str'.format(type(mean_option)))
        if mean_option == 'manual':
            if not pl.isnumeric(mean):
                raise TypeError('`mean` ({}) must be a numeric'.format(type(mean)))
        elif mean_option in ['zero', 'auto']:
            mean = 0
        else:
            raise ValueError('`mean_option` ({}) not recognized'.format(mean_option))
        self.prior.mean.override_value(mean)

        # Set the variance
        if not pl.isstr(var_option):
            raise TypeError('`var_option` ({}) must be a str'.format(type(var_option)))
        if var_option == 'manual':
            if not pl.isnumeric(var):
                raise TypeError('`var` ({}) must be a numeric'.format(type(var)))
            if var <= 0:
                raise ValueError('`var` ({}) must be positive'.format(var))
        elif var_option in ['diffuse', 'auto']:
            var = 1e4
        elif var_option == 'tight':
            var = 1e2
        else:
            raise ValueError('`var_option` ({}) not recognized'.format(var_option))
        self.prior.var.override_value(var)

        # Set the value
        if not pl.isstr(value_option):
            raise TypeError('`value_option` ({}) must be a str'.format(type(value_option)))
        if value_option == 'manual':
            if not pl.isnumeric(value):
                raise TypeError('`value` ({}) must be a numeric'.format(type(value)))
        elif value_option in ['prior-mean', 'auto']:
            value = self.prior.mean.value
        elif value_option == 'zero':
            value = 0
        else:
            raise ValueError('`value_option` ({}) not recognized'.format(value_option))
        self.value = value
        
    def update(self):
        '''Update using a Gibbs update
        '''
        if self.sample_iter < self.delay:
            return

        x = self.perturbation.cluster_array(only_pos_ind=True)
        prec = 1/self.perturbation.magnitude.prior.var.value

        prior_prec = 1/self.prior.var.value
        prior_mean = self.prior.mean.value

        self.var.value = 1/(prior_prec + (len(x)*prec))
        self.mean.value = self.var.value * ((prior_mean * prior_prec) + (np.sum(x)*prec))
        self.sample()
