'''Logistic growth parameters for the posterior
'''
import logging
import time
import numpy as np
import numba
import numpy.random as npr

import matplotlib.pyplot as plt

from .util import expected_n_clusters, build_prior_covariance, build_prior_mean, sample_categorical_log, \
    log_det, pinv
from .perturbations import PerturbationMagnitudes
from .names import STRNAMES, REPRNAMES

from . import pylab as pl

class PriorVarInteractions(pl.variables.SICS):
    '''This is the posterior of the prior variance of regression coefficients
    for the interaction (off diagonal) variables
    '''
    def __init__(self, prior, value=None, **kwargs):

        kwargs['name'] = STRNAMES.PRIOR_VAR_INTERACTIONS
        pl.variables.SICS.__init__(self, value=value,
            dtype=float, **kwargs)
        self.add_prior(prior)

    def initialize(self, value_option, dof_option, scale_option, value=None,
        mean_scaling_factor=None, dof=None, scale=None, delay=0):
        '''Initialize the hyperparameters of the self interaction variance based on the
        passed in option

        Parameters
        ----------
        value_option : str
            - Initialize the value based on the specified option
            - Options
                - 'manual'
                    - Set the value manually, `value` must also be specified
                - 'auto', 'prior-mean'
                    - Set the value to the mean of the prior
        scale_option : str
            - Initialize the scale of the prior
            - Options
                - 'manual'
                    - Set the value manually, `scale` must also be specified
                - 'auto', 'same-as-aii'
                    - Set the mean the same as the self-interactions
        dof_option : str
            Initialize the dof of the parameter
            Options:
                'manual': Set the value with the parameter `dof`
                'diffuse': Set the value to 2.01
                'strong': Set the valuye to the expected number of interactions
                'auto': Set to diffuse
        dof, scale : int, float
            - User specified values
            - Only necessary if `hyperparam_option` == 'manual'
        '''
        if not pl.isint(delay):
            raise TypeError('`delay` ({}) must be an int'.format(type(delay)))
        if delay < 0:
            raise ValueError('`delay` ({}) must be >= 0'.format(delay))
        self.delay = delay

        self.interactions = self.G[REPRNAMES.CLUSTER_INTERACTION_VALUE]

        if not pl.isstr(dof_option):
            raise TypeError('`dof_option` ({}) must be a str'.format(type(dof_option)))
        if dof_option == 'manual':
            if not pl.isnumeric(dof):
                raise TypeError('`dof` ({}) must be a numeric'.format(type(dof)))
            if dof < 0:
                raise ValueError('`dof` ({}) must be > 0 for it to be a valid prior'.format(dof))
        elif dof_option in ['diffuse', 'auto']:
            dof = 2.01
        elif dof_option == 'strong':
            N = expected_n_clusters(G=self.G)
            dof = N * (N - 1)
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
        elif scale_option in ['auto', 'same-as-aii']:
            mean = self.G[REPRNAMES.PRIOR_VAR_SELF_INTERACTIONS].prior.mean()
            scale = mean * (self.prior.dof.value - 2) /(self.prior.dof.value)
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
            if not pl.isnumeric(mean_scaling_factor):
                raise ValueError('`mean_scaling_factor` ({}) must be a numeric type ' \
                    '(float,int)'.format(mean_scaling_factor.__class__))
            self.value = self.prior.mean()
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

        x = self.interactions.obj.get_values(use_indicators=True)
        mu = self.G[REPRNAMES.PRIOR_MEAN_INTERACTIONS].value

        se = np.sum(np.square(x - mu))
        n = len(x)

        self.dof.value = self.prior.dof.value + n
        self.scale.value = ((self.prior.scale.value * self.prior.dof.value) + \
           se)/self.dof.value
        self.sample()


class PriorMeanInteractions(pl.variables.Normal):
    '''This is the posterior mean for the interactions
    '''

    def __init__(self, prior, **kwargs):
        kwargs['name'] = STRNAMES.PRIOR_MEAN_INTERACTIONS
        pl.variables.Normal.__init__(self, mean=None, var=None, dtype=float, **kwargs)
        self.add_prior(prior)

    def __str__(self):
        # If this fails, it is because we are dividing by 0 sampler_iter
        # If which case we just return the value 
        try:
            s = 'Value: {}, Acceptance rate: {}'.format(
                self.value, np.mean(self.acceptances[
                    np.max([self.sample_iter-50, 0]):self.sample_iter]))
        except:
            s = str(self.value)
        return s

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
            'same-as-aii', 'auto'
                Set as the same variance as the self-interactions
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
        elif var_option in ['same-as-aii', 'auto']:
            var = self.G[STRNAMES.PRIOR_VAR_SELF_INTERACTIONS].value
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
        '''Update using Gibbs sampling
        '''
        if self.sample_iter < self.delay:
            return

        if self.G[REPRNAMES.CLUSTER_INTERACTION_INDICATOR].num_pos_indicators == 0:
            # sample from the prior
            self.value = self.prior.sample()
            return

        x = self.G[REPRNAMES.CLUSTER_INTERACTION_VALUE].value
        prec = 1/self.G[REPRNAMES.PRIOR_VAR_INTERACTIONS].value

        prior_prec = 1/self.prior.var.value
        prior_mean = self.prior.mean.value

        self.var.value = 1/(prior_prec + (len(x)*prec))
        self.mean.value = self.var.value * ((prior_mean * prior_prec) + (np.sum(x)*prec))
        self.sample()


class ClusterInteractionValue(pl.variables.MVN):
    '''Interactions of Lotka-Voltera

    Since we initialize the interactions object in the `initialize` function,
    make sure that you have initialized the prior of the values of the interactions
    and of the indicators of the interactions before you call the initialization of
    this class
    '''
    def __init__(self, prior, clustering, **kwargs):
        kwargs['name'] = STRNAMES.CLUSTER_INTERACTION_VALUE
        pl.variables.MVN.__init__(self, dtype=float, **kwargs)
        self.set_value_shape(shape=(len(self.G.data.asvs),len(self.G.data.asvs)))
        self.add_prior(prior)
        self.clustering = clustering
        self.obj = pl.contrib.Interactions(
            clustering=self.clustering,
            use_indicators=True,
            name=STRNAMES.INTERACTIONS_OBJ, G=self.G,
            signal_when_clusters_change=False)
        self._strr = 'None'

    def __str__(self):
        return self._strr

    def __len__(self):
        # Return the number of on interactions
        return self.obj.num_pos_indicators()

    def set_values(self, *args, **kwargs):
        '''Set the values from an array
        '''
        self.obj.set_values(*args, **kwargs)

    def initialize(self, value_option, hyperparam_option=None, value=None,
        indicators=None, delay=0):
        '''Initialize the interactions object.

        Parameters
        ----------
        value_option : str
            This is how to initialize the values
            Options:
                'manual'
                    Set the values of the interactions manually. `value` and `indicators`
                    must also be specified. We assume the values are only set for when
                    `indicators` is True, and that the order of the `indicators` and `values`
                    correspond to how we iterate over the interactions
                    Example
                        3 Clusters
                        indicators = [True, False, False, True, False, True]
                        value = [0.2, 0.8, -0.35]
                'all-off', 'auto'
                    Set all of the interactions and the indicators to 0
                'all-on'
                    Set all of the indicators to on and all the values to 0
        delay : int
            How many MCMC iterations to delay starting to update
        See also
        --------
        `pylab.cluster.Interactions.set_values`
        '''
        self.obj.set_signal_when_clusters_change(True)
        self.G[REPRNAMES.INTERACTIONS_OBJ].value_initializer = self.prior.sample
        self.G[REPRNAMES.INTERACTIONS_OBJ].indicator_initializer = self.G[REPRNAMES.INDICATOR_PROB].prior.sample

        self._there_are_perturbations = self.G.perturbations is not None

        if not pl.isint(delay):
            raise TypeError('`delay` ({}) must be an int'.format(type(delay)))
        if delay < 0:
            raise ValueError('`delay` ({}) must be >= 0'.format(delay))
        self.delay = delay

        if not pl.isstr(value_option):
            raise TypeError('`value_option` ({}) must be a str'.format(type(value_option)))
        if value_option in ['auto', 'all-off']:
            for interaction in self.obj:
                interaction.value = 0
                interaction.indicator = False
        elif value_option == 'all-on':
            for interaction in self.obj:
                interaction.value = 0
                interaction.indicator = True
        elif value_option == 'manual':
            if not np.all(pl.itercheck([value, indicators], pl.isarray)):
                raise TypeError('`value` ({}) and `indicators` ({}) must be arrays'.format(
                    type(value), type(indicators)))
            if len(value) != np.sum(indicators):
                raise ValueError('Length of `value` ({}) must equal the number of positive ' \
                    'values in `indicators` ({})'.format(len(value), np.sum(indicators)))
            if len(indicators) != self.obj.size:
                raise ValueError('The length of `indicators` ({}) must be the same as the ' \
                    'number of possible interactions ({})'.format(len(indicators), self.obj.size))
            ii = 0
            for i,interaction in enumerate(self.obj):
                interaction.indicator = indicators[i]
                if interaction.indicator:
                    interaction.value = value[ii]
                    ii += 1
        else:
            raise ValueError('`value_option` ({}) not recognized'.format(value_option))

        self._strr = str(self.obj.get_values(use_indicators=True))
        self.value = self.obj.get_values(use_indicators=True)

    def update(self):
        '''Update the values (where the indicators are positive) using a multivariate normal
        distribution - call this from regress coeff if you want to update the interactions
        conditional on all the other parameters.
        '''
        if self.obj.sample_iter < self.delay:
            return
        if self.obj.num_pos_indicators() == 0:
            # logging.info('No positive indicators, skipping')
            self._strr = '[]'
            return

        rhs = [
            REPRNAMES.CLUSTER_INTERACTION_VALUE]
        lhs = [
            REPRNAMES.GROWTH_VALUE,
            REPRNAMES.SELF_INTERACTION_VALUE]
        X = self.G.data.construct_rhs(keys=rhs)
        y = self.G.data.construct_lhs(keys=lhs,
            kwargs_dict={REPRNAMES.GROWTH_VALUE:{
                'with_perturbations':self._there_are_perturbations}})
        process_prec = self.G[REPRNAMES.PROCESSVAR].build_matrix(
            cov=False, sparse=True)
        prior_prec = build_prior_covariance(G=self.G, cov=False,
            order=rhs, sparse=True)

        pm = prior_prec @ (self.prior.mean.value * np.ones(prior_prec.shape[0]).reshape(-1,1))

        prec = X.T @ process_prec @ X + prior_prec
        cov = pinv(prec, self)
        mean = (cov @ (X.T @ process_prec.dot(y) + pm)).ravel()

        # print(np.hstack((y, self.G.data.lhs.vector.reshape(-1,1))))
        # for perturbation in self.G.perturbations:
        #     print()
        #     print(perturbation.magnitude.cluster_array())
        #     print(perturbation.indicator.cluster_array())

        self.mean.value = mean
        self.cov.value = cov
        value = self.sample()
        self.obj.set_values(arr=value, use_indicators=True)
        self.update_str()

        if np.any(np.isnan(self.value)):
            logging.critical('mean: {}'.format(self.mean.value))
            logging.critical('nan in cov: {}'.format(np.any(np.isnan(self.cov.value))))
            logging.critical('value: {}'.format(self.value))
            raise ValueError('`Values in {} are nan: {}'.format(self.name, self.value))

    def update_str(self):
        self._strr = str(self.obj.get_values(use_indicators=True))

    def set_trace(self):
        self.obj.set_trace()

    def add_trace(self):
        self.obj.add_trace()


class ClusterInteractionIndicatorProbability(pl.variables.Beta):
    '''This is the posterior for the probability of a cluster being on
    '''
    def __init__(self, prior, **kwargs):
        '''Parameters

        prior (pl.variables.Beta)
            - prior probability
        **kwargs
            - Other options like graph, value
        '''
        kwargs['name'] = STRNAMES.INDICATOR_PROB
        pl.variables.Beta.__init__(self, a=prior.a.value, b=prior.b.value,
            dtype=float, **kwargs)
        self.add_prior(prior)

    def initialize(self, value_option, hyperparam_option, a=None, b=None, value=None,
        N='auto', delay=0):
        '''Initialize the hyperparameters of the beta prior

        Parameters
        ----------
        value_option : str
            - Option to initialize the value by
            - Options
                - 'manual'
                    - Set the values manually, `value` must be specified
                - 'auto'
                    - Set to the mean of the prior
        hyperparam_option : str
            - If it is a string, then set it by the designated option
            - Options
                - 'manual'
                    - Set the value manually. `a` and `b` must also be specified
                - 'weak-agnostic'
                    - a=b=0.5
                - 'strong-dense'
                    - a = N(N-1), N are the expected number of clusters
                    - b = 0.5
                - 'strong-sparse'
                    - a = 0.5
                    - b = N(N-1), N are the expected number of clusters
                - 'very-strong-sparse'
                    - a = 0.5
                    - b = n_asvs * (n_asvs-1)
        N : str, int
            This is the number of clusters to set the hyperparam options to 
            (if they are dependent on the number of cluster). If 'auto', set to the expected number
            of clusters from a dirichlet process. Else use this number (must be an int).
        a, b : int, float
            - User specified values
            - Only necessary if `hyperparam_option` == 'manual'
        '''
        if not pl.isint(delay):
            raise TypeError('`delay` ({}) must be an int'.format(type(delay)))
        if delay < 0:
            raise ValueError('`delay` ({}) must be >= 0'.format(delay))
        self.delay = delay

        if hyperparam_option == 'manual':
            if pl.isnumeric(a) and pl.isnumeric(b):
                self.prior.a.override_value(a)
                self.prior.b.override_value(b)
            else:
                raise ValueError('a ({}) and b ({}) must be numerics (float, int)'.format(
                    a.__class__, b.__class__))
        elif hyperparam_option in ['weak-agnostic', 'auto']:
            self.prior.a.override_value(0.5)
            self.prior.b.override_value(0.5)
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
            self.prior.a.override_value(N * (N - 1))
            self.prior.b.override_value(0.5)
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
            self.prior.a.override_value(0.5)
            self.prior.b.override_value((N * (N - 1)))
        elif hyperparam_option == 'very-strong-sparse':
            N = self.G.data.n_asvs
            self.prior.a.override_value(0.5)
            self.prior.b.override_value((N * (N - 1)))
        else:
            raise ValueError('option `{}` not recognized'.format(hyperparam_option))

        if value_option == 'manual':
            if pl.isnumeric(value):
                self.value = value
            else:
                raise ValueError('`value` ({}) must be a numeric (float,int)'.format(
                    value.__class__))
        elif value_option == 'auto':
            self.value = self.prior.mean()/100000
        else:
            raise ValueError('value option "{}" not recognized for indicator prob'.format(
                value_option))

        self.a.value = self.prior.a.value
        self.b.value = self.prior.b.value
        logging.info('Indicator Probability initialization results:\n' \
            '\tprior a: {}\n' \
            '\tprior b: {}\n' \
            '\tvalue: {}'.format(
                self.prior.a.value, self.prior.b.value, self.value))

    def update(self):
        '''Sample the posterior given the data
        '''
        if self.sample_iter < self.delay:
            return
        self.a.value = self.prior.a.value + \
            self.G[REPRNAMES.CLUSTER_INTERACTION_INDICATOR].num_pos_indicators
        self.b.value = self.prior.b.value + \
            self.G[REPRNAMES.CLUSTER_INTERACTION_INDICATOR].num_neg_indicators
        self.sample()
        return self.value


class ClusterInteractionIndicators(pl.variables.Variable):
    '''This is the posterior of the Indicator variables on the interactions
    between clusters. These clusters are not fixed.
    If `value` is not `None`, then we set that to be the initial indicators
    of the cluster interactions
    '''
    def __init__(self, prior, mp=None, relative=True, **kwargs):
        '''Parameters

        prior : pl.variables.Beta
            This is the prior of the variable
        mp : str, None
            If `None`, then there is no multiprocessing.
            If it is a str, then there are two options:
                'debug': pool is done sequentially and not sent to processors
                'full': pool is done at different processors
        relative : bool
            Whether you update using the relative marginal likelihood or not.
        '''
        if not pl.isbool(relative):
            raise TypeError('`relative` ({}) must be a bool'.format(type(relative)))
        if relative:
            if mp is not None:
                raise ValueError('Multiprocessing is slower for rel. Turn mp off')
            self.update = self.update_relative
        else:
            self.update = self.update_direct

        if mp is not None:
            if not pl.isstr(mp):
                raise TypeError('`mp` ({}) must be a str'.format(type(mp)))
            if mp not in ['full', 'debug']:
                raise ValueError('`mp` ({}) not recognized'.format(mp))

        kwargs['name'] = STRNAMES.CLUSTER_INTERACTION_INDICATOR
        pl.variables.Variable.__init__(self, dtype=bool, **kwargs)
        self.n_asvs = len(self.G.data.asvs)
        self.set_value_shape(shape=(self.n_asvs, self.n_asvs))
        self.add_prior(prior)
        self.clustering = self.G[STRNAMES.CLUSTERING_OBJ]
        self.mp = mp
        self.relative = relative

        # parameters used during update
        self.X = None
        self.y = None
        self.process_prec_matrix = None
        self._strr = 'None'

    def initialize(self, delay=0, run_every_n_iterations=1):
        '''Do nothing, the indicators are set in `ClusterInteractionValue`.

        Parameters
        ----------
        delay : int
            How many iterations to delay starting to update the values
        run_every_n_iterations : int
            Which iteration to run on
        '''
        if not pl.isint(delay):
            raise TypeError('`delay` ({}) must be an int'.format(type(delay)))
        if delay < 0:
            raise ValueError('`delay` ({}) must be >= 0'.format(delay))
        if not pl.isint(run_every_n_iterations):
            raise TypeError('`run_every_n_iterations` ({}) must be an int'.format(
                type(run_every_n_iterations)))
        if run_every_n_iterations <= 0:
            raise ValueError('`run_every_n_iterations` ({}) must be > 0'.format(
                run_every_n_iterations))

        self.delay = delay
        self.run_every_n_iterations = run_every_n_iterations
        self._there_are_perturbations = self.G.perturbations is not None
        self.update_cnt_indicators()
        self.interactions = self.G[REPRNAMES.INTERACTIONS_OBJ]
        self.n_asvs = len(self.G.data.asvs)

        # These are for the function `self._make_idx_for_clusters`
        self.ndts_bias = []
        self.n_replicates = self.G.data.n_replicates
        self.n_dts_for_replicate = self.G.data.n_dts_for_replicate
        self.total_dts = np.sum(self.n_dts_for_replicate)
        self.replicate_bias = np.zeros(self.n_replicates, dtype=int)
        for ridx in range(1, self.n_replicates):
            self.replicate_bias[ridx] = self.replicate_bias[ridx-1] + \
                self.n_asvs * self.n_dts_for_replicate[ridx - 1]
        for ridx in range(self.G.data.n_replicates):
            self.ndts_bias.append(
                np.arange(0, self.G.data.n_dts_for_replicate[ridx] * self.n_asvs, self.n_asvs))

        # Makes a dictionary that maps the asv index to the rows that it the ASV in
        self.oidx2rows = {}
        for oidx in range(self.n_asvs):
            idxs = np.zeros(self.total_dts, dtype=int)
            i = 0
            for ridx in range(self.n_replicates):
                temp = np.arange(0, self.n_dts_for_replicate[ridx] * self.n_asvs, self.n_asvs)
                temp = temp + oidx
                temp = temp + self.replicate_bias[ridx]
                l = len(temp)
                idxs[i:i+l] = temp
                i += l
            self.oidx2rows[oidx] = idxs

    def add_trace(self):
        self.value = self.G[REPRNAMES.INTERACTIONS_OBJ].get_datalevel_indicator_matrix()
        pl.variables.Variable.add_trace(self)

    def update_cnt_indicators(self):
        self.num_pos_indicators = self.G[REPRNAMES.INTERACTIONS_OBJ].num_pos_indicators()
        self.num_neg_indicators = self.G[REPRNAMES.INTERACTIONS_OBJ].num_neg_indicators()

    def __str__(self):
        return self._strr

    # @profile
    def update_direct(self):
        '''Permute the order that the indices that are updated.

        Build the full master interaction matrix that we can then slice
        '''
        start = time.time()
        if self.sample_iter < self.delay:
            # for interaction in self.interactions:
            #     interaction.indicator=False
            self._strr = '{}\ntotal time: {}'.format(
                self.interactions.get_indicators(), time.time()-start)
            return
        if self.sample_iter % self.run_every_n_iterations != 0:
            return

        # keys = npr.permutation(self.interactions.key_pairs())
        idxs = npr.permutation(self.interactions.size)
        for idx in idxs:
            # print('indicator {}/{}'.format(iii, len(keys)))
            self.update_single_idx_slow(idx=idx)

        self.update_cnt_indicators()
        # Since slicing is literally so slow, it is faster to build than just slicing M
        self.G.data.design_matrices[REPRNAMES.CLUSTER_INTERACTION_VALUE].M.build(
            build=True, build_for_neg_ind=False)
        iii = self.interactions.get_indicators()
        n_on = np.sum(iii)
        self._strr = '{}\ntotal time: {}, n_interactions: {}/{}, {:.2f}'.format(
            iii, time.time()-start, n_on, len(iii), n_on/len(iii))

    def update_single_idx_slow(self, idx):
        '''Update the likelihood for interaction `idx`

        Parameters
        ----------
        idx : int
            This is the index of the interaction we are updating
        '''
        prior_ll_on = np.log(self.G[REPRNAMES.INDICATOR_PROB].value)
        prior_ll_off = np.log(1 - self.G[REPRNAMES.INDICATOR_PROB].value)

        d_on = self.calculate_marginal_loglikelihood(idx=idx, val=True)
        d_off = self.calculate_marginal_loglikelihood(idx=idx, val=False)

        ll_on = d_on['ret'] + prior_ll_on
        ll_off = d_off['ret'] + prior_ll_off

        # print('slow\n\ttotal: {}\n\tbeta_logdet_diff: {}\n\t' \
        #     'priorvar_logdet_diff: {}\n\tbEb_diff: {}\n\t' \
        #     'bEbprior_diff: {}\n\tn_on_when_off: {}'.format(
        #         ll_on - ll_off,
        #         d_on['beta_logdet'] - d_off['beta_logdet'],
        #         d_on['priorvar_logdet'] - d_off['priorvar_logdet'],
        #         d_on['bEb'] - d_off['bEb'],
        #         d_on['bEbprior'] - d_off['bEbprior'],
        #         self.interactions.num_pos_indicators()))

        dd = [ll_off, ll_on]

        res = bool(sample_categorical_log(dd))
        self.interactions.iloc(idx).indicator = res
        self.update_cnt_indicators()

    # @profile
    def _make_idx_vector_for_clusters(self):
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
            rows_to_include = self.G.data.rows_to_include_zero_inflation
            for cid in d:
                arr = d[cid]
                new_arr = np.zeros(len(arr), dtype=int)
                n = 0
                for idx in arr:
                    if rows_to_include[idx]:
                        new_arr[n] = idx - prevoff_arr[idx]
                        n += 1

                new_arr = new_arr[:n]
                d[cid] = new_arr
        return d

    # @profile
    def make_rel_params(self):
        '''We make the parameters needed to update the relative log-likelihod.
        This function is called once at the beginning of the update.

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
            Maps the target cluster id to the matrix of the design matrix of the
            interactions. Only includes the rows that correspond to the ASVs in the
            target cluster. It includes every single column as if all of the indicators
            are on. We only index out the columns when we are doing the marginalization.
        prior_prec_interaction : float
            Prior precision of the interaction value. We then use this
            value to make the diagonal of the prior precision.
        prior_var_interaction : float
            Prior variance of the interaction value.
        prior_mean_interaction : float
            Prior mean of the interaction values. We use this value
            to make the prior mean vector during the marginalization.
        n_on_master : int
            How many interactions are on at any one time. We adjust this
            throughout the update depending on what interactions we turn off and
            on.
        prior_ll_on : float
            Prior log likelihood of a positive interaction
        prior_ll_off : float
            Prior log likelihood of the negative interaction
        priorvar_logdet_diff : float
            This is the prior variance log determinant that we add when the indicator
            is positive.

        Parameters created if there are perturbations
        ---------------------------------------------
        perturbationsXs : dict (int -> np.ndarray)
            Maps the target cluster id to the design matrix that corresponds to 
            the on perturbations of the target clusters. This is preindexed in
            both rows and columns
        prior_prec_perturbations : dict (int -> np.ndarray)
            Maps the target cluster id to the diagonal of the prior precision
            of the perturbations
        prior_var_perturbations : dict (int -> np.ndarray)
            Maps the target cluster id to the diagonal of the prior variance 
            of the perturbations
        prior_mean_perturbations : dict (int -> np.ndarray)
            Maps the target cluster id to the vector of the prior mean of the
            perturbations
        '''
        # Get the row indices for each cluster
        row_idxs = self._make_idx_vector_for_clusters()

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
        self.G.data.design_matrices[REPRNAMES.CLUSTER_INTERACTION_VALUE].M.build(
            build=True, build_for_neg_ind=True)
        XM_master = self.G.data.design_matrices[REPRNAMES.CLUSTER_INTERACTION_VALUE].toarray()
        for tcid in self.clustering.order:
            self.interactionXs[tcid] = XM_master[row_idxs[tcid], :]

        # Make prior parameters
        self.prior_var_interaction = self.G[REPRNAMES.PRIOR_VAR_INTERACTIONS].value
        self.prior_prec_interaction = 1/self.prior_var_interaction
        self.prior_mean_interaction = self.G[REPRNAMES.PRIOR_MEAN_INTERACTIONS].value
        self.prior_ll_on = np.log(self.prior.value)
        self.prior_ll_off = np.log(1 - self.prior.value)
        self.n_on_master = self.interactions.num_pos_indicators()

        # Make priorvar_logdet
        self.priorvar_logdet = np.log(self.prior_var_interaction)

        if self._there_are_perturbations:
            XMpert_master = self.G.data.design_matrices[REPRNAMES.PERT_VALUE].toarray()

            # Make perturbationsXs
            self.perturbationsXs = {}
            for tcid in self.clustering.order:
                rows = row_idxs[tcid]
                cols = []
                i = 0
                for perturbation in self.G.perturbations:
                    for cid in perturbation.indicator.value:
                        if perturbation.indicator.value[cid]:
                            if cid == tcid:
                                cols.append(i)
                            i += 1
                cols = np.asarray(cols, dtype=int)

                self.perturbationsXs[tcid] = pl.util.fast_index(M=XMpert_master,
                    rows=rows, cols=cols)

            # Make prior perturbation parameters
            self.prior_mean_perturbations = {}
            self.prior_var_perturbations = {}
            self.prior_prec_perturbations = {}
            for tcid in self.clustering.order:
                mean = []
                var = []
                for perturbation in self.G.perturbations:
                    if perturbation.indicator.value[tcid]:
                        # This is on, get the parameters
                        mean.append(perturbation.magnitude.prior.mean.value)
                        var.append(perturbation.magnitude.prior.var.value)
                self.prior_mean_perturbations[tcid] = np.asarray(mean)
                self.prior_var_perturbations[tcid] = np.asarray(var)
                self.prior_prec_perturbations[tcid] = 1/self.prior_var_perturbations[tcid]

            # Make priorvar_det_perturbations
            self.priorvar_det_perturbations = 0
            for perturbation in self.G.perturbations:
                self.priorvar_det_perturbations += \
                    perturbation.indicator.num_on_clusters() * \
                    perturbation.magnitude.prior.var.value

    # @profile
    def update_relative(self):
        '''Update the indicators variables by calculating the relative loglikelihoods
        of it being on as supposed to off. Because this is a relative loglikelihood,
        we only need to take into account the following parameters of the model:
            - Only the ASVs in the target cluster of the interaction
            - Only the positively indicated interactions going into the
              target cluster.

        This is 1000's of times faster than `update` because we are operating on matrices
        that are MUCH smaller than in a full system. These matrices are also considered dense
        so we do all of our computations without sparse matrices.

        We permute the order that the indices are updated for more robust mixing.
        '''
        start = time.time()
        if self.sample_iter < self.delay:
            self._strr = '{}\ntotal time: {}'.format(
                self.interactions.get_indicators(), time.time()-start)
            return
        if self.sample_iter % self.run_every_n_iterations != 0:
            return

        idxs = npr.permutation(self.interactions.size)

        self.make_rel_params()
        for idx in idxs:
            self.update_single_idx_fast(idx=idx)

        self.update_cnt_indicators()
        # Since slicing is literally so slow, it is faster to build than just slicing M
        self.G.data.design_matrices[REPRNAMES.CLUSTER_INTERACTION_VALUE].M.build(
            build=True, build_for_neg_ind=False)
        iii = self.interactions.get_indicators()
        n_on = np.sum(iii)
        self._strr = '{}\ntotal time: {}, n_interactions: {}/{}, {:.2f}'.format(
            iii, time.time()-start, n_on, len(iii), n_on/len(iii))

    def calculate_marginal_loglikelihood(self, idx, val):
        '''Calculate the likelihood of interaction `idx` with the value `val`
        '''
        # Build and initialize
        self.interactions.iloc(idx).indicator = val
        self.update_cnt_indicators()
        self.G.data.design_matrices[REPRNAMES.CLUSTER_INTERACTION_VALUE].M.build()

        lhs = [REPRNAMES.GROWTH_VALUE, REPRNAMES.SELF_INTERACTION_VALUE]
        if self._there_are_perturbations:
            rhs = [REPRNAMES.PERT_VALUE, REPRNAMES.CLUSTER_INTERACTION_VALUE]
        else:
            rhs = [REPRNAMES.CLUSTER_INTERACTION_VALUE]

        y = self.G.data.construct_lhs(lhs, 
            kwargs_dict={REPRNAMES.GROWTH_VALUE:{'with_perturbations': False}})
        X = self.G.data.construct_rhs(rhs, toarray=True)

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

    def update_single_idx_fast(self, idx):
        '''Calculate the relative log likelihood of changing the indicator of the
        interaction at index `idx`.

        This is about 20X faster than `update_single_idx_slow`.

        Parameters
        ----------
        idx : int
            This is the index of the interaction index we are sampling
        '''
        # Get the current interaction by the index
        self.curr_interaction = self.interactions.iloc(idx)
        start_sign = self.curr_interaction.indicator
        
        tcid = self.curr_interaction.target_cid
        self.curr_interaction.indicator = False
        self.col_idxs = np.asarray(
            self.interactions.get_arg_indicators(target_cid=tcid),
            dtype=int)

        if not start_sign:
            self.n_on_master += 1
            self.num_pos_indicators += 1
            self.num_neg_indicators -= 1

        d_on = self.calculate_relative_marginal_loglikelihood(idx=idx, val=True)

        self.n_on_master -= 1
        self.num_pos_indicators -= 1
        self.num_neg_indicators += 1
        d_off = self.calculate_relative_marginal_loglikelihood(idx=idx, val=False)

        ll_on = d_on + self.prior_ll_on
        ll_off = d_off + self.prior_ll_off

        dd = [ll_off, ll_on]

        # print('\nindicator', idx)
        # print('fast\n\ttotal: {}\n\tbeta_logdet_diff: {}\n\t' \
        #     'priorvar_logdet_diff: {}\n\tbEb_diff: {}\n\t' \
        #     'bEbprior_diff: {}\n\tn_on_when_off: {}'.format(
        #         ll_on - ll_off,
        #         d_on['beta_logdet'] - d_off['beta_logdet'],
        #         d_on['priorvar_logdet'] - d_off['priorvar_logdet'],
        #         d_on['bEb'] - d_off['bEb'],
        #         d_on['bEbprior'] - d_off['bEbprior'],
        #         self.n_on_master))
        # print('log(prior_var)', np.log(self.prior_var_interaction))
        # self.update_single_idx_slow(idx)

        res = bool(sample_categorical_log(dd))
        if res:
            self.n_on_master += 1
            self.num_pos_indicators += 1
            self.num_neg_indicators -= 1
            self.curr_interaction.indicator = True

    def calculate_relative_marginal_loglikelihood(self, idx, val):
        '''Calculate the relative marginal log likelihood for the interaction index
        `idx` with the indicator `val`

        Parameters
        ----------
        idx : int
            This is the index of the interaction
        val : bool
            This is the value to calculate it as
        '''
        tcid = self.curr_interaction.target_cid

        y = self.ys[tcid]
        process_prec = self.process_precs[tcid]

        # Make X, prior mean, and prior_var
        if val:
            cols = np.append(self.col_idxs, idx)
        else:
            cols = self.col_idxs
        
        X = self.interactionXs[tcid][:, cols]
        prior_mean = np.full(len(cols), self.prior_mean_interaction)
        prior_prec_diag = np.full(len(cols), self.prior_prec_interaction)

        if self._there_are_perturbations:
            Xpert = self.perturbationsXs[tcid]
            X = np.hstack((X, Xpert))

            prior_mean = np.append(
                prior_mean,
                self.prior_mean_perturbations[tcid])
            
            prior_prec_diag = np.append(
                prior_prec_diag,
                self.prior_prec_perturbations[tcid])

        if X.shape[1] == 0:
            # return {
            #     'ret': 0,
            #     'beta_logdet': 0,
            #     'priorvar_logdet': 0,
            #     'bEb': 0,
            #     'bEbprior': 0}
            return 0

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

            logging.critical('here')
            print('y')
            print(y.shape)
            print(y)
            print('process_prec')
            print(process_prec.shape)
            print('X')
            print(X.shape)
            print(X)
            print('priors')
            print(prior_mean)
            print(prior_prec_diag)
            print('self-interactions')
            X = pl.toarray(self.G.data.design_matrices[REPRNAMES.SELF_INTERACTION_VALUE].matrix)
            print(X.shape)
            print(np.any(np.isnan(X)))
            print('growth')
            X = pl.toarray(self.G.data.design_matrices[REPRNAMES.GROWTH_VALUE].matrix_without_perturbations)
            print(X.shape)
            print(np.any(np.isnan(X)))
            print('orig y')
            y = self.G.data.lhs.vector
            print(y.shape)
            print(np.any(np.isnan(y)))
            print('cluster-interactions')
            X = pl.toarray(self.G.data.design_matrices[REPRNAMES.CLUSTER_INTERACTION_VALUE].matrix)
            print(X.shape)
            print(np.any(np.isnan(X)))

            n_on = 0
            for row in range(X.shape[0]):
                n_on += np.any(np.isnan(X[row]))

            print('nans on {}/{} rows'.format(n_on, X.shape[0]))
                    

            raise
        
        if val:
            bEbprior = (self.prior_mean_interaction**2)/self.prior_var_interaction
            priorvar_logdet = self.priorvar_logdet
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
            
    def kill(self):
        pass
