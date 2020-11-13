'''Process variance parameters for the posterior
'''
import logging
import time
import numpy as np
import scipy.sparse
import scipy

from .util import build_prior_covariance, build_prior_mean
from .names import STRNAMES, REPRNAMES
from . import pylab as pl

class ProcessVarGlobal(pl.variables.SICS):
    '''Learn a Process variance where we learn th same process variance
    for each ASV. This assumes that the model we're using uses the logscale
    of the data.
    '''
    def __init__(self, prior, **kwargs):
        '''
        Parameters
        ----------
        prior : pl.variables.SICS
            This is the prior of the distribution
        kwargs : dict
            These are the extra parameters for the Variable class
        '''
        kwargs['name'] = STRNAMES.PROCESSVAR
        pl.variables.SICS.__init__(self, dtype=float, **kwargs)
        self.add_prior(prior)
        self.global_variance = True
        self._strr = 'NA'

    def __str__(self):
        return self._strr

    def initialize(self, dof_option, scale_option, value_option, 
        dof=None, scale=None, value=None, variance_scaling=1,
        delay=0):
        '''Initialize the value and hyperparameter.

        Parameters
        ----------
        dof_option : str
            How to initialize the `dof` parameter. Options:
                'manual'
                    Manually specify the dof with the parameter `dof`
                'half'
                    Set the degrees of freedom to the number of data points
                'auto', 'diffuse'
                    Set the degrees of freedom to a sparse number (2.5)
        scale_option : str
            How to initialize the scale of the parameter. Options:
                'manual'
                    Need to also specify `scale` parameter
                'med', 'auto'
                    Set the scale such that mean of the distribution
                    has medium noise (20%)
                'low'
                    Set the scale such that mean of the distribution
                    has low noise (10%)
                'high'
                    Set the scale such that mean of the distribution
                    has high noise (30%)
        value_option : str
            How to initialize the value
                'manual'
                    Set the value with the `value` parameter
                'prior-mean', 'auto'
                    Set the value to the mean of the prior
        variance_scaling : float, None
            How much to inflate the variance
        '''
        if not pl.isint(delay):
            raise TypeError('`delay` ({}) must be an int'.format(type(delay)))
        if delay < 0:
            raise ValueError('`delay` ({}) must be >= 0'.format(delay))
        self.delay = delay

        # Set the dof
        if not pl.isstr(dof_option):
            raise TypeError('`dof_option` ({}) must be a str'.format(type(dof_option)))
        if dof_option == 'manual':
            if not pl.isnumeric(dof):
                raise TypeError('`dof` ({}) must be a numeric'.format(type(dof)))
            if dof <= 0:
                raise ValueError('`dof` must be > 0'.format(dof))
            if dof <= 2:
                logging.critical('Process Variance dof ({}) is set unproper'.format(dof))
        elif dof_option == 'half':
            dof = len(self.G.data.lhs)
        elif dof_option in ['auto', 'diffuse']:
            dof = 2.5
        else:
            raise ValueError('`dof_option` ({}) not recognized'.format(dof_option))
        self.prior.dof.override_value(dof)

        # Set the scale
        if not pl.isstr(scale_option):
            raise TypeError('`scale_option` ({}) must be a str'.format(type(scale_option)))
        if scale_option == 'manual':
            if not pl.isnumeric(scale):
                raise TypeError('`scale` ({}) must be a numeric'.format(type(scale)))
            if scale <= 0:
                raise ValueError('`scale` ({}) must be > 0'.format(scale))
        elif scale_option in ['auto', 'med']:
            scale = (0.2 ** 2) * (self.prior.dof.value - 2) / self.prior.dof.value
        elif scale_option ==  'low':
            scale = (0.1 ** 2) * (self.prior.dof.value - 2) / self.prior.dof.value
        elif scale_option ==  'high':
            scale = (0.3 ** 2) * (self.prior.dof.value - 2) / self.prior.dof.value
        else:
            raise ValueError('`scale_option` ({}) not recognized'.format(scale_option))
        self.prior.scale.override_value(scale)

        # Set the value
        if not pl.isstr(value_option):
            raise TypeError('`value_option` ({}) must be a str'.format(type(value_option)))
        if value_option == 'manual':
            if not pl.isnumeric(dof):
                raise TypeError('`value` ({}) must be a numeric'.format(type(value)))
            if value <= 0:
                raise ValueError('`value` ({}) must be > 0'.format(value))
        elif value_option in ['prior-mean', 'auto']:
            value = self.prior.mean()
        else:
            raise ValueError('`value_option` ({}) not recognized'.format(value_option))
        self.value = value
        
        self.rebuild_diag()
        self._there_are_perturbations = self.G.perturbations is not None

    def build_matrix(self, cov, sparse=True):
        '''Builds the process variance as a covariance or precision
        matrix.

        Parameters
        ----------
        cov : bool
            If True, make the covariance matrix
        sparse : bool
            If True, return the matrix as a sparse matrix

        Returns
        -------
        np.ndarray or scipy.sparse
            Either sparse or dense covariance/precision matrix
        '''
        a = self.diag
        if not cov:
            a = 1/a
        if sparse:
            return scipy.sparse.dia_matrix((a,[0]), shape=(len(a),len(a))).tocsc()
        else:
            return np.diag(a)

    def rebuild_diag(self):
        '''Builds up the process variance diagonal that we use to make the matrix
        '''
        a = self.value / self.G.data.dt_vec
        if self.G.data.zero_inflation_transition_policy is not None:
            a = a[self.G.data.rows_to_include_zero_inflation]
        
        self.diag = a
        self.prec = 1/a

    def update(self):
        '''Update the process variance

        y = (log(x_{k+1}) - log(x_k))/dt
        Xb = a_1 (1 + \gamma) + A x

        % These are our dynamics with the process variance
        y ~ Normal(Xb , \sigma^2_w / dt)

        % Subtract the mean
        y - Xb ~ Normal( 0, \sigma^2_w / dt)
        
        % Substitute
        z = y - Xb 

        z ~ Normal (0, \sigma^2_w / dt)
        z * \sqrt{dt} ~ Normal (0, \sigma^2_w)

        This is now in a form we can use to calculate the posterior
        '''
        if self._there_are_perturbations:
            lhs = [
                REPRNAMES.GROWTH_VALUE, 
                REPRNAMES.SELF_INTERACTION_VALUE,
                REPRNAMES.CLUSTER_INTERACTION_VALUE]
        else:
            lhs = [
                REPRNAMES.GROWTH_VALUE, 
                REPRNAMES.SELF_INTERACTION_VALUE,
                REPRNAMES.CLUSTER_INTERACTION_VALUE]
        
        # This is the residual z = y - Xb
        z = self.G.data.construct_lhs(lhs, 
            kwargs_dict={REPRNAMES.GROWTH_VALUE:{
                'with_perturbations': self._there_are_perturbations}})
        z = np.asarray(z).ravel()
        if self.G.data.zero_inflation_transition_policy is not None:
            z = z * self.G.data.sqrt_dt_vec[self.G.data.rows_to_include_zero_inflation]
        else:
            z = z * self.G.data.sqrt_dt_vec
        residual = np.sum(np.square(z))

        self.dof.value = self.prior.dof.value + len(z)
        self.scale.value = ((self.prior.scale.value * self.prior.dof.value) + \
           residual)/self.dof.value
        
        # shape = 2 + (len(residual)/2)
        # scale = 0.00001 + np.sum(np.square(residual))/2
        # self.value = pl.random.invgamma.sample(shape=shape, scale=scale)
        self.sample()
        self.rebuild_diag()

        self._strr = '{}, empirical_variance: {:.5f}'.format(self.value, 
            residual/len(z))

