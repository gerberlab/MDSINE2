'''Make the Dynamics
'''
import numpy as np
import time

from . import pylab as pl

class gLVDynamicsSingleClustering(pl.dynamics.BaseDynamics):
    '''Discretized Generalized Lotka-Voltera Dynamics with 
    clustered interactions and perturbations. This class provides functionality
    to forward simulate the dynamics:

    log(x_{k+1}) = log(x_k) + \Delta_k * ( 
            a1*(1 + \gamma) + 
            x_k * a2 + 
            sum_{c_i != c_j} b_{c_i, c_j} x_{k,j})

    If you want to forward simulate, then pass this dynamics object into 
    `mdsine2.integrate`.

    Parameters
    ----------
    growth : np.ndarray((n,))
        This is the growth of the dynamics for each Taxa
    interactions : np.ndarray((n,n)) 
        Interactions is assumed are the Taxa-Taxa interactions (shape=(n,n)). The diagonal of
        the interactions is assumed to be the self interactions.
    perturbations : list(np.ndarray((n,))), None
        These are the perturbation magnitudes for each perturbation.  If `None` then we assume 
        there are no perturbations. If this is specified then `perturbation_starts` and 
        `perturbation_ends` must also be specified
    perturbation_starts : list(float), None
        These are the start time points of the perturbations.
    perturbation_ends : list(float), None
        These are the end time points of the perturbations.

    See also
    --------
    pylab.dynamics.dynamics
    '''

    def __init__(self, growth, interactions, perturbations=None, perturbation_starts=None,
        perturbation_ends=None, sim_max=None, start_day=0):

        self.growth = growth
        self.interactions = interactions
        self.perturbations = perturbations
        self.perturbation_starts = perturbation_starts
        self.perturbation_ends = perturbation_ends
        self.sim_max = sim_max
        self.start_day = start_day

        self._pert_intervals = None
        self._adjusted_growth = None

    def stability(self):
        '''This is the analytical solution for the stability
        '''
        return - np.linalg.pinv(self.interactions) @ (self.growth.reshape(-1,1))

    def init_integration(self):
        '''This is called internally from mdsine2.integrate
        '''
        self.growth = self.growth.reshape(-1,1)
        if self.perturbations is not None:
            self._adjust_growth = []
            for pert in self.perturbations:
                pert = pert.reshape(-1,1)
                self._adjust_growth.append(self.growth * (1 + pert))
            
    def integrate_single_timestep(self, x, t, dt):
        '''Integrate over a single step

        Parameters
        ----------
        x : np.ndarray((n,1))
            This is the abundance as a column vector for each Taxa
        t : numeric
            This is the time point we are integrating to
        dt : numeric
            This is the amount of time from the previous time point we
            are integrating from
        '''
        growth = self.growth

        if self.perturbations is not None:
            # Initialize pert_intervals
            if self._pert_intervals is None:
                # float -> int
                # timepoint -> perturbation index

                self._pert_intervals = {}
                for pidx in range(len(self.perturbation_ends)):
                    start = self.perturbation_starts[pidx]
                    end = self.perturbation_ends[pidx]
                    rang = np.arange(start, end, step=dt)

                    for t in rang:
                        self._pert_intervals[t] = pidx
            
            if t-dt in self._pert_intervals:
                growth = self._adjust_growth[self._pert_intervals[t]]

        # Integrate
        logret = np.log(x) + (growth + self.interactions @ x) * dt
        ret = np.exp(logret).ravel()
        ret[ret >= self.sim_max] = self.sim_max
        return ret

    def finish_integration(self):
        '''This is the function that
        '''
        self._pert_intervals = None
        self._adjusted_growth = None
        

class MultiplicativeGlobal(pl.dynamics.BaseProcessVariance):
    '''This is multiplicative noise used in a lognormal model
    '''
    def __init__(self, value):
        self.value = value

    def init_integration(self):
        if self.value is None:
            raise ValueError('`value` needs to be initialized (it is None).')
        if not pl.isnumeric(self.value):
            raise TypeError('`value` ({}) must be a numeric'.format(type(self.value)))
        if self.value <= 0:
            raise ValueError('`value` ({}) must be > 0'.format(self.value))

    def integrate_single_timestep(self, x, t, dt):
        std = np.sqrt(dt * self.value)
        return np.exp(pl.random.normal.sample(mean=np.log(x), std=std))

    def finish_integration(self):
        pass

