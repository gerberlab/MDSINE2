'''Make the Dynamics
'''
import numpy as np
import time
import logging

from typing import Union, Dict, Iterator, Tuple, List, Any

from . import pylab as pl
from .pylab import BaseMCMC, Subject
from .names import STRNAMES

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
    growth : np.ndarray((n,)), None
        This is the growth of the dynamics for each taxon
    interactions : np.ndarray((n,n)), None
        Interactions is assumed are the Taxon-Taxon interactions (shape=(n,n)). The diagonal of
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
    def __init__(self, growth: Union[np.ndarray, type(None)], interactions: Union[np.ndarray, type(None)], 
        perturbations: Iterator[np.ndarray]=None, perturbation_starts: Iterator[float]=None,
        perturbation_ends: Iterator[float]=None, sim_max: float=None, start_day: float=0):

        self.growth = growth
        self.interactions = interactions
        self.perturbations = perturbations
        self.perturbation_starts = perturbation_starts
        self.perturbation_ends = perturbation_ends
        self.sim_max = sim_max
        self.start_day = start_day

        self._pert_intervals = None
        self._adjusted_growth = None

    def stability(self) -> np.ndarray:
        '''This is the analytical solution for the stability
        '''
        return - np.linalg.pinv(self.interactions) @ (self.growth.reshape(-1,1))

    def init_integration(self, dt: float):
        '''This is called internally from mdsine2.integrate

        Pre-multiply the growth and interactions with dt. Improves
        efficiency by 25%.

        Parameters
        ----------
        dt : numeric
            This is the amount of time from the previous time point we
            are integrating from
        '''
        self.growth = self.growth.reshape(-1,1)
        self._dtgrowth = self.growth * dt
        if self.perturbations is not None:
            self._adjust_growth = []
            for pert in self.perturbations:
                pert = pert.reshape(-1,1)
                self._adjust_growth.append(self._dtgrowth * (1 + pert))
        if self.sim_max is not None:
            self.record = {}
        else:
            self.record = None

        self._dtinteractions = self.interactions * dt

        # Initialize pert_intervals
        # float -> int
        # timepoint -> perturbation index
        if self.perturbations is not None:
            self._pert_intervals = {}
            for pidx in range(len(self.perturbation_ends)):
                start = self.perturbation_starts[pidx]
                end = self.perturbation_ends[pidx]
                rang = np.arange(start, end, step=dt)

                for t in rang:
                    self._pert_intervals[round(t, ndigits=2)] = pidx

    def integrate_single_timestep(self, x: np.ndarray, t: float, dt: float) -> np.ndarray:
        '''Integrate over a single step

        Parameters
        ----------
        x : np.ndarray((n,1))
            This is the abundance as a column vector for each taxon
        t : numeric
            This is the time point we are integrating to
        dt : numeric
            This is the amount of time from the previous time point we
            are integrating from
        '''
        growth = self._dtgrowth
        t = round(t, ndigits=2)

        if self.perturbations is not None:
            if t in self._pert_intervals:
                growth = self._adjust_growth[self._pert_intervals[t]]

        # Integrate
        logret = np.log(x) + growth + self._dtinteractions @ x
        ret = np.exp(logret).ravel()
        if self.record is not None:
            oidxs = np.where(ret >= self.sim_max)[0]
            if len(oidxs) > 0:
                for oidx in oidxs:
                    if oidx not in self.record:
                        self.record[oidx] = []
                    self.record[oidx].append(ret[oidx])
            ret[ret >= self.sim_max] = self.sim_max
        return ret

    def finish_integration(self):
        '''This is the function that
        '''
        self._pert_intervals = None
        self._adjusted_growth = None
        self._dtgrowth = None
        self._dtinteractions = None

    @staticmethod
    def forward_sim_from_chain(mcmc: BaseMCMC, initial_conditions: np.ndarray, times: np.ndarray, simulation_dt: float,
        subj: Subject=None, sim_max: float=None, section: str='posterior') -> np.ndarray:
        '''Forward simulate the dynamics from a chain. This assumes that the
        initial conditions occur at time `times[0]`

        Parameters
        ----------
        mcmc : md2.BaseMCMC
            MCMC chain with all of the traces of the parameters
        subj : md2.Subject
            This is the subject we are forward simulating for. We need this
            to get the start and end times for each perturbation. If this is None,
            then we assume there are no perturbations
        initial_conditions : np.ndarray(n_taxa)
            Initial conditions for each taxon
        times : np.ndarray
            These are the times to forward simulate for
        simulation_dt : float
            This is the step size to forward simulate with
        sim_max : float
            This is the maximum value for inference that we clip at
        section : str
            This is the part of the trace that we are forward simulating from

        Returns
        -------
        np.ndarray(n_gibbs, n_taxa, len(times))
            These are the forward imsualtions for each gibb step
        '''
        # Get the parameters from the chain
        growths = mcmc.graph[STRNAMES.GROWTH_VALUE].get_trace_from_disk(section=section)
        self_interactions = mcmc.graph[STRNAMES.SELF_INTERACTION_VALUE].get_trace_from_disk(section=section)
        interactions = mcmc.graph[STRNAMES.INTERACTIONS_OBJ].get_trace_from_disk(section=section)

        si = -np.absolute(self_interactions)
        for i in range(len(mcmc.graph.data.taxa)):
            interactions[:, i, i] = si[:, i]
        interactions[np.isnan(interactions)] = 0

        if mcmc.graph.perturbations is not None and subj is not None:
            perturbations = []
            for pert in mcmc.graph.perturbations:
                perturbations.append(pert.get_trace_from_disk(section=section))
                perturbations[-1][np.isnan(perturbations[-1])] = 0
            
            perturbation_starts = []
            perturbation_ends = []
            for pert in mcmc.graph.perturbations:
                if pert.name in subj.perturbations:
                    perturbation_starts.append(subj.perturbations[pert.name].starts[subj.name])
                    perturbation_ends.append(subj.perturbations[pert.name].ends[subj.name])  

        else:
            perturbation_starts = None
            perturbation_ends = None
            perturbations = None

        # Forward simulate for every gibb step
        pred_matrix = np.zeros(shape=(growths.shape[0], growths.shape[1], len(times)))

        dyn = gLVDynamicsSingleClustering(growth=None, interactions=None, sim_max=sim_max,
            start_day=times[0], perturbation_ends=perturbation_ends, 
            perturbation_starts=perturbation_starts)
        start_time = time.time()
        initial_conditions = initial_conditions.reshape(-1,1)
        for gibb in range(growths.shape[0]):
            if gibb % 5 == 0 and gibb > 0:
                logging.info('{}/{} - {}'.format(gibb,growths.shape[0],
                    time.time()-start_time))
                start_time = time.time()
            dyn.growth = growths[gibb]
            dyn.interactions = interactions[gibb]
            if perturbations is not None:
                dyn.perturbations = [pert[gibb] for pert in perturbations]
            
            X = pl.dynamics.integrate(dynamics=dyn, initial_conditions=initial_conditions, 
                dt=simulation_dt, n_days=times[-1]+simulation_dt, processvar=None,
                subsample=True, times=times, log_every=10000)
            pred_matrix[gibb] = X['X']
        return pred_matrix

        

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

    def integrate_single_timestep(self, x: np.ndarray, t: float, dt: float) -> np.ndarray:
        std = np.sqrt(dt * self.value)
        return np.exp(pl.random.normal.sample(loc=np.log(x), scale=std))

    def finish_integration(self):
        pass

