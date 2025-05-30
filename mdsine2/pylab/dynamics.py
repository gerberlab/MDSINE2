'''This module is for specifying dynamics. Additionally, we can integrate the 
dynamics with the function `integrate`.

Changes to Make in pylab 1.0
    - Fix Negative Binomial
    - Type hinting - I think this should be super important
        * mypy?
    - Make Metropolis better
        * look at implementation in posterior
    - Distinguish between TypeError and ValueError in initializations
    - Redo how you make the SubjsetSet - idk how but it's messy
'''
import numpy as np
from mdsine2.logger import logger

# Typing
from typing import Any, Dict


from . import util as plu
from .base import *

_ADDITIVE = 0
_MULTIPLICATIVE = 1


def isdynamics(x: Any) -> bool:
    '''Checks if `a` is a (subclass of) BaseDynamics

    Parameters
    ----------
    x : any
        Instance we are checking

    Returns
    -------
    bool
        True if `x` is a subclass of a BaseDynamics
    '''
    return x is not None and issubclass(x.__class__, BaseDynamics)

def isprocessvariance(x: Any) -> bool:
    '''Checks if `a` is a (subclass of) ProcessVariance

    Parameters
    ----------
    x : any
        Instance we are checking

    Returns
    -------
    bool
        True if `x` is a subclass of a BaseProcessVariance
    '''
    return x is not None and issubclass(x.__class__, BaseProcessVariance)

def isintegratable(x: Any) -> bool:
    '''Checks if `a` is a (subclass of) Integratable

    Parameters
    ----------
    x : any
        Instance we are checking

    Returns
    -------
    bool
        True if `x` is a subclass of a Integratable
    '''
    return x is not None and issubclass(x.__class__, Integratable)

class Integratable:
    '''These are the functions needed for integrating
    '''  
    def init_integration(self):
        '''This is the function that `integrate` calls to start the integration
        '''
        pass

    def integrate_single_timestep(self, x: np.ndarray, t: float, dt: float):
        '''This is the function that `integrate` calls to propagate the underlying
        dynamics a single time step. There is no type checking for computational
        efficiency

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
        raise NotImplementedError('User needs to implement this function')

    def finish_integration(self):
        '''This is the function that
        '''
        pass


class BaseDynamics(Integratable):
    '''This is the base class for the dynamics
    '''
    def __init__(self, **kwargs):
        Integratable.__init__(self, **kwargs)

    def stability(self):
        raise NotImplementedError('User needs to implement this function')

    def integrate(self, *args, **kwargs):
        return integrate(dynamics=self, *args, **kwargs)


class BaseProcessVariance(Integratable):
    '''
    '''
    def __init__(self, *args, **kwargs):
        Integratable.__init__(self, *args, **kwargs)


class _NoProcessVariance(Integratable):
    '''This is when you do not want process variance during integration.
    For inner use only.
    '''
    def __init__(self, *args, **kwargs):
        Integratable.__init__(self, *args, **kwargs)

    def integrate_single_timestep(self, x: np.ndarray, *args, **kwargs):
        '''Do nothing
        '''
        return x


def integrate(dynamics: BaseDynamics, initial_conditions: np.ndarray, dt: float, final_day: float,
    processvar: BaseProcessVariance=None, subsample: bool=False, times: np.ndarray=None) -> Dict[str, np.ndarray]:
    '''Numerically integrates the ODE given the dynamics and the initial 
    conditions. If the process variance is not None, then this integrates
    a stochastic ODE.

    Subsampling
    -----------
    If the dynamics are complex and the process variance is high, numerical
    integration with large time steps can lead to numerical instability, so we 
    integrate at smaller time steps than what our data is. This smaller time
    scale in this function is specified by `dt`.

    We can then subsample our densely integrated trajectories with the flag
    `subsample`. If `subsample=False`, we return the whole trajectory. If
    `subsample` is True, then we must also specify `times`.

    Parameters
    ----------
    dynamics : BaseDynamics
        These are the dynamics that we want to integrate.
    processvar : BaseProcessVariance, None
        This is the process variance we want to inject into the integration.
        If None, we assume there is no process variance and it is a normal ODE
    initial_conditions : np.ndarray((n_taxa,1), dtype=float)
        These are the initial conditions to integrate from. This must be a 
        column array.
    dt : float
        Time between each time step (in days) during integration
    final_day : float
        the last timepoint to simulate.
    subsample : bool
        If True, we subsample the integration at the time points indicated in
        `times`. If False we do not subsample
    times : int, np.ndarray((t,), dtype=numeric), None
        `times` must be a list of floats/ints, where each element 
        corresponds to a time of day to take the sample at. The last time 
        point must not exceed `final_day`, each time must be >= 0, and there
        must not be duplicates.
        Example: (assuming `subsample` is True)
            final_day = 6
            times = [0, 0.1, 1.1, 3, 5] - This is valid
            times = [0.1, 0, 1.1, 3, 5] - This is valid:
                (0 is automatically reordered)
            times = [0, 0.1, 1.1, 3, 6] - This is invalid: 
                (6 is not inclusive)
            times = [-0.1, 0, 1.1, 3, 5] - This is invalid:
                (-0.1 < 0)
            times = [0, 0, 1.1, 3, 5] - This is valid:
                (duplicate points for `0`, but we discard 1)

        If `subsample` is True and `times` is None, we automatically 
        return 1 timepoint per day.
    log_every : int, None
        - This is how oftent o log the progress of the integration. If None, it 
          will never log

    Returns
    -------
    dict
        'X': np.ndarray((n_taxa, k), dtype=float)
            The abundances for each taxon (row) for each time (column)
        'times': np.ndarray((k, ), dtype=float)
            These are the times, in days, for each column
    '''
    # Type and format checking
    if not isdynamics(dynamics):
        raise TypeError('`dynamics` ({}) must be a (subclass of) ' \
            'BaseDynamics'.format(type(dynamics)))

    if processvar is None:
        processvar = _NoProcessVariance()
    elif not isprocessvariance(processvar):
        raise TypeError('`processvar` ({}) must be a (subclass of) ' \
            'BaseProcessVariance'.format(type(processvar)))

    if not plu.isarray(initial_conditions):
        raise TypeError('`initial_conditions` ({}) must be an array'.format( 
            type(initial_conditions)))
    initial_conditions = np.asarray(initial_conditions, dtype=float)
    if initial_conditions.ndim != 2:
        raise ValueError('`initial_conditions` ({}) must be a column vector'.format( 
            initial_conditions.shape))
    if initial_conditions.shape[1] != 1:
        raise ValueError('`initial_conditions` ({}) must be a column vector'.format( 
            initial_conditions.shape))

    if not plu.isnumeric(dt):
        raise TypeError('`dt` ({}) must be a numeric'.format(type(dt)))
    if dt <= 0:
        raise ValueError('`dt` ({}) must be strictly greater than 0'.format(dt))

    if not plu.isbool(subsample):
        raise TypeError('`subsample` ({}) must be a bool'.format(type(subsample)))
    
    if not plu.isnumeric(final_day):
        raise TypeError('`final_day` ({}) must be an int'.format(type(final_day)))
    if final_day <= 0:
        raise ValueError('`final_day` ({}) must be > 0'.format(final_day))

    if subsample:
        if times is None:
            times = np.arange(int(final_day), dtype=float)
        elif not plu.isarray(times):
            raise TypeError('If `subsample` is True, then `times` ({}) must either ' \
                'be an array or None'.format(type(times)))
        times = np.asarray(times, dtype=float).ravel()
        times = np.sort(np.unique(times))
        if np.any(times < 0):
            raise ValueError('All `times` ({}) must be > 0 '.format(times))
        if np.any(times > final_day):
            raise ValueError('All `times` ({}) must be < `final_day` ({})'.format(
                times, final_day))

    # Everything is good - initialize then start integrating
    dynamics.init_integration(dt=dt)
    processvar.init_integration()

    sim_times = np.arange(start=dynamics.start_day, stop=final_day+dt, step=dt)
    X = np.zeros(shape=(initial_conditions.shape[0], len(sim_times)), dtype=float)
    X[:, 0] = initial_conditions.ravel()

    for t_idx, t in enumerate(sim_times):
        if t_idx == len(sim_times) - 1:
            continue  # nothing to simulate for last timepoint.
        cur = X[:, [t_idx]]
        a = dynamics.integrate_single_timestep(x=cur, t=t, dt=dt)
        X[:, t_idx+1] = processvar.integrate_single_timestep(x=a, t=t, dt=dt)
    
    dynamics.finish_integration()
    processvar.finish_integration()

    if subsample:
        numerical_tol = 1e-4 * dt
        locs = np.bitwise_or.reduce(
            np.stack([
                np.abs(sim_times - t) < numerical_tol
                for t in times
            ], axis=0),
            0
        )
        if np.sum(locs) != len(times):
            raise ValueError("Simulation times did not contain all requested timepoints, presumably due to "
                             "floating-point precision weirdness. This is an unexpected bug.")
        return {'X': X[:, locs], 'times': times}
    else:
        return {'X': X, 'times': sim_times}
