'''Metropolis-Hasting Classes

These track the acceptance rate and sets backend pointers

These are only for scalars.
'''

import numpy.random as npr
import numpy as np
import sys
from mdsine2.logger import logger

# Typing
from typing import TypeVar, Generic, Any, Union, Dict, Iterator, Tuple

from . import util
from . import variables
from .errors import MathError

# Constants
DEFAULT_TARGET_ACCEPTANCE_RATE = 'auto'
DEFAULT_END_TUNING = 'auto'

def isMetropKernel(a: Any) -> bool:
    '''Checks if `a` is a MetropolisKernel

    Parameters
    ----------
    a : any
        Instance we are checking

    Returns
    -------
    bool
        True if `a` is a MetropolisKernel
    '''
    return a is not None and issubclass(a.__class__, _BaseKernel)

def _sum_look_back(trace: np.ndarray) -> int:
    '''Inner function
    
    Makes a boolean array for the accept/reject based on the values in the
    array. Always say the first element was accepted.

    Parameters
    ----------
    trace : array
        Trace to look at

    Returns
    -------
    int
        Number of accepts over the trace
    '''
    ret = 1
    for i in range(1,len(trace)):
        ret += trace[i]!=trace[i-1]
    return ret

def acceptance_rate(x: np.ndarray, start: int, end: int) -> Iterator[float]:
    '''Calculate the acceptance rate from `start` to `end`

    Parameters
    ----------
    x : np.ndarray
        - If it is a numpy array, just do regular slicing
    start, end : int
        - start and end indices

    Returns
    -------
    list(float)
        This is an array of the acceptance rates over the trace
    '''
    l = end-start
    if l == 0:
        return 0
    if type(x) == np.ndarray:
        return _sum_look_back(x[start:end])/l
    elif variables.isVariable(x):
        return _sum_look_back(x.trace[start:end])/l
    else:
        raise ValueError('`x` ({}) must be an array or a pylab Variable'.format(
            type(x)))

class _BaseKernel:
    '''DEPRECIATED - THIS IS NOT USED IN THE CODE
    
    This is the base Kernel jumping class for both symmetric and nonsymmetric
    jumping kernels.

    ###########################################################################
    For nonsymmetric proposals:
    The proposal normalization (proposal_norm) attribute is defined as
        $\frac
            {P_{prop}(x^{(-1)})}
            {P_{prop}(x^{(*)})}$
    where $P_{prop}(x)$ is the pdf of the proposal distribution at x

    Assuming that the jumping probability is defined as:
        $r = \frac{P(x^{(*)})}{P(x^{(-1)})}$
    You multiply r by `self.proposal_norm`. This is unnecessary for symmetric
    proposals.
    ###########################################################################

    Parameters
    ----------
     x : pylab.variables.Variable
            - Variable we are doing the Metropolis-Hasting steps on
    '''
    def __init__(self, x: variables.Variable):
        if not variables.isVariable(x):
            raise ValueError('`x` ({}) must be a pylab Variable'.format(type(x)))

        raise NotImplementedError('Need to change')
        self.x = x
        self._n_accepted_temp = 0
        self._n_accepted_total = 0
        self._prev_value = None
        self.proposal = None
        self._value = None

        # Set metropolis pointer for node object itself
        x._metropolis = self

    @property
    def value(self) -> np.ndarray:
        return self._value

    def acceptance_rate(self, prev: int=None) -> np.ndarray:
        '''Calculate the acceptance rate over the previous iterations.

        It does this by looking at the previous time step and seeing if the
        value is different. If it is different then it accepted, if not
        the proposal got rejected.

        Parameters
        ----------
        prev : int, None, Optional
            - Calculate the acceptance rate based on the most recent `prev` iterations
            - If this is not specified, it will calculate the acceptance rate over all
              iterations
        '''
        if self.x.sample_iter == 0:
            return 0
        if prev is None:
            return self._n_accepted_total / self.x.sample_iter
        else:
            start = int(np.max([0, self.x.sample_iter - prev]))
            end = self.x.sample_iter

            return acceptance_rate(self.x, start, end)

    def step(self):
        '''Make a Metropolis-Hastings step
        '''
        self._prev_value = self.x.value
        self.x.value = self._propose()
        self._value = None

    def proposal_norm(self, log: bool=True) -> float:
        '''Calculate the proposal normalization factor to multiply (or add) the
        acceptance ratio r

        Parameters
        ----------
        log : bool
            - If True, return the normalization factor in log space
            - If False, do not return in log space
        '''
        if self.value is not None:
            accept = 'accepted' if self.value else 'rejected'
            raise MathError('You already {} the proposal ({metropolis.value = True}), ' \
                'you can only calculate the proposal normalization factor if you have ' \
                'not already {} the value'.format(accept,accept))
        if not util.isbool(log):
            raise ValueError('`log` ({}) must be of type bool'.format(type(log)))

        if log:
            func = self.proposal.logpdf
        else:
            func = self.proposal.pdf
        self.proposal.mean.value = self._prev_value
        self.proposal.value = self.x.value
        forward = func()

        self.proposal.mean.value = self.x.value
        self.proposal.value = self._prev_value
        reverse = func()

        if log:
            return reverse - forward
        else:
            return reverse/forward

    def accept(self):
        '''Accept the new value
        '''
        self._n_accepted_temp += 1
        self._n_accepted_total += 1
        self._value = True
        self._prev_value = None

    def reject(self):
        '''Revert to original value
        '''
        self.x.value = self._prev_value
        self._value = False
        self._prev_value = None


class _TunableKernel(_BaseKernel):
    '''DEPRECIATED - THIS CLASS IS NOT USED IN THE CODE
    
    Class for tuning the variance of the distribution
    
    Parameters
    ----------
    tune : int, bool, Optional
        - If an int, this is the tuning interval to adjust the scaling over the
            iterations
        - If a bool,
            - If True we set the tuning interval to 50
            - If False we do not tune
        - Example, if you want to tune the scaling every 50 iterations, set `tune`=50
        - If None, we do no tuning
    target_acceptance_rate : float (0,1), str, Optional
        - If it is a string:
            - Options:
                - 'fixed'
                    - Set the target rate to be 0.44
                - 'range'
                    - Set the target range to be between 0.2-0.5
                - 'auto'
                    - Set to 'fixed'
                - 'default'
                    - Set to the default
        - If it is a float:
            - A float indicating the target acceptance rate
            - The default value is 0.44 - the optimal value for single variable Metropolis-
                Hastings steps [1]. If the acceptance rate is below the target, it will
                scale the variance by 2/3. If the acceptance rate is above the target, it
                will scale the variance by 3/2.
        - If None, it will adapt the scale with the following rules (taken from PyMC3):
                Rate    Variance adaptation
                ----    -------------------
                <0.001        x 0.1
                <0.05         x 0.5
                <0.2          x 0.9
                >0.5          x 1.1
                >0.75         x 2
                >0.95         x 10
        - Only necessary is `tune` is not None
    end_tuning : str, int, float, None, Optional
        - When to end the tuning
        - If int
            - This is the iteration to stop the tuning
        - If str
            - Options
                - 'after burnin'
                    - Ends the tuning after the burnin period
                - 'half burnin'
                    - Ends tuning half way through burnin
                - 'auto'
                    - Set to 'half burnin'
                - 'default'
                    - Set to the default
        If float
            - Must be in (0,1), which represents the percent of burnin to tune
                over. If `end_tuning`*`burnin` is not an int, it will round down.
        - If None, set to 'default'
        - Only necessary is `tune` is not None
    start_tuning : int
        This is the iteration to start the tuning process

    References
    ----------
    [1] A. Gelman, H. S. Stern, J. B. Carlin, D. B. Dunson, A. Vehtari, and D. B. Rubin,
    Bayesian Data Analysis Third Edition. Chapman and Hall/CRC, 2013.
    '''
    def __init__(self, x: variables.Variable, tune: int=None, target_acceptance_rate: Union[str, float]='default', 
        end_tuning: Union[str, int]='default', start_tuning: int=0):
        _BaseKernel.__init__(self, x=x)

        self._scaling = 1
        if tune is not None:
            if util.isbool(tune):
                if tune:
                    self.tune = 50
                else:
                    tune = None
                    self.tune = float('inf')
            elif not util.isint(tune):
                raise ValueError('`tune` ({}) must be a bool or an int'.format(type(tune)))
            else:
                self.tune = tune
        else:
            self.tune = float('inf')

        if tune is not None:
            if target_acceptance_rate is not None:
                if type(target_acceptance_rate) == str:
                    if target_acceptance_rate == 'default':
                        target_acceptance_rate = DEFAULT_TARGET_ACCEPTANCE_RATE
                    if target_acceptance_rate in ['auto', 'fixed']:
                        target_acceptance_rate = 0.44
                    elif target_acceptance_rate == 'range':
                        target_acceptance_rate = None
                    else:
                        raise ValueError('`target_acceptance_rate` ({}) not recognized'.format(
                            target_acceptance_rate))
                elif util.isfloat(target_acceptance_rate):
                    if target_acceptance_rate <= 0 or target_acceptance_rate >=1:
                        raise ValueError('`target_acceptance_rate` ({}) must be between 0 and 1'.format(
                            target_acceptance_rate))

            if util.isint(end_tuning):
                # if end_tuning > self.x.n_samples:
                #     raise ValueError('`end_tuning` ({}) cannot be greater than the total ' \
                #         'number of inference steps ({})'.format(end_tuning, self.x.n_samples))
                if end_tuning < 0:
                    raise ValueError('`end_tuning` ({}) cannot be negative'.format(end_tuning))
            elif util.isfloat(end_tuning):
                if end_tuning < 0 or end_tuning > 1:
                    raise ValueError('`end_tuning` ({}) must be between 0 and 1'.format(end_tuning))
                end_tuning = int(end_tuning * self.x.burnin)
            elif type(end_tuning) == str:
                if end_tuning == 'default':
                    end_tuning = DEFAULT_END_TUNING
                if end_tuning == 'after burnin':
                    end_tuning = self.x.burnin
                elif end_tuning in ['auto', 'half burnin']:
                    end_tuning = int(self.x.burnin/2)
                else:
                    raise ValueError('`end_tuning` ({}) not recognized'.format(end_tuning))
            elif end_tuning is None:
                end_tuning = int(self.x.burnin/2)
            else:
                raise ValueError('`end_tuning` ({}) type not recognized'.format(type(end_tuning)))

            if not util.isint(start_tuning):
                raise ValueError('`start_tuning` ({}) must be an int'.format( 
                    type(start_tuning)))
            
        
        else:
            end_tuning = None
            target_acceptance_rate = None
            start_tuning = None

        self._steps_until_tune = self.tune
        self.target_acceptance_rate = target_acceptance_rate
        self.end_tuning = end_tuning
        self.start_tuning = start_tuning

    def step(self):
        '''Override the base `step` method to incorporate variance tuning
        Take a single step, propose a new value
        '''
        self._prev_value = self.x.value

        if self.x.sample_iter > self.start_tuning:
            if self._steps_until_tune == 0:
                self._tune_variance()
            self._steps_until_tune -= 1

        self.x.value = self._propose()
        self._value = None
        
    def _tune_variance(self):
        '''Tune the variance if applicable
        '''
        self._n_accpeted_temp = 0
        self._steps_until_tune = self.tune
        if self.x.sample_iter > self.end_tuning:
            self._steps_until_tune = float('inf')
            return
        acc_rate = self.acceptance_rate(prev=self.tune)

        # logger.info('Tuning the variance of {}. Acceptance rate: {}'.format(
        #     self.x.name, acc_rate))

        
        if self.target_acceptance_rate is None:
            raise NotImplementedError('Check the conditions?')
            if acc_rate < 0.001:
                # reduce by 90 percent
                self.proposal.scale2.value *= 0.1
            elif acc_rate < 0.05:
                # reduce by 50 percent
                self.proposal.scale2.value *= 0.5
            elif acc_rate < 0.2:
                # reduce by ten percent
                self.proposal.scale2.value *= 0.9
            elif acc_rate > 0.95:
                # increase by factor of ten
                self.proposal.scale2.value *= 10.0
            elif acc_rate > 0.75:
                # increase by double
                self.proposal.scale2.value *= 2.0
            elif acc_rate > 0.5:
                # increase by ten percent
                self.proposal.scale2.value *= 1.1
        else:
            # print('acc_rate', acc_rate)
            if acc_rate > self.target_acceptance_rate:
                # print('increase')
                self.proposal.scale2.value *= 2
            else:
                # print('decrease')
                self.proposal.scale2.value /= 2


class NormalKernel(_TunableKernel):
    '''This is either a normal or truncated normal proposal to use for the
    proposal. If `low` and `high` are both None, it assumes it is a normal
    distribution. Else it is a truncated distribution

    Parameters
    ----------
    x : pylab.variables.Variable
        - Variable we are tuning
    var : numeric (float,int)
        - Variance for the proposal
    low, high : numeric, Optional
        - If specified makes the jumping distribution a truncated distribution
    '''
    def __init__(self, x: variables.Variable, var: Union[float, int], low: Union[float, int]=None, 
        high: Union[float, int]=None, **kwargs):
        _TunableKernel.__init__(self, x=x, **kwargs)

        if not util.isnumeric(var):
            raise ValueError('`var` ({}) must be a numeric type (float, int)'.format(
                type(var)))
        if var  <= 0:
            raise ValueError('`var` ({}) must be greater than 0'.format(var))

        name = self.x.name + '_proposal'
        if low is None and high is None:
            self.proposal = variables.Normal(loc=0, scale2=var, G=self.x.G, name=name)
        else:
            if low is not None:
                if not util.isnumeric(low):
                    raise ValueError('If `low` ({}) is specified, it must be a numeric type (float, int)'.format(
                        type(low)))
            else:
                low = float('-inf')
            if high is not None:
                if not util.isnumeric(high):
                    raise ValueError('If `high` ({}) is specified, it must be a numeric type (float, int)'.format(
                        type(high)))
            else:
                high = float('inf')
            self.proposal = variables.TruncatedNormal(loc=0, scale2=var, low=low, high=high,
                G=self.x.G, name=name)

    def _propose(self) -> float:
        '''propose a new value. Sets the mean to the value of x
        '''
        self.proposal.mean.value = self.x.value
        return self.proposal.sample()
