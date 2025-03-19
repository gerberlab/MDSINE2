'''Sample from efficient distributions.

The difference between these classes and the classes in `pylab.variables` is that
these are just efficnet wrappers of functions that do not require instantiating a
node whereas the classes in `pylab.variables` are meant to be inherited for posteriors
and to instantiate as objects. The functions of the distributions in `pylab.variables`
are implemented here
'''

import os
import numpy as np
import math
import numpy.random as npr
import random
from mdsine2.logger import logger
import scipy.stats

# Typing
from typing import Any, Union, Tuple, List
from .errors import MathError, UndefinedError


# For caluclating pdf, logpdf, cdf, logcdf - faster access and precomputation
import numba # for compiling and forcing function to stay in cache
from math import sqrt as SQRT
from math import pi as _PI
from math import erf as ERF
from math import gamma as GAMMA
from math import lgamma as LGAMMA
from numpy import exp as EXP
from numpy import log as LOG

_INV_SQRT_2PI = 1/SQRT(2*_PI)
_LOG_INV_SQRT_2PI = LOG(1/SQRT(2*_PI))
_LOG_2PI = LOG(2*_PI)
_INV_SQRT_2 = 1/SQRT(2)
_LOG_INV_SQRT_2 = LOG(1/SQRT(2))
_LOG_ONE_HALF = LOG(0.5)
_NEGINF = float('-inf')

def israndom(x: Any) -> bool:
    '''Checks whether the input is a subclass of BaseRandom (not a
    random variable (isRandomVariable))

    Parameters
    ----------
    x : any
        Input instance to check the type of BaseRandom

    Returns
    -------
    bool
        True if `x` is of type BaseRandom, else False
    '''
    return x is not None and issubclass(x.__class__, _BaseSample)

def seed(x: int):
    '''Sets all of the seeds with the given seed `x`

    Parameters
    ----------
    x (int)
        Seed to set everything at
    '''
    np.random.seed(x)
    random.seed(x)

def _safe_cholesky(M: np.ndarray, jitter: bool=False, save_if_crash: bool=False):
    # if scipy.sparse.issparse(M):
    #     M = M.toarray()

    try:
        # if type(M) == torch.Tensor:
        #     L = torch.cholesky(M)
        # else:
        return np.linalg.cholesky(M)
    except:
        try:
            return scipy.linalg.cholesky(M)
        except:
            if not jitter:
                if save_if_crash:
                    saveloc = 'this_is_what_made_cholesky_crash_{}.npy'.format(os.getpid())
                    np.save(saveloc, M)
                raise
        jitter = 1e-9
        while jitter < 1.0:
            try:
                L = np.linalg.cholesky(M + np.diag(jitter*np.ones(M.shape[0])))
                logger.warning('jitter threshold: {}'.format(jitter))
                return L
            except:
                jitter *= 10

    if save_if_crash:
        saveloc = 'this_is_what_made_cholesky_crash_{}.npy'.format(os.getpid())
        np.save(saveloc, M)
    raise MathError('Cholesky could not be calculated with jitter. Array that ' \
        'crashed the system saved as `this_is_what_made_cholesky_crash.npy`')

def _log_det_func(M: np.ndarray) -> float:
    # if scipy.sparse.issparse(M):
    #     M_ = M.toarray()
    # else:
    #     M_ = M
    L = _safe_cholesky(M)
    return 2*np.sum(np.log(np.diag(L)))

class misc:
    '''These are miscellaneus methods
    '''
    @staticmethod
    def multivariate_normal_fast_2d(mean: Union[np.ndarray, List], 
        cov: np.ndarray) -> Union[np.ndarray, List]:
        '''Sample from 2d normal.

        Parameters
        ----------
        mean : 1d array
        cov : 2 x 2 array

        Returns
        -------
        np.ndarray (2,)
            Two samples
        '''
        z0 = np.random.normal()
        SQRT_COV_00 = math.sqrt(cov[0,0])
        mean[0] += SQRT_COV_00 * z0
        mean[1] += cov[1,0] / SQRT_COV_00 * z0 + math.sqrt(cov[1,1] - \
            cov[1,0]**2 / cov[0,0]) * np.random.normal()
        return mean

    @staticmethod
    def fast_sample_standard_uniform() -> float:
        '''Sample from a uniform distribution on [0,1)
        '''
        return np.random.uniform()

    @staticmethod
    def fast_sample_normal(loc: float, scale: float) -> float:
        '''Sample from a c_implementation of a normal distribution.
        Only accepts floats

        Parameters
        ----------
        loc, scale : float
            Mean and standard devition, respectively

        Returns
        -------
        float
        '''
        return np.random.normal(loc=loc, scale=scale)


class _BaseSample:

    @staticmethod
    def sample(*args, **kwargs):
        '''Sample a random variable from the distribution
        '''
        raise UndefinedError('This function is undefined.')

    @staticmethod
    def pdf(*args, **kwargs):
        '''Calculate the pdf
        '''
        raise UndefinedError('This function is undefined.')

    @staticmethod
    def logpdf(*args, **kwargs):
        '''Calculate the logpdf
        '''
        raise UndefinedError('This function is undefined.')

    @staticmethod
    def cdf(*args, **kwargs):
        '''Calculate the cdf
        '''
        raise UndefinedError('This function is undefined.')

    @staticmethod
    def logcdf(*args, **kwargs):
        '''Calculate the logcdf
        '''
        raise UndefinedError('This function is undefined.')


class normal(_BaseSample):
    '''Normal distribution parameterized by a mean and standard deviation
    '''
    @staticmethod
    def sample(loc: Union[float, np.ndarray]=0, scale: Union[float, np.ndarray]=1,
        size: int=None) -> Union[float, np.ndarray]:
        '''Sample from a normal random distribution. This can be vectorized

        NOTE: If you want to sample a single scalar value with a normal distribution,
        use the function `mdsine2.random.misc.fast_sample_normal`

        Parameters
        ----------
        loc : np.ndarray, float
            This is the mean
        scale : np.ndarray, float
            This is the scale
        size : int
            Number of samples to return

        Returns
        -------
        np.ndarray, float

        See Also
        --------
        mdsine2.random.misc.fast_sample_normal
        '''
        return npr.normal(loc, scale, size=size)

    @staticmethod
    @numba.jit(nopython=True, fastmath=True, cache=True)
    def pdf(value: float, loc: float, scale: float) -> float:
        '''Returns the probability density function of a normal distribution

        Parameters
        ----------
        value : float
            This is the value we are calculating at
        loc : float
            This is the mean
        scale : float
            This is the scale

        Returns
        -------
        float
        '''
        return _INV_SQRT_2PI * EXP(-0.5*((value-loc)/scale)**2) / scale

    @staticmethod
    @numba.jit(nopython=True, fastmath=True, cache=True)
    def logpdf(value: float, loc: float, scale: float) -> float:
        '''Returns the log probability density function of a normal distribution

        Parameters
        ----------
        value : float
            This is the value we are calculating at
        loc : float
            This is the mean
        scale : float
            This is the scale

        Returns
        -------
        float
        '''
        return _LOG_INV_SQRT_2PI + (-0.5*((value-loc)/scale)**2) - LOG(scale)


    @staticmethod
    @numba.jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def cdf(value: float, loc: float, scale: float) -> float:
        '''Returns the cumulative density function of a normal distribution

        Parameters
        ----------
        value : float
            This is the value we are calculating at
        loc : float
            This is the mean
        scale : float
            This is the scale

        Returns
        -------
        float
        '''
        return 0.5 * (1 + ERF(_INV_SQRT_2 * ((value-loc)/scale)))

    @staticmethod
    @numba.jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def logcdf(value: float, loc: float, scale: float) -> float:
        '''Returns the log cumulative density function of a normal distribution

        Parameters
        ----------
        value : float
            This is the value we are calculating at
        loc : float
            This is the mean
        scale : float
            This is the scale

        Returns
        -------
        float
        '''
        return _LOG_ONE_HALF + LOG(1 + ERF(_INV_SQRT_2 * ((value-loc)/scale)))


class lognormal(_BaseSample):
    '''Log-normal distribution:
    X = exp(\mu + \sigma Z), Z ~ Normal(0,1)
    ''' 
    @staticmethod
    def sample(loc: Union[float, np.ndarray], scale: Union[float, np.ndarray], 
        size: int=None) -> Union[float, np.ndarray]:
        '''Sample from a log-normal random distribution. This can be vectorized

        Parameters
        ----------
        loc : np.ndarray, float
            This is the mean
        scale : np.ndarray, float
            This is the scale
        size : int
            Number of samples to return

        Returns
        -------
        np.ndarray, float
        '''
        return np.exp(loc + scale * npr.normal(0,1,size=size))

    @staticmethod
    @numba.jit(nopython=True, fastmath=True, cache=True)
    def pdf(value: float, loc: float, scale: float) -> float:
        '''Returns the probability density function of a log-normal distribution

        Parameters
        ----------
        value : float
            This is the value we are calculating at
        loc : float
            This is the mean
        scale : float
            This is the scale

        Returns
        -------
        float
        '''
        return _INV_SQRT_2PI * (1/(scale*value)) * EXP(-0.5 * \
            ((LOG(value)-loc)/scale) ** 2)

    @staticmethod
    @numba.jit(nopython=True, fastmath=True, cache=True)
    def logpdf(value: float, loc: float, scale: float) -> float:
        '''Returns the log probability density function of a log-normal distribution

        Parameters
        ----------
        value : float
            This is the value we are calculating at
        loc : float
            This is the mean
        scale : float
            This is the scale

        Returns
        -------
        float
        '''
        return _LOG_INV_SQRT_2PI - LOG(scale) - LOG(value) + \
            (-0.5*((LOG(value)-loc)/scale)**2)


class truncnormal(_BaseSample):
    '''Truncated normal distribution.

    We parameterize the truncated normal distribution with the mean `loc` and 
    standard deviation `scale` of the underlying normal distirbution and then we
    specified the truncation bounds:    
    
    For example:
        mdsine2.random.truncnormal.sample(0, 3, low=-2, high=10)
            Here, the mean in 0, the standard deviation is 3, the lower bound is -2,
            and the high is 10.

    NOTE: THIS IS A DIFFERENT PARAMETERIZATION THAN SCIPY
    '''
    @staticmethod
    def sample(loc: Union[float, np.ndarray], scale: Union[float, np.ndarray], 
        low: float=float('-inf'), high: float=float('inf'), size: int=None) -> Union[float, np.ndarray]:
        '''Sample from a truncated normal random distribution defined on
        [low, high] with mean `loc` and standard deviation `scale`

        Parameters
        ----------
        loc : np.ndarray, float
            This is the mean
        scale : np.ndarray, float
            This is the scale
        low, high : float
            Truncation points of normal distribution
        size : int
            Number of samples to return
        
        Returns
        -------
        np.ndarray, float
        '''
        if size is not None:
            value = scipy.stats.truncnorm(
                a=(low-loc)/scale,
                b=(high-loc)/scale,
                loc=loc,
                scale=scale).rvs(size=size)
        else:
            value = scipy.stats.truncnorm(
                a=(low-loc)/scale,
                b=(high-loc)/scale,
                loc=loc,
                scale=scale).rvs(size=size)
        return value

    @staticmethod
    def pdf(value: float, loc: float, scale: float, low: float, high: float) -> float:
        '''Returns the probability density function of a truncated normal distribution

        Parameters
        ----------
        value : float
            This is the value we are calculating at
        loc : float
            This is the mean
        scale : float
            This is the scale
        low, high : float
            Truncation points of normal distribution

        Returns
        -------
        float
        '''
        return scipy.stats.truncnorm.pdf(value, (low-loc)/scale, (high-loc)/scale, loc, scale)

    @staticmethod
    def logpdf(value: float, loc: float, scale: float, low: float, high: float) -> float:
        '''Returns the log probability density function of a truncated normal distribution

        Parameters
        ----------
        value : float
            This is the value we are calculating at
        loc : float
            This is the mean
        scale : float
            This is the scale
        low, high : float
            Truncation points of normal distribution

        Returns
        -------
        float
        '''
        return scipy.stats.truncnorm.logpdf(value, (low-loc)/scale, (high-loc)/scale, loc, scale)


class multivariate_normal(_BaseSample):
    '''Multivariate normal distribution - this is the same sampling methods as Numpy
    '''

    @staticmethod
    def sample(mean: np.ndarray, cov: np.ndarray, size: int=None) -> np.ndarray:
        '''Sample from a multivariate normal random distribution.

        Parameters
        ----------
        mean : np.ndarray
            This is the mean
        cov : np.ndarray
            This is the covaraiance 
        size : int
            Number of samples to return

        Returns
        -------
        np.ndarray
        '''
        return npr.multivariate_normal(mean=mean, cov=cov, size=size)

    @staticmethod
    def logpdf(value: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        '''Returns the probability density function of a multivariate normal distribution

        Parameters
        ----------
        value : np.ndarray
            This is the value we are calculating at
        mean : np.ndarray
            This is the mean
        cov : np.ndarray
            This is the scale

        Returns
        -------
        np.ndarray
        '''
        k = cov.shape[0]
        logdet = _log_det_func(cov)
        prec = np.linalg.pinv(cov)
        vmm = value - mean
        a = -k * 0.5 * _LOG_2PI
        b = -0.5 * logdet
        c = -0.5 * ( vmm.T @ prec @ vmm)
        return np.squeeze(a + b + c)


class gamma(_BaseSample):
    '''Gamma random distribution - this is the same parameterization as
    Numpy and scipy
    '''
    @staticmethod
    def sample(shape: Union[float, np.ndarray], scale: Union[float, np.ndarray], 
        size: int=None) -> Union[float, np.ndarray]:
        '''Sample from a gamma random distribution. This can be vectorized

        Parameters
        ----------
        shape : np.ndarray, float
            This is the shape parameter
        scale : np.ndarray, float
            This is the scale parameter
        size : int
            Number of samples to return

        Returns
        -------
        np.ndarray, float
        '''
        return npr.gamma(shape=shape, scale=scale, size=size)

    @staticmethod
    def pdf(value: float, shape: float, scale: float) -> float:
        '''Returns the probability density function of a gamma distribution

        Parameters
        ----------
        value : float
            This is the value we are calculating at
        shape : float
            This is the shape parameter
        scale : float
            This is the scale parameter

        Returns
        -------
        float
        '''
        return scipy.stats.gamma.pdf(x=value, a=shape, scale=scale)


class beta(_BaseSample):
    '''Beta random distribution - same parameterization as numpy
    '''
    @staticmethod
    def sample(a: Union[float, np.ndarray], b: Union[float, np.ndarray], 
        size: int=None) -> Union[float, np.ndarray]:
        '''Sample from a beta random distribution. This can be vectorized

        Parameters
        ----------
        a, b : np.ndarray, float
            These are the a and b parmeters of the distribution

        Returns
        -------
        np.ndarray, float
        '''
        return npr.beta(a=a, b=b, size=size)


class sics(_BaseSample):
    '''Scaled Inverse Chi^2 distribution.
    '''
    @staticmethod
    @numba.jit(nopython=True, fastmath=True, cache=True)
    def pdf(value: Union[float, np.ndarray], dof: Union[float, np.ndarray], 
        scale: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        dofdiv2 = dof/2
        a = ((scale * dofdiv2)**(dofdiv2))/(GAMMA(dofdiv2))
        b = EXP(-scale*dofdiv2/(value)) / (value ** (1 + (dofdiv2)))
        return a*b
        
    @staticmethod
    @numba.jit(nopython=True, fastmath=True, cache=True)
    def logpdf(value: Union[float, np.ndarray], dof: Union[float, np.ndarray], 
        scale: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        dofdiv2 = dof/2
        a = dofdiv2*LOG(scale*dofdiv2)
        b = -LGAMMA(dofdiv2)
        c = -scale * dofdiv2 / value
        d = -(1+dofdiv2) * LOG(value)
        return a + b + c + d

    @staticmethod
    def sample(dof: Union[float, np.ndarray], scale: Union[float, np.ndarray], 
        size: int=None) -> Union[float, np.ndarray]:
        return invgamma.sample(shape=dof/2, scale=dof*scale/2, size=size)


class invchisquared(_BaseSample):

    @staticmethod
    def sample(nu: Union[float, np.ndarray], size: int=None) -> Union[float, np.ndarray]:
        return invgamma.sample(shape=nu/2, scale=0.5, size=size)


class invgamma(_BaseSample):

    @staticmethod
    def sample(shape: Union[float, np.ndarray], scale: Union[float, np.ndarray], 
        size: int=None) -> Union[float, np.ndarray]:
        return 1/npr.gamma(shape=shape, scale=1/scale, size=size)

    @staticmethod
    def pdf(value: Union[float, np.ndarray], shape: Union[float, np.ndarray], 
        scale: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return scipy.stats.invgamma.pdf(value, a=shape, scale=scale)

    @staticmethod
    def logpdf(value: Union[float, np.ndarray], shape: Union[float, np.ndarray], 
        scale: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return scipy.stats.invgamma.logpdf(value, a=shape, scale=scale)


class uniform(_BaseSample):
    @staticmethod
    def sample(low: Union[float, np.ndarray]=0, high: Union[float, np.ndarray]=1, 
        size: int=None) -> Union[float, np.ndarray]:
        return npr.uniform(low=low, high=high, size=size)

    @staticmethod
    def pdf(value: Union[float, np.ndarray], low: Union[float, np.ndarray], 
        high: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if value < low or value > high:
            return 0
        else:
            return 1/(high-low)

    @staticmethod
    def logpdf(value: Union[float, np.ndarray], low: Union[float, np.ndarray], 
        high: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if value < low or value > high:
            return 0
        else:
            return -LOG(high-low)

    @staticmethod
    def cdf(value: Union[float, np.ndarray], low: Union[float, np.ndarray], 
        high: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if value < low:
            return 0
        elif value >= high:
            return 1
        else:
            return (value-low)/(high-value)

    @staticmethod
    def logcdf(value: Union[float, np.ndarray], low: Union[float, np.ndarray], 
        high: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if value < low:
            return float('-inf')
        elif value >= high:
            return 0
        else:
            return LOG(value-low) - LOG(high-value)


class negative_binomial(_BaseSample):
    '''Parameterization of the negative binomial with mean $\\phi$ and dispersion
    $\\epsilon$:
        $\\text{NegBin}(y; \\phi, \\epsilon) =
            \\Gamma(r+y) / (y! * \\Gamma(r)) *
            (\\phi / (\\phi + r))^y *
            (r / (r + \\phi))^r$, where $r = 1/\\epsilon$
    We reparameterize the inputs so we can use the scipy implementation
    '''
    @staticmethod
    def sample(mean: float, dispersion: float, size: int=None) -> float:
        '''Sample
        '''
        n,p = negative_binomial.convert_params(mean, dispersion)
        return scipy.stats.nbinom(n=n, p=p).rvs(size=size)

    @staticmethod
    def pmf(value: float, mean: float, dispersion: float) -> float:
        '''Calculate the pmf
        '''
        n,p = negative_binomial.convert_params(mean, dispersion)
        return scipy.stats.nbinom.pmf(value, n, p)

    @staticmethod
    def logpmf(value: float, mean: float, dispersion: float) -> float:
        '''Calculate the logpmf
        '''
        n,p = negative_binomial.convert_params(mean, dispersion)
        return scipy.stats.nbinom.logpmf(value, n, p)

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def convert_params(mu: float, theta: float) -> Tuple[float, float]:
        """
        Convert mean/dispersion parameterization of a negative binomial to the ones scipy supports

        Parameters
        ----------
        mu : float
           Mean of NB distribution.
        theta : float
           Dispersion parameter used for variance calculation.

        Returns
        -------
        float, float
            Returns n, p

        See Also
        --------
        https://en.wikipedia.org/wiki/Negative_binomial_distribution#Alternative_formulations
        """
        r = 1/theta
        var = mu + 1 / r * mu ** 2
        p = (var - mu) / var
        return r, 1-p


class bernoulli:

    @staticmethod
    def sample(p: float=0.5, size: int=None) -> float:
        '''Sample a random variable from the distribution
        '''
        return npr.binomial(n=1, p=p, size=size)
