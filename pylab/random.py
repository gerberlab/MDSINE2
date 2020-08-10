'''Sample from efficient distributions.

The difference between these classes and the classes in `pylab.variables` is that
these are just efficnet wrappers of functions that do not require instantiating a
node whereas the classes in `pylab.variables` are meant to be inherited for posteriors
and to instantiate as objects. The functions of the distributions in `pylab.variables`
are implemented here
'''

import numpy as np
import math
import numpy.random as npr
import pickle
import random
import logging
import warnings
import sys
import scipy.stats
import warnings

from .base import Saveable
from .errors import MathError, UndefinedError
try:
    import _sample as c_sample
    # warnings.warn('PYLAB - USING C VERSIONS FOR DISTRIBUTIONS')

    C_SAMPLE = c_sample.Sample()
    CUSTOM_DIST_AVAIL = True
except ImportError:
    warnings.warn('PYLAB - WAS NOT ABLE TO IMPORT C VERSIONS OF DISTRIBUTIONS.' \
        ' USING DEFAULT PYTHON INSTEAD.')
    CUSTOM_DIST_AVAIL = False


# For caluclating pdf, logpdf, cdf, logcdf - faster access and precomputation
import numba # for compiling and forcing function to stay in cache
from math import sqrt as SQRT
from math import pi as _PI
from math import erf as ERF
from math import gamma as GAMMA
from math import lgamma as LGAMMA
from numpy import exp as EXP
from numpy import log as LOG
from numpy import square as SQD

_INV_SQRT_2PI = 1/SQRT(2*_PI)
_LOG_INV_SQRT_2PI = LOG(1/SQRT(2*_PI))
_LOG_2PI = LOG(2*_PI)
_INV_SQRT_2 = 1/SQRT(2)
_LOG_INV_SQRT_2 = LOG(1/SQRT(2))
_LOG_ONE_HALF = LOG(0.5)
_NEGINF = float('-inf')

def israndom(x):
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

def seed(x):
    '''Sets all of the seeds with the given seed `x`

    Parameters
    ----------
    x (int)
        Seed to set everything at
    '''
    np.random.seed(x)
    random.seed(x)
    if CUSTOM_DIST_AVAIL:
        C_SAMPLE.seed(x)

def _safe_cholesky(M, jitter=False, save_if_crash=False):
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
                logging.warning('jitter threshold: {}'.format(jitter))
                return L
            except:
                jitter *= 10

    if save_if_crash:
        saveloc = 'this_is_what_made_cholesky_crash_{}.npy'.format(os.getpid())
        np.save(saveloc, M)
    raise MathError('Cholesky could not be calculated with jitter. Array that ' \
        'crashed the system saved as `this_is_what_made_cholesky_crash.npy`')

def _log_det_func(M):
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
    def multivariate_normal_fast_2d(mean, cov):
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
        z0 = C_SAMPLE.c_standard_normal()
        SQRT_COV_00 = math.sqrt(cov[0,0])
        mean[0] += SQRT_COV_00 * z0
        mean[1] += cov[1,0] / SQRT_COV_00 * z0 + math.sqrt(cov[1,1] - \
            cov[1,0]**2 / cov[0,0]) * C_SAMPLE.c_standard_normal()
        return mean

    @staticmethod
    def fast_sample_standard_uniform():
        '''Sample from a uniform distribution on [0,1)
        '''
        return C_SAMPLE.c_standard_uniform()

    @staticmethod
    def fast_sample_normal(mean, std):
        '''Sample from a c_implementation of a normal distribution.
        Only accepts floats

        Parameters
        ----------
        mean, std : float
            Mean and standard devition, respectively

        Returns
        -------
        float
        '''
        return C_SAMPLE.c_normal(mean, std)


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
    '''Scalar normal distribution parameterized by a mean and standard deviation
    '''
    @staticmethod
    def sample(mean=0, std=1, size=None):
        '''Sample from a normal random distribution
        '''
        return npr.normal(mean, std, size=size)

    @staticmethod
    @numba.jit(nopython=True, fastmath=True, cache=True)
    def pdf(value, mean, std):
        return _INV_SQRT_2PI * EXP(-0.5*((value-mean)/std)**2) / std

    @staticmethod
    @numba.jit(nopython=True, fastmath=True, cache=True)
    def logpdf(value, mean, std):
        return _LOG_INV_SQRT_2PI + (-0.5*((value-mean)/std)**2) - LOG(std)


    @staticmethod
    @numba.jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def cdf(value, mean, std):
        return 0.5 * (1 + ERF(_INV_SQRT_2 * ((value-mean)/std)))

    @staticmethod
    @numba.jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def logcdf(value, mean, std):
        return _LOG_ONE_HALF + LOG(1 + ERF(_INV_SQRT_2 * ((value-mean)/std)))


class lognormal(_BaseSample):
    '''Sample from a log-normal distribution:

    X = exp(\mu + \sigma Z), Z ~ Normal(0,1)
    ''' 
    @staticmethod
    def sample(mean, std, size=None):
        return np.exp(mean + std * npr.normal(0,1,size=size))

    @staticmethod
    @numba.jit(nopython=True, fastmath=True, cache=True)
    def pdf(value, mean, std):
        return _INV_SQRT_2PI * (1/(std*value)) * EXP(-0.5 * \
            ((LOG(value)-mean)/std) ** 2)

    @staticmethod
    @numba.jit(nopython=True, fastmath=True, cache=True)
    def logpdf(value, mean, std):
        return _LOG_INV_SQRT_2PI - LOG(std) - LOG(value) + \
            (-0.5*((LOG(value)-mean)/std)**2)


class truncnormal(_BaseSample):

    @staticmethod
    def sample(mean, std, low=float('-inf'), high=float('inf'), size=None):
        '''Sample from a truncated normal random distribution defined on
        [low, high] with mean `mean` and standard deviation `std`
        '''
        if size is not None:
            if CUSTOM_DIST_AVAIL:
                try:
                    value = np.asarray([C_SAMPLE.c_truncated_normal(
                        mean,std,low,high) for i in range(len(size))])
                except:
                    value = scipy.stats.truncnorm(
                        a=(low-mean)/std,
                        b=(high-mean)/std,
                        loc=mean,
                        scale=std).rvs(size=size)
            else:
                value = scipy.stats.truncnorm(
                    a=(low-mean)/std,
                    b=(high-mean)/std,
                    loc=mean,
                    scale=std).rvs(size=size)
        else:
            if CUSTOM_DIST_AVAIL:
                try:
                    value = C_SAMPLE.c_truncated_normal(mean, std, low, high)
                except:
                    # likely because the mean and std are vectors
                    # try vectorizing it
                    value = np.asarray([C_SAMPLE.c_truncated_normal(
                        mean[i],std[i],low,high) for i in range(len(mean))])
            else:
                value = scipy.stats.truncnorm(
                    a=(low-mean)/std,
                    b=(high-mean)/std,
                    loc=mean,
                    scale=std).rvs(size=size)
        return value

    @staticmethod
    def sample_vec(mean, std, low=float('-inf'), high=float('inf'), size=None):
        '''Sample from a truncated normal random distribution defined on
        [low, high] with mean `mean` and standard deviation `std`
        '''
        return np.asarray([C_SAMPLE.c_truncated_normal(
            mean[i],std[i],low,high) for i in range(len(mean))])

    @staticmethod
    # @numba.jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def pdf(value, mean, std, low, high):
        return scipy.stats.truncnorm.pdf(value, (low-mean)/std, (high-mean)/std, mean, std)

    @staticmethod
    # @numba.jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def logpdf(value, mean, std, low, high):
        return scipy.stats.truncnorm.logpdf(value, (low-mean)/std, (high-mean)/std, mean, std)


class multivariate_normal(_BaseSample):

    @staticmethod
    def sample(mean, cov, size=None):
        return npr.multivariate_normal(mean=mean, cov=cov, size=size)

    @staticmethod
    def logpdf(value, mean, cov):
        k = cov.shape[0]
        logdet = _log_det_func(cov)
        prec = np.linalg.pinv(cov)
        vmm = value - mean
        a = -k * 0.5 * _LOG_2PI
        b = -0.5 * logdet
        c = -0.5 * ( vmm.T @ prec @ vmm)
        return np.squeeze(a + b + c)


class gamma(_BaseSample):

    @staticmethod
    def sample(shape, scale, size=None):
        return npr.gamma(shape=shape, scale=scale, size=size)

    @staticmethod
    def pdf(value, shape, scale):
        return scipy.stats.gamma.pdf(x=value, a=shape, scale=scale)


class beta(_BaseSample):

    @staticmethod
    def sample(a, b, size=None):
        return npr.beta(a=a, b=b, size=size)


class sics(_BaseSample):
    '''Scaled Inverse Chi^2 distribution.
    '''
    @staticmethod
    @numba.jit(nopython=True, fastmath=True, cache=True)
    def pdf(value, dof, scale):
        dofdiv2 = dof/2
        a = ((scale * dofdiv2)**(dofdiv2))/(GAMMA(dofdiv2))
        b = EXP(-scale*dofdiv2/(value)) / (value ** (1 + (dofdiv2)))
        return a*b
        
    @staticmethod
    @numba.jit(nopython=True, fastmath=True, cache=True)
    def logpdf(value, dof, scale):
        dofdiv2 = dof/2
        a = dofdiv2*LOG(scale*dofdiv2)
        b = -LGAMMA(dofdiv2)
        c = -scale * dofdiv2 / value
        d = -(1+dofdiv2) * LOG(value)
        return a + b + c + d

    @staticmethod
    def sample(dof, scale, size=None):
        return invgamma.sample(shape=dof/2, scale=dof*scale/2, size=size)


class invchisquared(_BaseSample):

    @staticmethod
    def sample(nu, size=None):
        return invgamma.sample(shape=nu/2, scale=0.5, size=size)


class invgamma(_BaseSample):

    @staticmethod
    def sample(shape, scale, size=None):
        return 1/npr.gamma(shape=shape, scale=1/scale, size=size)

    @staticmethod
    def pdf(value, shape, scale):
        return scipy.stats.invgamma.pdf(value, a=shape, scale=scale)

    @staticmethod
    def logpdf(value, shape, scale):
        return scipy.stats.invgamma.logpdf(value, a=shape, scale=scale)


class uniform(_BaseSample):


    @staticmethod
    def sample(low=0, high=1, size=None):
        return npr.uniform(low=low, high=high, size=size)

    @staticmethod
    def pdf(value, low, high):
        if value < low or value > high:
            return 0
        else:
            return 1/(high-low)

    @staticmethod
    def logpdf(value, low, high):
        if value < low or value > high:
            return 0
        else:
            return -LOG(high-low)

    @staticmethod
    def cdf(value, low, high):
        if value < low:
            return 0
        elif value >= high:
            return 1
        else:
            return (value-low)/(high-value)

    @staticmethod
    def logcdf(value, low, high):
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
    def sample(mean, dispersion, size=None):
        '''Sample
        '''
        n,p = negative_binomial.convert_params(mean, dispersion)
        return scipy.stats.nbinom(n=n, p=p).rvs(size=size)

    @staticmethod
    # @numba.jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def pmf(value, mean, dispersion):
        '''Calculate the pmf
        '''
        n,p = negative_binomial.convert_params(mean, dispersion)
        return scipy.stats.nbinom.pmf(value, n, p)

    @staticmethod
    # @numba.jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def logpmf(value, mean, dispersion):
        '''Calculate the logpmf
        '''
        n,p = negative_binomial.convert_params(mean, dispersion)
        return scipy.stats.nbinom.logpmf(value, n, p)

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def convert_params(mu, theta):
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
    def sample(p=0.5, size=None):
        '''Sample a random variable from the distribution
        '''
        return npr.binomial(n=1, p=p, size=size)
