'''This module defines constant, variable, and random classes.

Tracing
-------
There are two kinds of tracing:
Variable.trace
    This is a trace that stores the tracing on RAM. This can be accessed very fast.
    The local index to use this is called `ckpt_iter` (int). This is the corresponds 
    to the index in the preallocated space in Varibale.trace where to next add a variable.
Variable.G.tracer
    This is an object that stores the data of the variable on Disk. Reading and writing to 
    disk is slower but it is needed for inferences that are sufficiently large as you cannot
    store all of the trace data in RAM. `sample_iter` (int) corresponds to the overall sample
    iteration number that we are on in the inference, which includes the `ckpt_iter`. You must
    be careful when you read from disk as the tracer does not differentiate between the burnin
    and the regular samples.
When you call `add_trace`, the internal mechanism automatically pushes the local trace to
disk once it has reached the checkpoint to write to disk.
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

from .graph import get_default_graph, Node, isnode
from .base import Traceable, istraceable
from .errors import UndefinedError, MathError, InitializationError, \
    NeedToImplementError
from . import random
from .util import isarray, isbool, isstr, istype, istuple

# Constants
DEFAULT_VARIABLE_TYPE = float

DEFAULT_NORMAL_MEAN_SUFFIX = '_mean'
DEFAULT_NORMAL_VAR_SUFFIX = '_var'
DEFAULT_LOGNORMAL_STD_SUFFIX = '_std'
DEFAULT_UNIFORM_LOW_SUFFIX = '_low'
DEFAULT_UNIFORM_HIGH_SUFFIX = '_high'
DEFAULT_SICS_DOF_SUFFIX = '_dof'
DEFAULT_SICS_SCALE_SUFFIX = '_scale'
DEFAULT_GAMMA_SHAPE_SUFFIX = '_shape'
DEFAULT_GAMMA_SCALE_SUFFIX = '_scale'
DEFAULT_BERNOULLI_P_SUFFIX = '_p'
DEFAULT_BETA_ALPHA_SUFFIX = '_a'
DEFAULT_BETA_BETA_SUFFIX = '_b'
DEFAULT_MVN_MEAN_SUFFIX = '_mean'
DEFAULT_MVN_COV_SUFFIX = '_cov'


def isVariable(var):
    '''Checks if `a` is a subclass of a Variable

    Parameters
    ----------
    a : any
        Instance we are checking

    Returns
    -------
    bool
        True if `a` is a subclass of a Variable
    '''
    return var is not None and issubclass(var.__class__, Variable)

def isRandomVariable(var):
    '''Checks if `a` is a subclass of a RandomVariable

    Parameters
    ----------
    a : any
        Instance we are checking

    Returns
    -------
    bool
        True if `a` is a subclass of a RandomVariable
    '''
    return var is not None and issubclass(var.__class__, _RandomBase)

def summary(var, set_nan_to_0=False, section='posterior', only=None):
    '''Calculates different metrics about the given trace (mean, 
    median, 25th percentile, 75th percentile)

    Parameters
    ----------
    var : Traceable, np.ndarray
        Variable/trace we are doing the calculations on
    set_nan_to_0 : bool
        If True, we set the NaNs in the trace to zeros
    section : str
        This is the section of the MH samples for us to retreive
    only : None, list(str)
        If this is specified, then only calculate the types that are 
        in here. Accepted:
            'mean', 'median', '25th percentile', '75th percentile'

    Returns
    -------
    dict
        mean ('mean'), median ('median'), 25th percentile 
        ('25th percentile'), and 75th percentile ('75th percentile'), 
    '''
    if istraceable(var):
        if var.G.inference.ckpt is None:
            if section == 'burnin':
                var = var.trace[:var.G.inference.burnin, ...]
            elif section == 'posterior':
                var = var.trace[var.G.inference.burnin:, ...]
            elif section == 'entire':
                var = var.trace
            else:
                raise ValueError('`section` ({}) not recognized'.format(section))
        else:
            var = var.get_trace_from_disk(section=section)
    elif isarray(var):
        var = np.asarray(var)
    else:
        raise ValueError('`var` ({}) must either be a subclass of Traceable ' \
            'or an array'.format(type(var)))
    if not isbool(set_nan_to_0):
        raise ValueError('`set_nan_to_0` ({}) must be a bool'.format(type(set_nan_to_0)))
    if set_nan_to_0:
        var = np.nan_to_num(var)
    ret = {}
    try:
        do = True
        if only is not None:
            do = 'median' in only
        if do is True:
            ret['median'] = np.nan_to_num(np.nanmedian(var,axis=0))
    except:
        logging.error('median failed')
        ret['median'] = np.nan
    try:
        do = True
        if only is not None:
            do = 'mean' in only
        if do is True:
            ret['mean'] = np.nan_to_num(np.nanmean(var,axis=0))
    except:
        logging.error('mean failed')
        ret['mean'] = np.nan
    try:
        do = True
        if only is not None:
            do = '25th percentile' in only
        if do is True:
            ret['25th percentile'] = np.nan_to_num(np.nanpercentile(var,25,axis=0))
    except:
        logging.error('25th percentile failed')
        ret['25th percentile'] = np.nan
    try:
        do = True
        if only is not None:
            do = '75th percentile' in only
        if do is True:
            ret['75th percentile'] = np.nan_to_num(np.nanpercentile(var,75,axis=0))
    except:
        logging.error('75th percentile failed')
        ret['75th percentile'] = np.nan
    return ret


class _BaseArithmeticClass:
    '''This is a baseclass that lets us do arithmetic on the 
    value.

    Nothing is checked in this arithmetic for of speed
    If something is going to fail, other, faster libraries (numpy) will fail
    "Ask forgiveness, not permission"
    '''
    def __str__(self):
        return str(self.value)

    # ----------------
    # Binary operators
    # ----------------
    def __mul__(self, val):
        # self * val
        return self.value * val

    def __imul__(self, val):
        # self *= val
        self.value = self.value * val
        return self

    def __rmul__(self, val):
        # val * self
        return val * self.value

    def __add__(self,val):
        # self + val
        return self.value + val

    def __iadd__(self,val):
        # self += val
        self.value = self.value + val
        return self

    def __radd__(self,val):
        # val + self
        return self.value + val

    def __sub__(self,val):
        # self - val
        return self.value - val

    def __rsub__(self,val):
        # val - self
        return val - self.value

    def __isub__(self,val):
        # self -= val
        self.value = self.value - val
        return self

    def __truediv__(self,val):
        # self / val
        return self.value / val

    def __rtruediv__(self,val):
        # val / self
        return val / self.value

    def __itruediv__(self,val):
        # self /= val
        self.value = self.value / val
        return self

    def __floordiv__(self,val):
        # self // val
        return self.value // val

    def __rfloordiv__(self,val):
        # val // self
        return val // self.value

    def __ifloordiv__(self,val):
        # self //= val
        self.value = self.value // val
        return self

    def __mod__(self,val):
        # self % val
        return self.value % val

    def __rmod__(self,val):
        # val % self
        return val % self.value

    def __imod__(self, val):
        # self %= val
        self.value = self.value % val
        return self

    def __pow__(self,val):
        # self ** val
        return self.value ** val

    def __rpow__(self,val):
        # val ** self
        return val ** self.value

    def __ipow__(self,val):
        # self **= val
        self.value = self.value ** val
        return self

    # ------------------
    # Comparison methods
    # ------------------
    def __eq__(self,val):
        # self == val
        return self.value == val

    def __ne__(self,val):
        # self != val
        return self.value != val

    def __le__(self,val):
        # self < val=
        return self.value <= val

    def __ge__(self,val):
        # self >= val
        return self.value >= val

    def __lt__(self,val):
        # self < val
        return self.value < val

    def __gt__(self,val):
        # self > val
        return self.value > val

    # ----------------
    # Unary operations
    # ----------------
    def __invert__(self):
        # ~self
        return ~self.value

    def __abs__(self):
        # abs(self)
        return abs(self.value)

    def __neg__(self):
        # -self
        return -self.value

    def __pos__(self):
        # + self
        return self.value

    def __int__(self):
        # int(self)
        return int(self.value)

    def __float__(self):
        # float(self)
        return float(self.value)

    # ----------------
    # Matrix operations
    # ----------------
    def __matmul__(self,val):
        # self @ val
        return self.value @ np.asarray(val)

    def __rmatmul__(self,val):
        # val # self
        return np.array(val) @ self.value

    def __imatmul__(self,val):
        # self @= val
        self.value = self.value @ np.asarray(val)

    def __getitem__(self,*args, **kwargs):
        return self.value.__getitem__(*args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        self.value.__setitem__(*args, **kwargs)


class _RandomBase:
    def sample(self, *args, **kwargs):
        '''Sample with the given parameters
        '''
        raise NeedToImplementError('User needs to implement this function')

    def pdf(self, value=None):
        '''Calculate the pdf with the specified value. If `value` is not
        specified, then use the current value

        Parameters
        ----------
        value : float, int, array_like
            Value we are using instead of self.value

        Returns
        -------
        float
        '''
        raise NeedToImplementError('User needs to implement this function')

    def logpdf(self, value=None):
        '''Calculate the logpdf with the specified value. If `value` is not
        specified, then use the current value

        Parameters
        ----------
        value : float, int, array_like
            Value we are using instead of self.value

        Returns
        -------
        float
        '''
        raise NeedToImplementError('User needs to implement this function')

    def cdf(self, value=None):
        '''Calculate the cdf with the specified value. If `value` is not
        specified, then use the current value

        Parameters
        ----------
        value : float, int, array_like
            Value we are using instead of self.value

        Returns
        -------
        float
        '''
        raise NeedToImplementError('User needs to implement this function')

    def logcdf(self, value=None):
        '''Calculate the logcdf with the specified value. If `value` is not
        specified, then use the current value

        Parameters
        ----------
        value : float, int, array_like
            Value we are using instead of self.value

        Returns
        -------
        float
        '''
        raise NeedToImplementError('User needs to implement this function')


class Constant(Node, _BaseArithmeticClass):
    '''A value in the graph that does not change.
    This can be a scalar or a matrix. We can manually override the value
    by calling `override_value`. We do this for model safety.

    Parameters
    ----------
    value : any
        This is the value of the constant
    kwargs : dict
        These are the extra arguments for the Node class
    '''
    def __init__(self, value, **kwargs):
        self._value = value
        Node.__init__(self, **kwargs)

    def override_value(self, val):
        '''Override the value with `val`

        Parameters
        ----------
        val : any
            Value to override self.value with
        '''
        self._value = val

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        raise UndefinedError('{} - Cannot change the value of a constant. If you ' \
            'want to change the value, then you must call the function ' \
            '`override_value` explicitly'.format(self.name))


class Variable(Node, _BaseArithmeticClass, Traceable):
    '''Scalar values that can change over time and be traced

    Parameters
    ----------
    value : int, float, None
        This is the value of the scalar
    dtype : Type
        This is the type of the variable
    kwargs :dict
        These are the extra parameters for the Node class
    '''
    def __init__(self, value=None, dtype=None, shape=None, **kwargs):
        Node.__init__(self, **kwargs)
        if dtype is None:
            dtype = DEFAULT_VARIABLE_TYPE
        if not istype(dtype):
            raise TypeError('`dtype` ({}) must be a type object'.format(type(dtype)))
        if shape is not None:
            if not istuple(shape):
                raise TypeError('`shape` ({}) must be a tuple or None')
        self.dtype = dtype
        self.value = value
        self._shape = shape
        self._init_value = None

    def __len__(self):
        if self._shape is None:
            raise ValueError('`No length for scalar`')
        return self._shape[0]

    def set_trace(self):
        '''Initialize the trace arrays for the variable in the Tracer object. 

        It will initialize a buffer the size of the checkpoint size in Tracer
        '''
        if self.G.inference.tracer_filename is not None:
            self.G.tracer.set_trace(self.name, shape=self._shape, dtype=self.dtype)
            ckpt = self.G.tracer.ckpt
        else:
            ckpt = self.G.inference.n_samples + self.G.inference.burnin
        
        self.ckpt_iter = 0
        self.sample_iter = 0
        shape = (ckpt, )
        if self._shape is not None:
            shape += self._shape
        self.trace = np.full(shape=shape, fill_value=np.nan, dtype=self.dtype)

    def add_trace(self):
        '''Adds the current value to the trace. If the buffer is full
        it will end it to disk
        '''
        try:
            self.trace[self.ckpt_iter] = self.value
        except:
            logging.critical('{} - trace shape ({}), value shape ({})'.format(
                self.name, self.trace[self.ckpt_iter].shape, self.value.shape))
            raise
        self.ckpt_iter += 1
        self.sample_iter += 1
        if self.ckpt_iter == len(self.trace):
            if self.G.inference.ckpt is None:
                # No writing to disk
                return
            # We have gotten the largest we can in the local buffer, write to disk
            self.G.tracer.write_to_disk(name=self.name)
            shape = (self.G.tracer.ckpt, )
            if self._shape is not None:
                shape += self._shape
            self.trace = np.full(shape=shape, fill_value=np.nan, dtype=self.dtype)
            self.ckpt_iter = 0

        if self.ckpt_iter > len(self.trace):
            raise ValueError('Iteration {} too long for RAM trace {}'.format(self.ckpt_iter, 
                len(self.trace)))

    def set_value_shape(self, shape):
        '''Set the shape
        
        Parameters
        ----------
        shape : tuple
        '''
        if not istuple(shape):
            raise TypeError('`shape` ({}) must be a tuple'.format(type(shape)))
        self._shape = shape

    @property
    def T(self):
        '''Transpose
        '''
        return self.value.T


class Normal(Variable, _RandomBase):
    '''Scalar normal variable parameterized by the mean and variance

    Parameters
    ----------
    mean : float, int
        This is the mean of the distribution
    var : float, int
        This is the variance of the distribution
    kwargs : dict
        These are extra parameters for the Node class
    '''
    def __init__(self, mean=None, var=None, **kwargs):
        Variable.__init__(self, **kwargs)

        # Wrap parameters in nodes
        if not isnode(mean):
            self._mean = Variable(
                value=mean,
                name=self.name + DEFAULT_NORMAL_MEAN_SUFFIX,
                G=self.G)
        else:
            self._mean = mean
        if not isnode(var):
            self._var = Variable(
                value=var,
                name=self.name + DEFAULT_NORMAL_VAR_SUFFIX,
                G=self.G)
        else:
            self._var = var

        # Set graph with parents
        self.add_parent(self._mean)
        self.add_parent(self._var)

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        return self._var

    @property
    def std(self):
        return np.sqrt(self._var.value)

    def mode(self):
        return self._mean.value

    def sample(self, size=None):
        '''Sample the distirbution given `self.mean` and `self.var`

        Parameters
        ----------
        size : int, None, Optional
            How many samples the pull

        Returns
        -------
        float
        '''
        self.value = random.normal.sample(
            mean=self._mean.value, 
            std=np.sqrt(self._var.value),
            size=size)
        return self.value

    def pdf(self, value=None):
        '''Calculate the pdf given `self.value`, `self.mean`, and `self.var`.
        If `value` is provided we use `value` instead of `self.value`.

        Parameters
        ----------
        value : float, int, None
            Value to calculate from instead of `self.value`

        Returns
        -------
        float
        '''
        if value is None:
            value = self.value
        return random.normal.pdf(value=value, mean=self._mean.value,
            std=np.sqrt(self._var.value))

    def logpdf(self, value=None):
        '''Calculate the logpdf given `self.value`, `self.mean`, and `self.var`.
        If `value` is provided we use `value` instead of `self.value`.

        Parameters
        ----------
        value : float, int, None
            Value to calculate from instead of `self.value`

        Returns
        -------
        float
        '''
        if value is None:
            value = self.value
        return random.normal.logpdf(value=value, mean=self._mean.value,
            std=np.sqrt(self._var.value))

    def cdf(self, value=None):
        '''Calculate the cdf given `self.value`, `self.mean`, and `self.var`.
        If `value` is provided we use `value` instead of `self.value`.

        Parameters
        ----------
        value : float, int, None
            Value to calculate from instead of `self.value`

        Returns
        -------
        float
        '''
        if value is None:
            value = self.value
        return random.normal.cdf(value=value, mean=self._mean.value,
            std=np.sqrt(self._var.value))

    def logcdf(self, value=None):
        '''Calculate the logcdf given `self.value`, `self.mean`, and `self.var`.
        If `value` is provided we use `value` instead of `self.value`.

        Parameters
        ----------
        value : float, int, None
            Value to calculate from instead of `self.value`

        Returns
        -------
        float
        '''
        if value is None:
            value = self.value
        return random.normal.logcdf(value=value, mean=self._mean.value,
            std=np.sqrt(self._var.value))


class Lognormal(Variable, _RandomBase):
    '''Lognormal distribution

    Parameters
    ----------
    mean : numeric, array
        This is the mean of the distribution
    std : numeric array
        This is the standard deviation of the array
    kwargs
    '''
    def __init__(self, mean, std, **kwargs):
        Variable.__init__(self, **kwargs)

        # Wrap parameters in nodes
        if not isnode(mean):
            self._mean = Variable(
                value=mean,
                name=self.name + DEFAULT_NORMAL_MEAN_SUFFIX,
                G=self.G)
        else:
            self._mean = mean
        if not isnode(std):
            self._std = Variable(
                value=std,
                name=self.name + DEFAULT_LOGNORMAL_STD_SUFFIX,
                G=self.G)
        else:
            self._std = std

        # Set graph with parents
        self.add_parent(self._mean)
        self.add_parent(self._std)

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    def mode(self):
        return self._mean.value

    def sample(self, size=None):
        '''Sample the distirbution given `self.mean` and `self.var`

        Parameters
        ----------
        size : int, None, Optional
            How many samples the pull

        Returns
        -------
        float
        '''
        self.value = random.lognormal.sample(
            mean=self._mean.value, 
            std=self._std.value,
            size=size)
        return self.value

    def pdf(self, value=None):
        '''Calculate the pdf given `self.value`, `self.mean`, and `self.var`.
        If `value` is provided we use `value` instead of `self.value`.

        Parameters
        ----------
        value : float, int, None
            Value to calculate from instead of `self.value`

        Returns
        -------
        float
        '''
        if value is None:
            value = self.value
        return random.lognormal.pdf(value=value, mean=self._mean.value,
            std=self._std.value)

    def logpdf(self, value=None):
        '''Calculate the logpdf given `self.value`, `self.mean`, and `self.var`.
        If `value` is provided we use `value` instead of `self.value`.

        Parameters
        ----------
        value : float, int, None
            Value to calculate from instead of `self.value`

        Returns
        -------
        float
        '''
        if value is None:
            value = self.value
        return random.lognormal.logpdf(value=value, mean=self._mean.value,
            std=self._std.value)

    def cdf(self, value=None):
        '''Calculate the cdf given `self.value`, `self.mean`, and `self.var`.
        If `value` is provided we use `value` instead of `self.value`.

        Parameters
        ----------
        value : float, int, None
            Value to calculate from instead of `self.value`

        Returns
        -------
        float
        '''
        if value is None:
            value = self.value
        return random.lognormal.cdf(value=value, mean=self._mean.value,
            std=self._std.value)

    def logcdf(self, value=None):
        '''Calculate the logcdf given `self.value`, `self.mean`, and `self.var`.
        If `value` is provided we use `value` instead of `self.value`.

        Parameters
        ----------
        value : float, int, None
            Value to calculate from instead of `self.value`

        Returns
        -------
        float
        '''
        if value is None:
            value = self.value
        return random.lognormal.logcdf(value=value, mean=self._mean.value,
            std=self._std.value)


class Uniform(Variable, _RandomBase):
    '''Scalar Uniform variable parameterized by the low and high

    Parameters
    ----------
    low : float, int
        This is the low of the distribution
    high : float, int
        This is the high of the distribution
    kwargs : dict
        These are extra parameters for the Node class
    '''
    def __init__(self, low=None, high=None, **kwargs):
        Variable.__init__(self, **kwargs)
        # Wrap parameters in nodes
        if not isnode(low):
            self._low = Variable(
                value=low,
                name=self.name+DEFAULT_UNIFORM_LOW_SUFFIX,
                G=self.G)
        else:
            self._low = low
        if not isnode(high):
            self._high = Variable(
                value=high,
                name=self.name + DEFAULT_UNIFORM_HIGH_SUFFIX,
                G=self.G)
        else:
            self._high = high

        self.add_parent(self._low)
        self.add_parent(self._high)

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    def mean(self):
        return 0.5*(self._low.value + self._high.value)

    def median(self):
        return self.mean()

    def var(self):
        return (1/12)*(self._high.value - self._low.value)**2

    def variance(self):
        return self.var()

    def sample(self, size=None):
        '''Sample the distirbution given `self.low` and `self.high`

        Parameters
        ----------
        size : int, None, Optional
            How many samples the pull

        Returns
        -------
        float
        '''
        self.value = random.uniform.sample(
            low=self._low.value, 
            high=self._high.value,
            size=size)
        return self.value

    def pdf(self, value=None):
        '''Calculate the pdf given `self.value`, `self.low`, and `self.high`.
        If `value` is provided we use `value` instead of `self.value`.

        Parameters
        ----------
        value : float, int, None
            Value to calculate from instead of `self.value`

        Returns
        -------
        float
        '''
        if value is None:
            value = self.value
        return random.uniform.pdf(value=value, low=self._low.value,
            high=self._high.value)

    def logpdf(self, value=None):
        '''Calculate the logpdf given `self.value`, `self.low`, and `self.high`.
        If `value` is provided we use `value` instead of `self.value`.

        Parameters
        ----------
        value : float, int, None
            Value to calculate from instead of `self.value`

        Returns
        -------
        float
        '''
        if value is None:
            value = self.value
        return random.uniform.logpdf(value=value, low=self._low.value,
            high=self._high.value)

    def cdf(self, value=None):
        '''Calculate the cdf given `self.value`, `self.low`, and `self.high`.
        If `value` is provided we use `value` instead of `self.value`.

        Parameters
        ----------
        value : float, int, None
            Value to calculate from instead of `self.value`

        Returns
        -------
        float
        '''
        if value is None:
            value = self.value
        return random.uniform.cdf(value=value, low=self._low.value,
            high=self._high.value)

    def logcdf(self, value=None):
        '''Calculate the logcdf given `self.value`, `self.low`, and `self.high`.
        If `value` is provided we use `value` instead of `self.value`.

        Parameters
        ----------
        value : float, int, None
            Value to calculate from instead of `self.value`

        Returns
        -------
        float
        '''
        if value is None:
            value = self.value
        return random.uniform.logcdf(value=value, low=self._low.value,
            high=self._high.value)


class TruncatedNormal(Variable, _RandomBase):
    '''Scalar truncated normal variable parameterized by the mean, variance,
    low, and high

    Parameters
    ----------
    mean : float, int
        This is the mean of the distribution
    var : float, int
        This is the variance of the distribution
    low : float, int
        This is the lowest value that can be sampled
    high : float, int
        This is the highest value that can be sampled
    kwargs : dict
        These are extra parameters for the Node class
    '''
    def __init__(self, mean, var, low=None, high=None, **kwargs):
        Variable.__init__(self, **kwargs)

        if low is None:
            low = float('-inf')
        if high is None:
            high = float('inf')
        if low > high:
            raise ValueError('`low` ({}) cannot be larger than `high` ({})'.format(
                low,high))

        # Wrap parameters in nodes
        if not isnode(mean):
            self._mean = Variable(
                value=mean,
                name=self.name + DEFAULT_NORMAL_MEAN_SUFFIX,
                G=self.G)
        else:
            self._mean = mean
        if not isnode(var):
            self._var = Variable(
                value=var,
                name=self.name + DEFAULT_NORMAL_VAR_SUFFIX,
                G=self.G)
        else:
            self._var = var

        self.low=low
        self.high=high

        # Set graph with parents
        self.add_parent(self._mean)
        self.add_parent(self._var)

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        return self._var

    @property
    def std(self):
        return np.sqrt(self._var.value)

    def mode(self):
        if self.low <= self.mean and self.mean <= self.high:
            return self.mean
        elif self.mean < self.low:
            return self.low
        else:
            return self.high

    def sample(self, size=None):
        '''Sample the distirbution given `self.mean`, `self.var`, 
        `self.low`, and `self.high`

        Parameters
        ----------
        size : int, None, Optional
            How many samples the pull

        Returns
        -------
        float
        '''
        self.value = random.truncnormal.sample(
            low=self.low,
            high=self.high,
            mean=self._mean.value,
            std=np.sqrt(self._var.value),
            size=size)

        return self.value

    def pdf(self, value=None):
        '''Calculate the pdf given `self.value`, `self.mean`, `self.var`, 
        `self.low` and `self.high`. If `value` is provided we use `value` 
        instead of `self.value`.

        Parameters
        ----------
        value : float, int, None
            Value to calculate from instead of `self.value`

        Returns
        -------
        float
        '''
        if value is None:
            value = self.value
        return random.truncnormal.pdf(value=value, mean=self._mean.value,
            std=np.sqrt(self._var.value), low=self.low, high=self.high)

    def logpdf(self, value=None):
        '''Calculate the logpdf given `self.value`, `self.mean`, `self.var`, 
        `self.low` and `self.high`. If `value` is provided we use `value` 
        instead of `self.value`.

        Parameters
        ----------
        value : float, int, None
            Value to calculate from instead of `self.value`

        Returns
        -------
        float
        '''
        if value is None:
            value = self.value
        return random.truncnormal.logpdf(value=value, mean=self._mean.value,
            std=np.sqrt(self._var.value), low=self.low, high=self.high)

    def cdf(self, value=None):
        '''Calculate the cdf given `self.value`, `self.mean`, `self.var`, 
        `self.low` and `self.high`. If `value` is provided we use `value` 
        instead of `self.value`.

        Parameters
        ----------
        value : float, int, None
            Value to calculate from instead of `self.value`

        Returns
        -------
        float
        '''
        if value is None:
            value = self.value
        return random.truncnormal.cdf(value=value, mean=self._mean.value,
            std=np.sqrt(self._var.value), low=self.low, high=self.high)

    def logcdf(self, value=None):
        '''Calculate the logcdf given `self.value`, `self.mean`, `self.var`, 
        `self.low` and `self.high`. If `value` is provided we use `value` 
        instead of `self.value`.

        Parameters
        ----------
        value : float, int, None
            Value to calculate from instead of `self.value`

        Returns
        -------
        float
        '''
        if value is None:
            value = self.value
        return random.truncnormal.logcdf(value=value, mean=self._mean.value,
            std=np.sqrt(self._var.value), low=self.low, high=self.high)


class SICS(Variable, _RandomBase):
    '''Scaled Inverse Chi Square parameterized by degrees of freedom `dof` and 
    scale `scale`.
    
    Parameters
    ----------
    dof : float, int
        This is the dof of the distribution
    scale : float, int
        This is the scale of the distribution
    kwargs : dict
        These are extra parameters for the Node class
    '''
    def __init__(self, dof=None, scale=None, **kwargs):
        Variable.__init__(self, **kwargs)

        # Wrap parameters in nodes
        if not isnode(dof):
            self._dof = Variable(
                value=dof,
                name=self.name + DEFAULT_SICS_DOF_SUFFIX,
                G=self.G)
        else:
            self._dof = dof
        if not isnode(scale):
            self._scale = Variable(
                value=scale,
                name=self.name + DEFAULT_SICS_SCALE_SUFFIX,
                G=self.G)
        else:
            self._scale = scale

        # Set graph with parents
        self.add_parent(self._dof)
        self.add_parent(self._scale)

    @property
    def dof(self):
        return self._dof

    @property
    def scale(self):
        return self._scale

    def mean(self):
        return self._dof.value * self._scale.value/ \
            (self._dof.value-2)

    def mode(self):
        return self._dof.value * self._scale.value / \
            (self._dof.value + 2)

    def sample(self, size=None):
        '''Sample from a SICS distribution parameerized by the current
        values of `scale` and `dof`.

        Parameters
        ----------
        size : int, None
            This is how many samples to draw. If None then we only draw a 
            single sample.

        Returns
        -------
        float, np.ndarray(float)
        '''
        self.value = random.sics.sample(
            dof=self._dof.value,
            scale=self._scale.value,
            size=size)
        return self.value

    def pdf(self, value=None):
        '''Calculate the pdf given the internal state. If `value` is defined,
        calculate the pdf with the passed in value instead of `self.value`.

        Parameters
        ----------
        value : numeric
            Value to calculate the pdf of instead of `self.value`

        Returns
        -------
        float
        '''
        if value is None:
            value = self.value
        return random.sics.pdf(value=value,
            dof=self._dof.value, scale=self._scale.value)

    def logpdf(self, value=None):
        '''Calculate the logpdf given the internal state. If `value` is defined,
        calculate the logpdf with the passed in value instead of `self.value`.

        Parameters
        ----------
        value : numeric
            Value to calculate the logpdf of instead of `self.value`

        Returns
        -------
        float
        '''
        if value is None:
            value = self.value
        return random.sics.logpdf(value=value,
            dof=self._dof.value, scale=self._scale.value)


class Gamma(Variable, _RandomBase):
    '''Gamma Distribution but for multiple values

    `shape` is a pylab.variables.Variable, `scale` is a
    pylab.variables.Variable
    '''
    def __init__(self, shape, scale, **kwargs):
        Variable.__init__(self, **kwargs)

        # Wrap parameters in nodes
        if not isnode(shape):
            self._shape_var = Variable(
                value=shape,
                name=self.name + DEFAULT_GAMMA_SHAPE_SUFFIX,
                G=self.G)
        else:
            self._shape_var = shape
        if not isnode(scale):
            self._scale = Variable(
                value=scale,
                name=self.name + DEFAULT_GAMMA_SCALE_SUFFIX,
                G=self.G)
        else:
            self._scale = scale

        if isarray(shape):
            self._shape = np.asarray(shape).shape

        # Set graph with parents
        self.add_parent(self._shape_var)
        self.add_parent(self._scale)

    @property
    def shape(self):
        return self._shape_var

    @property
    def scale(self):
        return self._scale

    def mean(self):
        return self._shape_var.value * self._scale.value

    def variance(self):
        return self._shape_var.value * (self._scale.value) ** 2

    def sample(self, size=None):
        self.value = random.gamma.sample(
            shape=self._shape_var.value,
            scale=self._scale.value,
            size=size)
        return self.value

    def pdf(self, value=None):
        '''Calculate the pdf given `self.value`, `self.shape`, and `self.scale`. 
        If `value` is provided we use `value` instead of `self.value`.

        Parameters
        ----------
        value : float, int, None
            Value to calculate from instead of `self.value`

        Returns
        -------
        float
        '''
        if value is None:
            value = self.value
        return random.gamma.pdf(value=value, 
            shape=self._shape_var.value, scale=self._scale.value)


class InvGamma(Variable, _RandomBase):
    def __init__(self, shape, scale, **kwargs):
        Variable.__init__(self, **kwargs)

        # Wrap parameters in nodes
        if not isnode(shape):
            self._shape_var = Variable(
                value=shape,
                name=self.name + DEFAULT_GAMMA_SHAPE_SUFFIX,
                G=self.G)
        else:
            self._shape_var = shape
        if not isnode(scale):
            self._scale = Variable(
                value=scale,
                name=self.name + DEFAULT_GAMMA_SCALE_SUFFIX,
                G=self.G)
        else:
            self._scale = scale

        if isarray(shape):
            self._shape = np.asarray(shape).shape

        # Set graph with parents
        self.add_parent(self._shape_var)
        self.add_parent(self._scale)

    @property
    def shape(self):
        return self._shape_var

    @property
    def scale(self):
        return self._scale

    def mean(self):
        if self._shape_var.value <= 1:
            raise MathError('Mean for InvGamma is undefined for shape <= 1 ' \
                '({})'.format(self._shape_var.value))
        return self._scale.value/(self._shape_var.value-1)

    def mode(self):
        return self._scale.value/(self._shape_var.value + 1)

    def variance(self):
        if self._scale.value <= 2:
            raise MathError('Variance for InvGamma is undefined for ' \
                'scale <= 2 ({})'.format(self._scale.value))
        return (self._scale.value**2)/ \
            ((self._shape_var.value - 1)**2) * (self._shape_var.value - 2)

    def sample(self, size=None):
        self.value = random.invgamma.sample(
            shape=self._shape_var.value,
            scale=self._scale.value,
            size=size)
        return self.value

    def pdf(self, value=None):
        if value is None:
            value = self.value
        return random.invgamma.pdf(value=value, 
            shape=self._shape_var.value,
            scale=self._scale.value)

    def logpdf(self, value=None):
        if value is None:
            value = self.value
        return random.invgamma.logpdf(value=value, 
            shape=self._shape_var.value,
            scale=self._scale.value)


class NegativeBinomial(Variable, _RandomBase):
    def __init__(self, n, p, **kwargs):
        raise NotImplementedError('Need to switch to the mean and variance parameterization')
        Variable.__init__(self, **kwargs)
        # Wrap parameters in nodes
        if not isnode(p):
            self._p = Variable(
                value=p,
                name=self.name + '_p',
                G=self.G)
        else:
            self._p = p
        self.add_parent(self._p)
        if not isnode(n):
            self._r = Variable(
                value=n,
                name=self.name + '_n',
                G=self.G)
        else:
            self._n = n
        self.add_parent(self._n)

    @property
    def p(self):
        return self._p

    @property
    def n(self):
        return self._n

    def mean(self):
        return self._p.value * self._r.value / (1 - self._p.value)

    def mode(self):
        if self._r.value > 1:
            # cast as int to round down
            p = self._p.value
            r = self._r.value
            return int(p*(r-1)/(1-p))
        else:
            return 0

    def variance(self):
        p = self._p.value
        r = self._r.value
        return p*r/((1-p)**2)

    def sample(self, size=None):
        self.value = npr.negative_binomial(
            n=self._n.value,
            p=self._p.value,
            size=size)
        return self.value


class Bernoulli(Variable, _RandomBase):

    def __init__(self, p, **kwargs):

        Variable.__init__(self, **kwargs)

        # Wrap parameters in nodes
        if not isnode(p):
            self._p = Variable(
                value=p,
                name=self.name + DEFAULT_BERNOULLI_P_SUFFIX,
                G=self.G)
        else:
            self._p = p

        self.add_parent(self._p)

    @property
    def p(self):
        return self._p


    def mean(self):
        return self._p

    def variance(self):
        return (1 - self._p.value) * self._p.value

    def mode(self):
        return int(self._p.value >= 0.5)

    def sample(self, size=None):
        return random.bernoulli(p=self._p.value, size=size)


class Beta(Variable, _RandomBase):

    def __init__(self, a, b, **kwargs):
        Variable.__init__(self, **kwargs)

        # Wrap parameters in nodes
        if not isnode(a):
            self._a = Variable(
                value=a,
                name=self.name + DEFAULT_BETA_ALPHA_SUFFIX,
                G=self.G)
        else:
            self._a = a
        if not isnode(b):
            self._b = Variable(
                value=b,
                name=self.name + DEFAULT_BETA_BETA_SUFFIX,
                G=self.G)
        else:
            self._b = b

        self.add_parent(self._a)
        self.add_parent(self._b)

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    def mean(self):
        '''E[X] = a/(a+b)
        '''
        return self._a.value / (self._a.value + self._b.value)

    def variance(self):
        '''var[X] = (a*b)/((a+b)**2 * (a+b+1))
        '''
        return self._a.value * self._b.value / ((self._a.value * \
            self._b.value)**2 * (self._a.value+self._b.value+1))

    def sample(self,size=None):
        self.value = random.beta.sample(a=self._a.value, b=self._b.value, size=size)
        return self.value


class MVN(Variable, _RandomBase):

    def __init__(self, mean=None, cov=None, **kwargs):
        Variable.__init__(self, **kwargs)

        # Wrap parameters in nodes
        if not isnode(mean):
            self._mean = Variable(
                value=mean,
                name=self.name + DEFAULT_MVN_MEAN_SUFFIX,
                G=self.G)
        else:
            self._mean = mean
        if not isnode(cov):
            self._cov = Variable(
                value=cov,
                name=self.name + DEFAULT_MVN_COV_SUFFIX,
                G=self.G)
        else:
            self._cov = cov

        self.add_parent(self._mean)
        self.add_parent(self._cov)

    @property
    def mean(self):
        return self._mean

    @property
    def cov(self):
        return self._cov

    def mode(self):
        return self._mean.value

    def prec(self):
        return np.linalg.pinv(self._cov.value)

    def sample(self,idxs=None,size=None):
        '''
        Sample all indices or only sample at the target indices (idxs) and
        set NaN to everything else. You would want to do this if you are
        using indicator variables and you do not want to sample certain
        values

        Parameters

        idxs (array, Optional)
            - If `idxs` is specified, only set values for those indices specified
            - If an index is not specified, set it to nan
            - If nothing is sent in, it sets the values to all
        '''
        try:
            if idxs is None:
                self.value = random.multivariate_normal.sample(
                    mean=self._mean.value, cov=self._cov.value, size=size)
            else:
                self.value = np.empty(shape=self.value.shape)
                self.value.fill(np.nan)
                self.value[idxs] = random.multivariate_normal.sample(
                    mean=self._mean.value, cov=self._cov.value, size=size)
        except RuntimeWarning:
            logging.critical('covariance is not positive semi-definate')
            print('mean\n',np.squeeze(self._mean.value))
            print('cov\n',self._cov.value)
            raise

        return self.value
