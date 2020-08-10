import numpy as np
import scipy.sparse
import scipy
import scipy.linalg
# import torch

import sys
import logging
import os

import matplotlib.pyplot as plt
import math

from .cluster import isclustering
from .util import count_calls, issquare, inspect_trace
from .errors import MathError

# Constants
ERROR_TOLERANCE_FORCE_SYMMETRY = 1e-20

def log_det(M):
    '''Computes the log determinant using the cholesky decomposition trick

    Parameters
    ----------
    M : array_like((n,n))
        - Matrix we want to take the determinant of
    '''
    if scipy.sparse.issparse(M):
        M_ = M.toarray()
    else:
        M_ = M
    
    # if type(M_) == torch.Tensor:
    #     return torch.logdet(M_)
    # else:
    L = safe_cholesky(M_)
    return 2*np.sum(np.log(np.diag(L)))

# @inspect_trace(max_trace=None)
def safe_cholesky(M, jitter=False, save_if_crash=False):
    '''First try numpy, then do scipy

    Parameters
    ----------
    M : np.ndarray 2-dim
        Matrix to take the cholesky of
    jitter : bool
        If True, and the cholesky fails, then we add a small offset
        to the diagonal until it becomes numerically stable
    save_if_crash : bool
        If True, save the array if it fails
    '''
    if scipy.sparse.issparse(M):
        M = M.toarray()

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

class metrics:
    '''A class defining different metrics
    '''

    @staticmethod
    def RMSE(arr1, arr2, force_reshape=False):
        '''Root Mean Square Error between `arr1` and `arr2`

        Parameters
        ----------
        arr1, arr2 : np.ndarray, array_like
            These are the arrays that we are taking the RMSE of
        force_reshape : bool
            If True, we flatten both of the arrays and then take the RMSE.
            Else we do not do any reshaping

        Returns
        -------
        float
        '''
        arr1 = np.asarray(arr1)
        arr2 = np.asarray(arr2)
        if arr1.shape != arr2.shape:
            if not force_reshape:
                raise ValueError('arr1.shape ({}) != arr2.shape ({})'.format(
                    arr1.shape, arr2.shape))
            arr1 = arr1.ravel()
            arr2 = arr2.ravel()

            if len(arr1) != len(arr2):
                raise ValueError('len(arr1) ({}) != len(arr2) ({})'.format(
                    len(arr1), len(arr2)))
        return np.sqrt(np.nanmean((arr1 - arr2) ** 2))

    @staticmethod
    def variation_of_information(X, Y, n):
        '''Variation of information:

        .. math::
            VI(X,Y) = - \sum_{i,j} r_{ij} [\log(r_{ij}/p_i) + \log(r_{ij}/q_j)] \\
            q_j = |Y_j|/n \\
            p_i = |X_i|/n \\
            r_{ij} = |X_i \bigcap Y_j|/n

        Measures the entropy between two different cluster assignments. This implements 
        the metric described in [1]. Values range from [0, ln(n)]. This function is 
        invariant to the element id, cluster order, and the order of the clusters. This 
        is distinct from mutual information.

        Parameters
        ----------
        X,Y : list(list)
            Defines the clusters to compare

            Example:
                X = [[1,2,3,4], [5,6,7,8]]
                Y = [[1,2,3,4], [5,6,7,8]]
                >>> variation_of_information(X,Y,8)
                0.0

                X = [[0,1,3,2], [4,5,6,7]]
                Y = [[4,5,6,7], [0,1,2,3]]
                >>> variation_of_information(X,Y,8)
                0.0

                X = [[0,1] ,[2,3], [4,5,6,7]]
                Y = [[4,5,6,7], [0,1,2,3]]
                >>> variation_of_information(X,Y,8)
                0.34657359027997264

                X = [[0,1] ,[2,3,4,5,6,7]]
                Y = [[4,5,6,7], [0,1,2,3]]
                >>> variation_of_information(X,Y,8)
                0.8239592165010823

                X = [[0],[1],[2],[3],[4],[5],[6],[7]]
                Y = [[0, 1, 2, 3, 4, 5, 6, 7]]
                >>> variation_of_information(X,Y,8)
                2.0794415416798357

                >>> math.log(8)
                2.0794415416798357
        n : int
            Total number of elements in the cluster configuration. 
            Must be the same for both.

            Example:
                X = [[0,1,2,3,4,5]]
                Y = [[0,1,3], [2,4,5]]
                n = 6
        
        Returns
        -------
        float

        References
        ----------
        [1] Meila, M. (2007). Comparing clusterings-an information based distance. 
            Journal of Multivariate Analysis, 98, 873-895. 
            doi:10.1016/j.jmva.2006.11.013
        '''
        sigma = 0.0
        for x in X:
            p = len(x) / n
            for y in Y:
                q = len(y) / n
                r = len(set(x) & set(y)) / n
                if r > 0.0:
                    sigma += r * (math.log(r / p) + math.log(r / q))
        return abs(sigma)

    @staticmethod
    def PE(truth, predicted, fillvalue=None):
        '''Percent error between the truth and predicted. If 
        `truth=0`, then we substitute it with `fillvalue`

        Percent error is defined as follows:
            PE = | (truth-predicted) / truth |

        Paramters
        ---------
        truth : np.ndarray, numeric
            What the value is supposed to be
        predicted : np.ndarray, numeric
            What the value was predicted to be
        fillvalue : float, None, str
            If not None, we substitute 0s in the `truth`` array with this value.
            If `fillvalue = 'discard'`, then we do not compute on the 
            samples that are == 0.

        Returns
        -------
        float
        '''
        if fillvalue is not None:
            if fillvalue == 'discard':
                predicted = predicted[truth != 0]
                truth = truth[truth !=  0]
            else:
                truth[truth == 0] = fillvalue
        return np.mean(np.absolute((truth-predicted)/truth))

    @staticmethod
    def logPE(truth, predicted, fillvalue=None):
        '''Percent error between the truth and predicted in log space.

        If any of the values in truth or predicted are 0 or negative, we
        replace it with `fillvalue` if it is not None.

        Percent error is defined as follows:
            PE = | (log(truth)-log(predicted)) / log(truth) |

        Paramters
        ---------
        truth : np.ndarray, numeric
            What the value is supposed to be
        predicted : np.ndarray, numeric
            What the value was predicted to be
        fillvalue : float, None, str
            If not None, we substitute 0s and negative numbers in the 
            arrays this value.
            If `fillvalue = 'discard'`, then we do not compute on the 
            samples that are <= 0.

        Returns
        -------
        float
        '''
        truth = np.asarray(truth)
        predicted = np.asarray(predicted)
        if fillvalue is not None:
            if fillvalue == 'discard':
                mask = (truth > 0) & (predicted > 0)
                truth = truth[mask]
                predicted = predicted[mask]
            else:
                truth[truth <= 0] = fillvalue
                predicted[predicted <= 0] = fillvalue
        logtruth = np.log(truth)
        return np.mean(np.absolute((logtruth-np.log(predicted))/logtruth))

@count_calls(max_calls=None)
def force_symmetry(M, check=True):
    '''Forces symmetry for the input square matrix

    Parameters
    ----------
    M : np.ndarray((n,n))
        - Matrix to force the symmetry
    '''
    if check:
        if not issquare(M):
            raise ValueError('M ({}) is either not a square matrix or the ' \
                'the number of dimensions is not 2'.format(M.shape))
    M_ = (M + M.T)/2

    # Check if the forced symmetry is within tolerance
    # error = np.sum(np.absolute(M-M.T))
    # if error > ERROR_TOLERANCE_FORCE_SYMMETRY:
    #     raise ValueError('Forced symmetry too much error: {} > {}'.format(
    #         error, ERROR_TOLERANCE_FORCE_SYMMETRY))
    return M_
