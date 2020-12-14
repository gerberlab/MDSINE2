'''Alpha and Beta diversity measures. This is heavily influenced by skbio. Acts as a wrapper
for the beta diversity measures

Defined Alpha diversity measures:
    * entropy
    * normalized entropy
    * shannon_entropy
Defined Beta diversity measures
    * Bray Curtis
    * Jaccard
    * Hamming
'''
import numpy as np
from scipy.spatial import distance

class alpha:
    @staticmethod
    def entropy(counts):
        '''Calculate the entropy
        
        Entropy is defined as
            E = - \sum_i (b_i * \log(b_i))
        where
            b_i is the relative abundance of the ith Taxon
        
        Parameters
        ----------
        counts (array_like)
            - Vector of counts

        Returns
        -------
        double
        '''
        counts = _validate_counts(counts)
        rel = counts[counts>0]
        rel = rel / np.sum(rel)

        a = rel * np.log(rel)
        a = -np.sum(a)
        return a

    @staticmethod
    def normalized_entropy(counts):
        '''Calculate the normailized entropy
        
        Entropy is defined as
            E = - \sum_i (b_i * \log_n(b_i))
        where
            b_i is the relative abundance of the ith Taxon
        
        Parameters
        ----------
        counts (array_like)
            - Vector of counts

        Returns
        -------
        double
        '''
        counts = _validate_counts(counts)
        rel = counts[counts>0]
        rel = rel / np.sum(rel)

        a = rel * np.log(rel)
        a = -np.sum(a) / np.log(len(rel))
        return a

    @staticmethod
    def shannon_entropy(counts, base=2):
        '''Calculates the Shannon entropy

        Based on the description given in the SDR-IV online manual [1] except that
        the default logarithm base used here is 2 instead of `e`.

        [1] http://www.pisces-conservation.com/sdrhelp/index.html

        Parameters
        ----------
        counts (array_like)
            - Vector of counts

        Returns
        -------
        double
        '''
        counts = _validate_counts(counts)
        freqs = counts/counts.sum()
        nonzero_freqs = freqs[freqs.nonzero()]
        return -(nonzero_freqs*np.log(nonzero_freqs)).sum()/np.log(base)


class beta:
    @staticmethod
    def braycurtis(u,v):
        return distance.braycurtis(u,v)

    @staticmethod
    def jaccard(u,v):
        return distance.jaccard(u,v)

    @staticmethod
    def hamming(u, v, ignore_char='N'):
        '''Calculate the hamming distance between `u` and `v`

        Parameters
        ----------
        u, v : iterable
            Iterable objects of the same size that we are comparing
        ignore_Ns : str, None
            If not None, then we ignore positions which are this character.
            This is used if we want to skip over positions that are not a consensus.

        Returns
        -------
        int
        '''
        if len(u) != len(v):
            raise ValueError('Cannot compare different distances')
        result=0
        for i,j in zip(u,v):
            if ignore_char is not None:
                if i == ignore_char or j == ignore_char:
                    continue
            if i!=j:
                result+=1
        return result

    @staticmethod
    def unifrac(*args,**kwargs):
        raise NotImplementedError('Not implemented')


# Utility functions
def _validate_counts(counts, cast_as_ints=True):
    '''Checks dimensions, wraps as np array, and casts values as ints
    if necessary

    Parameters
    ----------
    counts (array_like)
        - 1D data
    cast_as_ints (bool, Optional)
        - If True, it will cast the counts array as an int
        - If False it will not cast

    Returns
    -------
    np.ndarray
    '''

    counts = np.asarray(counts)
    if cast_as_ints:
        counts = counts.astype(int, copy=False)

    if counts.ndim != 1:
        raise ValueError('counts ({}) must be a single dimension'.format(
            counts.shape))
    if np.any(counts < 0):
        raise ValueError('counts must not have any negative values')
    return counts