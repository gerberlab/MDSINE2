'''Utility functions for the posterior
'''
import logging
import itertools
import numpy as np
import numba
import scipy.sparse
import scipy
import math
import random
import copy

from .names import STRNAMES, REPRNAMES
from . import pylab as pl

# @numba.jit(nopython=True, fastmath=True, cache=True)
def negbin_loglikelihood(k,m,dispersion):
    '''Loglikelihood - with parameterization in [1]

    Parameters
    ----------
    k : int
        Observed counts
    m : int
        Mean
    phi : float
        Dispersion

    Returns
    -------
    float
        Negative Binomial Log Likelihood

    References
    ----------
    [1] TE Gibson, GK Gerber. Robust and Scalable Models of Microbiome Dynamics. ICML (2018)
    '''
    r = 1/dispersion
    return math.lgamma(k+r) - math.lgamma(k+1) - math.lgamma(r) \
            + r * (math.log(r) - math.log(r+m)) + k * (math.log(m) - math.log(r+m))

@numba.jit(nopython=True, fastmath=True, cache=False)
def negbin_loglikelihood_MH_condensed(k,m,dispersion):
        '''
        Loglikelihood - with parameterization in [1] - but condensed (do not calculate stuff
        we do not have to)

        Parameters
        ----------
        k : int
            Observed counts
        m : int
            Mean
        phi : float
            Dispersion

        Returns
        -------
        float
            Negative Binomial Log Likelihood

        References
        ----------
        [1] TE Gibson, GK Gerber. Robust and Scalable Models of Microbiome Dynamics. ICML (2018)
        '''
        r = 1/dispersion
        rm = r+m
        return math.lgamma(k+r) - math.lgamma(r) \
            + r * (math.log(r) - math.log(rm)) + k * (math.log(m) - math.log(rm))

def negbin_loglikelihood_MH_condensed_not_fast(k,m,dispersion):
        '''
        Loglikelihood - with parameterization in [1] - but condensed (do not calculate stuff
        we do not have to). We use this function if `negbin_loglikelihood_MH_condensed` fails to
        compile, which can happen when doing jobs on the cluster

        Parameters
        ----------
        k : int
            Observed counts
        m : int
            Mean
        phi : float
            Dispersion

        Returns
        -------
        float
            Negative Binomial Log Likelihood

        References
        ----------
        [1] TE Gibson, GK Gerber. Robust and Scalable Models of Microbiome Dynamics. ICML (2018)
        '''
        r = 1/dispersion
        rm = r+m
        return math.lgamma(k+r) - math.lgamma(r) \
            + r * (math.log(r) - math.log(rm)) + k * (math.log(m) - math.log(rm))

def expected_n_clusters(G):
    '''Calculate the expected number of clusters given the number of ASVs

    Parameters
    ----------
    G : pl.Graph
        Graph object

    Returns
    -------
    int
        Expected number of clusters
    '''
    conc = G[STRNAMES.CONCENTRATION].prior.mean()
    return conc * np.log((G.data.n_asvs + conc) / conc)

def build_prior_covariance(G, cov, order, sparse=True, diag=False, cuda=False):
    '''Build basic prior covariance or precision for the variables
    specified in `order`

    Parameters
    ----------
    G : pylab.graph.Graph
        Graph to get the variables from
    cov : bool
        If True, build the covariance. If False, build the precision
    order : list(str)
        Which parameters to get the priors of
    sparse : bool
        If True, return as a sparse matrix
    diag : bool
        If True, returns the diagonal of the matrix. If this is True, it
        overwhelms the flag `sparse`
    cuda : bool
        If True, returns the array/matrix on the gpu (if there is one). Will not return 
        in sparse form - only dense.

    Returns
    -------
    arr : np.ndarray, scipy.sparse.dia_matrix, torch.DoubleTensor
        Prior covariance or precision matrix in either dense (np.ndarray) or
        sparse (scipy.sparse.dia_matrix) form
    '''
    n_asvs = G.data.n_asvs
    a = []
    for reprname in order:
        if reprname == REPRNAMES.GROWTH_VALUE:
            a.append(np.full(n_asvs, G[REPRNAMES.PRIOR_VAR_GROWTH].value))

        elif reprname == REPRNAMES.SELF_INTERACTION_VALUE:
            a.append(np.full(n_asvs, G[REPRNAMES.PRIOR_VAR_SELF_INTERACTIONS].value))

        elif reprname == REPRNAMES.CLUSTER_INTERACTION_VALUE:
            n_interactions = G[REPRNAMES.CLUSTER_INTERACTION_INDICATOR].num_pos_indicators
            a.append(np.full(n_interactions, G[REPRNAMES.PRIOR_VAR_INTERACTIONS].value))

        elif reprname == REPRNAMES.PERT_VALUE:
            for perturbation in G.perturbations:
                num_on = perturbation.indicator.num_on_clusters()
                a.append(np.full(
                    num_on,
                    perturbation.magnitude.prior.var.value))

        else:
            raise ValueError('reprname ({}) not recognized'.format(reprname))

    if len(a) == 1:
        arr = np.asarray(a[0])
    else:
        arr = np.asarray(list(itertools.chain.from_iterable(a)))
    if not cov:
        arr = 1/arr
    # if cuda:
    #     arr = torch.DoubleTensor(arr).to(_COMPUTE_DEVICE)
    if diag:
        return arr
    # if cuda:
    #     return torch.diag(arr)
    if sparse:
        return scipy.sparse.dia_matrix((arr,[0]), shape=(len(arr),len(arr))).tocsc()
    else:
        return np.diag(arr)

def build_prior_mean(G, order, shape=None, cuda=False):
    '''Builds the prior mean vector for all the variables in `order`.

    Parameters
    ----------
    G : pylab.grapg.Graph
        Graph to index the objects
    order : list
        list of objects to add the priors of. If the variable is the
        cluster interactions or cluster perturbations, then we assume the
        prior mean is a scalar and we set that value for every single value.
    shape : tuple, None
        Shape to cast the array into
    cuda : bool
        If True, returns the array/matrix on the gpu (if there is one)

    Returns
    -------
    np.ndarray, torch.DoubleTensor
    '''
    a = []
    for name in order:
        v = G[name]
        if v.id == REPRNAMES.GROWTH_VALUE:
            a.append(v.prior.mean.value * np.ones(G.data.n_asvs))
        elif v.id == REPRNAMES.SELF_INTERACTION_VALUE:
            a.append(v.prior.mean.value * np.ones(G.data.n_asvs))
        elif v.id == REPRNAMES.CLUSTER_INTERACTION_VALUE:
            a.append(
                np.full(
                    G[REPRNAMES.CLUSTER_INTERACTION_INDICATOR].num_pos_indicators,
                    v.prior.mean.value))
        elif v.id == REPRNAMES.PERT_VALUE:
            for perturbation in G.perturbations:
                a.append(np.full(
                    perturbation.indicator.num_on_clusters(),
                    perturbation.magnitude.prior.mean.value))
        else:
            raise ValueError('`name` ({}) not recognized'.format(name))
    if len(a) == 1:
        a = np.asarray(a[0])
    else:
        a = np.asarray(list(itertools.chain.from_iterable(a)))
    if shape is not None:
        a = a.reshape(*shape)
    # if cuda:
    #     a = torch.DoubleTensor(a).to(_COMPUTE_DEVICE)
    return a

def sample_categorical_log(log_p):
    '''Generate one sample from a categorical distribution with event
    probabilities provided in unnormalized log-space.

    Parameters
    ----------
    log_p : array_like
        logarithms of event probabilities, ***which need not be normalized***

    Returns
    -------
    int
        One sample from the categorical distribution, given as the index of that
        event from log_p.
    '''
    try:
        exp_sample = math.log(random.random())
        events = np.logaddexp.accumulate(np.hstack([[-np.inf], log_p]))
        events -= events[-1]
        return next(x[0]-1 for x in enumerate(events) if x[1] >= exp_sample)
    except:
        logging.critical('CRASHED IN `sample_categorical_log`:\nlog_p{}'.format(
            log_p))
        raise

def log_det(M, var):
    '''Computes pl.math.log_det but also saves the array if it crashes

    Parameters
    ----------
    M : nxn matrix (np.ndarray, scipy.sparse)
        Matrix to calculate the log determinant
    var : pl.variable.Variable subclass
        This is the variable that `log_det` was called from

    Returns
    -------
    np.ndarray
        Log determinant of matrix
    '''
    if scipy.sparse.issparse(M):
        M_ = np.zeros(shape=M.shape)
        M.toarray(out=M_)
        M = M_
    try:
        # if type(M) == torch.Tensor:
        #     return torch.inverse(M)
        # else:
        return pl.math.log_det(M)
    except:
        try:
            sample_iter = var.sample_iter
        except:
            sample_iter = None
        filename = 'crashes/logdet_error_iter{}_var{}pinv_{}.npy'.format(
            sample_iter, var.name, var.G.name)
        logging.critical('\n\n\n\n\n\n\n\nSaved array at "{}" - now crashing\n\n\n'.format(
                filename))
        os.makedirs('crashes/', exist_ok=True)
        np.save(filename, M)
        raise

def pinv(M, var):
    '''Computes np.linalg.pinv but it also saves the array that crashed it if
    it crashes.

    Parameters
    ----------
    M : nxn matrix (np.ndarray, scipy.sparse)
        Matrix to invert
    var : pl.variable.Variable subclass
        This is the variable that `pinv` was called from

    Returns
    -------
    np.ndarray
        Inverse of the matrix
    '''
    if scipy.sparse.issparse(M):
        M_ = np.zeros(shape=M.shape)
        M.toarray(out=M_)
        M = M_
    try:
        # if type(M) == torch.Tensor:
        #     return torch.inverse(M)
        # else:
        try:
            return np.linalg.pinv(M)
        except:
            try:
                return scipy.linalg.pinv(M)
            except:
                return scipy.linalg.inv(M)
    except:
        try:
            sample_iter = var.sample_iter
        except:
            sample_iter = None
        filename = 'crashes/pinv_error_iter{}_var{}pinv_{}.npy'.format(
            sample_iter, var.name, var.G.name)
        logging.critical('\n\n\n\n\n\n\n\nSaved array at "{}" - now crashing\n\n\n'.format(
                filename))
        os.makedirs('crashes/', exist_ok=True)
        np.save(filename, M)
        raise

# @numba.jit(nopython=True, fastmath=True, cache=True)
def prod_gaussians(means, variances):
    '''Product of Gaussians

    $\mu = [\mu_1, \mu_2, ..., \mu_n]$
    $\var = [\var_1, \var_2, ..., \var_3]$

    Means and variances must be in the same order.

    Parameters
    ----------
    means : np.ndarray
        All of the means
    variances : np.ndarray
        All of the means
    '''
    mu = means[0]
    var = variances[0]
    for i in range(1,len(means)):
        mu, var = _calc_params(mu1=mu, mu2=means[i], var1=var, var2=variances[i])
    return mu, var

# @numba.jit(nopython=True, fastmath=True, cache=True)
def _calc_params(mu1, mu2, var1, var2):
    v = var1+var2
    mu = ((var1*mu2) + (var2*mu1))/(v)
    var = (var1*var2)/v
    return mu,var

def _tricubic(x):
    y = np.zeros_like(x)
    idx = (x >= -1) & (x <= 1)
    y[idx] = np.power(1.0 - np.power(np.abs(x[idx]), 3), 3)
    return y

class Loess(object):
    '''LOESS - Locally Estimated Scatterplot Smoothing
    This module was created by João Paulo Figueira and copied from the 
    repository: https://github.com/joaofig/pyloess.git
    There are a few modifications, mostly to handle edge cases, i.e., what 
    happens when the entire thing is zero
    '''
    @staticmethod
    def normalize_array(array):
        min_val = np.min(array)
        max_val = np.max(array)
        return (array - min_val) / (max_val - min_val), min_val, max_val

    def __init__(self, xx, yy, degree=1):
        self.n_xx, self.min_xx, self.max_xx = self.normalize_array(xx)
        self.n_yy, self.min_yy, self.max_yy = self.normalize_array(yy)
        self.degree = degree

        self.input_yy = yy

    @staticmethod
    def get_min_range(distances, window):
        min_idx = np.argmin(distances)
        n = len(distances)
        if min_idx == 0:
            return np.arange(0, window)
        if min_idx == n-1:
            return np.arange(n - window, n)

        min_range = [min_idx]
        while len(min_range) < window:
            i0 = min_range[0]
            i1 = min_range[-1]
            if i0 == 0:
                min_range.append(i1 + 1)
            elif i1 == n-1:
                min_range.insert(0, i0 - 1)
            elif distances[i0-1] < distances[i1+1]:
                min_range.insert(0, i0 - 1)
            else:
                min_range.append(i1 + 1)
        return np.array(min_range)

    @staticmethod
    def get_weights(distances, min_range):
        max_distance = np.max(distances[min_range])
        weights = _tricubic(distances[min_range] / max_distance)
        return weights

    def normalize_x(self, value):
        return (value - self.min_xx) / (self.max_xx - self.min_xx)

    def denormalize_y(self, value):
        return value * (self.max_yy - self.min_yy) + self.min_yy

    def estimate(self, x, window, use_matrix=False, degree=1):
        n_x = self.normalize_x(x)
        distances = np.abs(self.n_xx - n_x)
        min_range = self.get_min_range(distances, window)
        weights = self.get_weights(distances, min_range)

        if use_matrix or degree > 1:
            wm = np.multiply(np.eye(window), weights)
            xm = np.ones((window, degree + 1))

            xp = np.array([[math.pow(n_x, p)] for p in range(degree + 1)])
            for i in range(1, degree + 1):
                xm[:, i] = np.power(self.n_xx[min_range], i)

            ym = self.n_yy[min_range]
            xmt_wm = np.transpose(xm) @ wm
            beta = np.linalg.pinv(xmt_wm @ xm) @ xmt_wm @ ym
            y = (beta @ xp)[0]
        else:
            xx = self.n_xx[min_range]
            yy = self.n_yy[min_range]
            sum_weight = np.sum(weights)
            sum_weight_x = np.dot(xx, weights)
            sum_weight_y = np.dot(yy, weights)
            sum_weight_x2 = np.dot(np.multiply(xx, xx), weights)
            sum_weight_xy = np.dot(np.multiply(xx, yy), weights)

            mean_x = sum_weight_x / sum_weight
            mean_y = sum_weight_y / sum_weight

            b = (sum_weight_xy - mean_x * mean_y * sum_weight) / \
                (sum_weight_x2 - mean_x * mean_x * sum_weight)
            a = mean_y - b * mean_x
            y = a + b * n_x
        
        ret = self.denormalize_y(y)
        if np.isnan(ret):
            if np.all(self.input_yy == 0):
                return 0
            else:
                raise ValueError('`Returning `np.nan`')
        return ret

def asvname_for_paper(asv, asvs):
    '''Makes the name in the format needed for the paper

    Parameters
    ----------
    asv : pylab.base.ASV
        This is the ASV we are making the name for
    asvs : pylab.base.ASVSet
        This is the ASVSet object that contains the ASV

    Returns
    -------
    '''
    if asv.tax_is_defined('species'):
        species = asv.taxonomy['species']
        species = species.split('/')
        if len(species) >= 3:
            species = species[:2]
        species = '/'.join(species)
        label = pl.asvname_formatter(
            format='%(genus)s {spec} %(name)s'.format(
                spec=species), 
            asv=asv, asvs=asvs)
    elif asv.tax_is_defined('genus'):
        label = pl.asvname_formatter(
            format='* %(genus)s %(name)s',
            asv=asv, asvs=asvs)
    elif asv.tax_is_defined('family'):
        label = pl.asvname_formatter(
            format='** %(family)s %(name)s',
            asv=asv, asvs=asvs)
    elif asv.tax_is_defined('order'):
        label = pl.asvname_formatter(
            format='*** %(order)s %(name)s',
            asv=asv, asvs=asvs)
    elif asv.tax_is_defined('class'):
        label = pl.asvname_formatter(
            format='**** %(class)s %(name)s',
            asv=asv, asvs=asvs)
    elif asv.tax_is_defined('phylum'):
        label = pl.asvname_formatter(
            format='***** %(phylum)s %(name)s',
            asv=asv, asvs=asvs)
    elif asv.tax_is_defined('kingdom'):
        label = pl.asvname_formatter(
            format='****** %(kingdom)s %(name)s',
            asv=asv, asvs=asvs)
    else:
        raise ValueError('Something went wrong - no taxnonomy: {}'.format(str(asv)))

    return label

def is_gram_negative(asv):
    '''Return true if the asv is gram - or gram positive
    '''
    if not asv.tax_is_defined('phylum'):
        return None
    elif asv.taxonomy['phylum'].lower() == 'bacteroidetes':
        return True
    elif asv.taxonomy['phylum'].lower() == 'firmicutes':
        return False
    elif asv.taxonomy['phylum'].lower() == 'verrucomicrobia':
        return True
    elif asv.taxonomy['phylum'].lower() == 'proteobacteria':
        return True
    else:
        raise ValueError('{} phylum not specified. If not bacteroidetes, firmicutes, verrucomicrobia, or ' \
            'proteobacteria, you must add another phylum'.format(str(asv)))

def is_gram_negative_taxa(taxa, taxalevel, asvs):
    '''Checks if the taxa `taxa` at the taxonomic level `taxalevel`
    is a gram negative or gram positive
    '''
    for asv in asvs:
        if asv.taxonomy[taxalevel] == taxa:
            return is_gram_negative(asv)

    else:
        raise ValueError('`taxa` ({}) not found at taxonomic level ({})'.format(
            taxa. taxalevel))

def generate_interation_bayes_factors_posthoc(mcmc, section='posterior'):
    '''Generates the bayes factors on an item-item level for the interactions, 
    given the passed in prior. All negative indicators are set as `np.nan`s in 
    the trace, so we do `~np.isnan` to get the indicators.

    Since the prior is conjugate, we can fully integrate out the prior of the 
    calculation:
        bf[i,j] = (count_on[i,j] * b) / (count_off[i,j] * a)

    Parameters
    ----------
    mcmc : mdsine2.BaseMCMC
        This is the inference object containing the traces
    section : str
        This is the section of the MH samples to take the samples

    Returns
    -------
    np.ndarray((n,n), dtpye=float)
        These are the bayes factors for each of the interactions on an item-item level
    '''
    interactions = mcmc.graph[STRNAMES.INTERACTIONS_OBJ]
    trace = interactions.get_trace_from_disk(section=section)
    trace = ~ np.isnan(trace)
    cnts_1 = np.sum(trace, axis=0)
    cnts_0 = np.sum(1-trace, axis=0)

    a = mcmc.graph[STRNAMES.INDICATOR_PROB].prior.a.value
    b = mcmc.graph[STRNAMES.INDICATOR_PROB].prior.a.value

    return (cnts_1 * b) / (cnts_0 * a)

def generate_perturbation_bayes_factors_posthoc(mcmc, perturbation, section='posterior'):
    '''Generates the bayes factors on an item-item level for the perturbations, 
    given the passed in prior. All negative indicators are set as `np.nan`s in 
    the trace, so we do `~np.isnan` to get the indicators.

    Since the prior is conjugate, we can fully integrate out the prior of the 
    calculation:
        bf[i,j] = (count_on[i] * b) / (count_off[i] * a)

    Parameters
    ----------
    mcmc : mdsine2.BaseMCMC
        This is the inference object containing the traces
    perturbation : mdsine2.Perturbation
        Perturbation object we are calculating from
    section : str
        This is the section of the MH samples to take the samples

    Returns
    -------
    np.ndarray((n,), dtpye=float)
        These are the bayes factors for each of the perturbations on an item level
    '''
    trace = ~ np.isnan(perturbation.get_trace_from_disk())
    cnts_1 = np.sum(trace, axis=0)
    cnts_0 = np.sum(1-trace, axis=0)

    a = perturbation.probability.prior.a.value
    b = perturbation.probability.prior.b.value

    return (cnts_1 * b) / (cnts_0 * a)

def generate_cluster_assignments_posthoc(clustering, n_clusters='mode', linkage='average',
    set_as_value=False, section='posterior'):
    '''Once the inference is complete, compute the clusters posthoc using
    sklearn's AgglomerativeClustering function with distance matrix being
    1 - cocluster matrix (we subtrace the cocluster matrix from 1 because
    the cocluster matrix describes similarity, not distance).

    Parameters
    ----------
    clustering : mdsine2.Clustering
        Clustering object
    n_clusters : str, int, callable, Optional
        This specifies the number of clusters that are used during
        Agglomerative clustering.
        If `n_clusters` is of type int, it will use that number as the number of
        clusters.
        If `n_clusters` is of type str, it calculates the number of clusters
        based on the trace for the number of clusters (clustering.n_clusters.trace).
        Possible calculation types are:
            * 'median', 'mode', and 'mean'.
        If `n_clusters` is callable, it will calculate n given the trace of n_clusters
        Default is 'mode'.
    linkage : str, Optional
        Which linkage criterion to use. Determines which distance to use
        between sets of observation. The AgglomerativeClustering algorithm
        will merge the pairs of cluster that minimize the linkage criterion.
        Possible types:
    set_as_value : bool
        If True then set the result as the value of the clustering object
    section : str
        What part of the chain to take the samples from
    
    Returns
    -------
    np.ndarray(size=(len(items), ), dtype=int)
        Each value is the cluster assignment for index i
    '''
    from sklearn.cluster import AgglomerativeClustering
    import scipy.stats


    trace = clustering.n_clusters.get_trace_from_disk(section=section)
    if callable(n_clusters):
        n = n_clusters(trace)
    elif type(n_clusters) == int:
        n = n_clusters
    elif type(n_clusters) == str:
        if n_clusters == 'mode':
            n = scipy.stats.mode(trace)[0][0]
        else:
            raise ValueError('`n_clusters` ({}) not recognized.'.format(n_clusters))
    else:
        raise ValueError('Type `n_clusters` ({}) not recognized. Must be of '\
            'type `str`, `int`, or callable.'.format(type(n_clusters)))
    if not pl.isbool(set_as_value):
        raise TypeError('`set_as_value` ({}) must be a bool'.format(type(set_as_value)))

    A = pl.summary(clustering.coclusters, section=section)['mean']
    A = 1 - A
    logging.info('Number of clusters: {}'.format(int(n)))
    c = AgglomerativeClustering(
        n_clusters=int(n),
        affinity='precomputed',
        linkage=linkage)
    ret = c.fit_predict(A)
    logging.info(ret)
    if set_as_value:
        ca = {}
        for idx, cidx in enumerate(ca):
            if cidx in ca:
                ca[cidx].append(idx)
            else:
                ca[cidx] = [idx]
        for cluster in ca:
            cid = clustering.make_new_cluster_with(idx=cluster[0])
            for oidx in cluster[1:]:
                clustering.move_item(idx=oidx, cid=cid)
    return ret

def aggregate_items(subjset, hamming_dist):
    '''Aggregate ASVs that have an average hamming distance of `hamming_dist`

    Parameters
    ----------
    subjset : mdsine2.Study
        This is the `mdsine2.Study` object that we are aggregating
    hamming_dist : int
        This is the hamming radius from one ASV to the next where we
        are aggregating

    Returns
    -------
    mdsine2.Study
    '''

    cnt = 0
    found = False
    iii = 0
    logging.info('Agglomerating asvs')
    while not found:
        for iii in range(iii, len(subjset.asvs)):
            if iii % 200 == 0:
                logging.info('{}/{}'.format(iii, len(subjset.asvs)))
            asv1 = subjset.asvs[iii]
            for asv2 in subjset.asvs.names.order[iii:]:
                asv2 = subjset.asvs[asv2]
                if asv1.name == asv2.name:
                    continue
                if len(asv1.sequence) != len(asv2.sequence):
                    continue

                dist = _avg_dist(asv1, asv2)
                if dist <= hamming_dist:
                    subjset.aggregate_items(asv1, asv2)
                    cnt += 1
                    found = True
                    break
            if found:
                break
        if found:
            found = False
        else:
            break
    logging.info('Aggregated {} asvs'.format(cnt))
    return subjset

def _avg_dist(asv1, asv2):
    dists = []
    if pl.isaggregatedasv(asv1):
        seqs1 = asv1.aggregated_seqs.values()
    else:
        seqs1 = [asv1.sequence]

    if pl.isaggregatedasv(asv2):
        seqs2 = asv2.aggregated_seqs.values()
    else:
        seqs2 = [asv2.sequence]

    for v1 in seqs1:
        for v2 in seqs2:
            dists.append(_hamming(v1, v2))
    return np.nanmean(dists)

def _hamming(s1,s2):
    result=0
    for i,j in zip(s1,s2):
        if i!=j:
            # print(f'char not math{i,j}in {x}')
            result+=1
    return result

def consistency_filtering(subjset, dtype, threshold, min_num_consecutive, min_num_subjects, 
    colonization_time=None, union_other_consortia=None):
    '''Filters the subjects by looking at the consistency of the 'dtype', which can
    be either 'raw' where we look for the minimum number of counts, 'rel', where we
    look for a minimum relative abundance, or 'abs' where we look for a minium 
    absolute abundance.

    There must be at least `threshold` for at least
    `min_num_consecutive` consecutive timepoints for at least
    `min_num_subjects` subjects for the ASV to be classified as valid.

    If a colonization time is specified, we only look after that timepoint

    Parameters
    ----------
    subjset : str, mdsine2.Study
        This is the Study object that we are doing the filtering on
        If it is a str, then it is the location of the saved object.
    dtype : str
        This is the string to say what type of data we are thresholding. Options
        are 'raw', 'rel', or 'abs'.
    threshold : numeric
        This is the threshold for either counts, relative abundance, or
        absolute abundance
    min_num_consecutive : int
        Number of consecutive timepoints to look for in a row
    colonization_time : numeric
        This is the time we are looking after for colonization. If None we assume 
        there is no colonization time.
    min_num_subjects : int, str
        This is the minimum number of subjects this needs to be valid for.
        If str, we accept 'all', which we set that automatically.
    union_other_consortia : mdsine2.Study, None
        If not None, take the union of the asvs passing the filtering of both
        `subjset` and `union_other_consortia`

    Returns
    -------
    mdsine2.Study
        This is the filtered subject set.

    Raises
    ------
    ValueError
        If types are not valid or values are invalid
    '''
    if not pl.isstr(dtype):
        raise TypeError('`dtype` ({}) must be a str'.format(type(dtype)))
    if dtype not in ['raw', 'rel', 'abs']:
        raise ValueError('`dtype` ({}) not recognized'.format(dtype))
    if not pl.isstudy(subjset):
        raise TypeError('`subjset` ({}) must be a mdsine2.Study'.format(
            type(subjset)))
    if not pl.isnumeric(threshold):
        raise TypeError('`threshold` ({}) must be a numeric'.format(type(threshold)))
    if threshold <= 0:
        raise ValueError('`threshold` ({}) must be > 0'.format(threshold))
    if not pl.isint(min_num_consecutive):
        raise TypeError('`min_num_consecutive` ({}) must be an int'.format(
            type(min_num_consecutive)))
    if min_num_consecutive <= 0:
        raise ValueError('`min_num_consecutive` ({}) must be > 0'.format(min_num_consecutive))
    if colonization_time is None:
        colonization_time = 0
    if not pl.isnumeric(colonization_time):
        raise TypeError('`colonization_time` ({}) must be a numeric'.format(
            type(colonization_time)))
    if colonization_time < 0:
        raise ValueError('`colonization_time` ({}) must be >= 0'.format(colonization_time))
    if min_num_subjects is None:
        min_num_subjects = 1
    if min_num_subjects == 'all':
        min_num_subjects = len(subjset)
    if not pl.isint(min_num_subjects):
        raise TypeError('`min_num_subjects` ({}) must be an int'.format(
            type(min_num_subjects)))
    if min_num_subjects > len(subjset) or min_num_subjects <= 0:
        raise ValueError('`min_num_subjects` ({}) value not valid'.format(min_num_subjects))
    if union_other_consortia is not None:
        if not pl.isstudy(union_other_consortia):
            raise TypeError('`union_other_consortia` ({}) must be a mdsine2.Study'.format(
                type(union_other_consortia)))

    subjset = copy.deepcopy(subjset)      
    
    if union_other_consortia is not None:
        asvs_to_keep = set()
        for subjset_temp in [subjset, union_other_consortia]:
            subjset_temp = consistency(subjset_temp, dtype=dtype,
                threshold=threshold, min_num_consecutive=min_num_consecutive,
                colonization_time=colonization_time, min_num_subjects=min_num_subjects,
                union_other_consortia=None)
            for asv_name in subjset_temp.asvs.names:
                asvs_to_keep.add(asv_name)
        to_delete = []
        for aname in subjset.asvs.names:
            if aname not in asvs_to_keep:
                to_delete.append(aname)
    else:
        # Everything is fine, now we can do the filtering
        talley = np.zeros(len(subjset.asvs), dtype=int)
        for i, subj in enumerate(subjset):
            matrix = subj.matrix()[dtype]
            tidx_start = None
            for tidx, t in enumerate(subj.times):
                if t >= colonization_time:
                    tidx_start = tidx
                    break
            if tidx_start is None:
                raise ValueError('Something went wrong')
            matrix = matrix[:, tidx_start:]

            for oidx in range(matrix.shape[0]):
                consecutive = 0
                for tidx in range(matrix.shape[1]):
                    if matrix[oidx,tidx] >= threshold:
                        consecutive += 1
                    else:
                        consecutive = 0
                    if consecutive >= min_num_consecutive:
                        talley[oidx] += 1
                        break

        invalid_oidxs = np.where(talley < min_num_subjects)[0]
        to_delete = subjset.asvs.ids.order[invalid_oidxs]
    subjset.pop_asvs(to_delete)
    return subjset

def conditional_consistency_filtering(subjset, other, dtype, threshold, min_num_consecutive_upper,  
    min_num_consecutive_lower, min_num_subjects, colonization_time):
    '''Filters the cohorts in `subjset` with the `mdsine2.consistency_filtering`
    filtering method but conditional on another cohort. If an ASV passes the filter
    in the cohort `other`, the ASV can only have `min_num_consecutive_lower` 
    consecutive timepoints instead of `min_num_consecutive_upper`. This potentially
    increases the overlap of the ASVs between cohorts, which is the reason why
    we would do this filtering over just `mdsine2.consistency_filtering`

    Algorithm
    ---------
    First, we apply the function `mdsine2.consistency_filtering` to the subjects
    `subjset` and `other` with minimum number of consectutive timepoints 
    `min_num_consecutive_upper` and `min_num_consecutive_lower`.

    For each ASV in cohort `subjset`
        If it passes filtering with `min_num_consecutive_upper` in `subjset`, it is included
        If it passes filtering with `min_num_consecutive_upper` in `other` AND it
            passes filtering  `min_num_consecutive_lower` in `subjset`, it is included.
        Otherwise it is excluded

    Parameters
    ----------
    subjset : str, mdsine2.Study
        This is the Study object that we are doing the filtering on
        If it is a str, then it is the location of the saved object.
    other : str, mdsine2.Study
        This is the other Study obejct that we are conditional on.
    dtype : str
        This is the string to say what type of data we are thresholding. Options
        are 'raw', 'rel', or 'abs'.
    threshold : numeric
        This is the threshold for either counts, relative abundance, or
        absolute abundance
    min_num_consecutive_upper, min_num_consecutive_lower : int
        Number of consecutive timepoints to look for in a row
    colonization_time : numeric
        This is the time we are looking after for colonization. If None we assume 
        there is no colonization time.
    min_num_subjects : int, str
        This is the minimum number of subjects this needs to be valid for.
        If str, we accept 'all', which we set that automatically.
    
    Returns
    -------
    mdsine2.Study
        This is the filtered subject set.

    See Also
    --------
    mdsine2.consistency_filtering
    '''
    # All of the checks are done within `consistency_filtering` so dont check here
    subjset_upper = consistency_filtering(subjset=copy.deepcopy(subjset), 
        dtype=dtype, threshold=threshold,
        min_num_consecutive=min_num_consecutive_upper, min_num_subjects=min_num_subjects, 
        colonization_time=colonization_time)
    subjset_lower = consistency_filtering(subjset=copy.deepcopy(subjset), 
        dtype=dtype, threshold=threshold,
        min_num_consecutive=min_num_consecutive_lower, min_num_subjects=min_num_subjects, 
        colonization_time=colonization_time)

    other_upper = consistency_filtering(subjset=copy.deepcopy(other), 
        dtype=dtype, threshold=threshold,
        min_num_consecutive=min_num_consecutive_upper, min_num_subjects=min_num_subjects, 
        colonization_time=colonization_time)

    # Conditional consistency filtering
    to_delete = []
    for asv in subjset_lower.asvs:
        if asv.name in subjset_upper.asvs:
            continue
        if asv.name in other_upper.asvs:
            continue
        to_delete.append(asv.name)

    subjset_lower.pop_asvs(to_delete)
    return subjset_lower