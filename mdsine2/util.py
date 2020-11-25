'''Utility functions for mdsine2
'''
import logging
import numpy as np
import scipy
import math
import copy

from .names import STRNAMES
from . import pylab as pl
from . import diversity

def is_gram_negative(taxa):
    '''Return true if the taxa is gram - or gram positive
    '''
    if not taxa.tax_is_defined('phylum'):
        return None
    elif taxa.taxonomy['phylum'].lower() == 'bacteroidetes':
        return True
    elif taxa.taxonomy['phylum'].lower() == 'firmicutes':
        return False
    elif taxa.taxonomy['phylum'].lower() == 'verrucomicrobia':
        return True
    elif taxa.taxonomy['phylum'].lower() == 'proteobacteria':
        return True
    else:
        raise ValueError('{} phylum not specified. If not bacteroidetes, firmicutes, verrucomicrobia, or ' \
            'proteobacteria, you must add another phylum'.format(str(taxa)))

def is_gram_negative_taxa(taxa, taxalevel, taxas):
    '''Checks if the taxa `taxa` at the taxonomic level `taxalevel`
    is a gram negative or gram positive
    '''
    for taxa in taxas:
        if taxa.taxonomy[taxalevel] == taxa:
            return is_gram_negative(taxa)

    else:
        raise ValueError('`taxa` ({}) not found at taxonomic level ({})'.format(
            taxa, taxalevel))

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

    # print(cnts_1)
    # print()
    # print(cnts_0)

    a = mcmc.graph[STRNAMES.CLUSTER_INTERACTION_INDICATOR_PROB].prior.a.value
    b = mcmc.graph[STRNAMES.CLUSTER_INTERACTION_INDICATOR_PROB].prior.b.value

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
    logging.info('Clusters assigned: {}'.format(ret))
    if set_as_value:
        for cidx in range(np.max(ret)):
            cluster = np.where(ret == cidx)[0]
            cid = clustering.make_new_cluster_with(idx=cluster[0])
            for oidx in cluster[1:]:
                clustering.move_item(idx=oidx, cid=cid)
    return ret

def aggregate_items(subjset, hamming_dist):
    '''Aggregate Taxas that have an average hamming distance of `hamming_dist`

    Parameters
    ----------
    subjset : mdsine2.Study
        This is the `mdsine2.Study` object that we are aggregating
    hamming_dist : int
        This is the hamming radius from one Taxa to the next where we
        are aggregating

    Returns
    -------
    mdsine2.Study
    '''

    cnt = 0
    found = False
    iii = 0
    logging.info('Agglomerating taxas')
    while not found:
        for iii in range(iii, len(subjset.taxas)):
            if iii % 200 == 0:
                logging.info('{}/{}'.format(iii, len(subjset.taxas)))
            taxa1 = subjset.taxas[iii]
            for taxa2 in subjset.taxas.names.order[iii:]:
                taxa2 = subjset.taxas[taxa2]
                if taxa1.name == taxa2.name:
                    continue
                if len(taxa1.sequence) != len(taxa2.sequence):
                    continue

                dist = _avg_dist(taxa1, taxa2)
                if dist <= hamming_dist:
                    subjset.aggregate_items(taxa1, taxa2)
                    cnt += 1
                    found = True
                    break
            if found:
                break
        if found:
            found = False
        else:
            break
    logging.info('Aggregated {} taxas'.format(cnt))
    return subjset

def _avg_dist(taxa1, taxa2):
    dists = []
    if pl.isotu(taxaa1):
        seqs1 = taxa1.aggregated_seqs.values()
    else:
        seqs1 = [taxa1.sequence]

    if pl.isotu(taxa2):
        seqs2 = taxa2.aggregated_seqs.values()
    else:
        seqs2 = [taxa2.sequence]

    for v1 in seqs1:
        for v2 in seqs2:
            dists.append(diversity.beta.hamming(v1, v2))
    return np.nanmean(dists)

def consistency_filtering(subjset, dtype, threshold, min_num_consecutive, min_num_subjects, 
    colonization_time=None, union_other_consortia=None):
    '''Filters the subjects by looking at the consistency of the 'dtype', which can
    be either 'raw' where we look for the minimum number of counts, 'rel', where we
    look for a minimum relative abundance, or 'abs' where we look for a minium 
    absolute abundance.

    There must be at least `threshold` for at least
    `min_num_consecutive` consecutive timepoints for at least
    `min_num_subjects` subjects for the Taxa to be classified as valid.

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
        If not None, take the union of the taxas passing the filtering of both
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
        taxas_to_keep = set()
        for subjset_temp in [subjset, union_other_consortia]:
            subjset_temp = consistency_filtering(subjset_temp, dtype=dtype,
                threshold=threshold, min_num_consecutive=min_num_consecutive,
                colonization_time=colonization_time, min_num_subjects=min_num_subjects,
                union_other_consortia=None)
            for taxa_name in subjset_temp.taxas.names:
                taxas_to_keep.add(taxa_name)
        to_delete = []
        for aname in subjset.taxas.names:
            if aname not in taxas_to_keep:
                to_delete.append(aname)
    else:
        # Everything is fine, now we can do the filtering
        talley = np.zeros(len(subjset.taxas), dtype=int)
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
        to_delete = subjset.taxas.ids.order[invalid_oidxs]
    subjset.pop_taxas(to_delete)
    return subjset

def conditional_consistency_filtering(subjset, other, dtype, threshold, min_num_consecutive_upper,  
    min_num_consecutive_lower, min_num_subjects, colonization_time):
    '''Filters the cohorts in `subjset` with the `mdsine2.consistency_filtering`
    filtering method but conditional on another cohort. If an Taxa passes the filter
    in the cohort `other`, the Taxa can only have `min_num_consecutive_lower` 
    consecutive timepoints instead of `min_num_consecutive_upper`. This potentially
    increases the overlap of the Taxas between cohorts, which is the reason why
    we would do this filtering over just `mdsine2.consistency_filtering`

    Algorithm
    ---------
    First, we apply the function `mdsine2.consistency_filtering` to the subjects
    `subjset` and `other` with minimum number of consectutive timepoints 
    `min_num_consecutive_upper` and `min_num_consecutive_lower`.

    For each Taxa in cohort `subjset`
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
    for taxa in subjset_lower.taxas:
        if taxa.name in subjset_upper.taxas:
            continue
        if taxa.name in other_upper.taxas:
            continue
        to_delete.append(taxa.name)

    subjset_lower.pop_taxas(to_delete)
    return subjset_lower