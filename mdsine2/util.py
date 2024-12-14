'''Utility functions for mdsine2
'''
import itertools

from mdsine2.logger import logger
import numpy as np
import copy
import pandas as pd

import scipy.stats
from sklearn.cluster import AgglomerativeClustering

from .names import STRNAMES
from . import pylab as pl

from typing import Union, Dict, List, Optional, Callable
from .pylab import BaseMCMC, diversity
from .base import *
from .dataset import parse

def is_gram_negative(taxon: Union[OTU, Taxon]) -> bool:
    '''Return true if the taxon is gram - or gram positive

    Parameters
    ----------
    taxon : md2.Taxon, md2.OTU
        Taxon object

    Returns
    -------
    bool
    '''
    if not taxon.tax_is_defined('phylum'):
        return None
    elif taxon.taxonomy['phylum'].lower() == 'bacteroidetes':
        return True
    elif taxon.taxonomy['phylum'].lower() == 'firmicutes':
        return False
    elif taxon.taxonomy['phylum'].lower() == 'verrucomicrobia':
        return True
    elif taxon.taxonomy['phylum'].lower() == 'proteobacteria':
        return True
    else:
        raise ValueError('{} phylum not specified. If not bacteroidetes, firmicutes, verrucomicrobia, or ' \
            'proteobacteria, you must add another phylum'.format(str(taxon)))

def generate_interation_bayes_factors_posthoc(mcmc: BaseMCMC, section: str='posterior') -> np.ndarray:
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
    np.ndarray((n,n), dtype=float)
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

def generate_perturbation_bayes_factors_posthoc(mcmc: BaseMCMC, perturbation: ClusterPerturbationEffect,
    section: str='posterior') -> np.ndarray:
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
    perturbation : mdsine2.ClusterPerturbationEffect
        Perturbation object we are calculating from
    section : str
        This is the section of the MH samples to take the samples

    Returns
    -------
    np.ndarray((n,), dtype=float)
        These are the bayes factors for each of the perturbations on an item level
    '''
    trace = ~ np.isnan(perturbation.get_trace_from_disk())
    cnts_1 = np.sum(trace, axis=0)
    cnts_0 = np.sum(1-trace, axis=0)

    a = perturbation.probability.prior.a.value
    b = perturbation.probability.prior.b.value

    return (cnts_1 * b) / (cnts_0 * a)

def generate_cluster_assignments_posthoc(clustering: Clustering, n_clusters: Union[str, int]='mode', linkage: str='average',
    set_as_value: bool=False, section: str='posterior') -> np.ndarray:
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
    trace = clustering.n_clusters.get_trace_from_disk(section=section)
    if callable(n_clusters):
        n = n_clusters(trace)
    elif type(n_clusters) == int:
        n = n_clusters
    elif type(n_clusters) == str:
        if n_clusters == 'mode':
            n = scipy.stats.mode(trace)[0]
        else:
            raise ValueError('`n_clusters` ({}) not recognized.'.format(n_clusters))
    else:
        raise ValueError('Type `n_clusters` ({}) not recognized. Must be of '\
            'type `str`, `int`, or callable.'.format(type(n_clusters)))
    if not pl.isbool(set_as_value):
        raise TypeError('`set_as_value` ({}) must be a bool'.format(type(set_as_value)))

    A = pl.summary(clustering.coclusters, section=section)['mean']
    A = 1 - A
    logger.info('Number of clusters: {}'.format(int(n)))
    c = AgglomerativeClustering(
        n_clusters=int(n),
        affinity='precomputed',
        linkage=linkage)
    ret = c.fit_predict(A)
    logger.info('Clusters assigned: {}'.format(ret))
    if set_as_value:
        for cidx in range(np.max(ret)):
            cluster = np.where(ret == cidx)[0]
            cid = clustering.make_new_cluster_with(idx=cluster[0])
            for oidx in cluster[1:]:
                clustering.move_item(idx=oidx, cid=cid)
    return ret

def generate_taxonomic_distribution_over_clusters_posthoc(mcmc: BaseMCMC, tax_fmt: str) -> pd.DataFrame:
    '''Make a table that shows the abundance of different taxonomies in every cluster

    Output:
        rows: taxonomic lables
        columns: clusters
        value: abundance of those taxonomies

    Value
    -----
    If there are perturbations, then the value is the mean abundance over the taxa
    before the first perturbation between all of the subjects. If there are no perturbations
    then this is just the mean abundance of the taxa over all the time points

    Parameters
    ----------
    mcmc : mdsine2.BaseMCMC
        This is the inference object containing the traces
    tax_fmt : str
        This is the format to generate the taxonomy names. See `mdsine2.taxaname_formatter`

    Returns
    -------
    pandas.DataFrame
    '''
    # Get the objects
    clustering = mcmc.graph[STRNAMES.CLUSTERING_OBJ]
    study = mcmc.graph.data.subjects

    # Get all the times
    times = study.times(agg='union')

    # Get the first time of the perturbation between all of the subjects
    first_time = None
    if study.perturbations is not None:
        for subj in study:
            if first_time is None:
                first_time = study.perturbations[0].starts[subj.name]
            else:
                curr_time = study.perturbations[0].starts[subj.name]
                if curr_time < first_time:
                    first_time = curr_time
        t_end = np.searchsorted(times, first_time)
    else:
        t_end = int(len(times))

    # Get mean abundance over subjects for times preperturbation then take mean
    times = times[:t_end]
    M = study.matrix(dtype='abs', agg='mean', times=times)
    abunds = np.mean(M, axis=1)

    # Get the clustering
    cidx_assign = generate_cluster_assignments_posthoc(clustering=clustering)
    clustering.from_array(cidx_assign)

    # Get the average abundances for each OTU in each cluster
    data = []
    for cluster in clustering:
        oidxs = list(cluster.members)
        temp = np.zeros(len(abunds))
        temp[oidxs] = abunds[oidxs]
        data.append(temp.reshape(-1,1))
    data = np.hstack(data).shape

    # Make the taxonomic heatmap as a dataframe
    df = condense_matrix_with_taxonomy(M, taxa=study.taxa, fmt=tax_fmt)
    return df

def condense_fixed_clustering_interaction_matrix(M: np.ndarray, clustering: Clustering) -> np.ndarray:
    '''Condense the interaction matrix `M` with the cluster assignments
    in `clustering`. Assume that the current cluster assignments is what
    is used. Assumes that the input matrix is run with a fixed clustering.

    Ignores the diagonal entrees

    Parameters
    ----------
    M : np.ndarrray(..., n_taxa, n_taxa)
        Taxon-Taxon interaction matrix
    clustering : mdsine2.Clustering
        Clustering object

    Returns
    -------
    np.ndarray(..., n_clusters, n_clusters)
        Cluster-cluster interaction matrix
    '''
    shape = M.shape[:-2]

    ret = np.zeros(shape=shape+(len(clustering), len(clustering)))
    for i1, cl1 in enumerate(clustering):
        for i2, cl2 in enumerate(clustering):
            if i1 == i2:
                continue
            aidx1 = list(cl1.members)[0]
            aidx2 = list(cl2.members)[0]
            ret[..., i1,i2] = M[..., aidx1, aidx2]
    return ret

def condense_fixed_clustering_perturbation(pert: np.ndarray, clustering: Clustering) -> np.ndarray:
    '''Condense the perturbation array passed in assuming that it was run with
    fixed clustering.

    Parameters
    ----------
    pert : np.ndarray(..., n_taxa)
        Perturbation values for each taxon
    clustering : mdsine2.Clustering
        Clustering object

    Returns
    -------
    np.ndarray(..., n_clusters)
        Cluster perturbations
    '''
    shape = pert.shape[:-1]
    ret = np.zeros(shape=shape+(len(clustering),))
    for cidx, cluster in enumerate(clustering):
        aidx = list(cluster.members)[0]
        ret[..., cidx] = pert[..., aidx]
    return ret


def aggregate_items(subjset: Study,
                    hamming_dist: int,
                    otu_naming: Callable[[int, List[Taxon]], str],
                    linkage: str = 'average',
                    sort_order='SIZE') -> Study:
    """
    Aggregate Taxa that have an average hamming distance of `hamming_dist`.

    Parameters
    ----------
    subjset : mdsine2.Study
        This is the `mdsine2.Study` object that we are aggregating
    hamming_dist : int
        This is the hamming radius from one taxon to the next where we are aggregating
    """

    # Compute the hamming dist matrix
    asvs = list(subjset.taxa)
    dists = np.zeros((len(asvs), len(asvs)), dtype=int)
    for i, j in itertools.combinations(subjset.taxa, r=2):
        d = diversity.beta.hamming(i.sequence, j.sequence)
        dists[i.idx, j.idx] = d
        dists[j.idx, i.idx] = d

    logger.info(f'Aggregating taxa with a hamming distance of {hamming_dist} (linkage: {linkage})')
    clustering = AgglomerativeClustering(
        metric='precomputed',
        n_clusters=None,
        linkage=linkage,  # min distance
        distance_threshold=hamming_dist
    ).fit(dists)

    subsets: List[List[Taxon]] = []
    oidx_set = set(clustering.labels_)
    for oidx in oidx_set:
        asv_subset: List[Taxon] = [asvs[i] for i in np.where(clustering.labels_ == oidx)[0]]
        subsets.append(asv_subset)

    if sort_order == 'SIZE':
        subsets = sorted(
            subsets,
            key=lambda x: len(x),
            reverse=True
        )
    elif sort_order == "MIN_ASV_IDX":
        subsets = sorted(
            subsets,
            key=lambda agg: min(taxa.idx for taxa in agg)
        )
    else:
        raise ValueError(f"Unrecognized sort_order argument `{sort_order}`")
    return subjset.aggregate_items(subsets, otu_naming=otu_naming)


def write_fixed_clustering_as_json(mcmc: BaseMCMC, output_filename: str):
    '''Export the posterior fixed topology as a json usable in Cytoscape

    Parameters
    ----------
    mcmc : BaseMCMC
        This is the chain that contains the traces
    output_filename : str
        This is the path to save the json file
    '''
    import networkx as nx
    from py2cytoscape.util import from_networkx
    import json

    def clusterize(labels: np.ndarray, taxa_list: List[str]) -> Dict[int, List[str]]:
        cluster = {}
        for i in range(len(labels)):
            if labels[i] not in cluster:
                cluster[labels[i]] = []
            cluster[labels[i]].append(taxa_list[i])
        return cluster

    def get_largest_weight(matrix: np.ndarray) -> float:
        '''Returns the largest non infinite weight (value) in the matrix
        '''
        new_matrix = np.where(matrix == np.inf, -np.inf, matrix)
        return np.amax(new_matrix)

    def get_bayes_category(bf: float) -> int:
        '''Classify bayes factor according to strength of evidence

        Parameters
        ----------
        bf : float
            Bayes factor

        Returns
        -------
        int
        '''
        category = 0
        #decisive
        if bf > 10 ** 2 :
            category = 3
        #strong
        elif 10 < bf <= 10 ** 2:
            category = 2
        # substantial + not worth mentioning
        elif 0 < bf <= 10:
            category = 1
        return category

    clustering = mcmc.graph[STRNAMES.CLUSTERING_OBJ]
    #consensus_cluster_labels = generate_cluster_assignments_posthoc(clustering=clustering, set_as_value=True)
    consensus_cluster_labels = clustering.toarray()

    taxa_names = []
    taxas = mcmc.graph.data.taxa
    for taxa in taxas:
        taxa_names.append(taxa.name)

    consensus_cluster = clusterize(consensus_cluster_labels, taxa_names)
    M = pl.summary(mcmc.graph[STRNAMES.INTERACTIONS_OBJ], set_nan_to_0=True,
        section='posterior')['mean']
    M_condensed = condense_fixed_clustering_interaction_matrix(M,
        clustering=clustering)

    int_mat = mcmc.graph[STRNAMES.INTERACTIONS_OBJ].get_trace_from_disk(
        section="posterior")
    int_mat_condensed = condense_fixed_clustering_interaction_matrix(int_mat,
        clustering=clustering)

    int_mat_condensed[np.isnan(int_mat_condensed)] = 0
    sign_mat = np.where(int_mat_condensed<0, -1, int_mat_condensed)
    sign_mat = np.where(sign_mat > 0, 1, sign_mat)
    sign_mat = np.sum(sign_mat, axis=0)


    bf = generate_interation_bayes_factors_posthoc(mcmc=mcmc,
          section='posterior')
    bf_condensed = condense_fixed_clustering_interaction_matrix(bf,
        clustering=clustering)

    columns = np.sort(list(consensus_cluster.keys()))
    # Take the transpose so that rows are the srouce and columns destination
    bayes_df = pd.DataFrame(bf_condensed.T, columns=columns, index=columns)
    graph_bayes = nx.from_pandas_adjacency(bayes_df, create_using=nx.DiGraph())

    largest = get_largest_weight(bf_condensed)
    all_edges = graph_bayes.edges()
    edge_attributes = {}

    for edge in graph_bayes.edges(data=True):
        # Column is the source and row is the destination
        int_strength = M_condensed[edge[1], edge[0]]
        coord = (edge[0], edge[1])
        sign = 0
        majority_sign = sign_mat[edge[1], edge[0]]
        weight = edge[2]['weight']

        if np.isinf(weight):
            weight = largest
        if majority_sign < 0:
            sign = -1
        elif majority_sign > 0:
            sign = 1

        category = get_bayes_category(weight)
        bend = False
        if (edge[1], edge[0]) in all_edges:
            bend = True

        edge_attributes[coord] = {'bayes_fac': category, 'sign': sign,
            'weight': weight, 'bend': bend}

    nx.set_edge_attributes(graph_bayes, edge_attributes)

    nodes_attributes = {}
    for keys in consensus_cluster:
        nodes_attributes[keys] = {'size': len(consensus_cluster[keys])}

    nx.set_node_attributes(graph_bayes, nodes_attributes)

    data_json = from_networkx(graph_bayes)
    if '.json' not in output_filename:
        output_filename += '.json'
    with open(output_filename, 'w') as f:
        json.dump(data_json, f)
    print("cyjs file exported to: {}".format(output_filename))


def consistency_filtering(subjset: Study, dtype: str, threshold: Union[float, int], min_num_consecutive: int, min_num_subjects: int,
    colonization_time: Union[float, int]=None, union_other_consortia: Optional[Study]=None) -> Study:
    '''Filters the subjects by looking at the consistency of the 'dtype', which can
    be either 'raw' where we look for the minimum number of counts, 'rel', where we
    look for a minimum relative abundance, or 'abs' where we look for a minium
    absolute abundance.

    There must be at least `threshold` for at least
    `min_num_consecutive` consecutive timepoints for at least
    `min_num_subjects` subjects for the taxon to be classified as valid.

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
        If not None, take the union of the taxa passing the filtering of both
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

    subjset = copy.deepcopy(subjset)

    if union_other_consortia is not None:
        taxa_to_keep = set()
        for subjset_temp in [subjset, union_other_consortia]:
            subjset_temp = consistency_filtering(subjset_temp, dtype=dtype,
                threshold=threshold, min_num_consecutive=min_num_consecutive,
                colonization_time=colonization_time, min_num_subjects=min_num_subjects,
                union_other_consortia=None)
            for taxon_name in subjset_temp.taxa.names:
                taxa_to_keep.add(taxon_name)
        to_delete = []
        for aname in subjset.taxa.names:
            if aname not in taxa_to_keep:
                to_delete.append(aname)
    else:
        # Everything is fine, now we can do the filtering
        talley = np.zeros(len(subjset.taxa), dtype=int)
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
        to_delete = subjset.taxa.ids.order[invalid_oidxs]
    subjset.pop_taxa(to_delete)
    return subjset

def conditional_consistency_filtering(subjset: Study, other: Study, dtype: str, threshold: Union[float, int],
    min_num_consecutive_upper: int, min_num_consecutive_lower: int, min_num_subjects: int,
    colonization_time: Union[float, int]) -> Study:
    '''Filters the cohorts in `subjset` with the `mdsine2.consistency_filtering`
    filtering method but conditional on another cohort. If a taxon passes the filter
    in the cohort `other`, the taxon can only have `min_num_consecutive_lower`
    consecutive timepoints instead of `min_num_consecutive_upper`. This potentially
    increases the overlap of the taxa between cohorts, which is the reason why
    we would do this filtering over just `mdsine2.consistency_filtering`

    Algorithm
    ---------
    First, we apply the function `mdsine2.consistency_filtering` to the subjects
    `subjset` and `other` with minimum number of consectutive timepoints
    `min_num_consecutive_upper` and `min_num_consecutive_lower`.

    For each taxon in cohort `subjset`
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
    for taxon in subjset_lower.taxa:
        if taxon.name in subjset_upper.taxa:
            continue
        if taxon.name in other_upper.taxa:
            continue
        to_delete.append(taxon.name)

    subjset_lower.pop_taxa(to_delete)
    return subjset_lower

def make_toy(metadata_f, qpcr_f, reads_f, taxa_f, perturbations_f=None, dataset_dir=None, n_taxa=15, ):
    """ Make small toy dataset from existing study.
    Adapted from MDSINE2_paper github repo.
    """

    dset = parse(
        name = metadata_f.parent.stem,
        metadata = metadata_f,
        taxonomy = taxa_f,
        reads = reads_f,
        qpcr = qpcr_f,
        perturbations = perturbations_f,
    )

    to_delete = []
    for taxon in dset.taxa:
        if taxon.idx >= n_taxa:   
            to_delete.append(taxon.name)
    dset.pop_taxa(to_delete)
    
    if dataset_dir is None:
        dataset_dir = metadata_f.parents[1]

    toy_dataset_path = dataset_dir / "{}-toy".format(dset.name)
    toy_dataset_path.mkdir(parents=True, exist_ok=True)

    tsv_files = [metadata_f, qpcr_f, reads_f, taxa_f, perturbations_f]
    toy_dataset_files = {f.stem : toy_dataset_path / f.name for f in tsv_files}
    
    dset.write_metadata_to_csv(path=toy_dataset_files[metadata_f.stem])
    dset.write_qpcr_to_csv(path=toy_dataset_files[qpcr_f.stem])
    if dset.perturbations is not None:
        dset.write_perturbations_to_csv(path=toy_dataset_files[perturbations_f.stem])
    dset.write_reads_to_csv(path=toy_dataset_files[reads_f.stem])
    dset.taxa.write_taxonomy_to_csv(path=toy_dataset_files[taxa_f.stem])

    toy_study = parse(
        name = toy_dataset_path.stem,
        metadata = toy_dataset_files[metadata_f.stem],
        taxonomy = toy_dataset_files[taxa_f.stem],
        reads = toy_dataset_files[reads_f.stem],
        qpcr = toy_dataset_files[qpcr_f.stem],
        perturbations = toy_dataset_files[perturbations_f.stem],
        )
    
    return toy_study
