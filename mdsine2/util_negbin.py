'''Stores the classes and functions used in the `main_negbin` module that
is used to learn the negative binomial dispersion parameters offline.
'''
import numpy as np
import logging
import sys
import time
import pandas as pd
import os
import os.path
import argparse

import numpy.random as npr
import random
import math
import numba

import matplotlib.pyplot as plt
import seaborn as sns

from . import pylab as pl
from . import config
from .names import REPRNAMES, STRNAMES


def build_synthetic_subjset(params):
    '''Build the `pylab.base.Study` object used to store the data
    for synthetic learning of the negative binomial dispersion parameters

    Parameters
    ----------
    params : config.NegBinConfig
        Parameters to construct the object

    Returns
    -------
    pylab.base.Study, pylab.base.Study
        First one is the one that we learn from with added noise, second one
        is the ground truth
    '''
    if type(params) != config.NegBinConfig:
        raise TypeError('`params` ({}) not recognized'.format(type(params)))

    reads = pd.read_csv(params.RAW_COUNTS_FILENAME, sep='\t', header=0)
    reads = reads.set_index('otuName')

    # Make the subjset object
    subjset_main_inference = pl.base.Study.load(params.MAIN_INFERENCE_SUBJSET_FILENAME)
    subjset = pl.base.Study(asvs=subjset_main_inference.asvs)

    # Ground truth object
    subjset_true = pl.base.Study(asvs=subjset_main_inference.asvs)

    days = params.SYNTHETIC_DAYS
    # Sample around each of the days using the specified amount of noise (a0 and a1)
    for day in days:
        logging.info('day {} in {}'.format(day, days))
        col = 'M2-D{}-1A'.format(day)
        d = reads[col].to_numpy()

        # Make a small offset so there are no math errors
        means = d + 1e-100
        rels = means/np.sum(means)
        dispersions = params.SYNTHETIC_A0 / rels + params.SYNTHETIC_A1
        counts = []

        for _ in range(params.SYNTHETIC_N_REPLICATES):
            temp = np.asarray([pl.random.negative_binomial.sample(means[i], dispersions[i]) for \
                    i in range(d.shape[0])])
            counts.append(temp.reshape(-1,1))
        counts = np.hstack(counts)

        # Get qPCR
        qpcr = subjset_main_inference['2'].qpcr[day]
        subjset = _build_subject_single_qPCR(subjectset=subjset, counts=counts, 
            qpcr_measurement=qpcr)
        subjset_true = _build_subject_single_qPCR(subjectset=subjset_true, 
            counts=d.reshape(-1,1), qpcr_measurement=qpcr)

    return subjset, subjset_true

def build_real_subjset(params):
    '''Build the `pylab.base.Study` object used to store the data
    for learning real data of the negative binomial dispersion parameters.

    Parameters
    ----------
    params : config.NegBinConfig
        Parameters to construct the object

    Returns
    -------
    pylab.base.Study
    '''
    if type(params) != config.NegBinConfig:
        raise TypeError('`params` ({}) not recognized'.format(type(params)))

    reads = pd.read_csv(params.RAW_COUNTS_FILENAME, sep='\t', header=0)
    reads = reads.set_index('otuName')

    # Make the subjset object
    subjset_main_inference = pl.base.Study.load(params.MAIN_INFERENCE_SUBJSET_FILENAME)
    subjset = pl.base.Study(asvs=subjset_main_inference.asvs)

    for day, data_cols in params.REPLICATE_DATA_COLS:
        counts = reads[data_cols].to_numpy()
        qpcr = subjset_main_inference['2'].qpcr[day]
        subjset = _build_subject_single_qPCR(subjectset=subjset, counts=counts, 
            qpcr_measurement=qpcr)

    return subjset

def _build_subject_single_qPCR(subjectset, counts, qpcr_measurement, name=None):
    '''Build a `pylab.base.Subject` object using the counts and qPCR measurement
    and adds it to the `pylab.base.Study` object `subjset`.
    
    This is a special case because we have no time. We assume that the counts are 
    all replicates of the same measurement at time t. And thus we only have one
    qPCR measurement.

    Parameters
    ----------
    subjectset : pylab.base.Study
        This is the subject set we are adding the subject to
    counts : np.ndarray (n_asvs x n_replicates) dtype int
        counts[i,k] = Number of counts for ith ASV and kth replicate.
        If it is a single replicate it must be a column vector.
    qPCR_measurement : pl.base.qPCRdata
        A single qPCR measurement
    name : str, None
        This is the name of the subject to add. If None it will be the string of
        the replicate index
    
    Returns
    -------
    pylab.base.Study
    '''
    if not pl.isstudy(subjectset):
        raise TypeError('`subjectset` ({}) must be a pylab.base.Study'.format(
            type(subjectset)))
    if not pl.isarray(counts):
        raise TypeError('`counts` ({}) must be an array'.format(type(counts)))
    counts = np.array(counts, dtype=int)
    if np.any(counts < 0):
        raise ValueError('All values in `counts` must be positive')
    if len(counts.shape) != 2:
        raise ValueError('Shape of `counts` must be 2 ({})'.format(len(counts.shape)))
    if counts.shape[0] != len(subjectset.asvs):
        raise ValueError('First dimension of `counts` ({}) must be the same as the ' \
            'number of ASVs in `subjectset` ({})'.format(counts.shape[0], len(subjectset.asvs)))
    if not pl.isqpcrdata(qpcr_measurement):
        raise TypeError('`qpcr_measurement` ({}) must be a pylab.base.qPCRdata object'.format(
            type(qpcr_measurement)))
    if name is None:
        name = str(len(subjectset))
    elif not pl.isstr(name):
        raise TypeError('`name` ({}) must be None or a str'.format(type(name)))
    
    subjectset.add(name=name)
    subj = subjectset[name]
    subj.times = np.arange(counts.shape[1])
    for t in subj.times:
        subj.qpcr[t] = qpcr_measurement
        subj.reads[t] = counts[:,t]

    return subjectset

def filter_out_zero_asvs(subjset):
    '''Filter out asvs that have only 0 counts

    Parameters
    ----------
    subjset : pl.base.Study
        Contains the data

    Returns
    -------
    pylab.base.Study
    '''
    if not pl.isstudy(subjset):
        raise TypeError('`subjset` ({}) must be a pylab.base.Study object'.format(
            type(subjset)))

    vs = []
    for subj in subjset:
        vs.append(subj.matrix()['raw'])

    M = np.hstack(vs)

    aidxs_to_remove = []
    for aidx in range(M.shape[0]):
        if np.all(M[aidx, :] == 0):
            aidxs_to_remove.append(aidx)
    
    anames = subjset.asvs.names.order[aidxs_to_remove]

    logging.info('Deleting {} asvs because they have 0 counts in all samples'.format(len(anames)))

    subjset.pop_asvs(anames)
    return subjset