'''This module holds classes for the logging and model configuration
parameters that are set manually in here. There are also the filtering
functions used to preprocess the data

Learned negative binomial dispersion parameters
-----------------------------------------------
a0
	median: 3.021173158076349e-05
	mean: 3.039336514482573e-05
	25th percentile: 2.8907661553542307e-05
	75th percentile: 3.1848563862236224e-05
	acceptance rate: 0.3636
a1
	median: 0.03610445832385458
	mean: 0.036163868481381596
	25th percentile: 0.034369620675005035
	75th percentile: 0.0378392670993046
	acceptance rate: 0.5324

'''
import logging
import numpy as np
import pandas as pd
import sys
import os.path

from .names import STRNAMES
from . import pylab as pl

# File locations
GRAPH_NAME = 'graph'
MCMC_FILENAME = 'mcmc.pkl'
SUBJSET_FILENAME = 'subjset.pkl'
VALIDATION_SUBJSET_FILENAME = 'validate_subjset.pkl'
SYNDATA_FILENAME = 'syndata.pkl'
GRAPH_FILENAME = 'graph.pkl'
HDF5_FILENAME = 'traces.hdf5'
TRACER_FILENAME = 'tracer.pkl'
PARAMS_FILENAME = 'params.pkl'
FPARAMS_FILENAME = 'filtering_params.pkl'
SYNPARAMS_FILENAME = 'synthetic_params.pkl'
MLCRR_RESULTS_FILENAME = 'mlcrr_results.pkl'
RESTART_INFERENCE_SEED_RECORD = 'restart_seed_record.tsv'
INTERMEDIATE_RESULTS_FILENAME = 'intermediate_results.tsv'


PHYLOGENETIC_TREE_FILENAME = 'raw_data/phylogenetic_tree_branch_len_preserved.nhx'

def isModelConfig(x):
    '''Checks if the input array is a model config object

    Parameters
    ----------
    x : any
        Instance we are checking

    Returns
    -------
    bool
        True if `x` is a a model config object
    '''
    return x is not None and issubclass(x.__class__, _BaseModelConfig)

class _BaseModelConfig(pl.Saveable):

    def __str__(self):
        s = '{}'.format(self.__class__.__name__)
        for k,v in vars(self).items():
            s += '\n\t{}: {}'.format(k,v)
        return s

    def suffix(self):
        raise NotImplementedError('Need to implement')


class MDSINE2ModelConfig(_BaseModelConfig):
    '''Configuration parameters for the model


    System initialization
    ---------------------

    Parameters
    ----------
    '''
    def __init__(self, basepath, data_seed, init_seed, burnin, n_samples,
        negbin_a0, negbin_a1, leave_out=None, max_n_asvs=None, 
        checkpoint=100):
        self.OUTPUT_BASEPATH = os.path.abspath(basepath)
        self.MODEL_PATH = None
        self.DATA_SEED = data_seed
        self.INIT_SEED = init_seed
        self.BURNIN = burnin
        self.N_SAMPLES = n_samples
        self.CHECKPOINT = checkpoint
        self.PROCESS_VARIANCE_TYPE = 'multiplicative-global'
        self.DATA_DTYPE = 'abs'

        self.QPCR_NORMALIZATION_MAX_VALUE = 100
        self.LEAVE_OUT = leave_out
        self.MAX_N_ASVS = max_n_asvs
        self.ZERO_INFLATION_TRANSITION_POLICY = None #'ignore'

        self.GROWTH_TRUNCATION_SETTINGS = 'positive'
        self.SELF_INTERACTIONS_TRUNCATION_SETTINGS = 'positive'

        self.MP_FILTERING = 'debug'
        self.MP_CLUSTERING = 'debug'

        self.NEGBIN_A0 = negbin_a0
        self.NEGBIN_A1 = negbin_a1
        self.N_QPCR_BUCKETS = 3

        self.INTERMEDIATE_VALIDATION_T = 1 * 3600 # Every hour
        self.INTERMEDIATE_VALIDATION_KWARGS = None

        self.LEARN = {
            STRNAMES.REGRESSCOEFF: True,
            STRNAMES.PROCESSVAR: True,
            STRNAMES.PRIOR_VAR_GROWTH: False,
            STRNAMES.PRIOR_VAR_SELF_INTERACTIONS: False,
            STRNAMES.PRIOR_VAR_INTERACTIONS: True,
            STRNAMES.PRIOR_VAR_PERT: True,
            STRNAMES.PRIOR_MEAN_GROWTH: True,
            STRNAMES.PRIOR_MEAN_SELF_INTERACTIONS: True,
            STRNAMES.PRIOR_MEAN_INTERACTIONS: True,
            STRNAMES.PRIOR_MEAN_PERT: True,
            STRNAMES.FILTERING: True,
            STRNAMES.ZERO_INFLATION: False,
            STRNAMES.CLUSTERING: True,
            STRNAMES.CONCENTRATION: True, 
            STRNAMES.CLUSTER_INTERACTION_INDICATOR: True,
            STRNAMES.INDICATOR_PROB: True,
            STRNAMES.PERT_INDICATOR: True,
            STRNAMES.PERT_INDICATOR_PROB: True,
            STRNAMES.QPCR_SCALES: False,
            STRNAMES.QPCR_DOFS: False,
            STRNAMES.QPCR_VARIANCES: False}

        self.INFERENCE_ORDER = [
            STRNAMES.CLUSTER_INTERACTION_INDICATOR,
            STRNAMES.INDICATOR_PROB,
            STRNAMES.PERT_INDICATOR,
            STRNAMES.PERT_INDICATOR_PROB,
            STRNAMES.REGRESSCOEFF,
            STRNAMES.PRIOR_MEAN_INTERACTIONS,
            STRNAMES.PRIOR_MEAN_PERT,
            STRNAMES.PRIOR_MEAN_GROWTH,
            STRNAMES.PRIOR_MEAN_SELF_INTERACTIONS,
            STRNAMES.PRIOR_VAR_GROWTH,
            STRNAMES.PRIOR_VAR_SELF_INTERACTIONS,
            STRNAMES.PRIOR_VAR_INTERACTIONS,
            STRNAMES.PRIOR_VAR_PERT,
            STRNAMES.PROCESSVAR,
            STRNAMES.ZERO_INFLATION,
            STRNAMES.QPCR_SCALES,
            STRNAMES.QPCR_DOFS,
            STRNAMES.QPCR_VARIANCES,
            STRNAMES.FILTERING,
            STRNAMES.CLUSTERING,
            STRNAMES.CONCENTRATION]

        self.INITIALIZATION_KWARGS = {
            STRNAMES.QPCR_VARIANCES: {
                'value_option': 'empirical'},
            STRNAMES.QPCR_SCALES: {
                'value_option': 'prior-mean',
                'scale_option': 'empirical',
                'dof_option': 'diffuse',
                'proposal_option': 'auto',
                'target_acceptance_rate': 'optimal',
                'end_tune': 'half-burnin',
                'tune': 50,
                'delay':0},
            STRNAMES.QPCR_DOFS: {
                'value_option': 'diffuse',
                'low_option': 'valid',
                'high_option': 'med',
                'proposal_option': 'auto',
                'target_acceptance_rate': 'optimal',
                'end_tune': 'half-burnin',
                'tune': 50,
                'delay': 0},
            STRNAMES.PERT_VALUE: {
                'value_option': 'prior-mean',
                'delay':0},
            STRNAMES.PERT_INDICATOR_PROB: {
                'value_option': 'prior-mean',
                'hyperparam_option': 'strong-sparse',
                'delay':0},
            STRNAMES.PERT_INDICATOR: {
                'value_option': 'all-off',
                'delay':0},
            STRNAMES.PRIOR_VAR_PERT: {
                'value_option': 'prior-mean',
                'scale_option': 'diffuse',
                'dof_option': 'diffuse',
                'delay': 0},
            STRNAMES.PRIOR_MEAN_PERT: {
                'value_option': 'prior-mean',
                'mean_option': 'zero',
                'var_option': 'diffuse',
                'delay':0},
            STRNAMES.PRIOR_VAR_GROWTH: {
                'value_option': 'prior-mean',
                'scale_option': 'inflated-median',
                'dof_option': 'diffuse',
                'proposal_option': 'tight',
                'target_acceptance_rate': 'optimal',
                'end_tune': 'half-burnin',
                'tune': 50,
                'delay':0},
            STRNAMES.PRIOR_MEAN_GROWTH: {
                'value_option': 'prior-mean',
                'mean_option': 'manual',
                'var_option': 'diffuse-linear-regression',
                'proposal_option': 'auto',
                'target_acceptance_rate': 0.44,
                'tune': 50,
                'end_tune': 'half-burnin',
                'truncation_settings': self.GROWTH_TRUNCATION_SETTINGS,
                'delay':0, 'mean': 1},
            STRNAMES.GROWTH_VALUE: {
                'value_option': 'linear-regression', #'prior-mean',
                'truncation_settings': self.GROWTH_TRUNCATION_SETTINGS,
                'delay': 0},
            STRNAMES.PRIOR_VAR_SELF_INTERACTIONS: {
                'value_option': 'prior-mean',
                'scale_option': 'inflated-median',
                'dof_option': 'diffuse',
                'proposal_option': 'tight',
                'target_acceptance_rate': 'optimal',
                'end_tune': 'half-burnin',
                'tune': 50,
                'delay':0},
            STRNAMES.PRIOR_MEAN_SELF_INTERACTIONS: {
                'value_option': 'prior-mean',
                'mean_option': 'median-linear-regression',
                'var_option': 'diffuse-linear-regression',
                'proposal_option': 'auto',
                'target_acceptance_rate': 0.44,
                'tune': 50,
                'end_tune': 'half-burnin',
                'truncation_settings': self.SELF_INTERACTIONS_TRUNCATION_SETTINGS,
                'delay':0},
            STRNAMES.SELF_INTERACTION_VALUE: {
                'value_option': 'linear-regression',
                'truncation_settings': self.SELF_INTERACTIONS_TRUNCATION_SETTINGS,
                'delay': 0},
            STRNAMES.PRIOR_VAR_INTERACTIONS: {
                'value_option': 'auto',
                'dof_option': 'diffuse',
                'scale_option': 'same-as-aii',
                'mean_scaling_factor': 1,
                'delay': 0},
            STRNAMES.PRIOR_MEAN_INTERACTIONS: {
                'value_option': 'prior-mean',
                'mean_option': 'zero',
                'var_option': 'same-as-aii',
                'delay':0},
            STRNAMES.CLUSTER_INTERACTION_VALUE: {
                'value_option': 'all-off',
                'delay': 0},
            STRNAMES.CLUSTER_INTERACTION_INDICATOR: {
                'delay':0,
                'run_every_n_iterations': 1},
            STRNAMES.INDICATOR_PROB: {
                'value_option': 'auto',
                'hyperparam_option': 'strong-sparse',
                'N': 25,
                'delay': 0},
            STRNAMES.FILTERING: {
                'x_value_option':  'loess',
                'tune': (int(self.BURNIN/2), 50),
                'a0': self.NEGBIN_A0,
                'a1': self.NEGBIN_A1,
                'v1': 1e-4,
                'v2': 1e-4,
                'proposal_init_scale':.001,
                'intermediate_interpolation': 'linear-interpolation',
                'intermediate_step': None, #('step', (1, None)), 
                'essential_timepoints': 'union',
                'delay': 1,
                'window': 6,
                'target_acceptance_rate': 0.44},
            STRNAMES.ZERO_INFLATION: {
                'value_option': None,
                'delay': 0},
            STRNAMES.CONCENTRATION: {
                'value_option': 'prior-mean',
                'hyperparam_option': 'diffuse',
                'delay': 0, 'n_iter': 20},
            STRNAMES.CLUSTERING: {
                'value_option': 'spearman', #'fixed-topology',
                'delay': 2,
                'n_clusters': 30,
                'run_every_n_iterations': 4},
            STRNAMES.REGRESSCOEFF: {
                'update_jointly_pert_inter': True,
                'update_jointly_growth_si': False,
                'tune': 50,
                'end_tune': 'half-burnin'},
            STRNAMES.PROCESSVAR: {
                # 'v1': 0.2**2,
                # 'v2': 1,
                # 'q_option': 'previous-t'}, #'previous-t'},
                'dof_option': 'diffuse', # 'half', 
                'scale_option': 'med',
                'value_option': 'prior-mean',
                'delay': 0}
        }

        self.INITIALIZATION_ORDER = [
            STRNAMES.FILTERING,
            STRNAMES.ZERO_INFLATION,
            STRNAMES.CONCENTRATION,
            STRNAMES.CLUSTERING,
            STRNAMES.PROCESSVAR,
            STRNAMES.PRIOR_MEAN_GROWTH,
            STRNAMES.PRIOR_VAR_GROWTH,
            STRNAMES.GROWTH_VALUE,
            STRNAMES.PRIOR_MEAN_SELF_INTERACTIONS,
            STRNAMES.PRIOR_VAR_SELF_INTERACTIONS,
            STRNAMES.SELF_INTERACTION_VALUE,
            STRNAMES.PRIOR_MEAN_INTERACTIONS,
            STRNAMES.PRIOR_VAR_INTERACTIONS,
            STRNAMES.CLUSTER_INTERACTION_VALUE,
            STRNAMES.CLUSTER_INTERACTION_INDICATOR,
            STRNAMES.INDICATOR_PROB,
            STRNAMES.PRIOR_MEAN_PERT,
            STRNAMES.PRIOR_VAR_PERT,
            STRNAMES.PERT_INDICATOR,
			STRNAMES.PERT_VALUE,
            STRNAMES.PERT_INDICATOR_PROB,
            STRNAMES.REGRESSCOEFF,
            STRNAMES.QPCR_SCALES,
            STRNAMES.QPCR_DOFS,
            STRNAMES.QPCR_VARIANCES]

    def suffix(self):
        '''Create a suffix with the parameters
        '''
        s = 'ds{}_is{}_b{}_ns{}_lo{}_mo{}'.format(
            self.DATA_SEED, self.INIT_SEED, self.BURNIN, self.N_SAMPLES, self.LEAVE_OUT,
            self.MAX_N_ASVS)
        return s

    def cv_suffix(self):
        '''Create a master suffix with the parameters
        '''
        s = 'ds{}_is{}_b{}_ns{}_mo{}'.format(
            self.DATA_SEED, self.INIT_SEED, self.BURNIN, self.N_SAMPLES,
            self.MAX_N_ASVS)
        return s

    def cv_single_suffix(self):
        '''Create a suffix for a single cv round
        '''
        return 'leave_out{}'.format(self.LEAVE_OUT)


class FilteringConfig(pl.Saveable):
    '''These are the parameters for Filtering

    Consistency filtering
    ----------------------------
    Filters the subjects by looking at the consistency of the counts.
    There must be at least `min_num_counts` for at least
    `min_num_consecutive` consecutive timepoints for at least
    `min_num_subjects` subjects for the ASV to be classified as valid.

    Parameters
    ----------
    min_num_consec: int
        This is the minimum number of consecutive timepoints that there
        must be at least `min_num_counts`
    threshold : float, int
        This is the minimum number of counts/abudnance/relative abundance that 
        there must be at each consecutive timepoint
    min_num_subjects : int, None
        This is how many subjects this must be true for for the ASV to be
        valid. If it is None then it only requires one subject.
    colonization_time : int
        How many days we consider colonization (ignore during filtering)
    '''
    def __init__(self, colonization_time, threshold, min_num_subj, min_num_consec):

        self.COLONIZATION_TIME = colonization_time
        self.THRESHOLD = threshold
        self.MIN_NUM_SUBJECTS = min_num_subj
        self.MIN_NUM_CONSECUTIVE = min_num_consec

    def __str__(self):
        return 'col{}_thresh{}_subj{}_consec{}'.format(
            self.COLONIZATION_TIME, self.THRESHOLD, self.MIN_NUM_SUBJECTS,
            self.MIN_NUM_CONSECUTIVE)

    def suffix(self):
        return str(self)


class LoggingConfig(pl.Saveable):
    '''These are the parameters for logging

    FORMAT : str
        This is the logging format for stdout
    LEVEL : logging constant, int
        This is the level to log at for stdout
    NUMPY_PRINTOPTIONS : dict
        These are the printing options for numpy.

    Parameters
    ----------
    basepath : str
        If this is specified, then we also want to log to a file. Set up a
        steam and a file
    fmt : str
        This is the format of the logging prefix for the `logging` module
    level : int
        This is the level of logging to log
    '''
    def __init__(self, basepath=None, level=logging.INFO, 
        fmt='%(levelname)s:%(module)s.%(lineno)s: %(message)s'):
        self.FORMAT = fmt
        self.LEVEL = level
        self.NUMPY_PRINTOPTIONS = {
            'threshold': sys.maxsize, 'linewidth': sys.maxsize}

        if basepath is not None:
            path = basepath + 'logging.log'
            self.PATH = path
            handlers = [
                logging.FileHandler(self.PATH, mode='w'),
                logging.StreamHandler()]
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            logging.basicConfig(level=self.LEVEL, format=self.FORMAT, handlers=handlers)
        else:
            self.PATH = None
            logging.basicConfig(format=self.FORMAT, level=self.LEVEL)
        
        np.set_printoptions(**self.NUMPY_PRINTOPTIONS)
        pd.set_option('display.max_columns', None)


class NegBinConfig(_BaseModelConfig):
    '''Configuration class for learning the negative binomial dispersion
    parameters. Note that these parameters are learned offline.

    Parameters
    ----------
    seed : int
        Seed to start the inderence
    burnin, n_samples : int
        How many iterations for burn-in and total samples, respectively.
    basepath : str
        This is the basepath to save the graph. A separate folder within
        `basepath` will be created for the specific graph.
    synth : bool
        If True, run with the synthetic data, where the parameters needed
        to learn are `SYNTHETIC_A0` AND `SYNTHETIC_A1`.
    '''

    def __init__(self, seed, burnin, n_samples, ckpt, basepath):
        if basepath[-1] != '/':
            basepath += '/'

        self.SEED = seed
        self.OUTPUT_BASEPATH = os.path.abspath(basepath)
        self.MODEL_PATH = None
        self.BURNIN = burnin
        self.N_SAMPLES = n_samples
        self.CKPT = ckpt
        self.MP_FILTERING = 'debug'

        self.INFERENCE_ORDER = [
            STRNAMES.NEGBIN_A0,
            STRNAMES.NEGBIN_A1,
            STRNAMES.FILTERING]

        self.LEARN = {
            STRNAMES.NEGBIN_A0: True,
            STRNAMES.NEGBIN_A1: True,
            STRNAMES.FILTERING: True}

        self.INITIALIZATION_ORDER = [
            STRNAMES.FILTERING,
            STRNAMES.NEGBIN_A0,
            STRNAMES.NEGBIN_A1]

        self.INITIALIZATION_KWARGS = {
            STRNAMES.NEGBIN_A0: {
                'value': 1e-10,
                'truncation_settings': (0, 1e5),
                'tune': 50,
                'end_tune': int(self.BURNIN/2),
                'target_acceptance_rate': 'optimal', 
                'proposal_option': 'auto',
                'delay': 0},
            STRNAMES.NEGBIN_A1: {
                'value': 0.1,
                'tune': 50,
                'truncation_settings': (0, 1e5),
                'end_tune': int(self.BURNIN/2),
                'target_acceptance_rate': 'optimal', 
                'proposal_option': 'auto',
                'delay': 0},
            STRNAMES.FILTERING: {
                'tune': 50,
                'end_tune': int(self.BURNIN/2),
                'target_acceptance_rate': 'optimal', 
                'qpcr_variance_inflation': 100,
                'delay': 100}}

    def suffix(self):
        return 'seed{}_nb{}_ns{}'.format(self.SEED, self.BURNIN, self.N_SAMPLES)
