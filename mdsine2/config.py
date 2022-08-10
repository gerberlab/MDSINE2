'''This module holds classes for the logging and model configuration
parameters that are set manually in here. There are also the filtering
functions used to preprocess the data
'''
import os.path
import json

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
    - OUTPUT_BASEPATH, MODEL_PATH : str
        - Path to save the model
    - SEED : int
        - Seed to initialize inference with
    - BURNIN : int
        - Number of initial Gibb steps to throw away
    - N_SAMPLES : int
        - Total number of Gibb steps
    - CHECKPOINT : int
        - How often to write to disk
    - PROCESS_VARIANCE_TYPE : str
        - What type of process variance to do
        - NOTE: There is only one option, do not change
    - DATA_DTYPE : str
        - What kind of data to do inference we
        - NOTE: Model assume you are using absolute abundance, do not change
    - QPCR_NORMALIZATION_MAX_VALUE : float
        - This is value to set the largest qPCR value to. Normalize
        all other qPCR measurements directly to this
    - LEAVE_OUT : str
        - Which subject to leave out, if necessary
    - ZERO_INFLATION_TRANSITION_POLICY : str
        - What type of zero inflation to do. Do not change
    - GROWTH_TRUNCATION_SETTINGS : str
        - How to initialize the truncation settings for the growth parameters
    - SELF_INTERACTIONS_TRUNCATION_SETTINGS : str
        - How to initialize the truncation settings for the self-interaction parameters
    - MP_FILTERING : str
        - How to do multiprocessing for filtering
    - MP_CLUSTERING : str
        - How to do multiprocessing for clustering
    - NEGBINB_A0, NEGBIN_A1 : float
        - Negative binomial dispersion parameters
    - N_QPCR_BUCKETS : int
        - Number of qPCR buckets. This is not learned in the model, do not change.
    - INTERMEDIATE_VALIDATION_T : float
        - How often to do the intermediate validation
    - INTERMEDIATE_VALIDATION_KWARGS : dict
        - Arguemnts for the intermediate valudation
    - LEARN : Dict[str, bool]
        - These are the dictionary of parameters which we are learning in the model.
        - If the name maps to True, then we learn it during inference. If it maps to 
          false, then we do not update its value duting inference.
    - INFERENCE_ORDER : list
        - This is the order to update the parameters during MCMC inference
    - INITIALIZATION_KWARGS : Dict[str, Dict[str, Any]]
        - These are the parameters to send into the `initialize` function for each variable
          that we are learning
    - INITIALIZATION_ORDER : list
        - This is the order to initialize the variables

    Parameters
    ----------
    basepath : str
        This is the base path to save the inference
    seed : int
        This is the seed to start the inference with
    burnin : int
        This is how many gibb steps to throw away originally
    n_samples : int
        This is the total number of Gibb steps to do for inference
    negbin_a0, negbin_a1 : float
        This is the negative binomial dispersion parameters
    leave_out : str
        This is the subject to leave out, if necesssary
    checkpoint : int
        This is how often we should write to disk
    '''
    def __init__(self, basepath: str, seed: int, burnin: int, n_samples: int,
        negbin_a0: float, negbin_a1: float, leave_out: str=None,
        checkpoint: int=100):
        self.OUTPUT_BASEPATH = os.path.abspath(basepath)
        self.MODEL_PATH = self.OUTPUT_BASEPATH
        self.SEED = seed
        self.BURNIN = burnin
        self.N_SAMPLES = n_samples
        self.CHECKPOINT = checkpoint
        self.PROCESS_VARIANCE_TYPE = 'multiplicative-global'
        self.DATA_DTYPE = 'abs'

        self.QPCR_NORMALIZATION_MAX_VALUE = 100
        self.LEAVE_OUT = leave_out
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
            STRNAMES.GLV_PARAMETERS: True,
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
            STRNAMES.CLUSTER_INTERACTION_INDICATOR_PROB: True,
            STRNAMES.PERT_INDICATOR: True,
            STRNAMES.PERT_INDICATOR_PROB: True,
            STRNAMES.QPCR_SCALES: False,
            STRNAMES.QPCR_DOFS: False,
            STRNAMES.QPCR_VARIANCES: False}

        self.INFERENCE_ORDER = [
            STRNAMES.CLUSTER_INTERACTION_INDICATOR,
            STRNAMES.CLUSTER_INTERACTION_INDICATOR_PROB,
            STRNAMES.PERT_INDICATOR,
            STRNAMES.PERT_INDICATOR_PROB,
            STRNAMES.GLV_PARAMETERS,
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
                'loc_option': 'zero',
                'scale2_option': 'diffuse',
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
                'loc_option': 'manual',
                'scale2_option': 'diffuse-linear-regression',
                'proposal_option': 'auto',
                'target_acceptance_rate': 0.44,
                'tune': 50,
                'end_tune': 'half-burnin',
                'truncation_settings': self.GROWTH_TRUNCATION_SETTINGS,
                'delay':0, 'loc': 1},
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
                'loc_option': 'median-linear-regression',
                'scale2_option': 'diffuse-linear-regression',
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
                'loc_option': 'zero',
                'scale2_option': 'same-as-aii',
                'delay':0},
            STRNAMES.CLUSTER_INTERACTION_VALUE: {
                'value_option': 'all-off',
                'delay': 0},
            STRNAMES.CLUSTER_INTERACTION_INDICATOR: {
                'delay':0,
                'run_every_n_iterations': 1},
            STRNAMES.CLUSTER_INTERACTION_INDICATOR_PROB: {
                'value_option': 'auto',
                'hyperparam_option': 'strong-sparse',
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
                'value_option': 'spearman', #'fixed-clustering',
                'delay': 2,
                'n_clusters': 30,
                'run_every_n_iterations': 4},
            STRNAMES.GLV_PARAMETERS: {
                'update_jointly_pert_inter': True,
                'update_jointly_growth_selfinter': False
            },
            STRNAMES.PROCESSVAR: {
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
            STRNAMES.CLUSTER_INTERACTION_INDICATOR_PROB,
            STRNAMES.PRIOR_MEAN_PERT,
            STRNAMES.PRIOR_VAR_PERT,
            STRNAMES.PERT_INDICATOR,
            STRNAMES.PERT_VALUE,
            STRNAMES.PERT_INDICATOR_PROB,
            STRNAMES.GLV_PARAMETERS,
            STRNAMES.QPCR_SCALES,
            STRNAMES.QPCR_DOFS,
            STRNAMES.QPCR_VARIANCES]

    def copy(self):
        cfg = MDSINE2ModelConfig(
            self.OUTPUT_BASEPATH,
            self.SEED,
            self.BURNIN,
            self.N_SAMPLES,
            self.NEGBIN_A0,
            self.NEGBIN_A1,
            self.LEAVE_OUT,
            self.CHECKPOINT
        )

        cfg.PROCESS_VARIANCE_TYPE = self.PROCESS_VARIANCE_TYPE
        cfg.DATA_DTYPE = self.DATA_DTYPE
        cfg.QPCR_NORMALIZATION_MAX_VALUE = self.QPCR_NORMALIZATION_MAX_VALUE
        cfg.ZERO_INFLATION_TRANSITION_POLICY = self.ZERO_INFLATION_TRANSITION_POLICY
        cfg.GROWTH_TRUNCATION_SETTINGS = self.GROWTH_TRUNCATION_SETTINGS
        cfg.SELF_INTERACTIONS_TRUNCATION_SETTINGS = self.SELF_INTERACTIONS_TRUNCATION_SETTINGS
        cfg.MP_FILTERING = self.MP_FILTERING
        cfg.MP_CLUSTERING = self.MP_CLUSTERING
        cfg.N_QPCR_BUCKETS = self.N_QPCR_BUCKETS
        cfg.INTERMEDIATE_VALIDATION_T = self.INTERMEDIATE_VALIDATION_T
        cfg.INTERMEDIATE_VALIDATION_KWARGS = self.INTERMEDIATE_VALIDATION_KWARGS

        cfg.LEARN = self.LEARN
        cfg.INFERENCE_ORDER = self.INFERENCE_ORDER
        cfg.INITIALIZATION_KWARGS = self.INITIALIZATION_KWARGS
        cfg.INITIALIZATION_ORDER = self.INITIALIZATION_ORDER
        return cfg

    def set_negbin_params(self, a0: float, a1: float):
        self.NEGBIN_A0 = a0
        self.NEGBIN_A1 = a1
        self.INITIALIZATION_KWARGS[STRNAMES.FILTERING]['a0'] = a0
        self.INITIALIZATION_KWARGS[STRNAMES.FILTERING]['a1'] = a1

    def suffix(self):
        '''Create a suffix with the parameters
        '''
        s = 's{}_b{}_ns{}_lo{}'.format(
            self.SEED, self.BURNIN, self.N_SAMPLES, self.LEAVE_OUT)
        return s

    def cv_suffix(self):
        '''Create a master suffix with the parameters
        '''
        s = 's{}_b{}_ns{}'.format(
            self.SEED, self.BURNIN, self.N_SAMPLES)
        return s

    def cv_single_suffix(self):
        '''Create a suffix for a single cv round
        '''
        return 'leave_out{}'.format(self.LEAVE_OUT)

    def make_metadata_file(self, fname):
        '''Make a metadata file that does an overview of the parameters in this class
        '''

        mystr = 'Global parameters\n' \
            '-----------------\n' \
            'Random seed: {seed}\n' \
            'Total number of Gibb steps: {n_samples}\n' \
            'Number of Gibb steps for burn-in: {burnin}\n' \
            'Saved location: {model_path}\n\n' \
            'Negative binomial dispersion parameters\n' \
            '---------------------------------------\n' \
            'a0: {a0:.4E}\n' \
            'a1: {a1:.4E}\n\n' \
            'Parameters learned and their order\n' \
            '----------------------------------\n' \
            '{params_learned}\n\n' \
            'Selected Initialization choices\n' \
            '-------------------------------\n' \
            'Cluster interaction probability prior: {clus_ind_prior}\n' \
            'Perturbation probability prior: {pert_ind_prior}\n' \
            'Filtering initialization: {filt}\n' \
            'Cluster initialization: {clus_init}\n\n'
        # params learned
        # --------------
        i = 0
        params_learned = ''
        for pname in self.INFERENCE_ORDER:
            if self.LEARN[pname]:
                params_learned += '{}: {}\n'.format(i, pname)
                i += 1
        
        # init choices
        # ------------
        if self.INITIALIZATION_KWARGS[STRNAMES.CLUSTERING]['value_option'] == 'spearman':
            clus_init = 'Spearman Correlation, {} clusters'.format(
                self.INITIALIZATION_KWARGS[STRNAMES.CLUSTERING]['n_clusters'])
        elif self.INITIALIZATION_KWARGS[STRNAMES.CLUSTERING]['value_option'] == 'no-clusters':
            clus_init = 'No clusters, everything in its own cluster'
        elif self.INITIALIZATION_KWARGS[STRNAMES.CLUSTERING]['value_option'] == 'fixed-clustering':
            clus_init = 'Same topology as {}'.format(
                self.INITIALIZATION_KWARGS[STRNAMES.CLUSTERING]['value'])
        elif self.INITIALIZATION_KWARGS[STRNAMES.CLUSTERING]['value_option'] == 'manual':
            clus_init = 'Manually with assignments {}'.format(
                self.INITIALIZATION_KWARGS[STRNAMES.CLUSTERING]['value'])
        elif self.INITIALIZATION_KWARGS[STRNAMES.CLUSTERING]['value_option'] == 'random':
            clus_init = 'Randomly, with {} clusters'.format(
                self.INITIALIZATION_KWARGS[STRNAMES.CLUSTERING]['n_clusters'])
        elif self.INITIALIZATION_KWARGS[STRNAMES.CLUSTERING]['value_option'] == 'taxonomy':
            clus_init = 'By taxonomic similarity, with {} clusters'.format(
                self.INITIALIZATION_KWARGS[STRNAMES.CLUSTERING]['n_clusters'])
        else:
            clus_init = 'Something slse'


        f = open(fname, 'w')
        f.write(mystr.format(
            seed=self.SEED, n_samples=self.N_SAMPLES,
            burnin=self.BURNIN, model_path=self.MODEL_PATH,
            a0=self.NEGBIN_A0, a1=self.NEGBIN_A1,
            params_learned=params_learned,
            clus_ind_prior=self.INITIALIZATION_KWARGS[STRNAMES.CLUSTER_INTERACTION_INDICATOR_PROB]['hyperparam_option'],
            pert_ind_prior=self.INITIALIZATION_KWARGS[STRNAMES.PERT_INDICATOR_PROB]['hyperparam_option'],
            filt=self.INITIALIZATION_KWARGS[STRNAMES.FILTERING]['x_value_option'],
            clus_init=clus_init))
        f.write("Complete initialization kwargs: \n")
        f.write(json.dumps(self.INITIALIZATION_KWARGS, indent=2))
        f.close()


class FilteringConfig(pl.Saveable):
    '''These are the parameters for Filtering

    Consistency filtering
    ----------------------------
    Filters the subjects by looking at the consistency of the counts.
    There must be at least `min_num_counts` for at least
    `min_num_consecutive` consecutive timepoints for at least
    `min_num_subjects` subjects for the  to be classified as valid.

    Parameters
    ----------
    min_num_consec: int
        This is the minimum number of consecutive timepoints that there
        must be at least `min_num_counts`
    threshold : float, int
        This is the minimum number of counts/abudnance/relative abundance that 
        there must be at each consecutive timepoint
    min_num_subjects : int, None
        This is how many subjects this must be true for for the Taxa to be
        valid. If it is None then it only requires one subject.
    colonization_time : float
        How many days we consider colonization (ignore during filtering)
    '''
    def __init__(self, colonization_time: float, threshold: float, 
        min_num_subj: int, min_num_consec: int):

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


class NegBinConfig(_BaseModelConfig):
    '''Configuration class for learning the negative binomial dispersion
    parameters. Note that these parameters are learned offline.

    System initialization
    ---------------------
    - OUTPUT_BASEPATH : str
        - Path to save the model
    - SEED : int
        - Seed to initialize inference with
    - BURNIN : int
        - Number of initial Gibb steps to throw away
    - N_SAMPLES : int
        - Total number of Gibb steps
    - CHECKPOINT : int
        - How often to write to disk
    - MP_FILTERING : str
        - How to do multiprocessing for filtering
    - LEARN : Dict[str, bool]
        - These are the dictionary of parameters which we are learning in the model.
        - If the name maps to True, then we learn it during inference. If it maps to 
          false, then we do not update its value duting inference.
    - INFERENCE_ORDER : list
        - This is the order to update the parameters during MCMC inference
    - INITIALIZATION_KWARGS : Dict[str, Dict[str, Any]]
        - These are the parameters to send into the `initialize` function for each variable
          that we are learning
    - INITIALIZATION_ORDER : list
        - This is the order to initialize the variables

    Parameters
    ----------
    seed : int
        Seed to start the inderence
    burnin, n_samples : int
        How many iterations for burn-in and total samples, respectively.
    checkpoint : int
        How often to write the trace in RAM to disk. Note that this must be
        a multiple of both `burnin` and `n_samples`
    basepath : str
        This is the basepath to save the graph. A separate folder within
        `basepath` will be created for the specific graph.
    synth : bool
        If True, run with the synthetic data, where the parameters needed
        to learn are `SYNTHETIC_A0` AND `SYNTHETIC_A1`.
    '''

    def __init__(self, seed: int, burnin: int, n_samples: int, checkpoint: int, 
        basepath: str):
        if basepath[-1] != '/':
            basepath += '/'

        self.SEED = seed
        self.OUTPUT_BASEPATH = os.path.abspath(basepath)
        self.MODEL_PATH = self.OUTPUT_BASEPATH
        self.BURNIN = burnin
        self.N_SAMPLES = n_samples
        self.CHECKPOINT = checkpoint
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
