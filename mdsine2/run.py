from mdsine2.logger import logger
import numpy as np
import h5py
import os

from typing import Union, Dict, Iterator, Tuple, List, Any, IO, Callable

# Custom modules
from . import config
from .names import STRNAMES
from . import design_matrices
from . import posterior

from . import pylab as pl
from .pylab import Study, BaseMCMC

def initialize_graph(params: config.MDSINE2ModelConfig, graph_name: str, subjset: Study, continue_inference: int=None, 
    intermediate_validation: Dict[str, Union[float, Callable, Dict[str, Any]]]=None) -> BaseMCMC:
    '''Builds the graph with the posterior classes and creates an
    mdsine2.BaseMCMC inference chain object that you ran run inference with

    Parameters
    ----------
    params : mdsine2.config.MDSINE2ModelConfig
        This class specifies all of the parameters of the model.
    graph_name : str
        Name of the graph you want to build
    subjset : mdsine2.Study
        This is the subjectset that contains all of the trajectories for inference.
        Note that this subjectset has already been filtered
    continue_inference : int, None
        Gibb sample to restart inference from
    intermediate_validation : dict
        A dictionary with the following arguments:
            't' : float
                This is how often, in seconds, you want to run this function
            'func' : callable
                This function is called every `t` seconds
            'kwargs' : dict
                These are additional arguments to pass into fucntion `func`

    Returns
    -------
    pl.inference.BaseMCMC
        Inference chain
    '''
    # Type Check
    # ----------
    if not config.isModelConfig(params):
        raise TypeError('`params` ({}) needs to be a config.ModelConfig object'.format(type(params)))
    if not pl.isstudy(subjset):
        raise TypeError('`subjset` ({}) must be a mdsine2.Study'.format(type(subjset)))
    if not pl.isstr(graph_name):
        raise TypeError('`graph_name` ({}) must be a str'.format(type(graph_name)))
    if continue_inference is not None:
        if not pl.isint(continue_inference):
            raise TypeError('`continue_inference` ({}) must be an int'.format(type(continue_inference)))
        if continue_inference <= 0:
            raise ValueError('`continue_inference` ({}0 must be > 0'.format(continue_inference))

        GRAPH = pl.graph.Graph.load(params.GRAPH_FILENAME)
    else:
        GRAPH = pl.Graph(name=graph_name, seed=params.SEED)
    GRAPH.as_default()
    pl.seed(params.SEED)

    # Continue inference if necessary
    # -------------------------------
    if continue_inference is not None:
        logger.info('Continuing inference at Gibb step {}'.format(continue_inference))
        mcmc = pl.inference.BaseMCMC.load(params.MCMC_FILENAME)
        mcmc.continue_inference(gibb_step_start=continue_inference)

        return mcmc

    # Make the basepath
    # -----------------
    basepath = params.OUTPUT_BASEPATH
    os.makedirs(basepath, exist_ok=True)

    # Normalize the qpcr measurements for numerical stability
    # -------------------------------------------------------
    if params.QPCR_NORMALIZATION_MAX_VALUE is not None:
        subjset.normalize_qpcr(max_value=params.QPCR_NORMALIZATION_MAX_VALUE)
        logger.info('Normalizing abundances for a max value of {}. Normalization ' \
            'constant: {:.4E}'.format(params.QPCR_NORMALIZATION_MAX_VALUE, 
            subjset.qpcr_normalization_factor))
        
        params.INITIALIZATION_KWARGS[STRNAMES.FILTERING]['v2'] *= subjset.qpcr_normalization_factor
        params.INITIALIZATION_KWARGS[STRNAMES.SELF_INTERACTION_VALUE]['rescale_value'] = \
            subjset.qpcr_normalization_factor
    
    subjset_filename = os.path.join(basepath, config.SUBJSET_FILENAME)
    subjset.save(subjset_filename)

    # Instantiate the posterior classes
    # ---------------------------------
    taxa = subjset.taxa
    d = design_matrices.Data(subjects=subjset, G=GRAPH, 
        zero_inflation_transition_policy=params.ZERO_INFLATION_TRANSITION_POLICY)
    clustering = pl.Clustering(clusters=None, items=taxa, G=GRAPH,
        name=STRNAMES.CLUSTERING_OBJ)

    # Interactions
    var_interactions = posterior.PriorVarInteractions(
        prior=pl.variables.SICS(
            dof=pl.Constant(None, G=GRAPH),
            scale=pl.Constant(None, G=GRAPH), 
        G=GRAPH), G=GRAPH)
    mean_interactions = posterior.PriorMeanInteractions(
        prior=pl.variables.Normal(
            loc=pl.Constant(None, G=GRAPH),
            scale2=pl.Constant(None, G=GRAPH), 
        G=GRAPH), G=GRAPH)
    interaction_value = pl.variables.Normal(
        loc=mean_interactions, scale2=var_interactions, G=GRAPH)

    interaction_indicator = posterior.ClusterInteractionIndicatorProbability(
        prior=pl.variables.Beta(a=pl.Constant(None, G=GRAPH), b=pl.Constant(None, G=GRAPH), G=GRAPH),
        G=GRAPH)
    interactions = posterior.ClusterInteractionValue(
        prior=interaction_value, clustering=clustering, G=GRAPH)
    Z = posterior.ClusterInteractionIndicators(prior=interaction_indicator, G=GRAPH)

    # Growth
    var_growth = posterior.PriorVarMH(
        prior=pl.variables.SICS(
            dof=pl.Constant(None, G=GRAPH),
            scale=pl.Constant(None, G=GRAPH), G=GRAPH),
        child_name=STRNAMES.GROWTH_VALUE, G=GRAPH)
    mean_growth = posterior.PriorMeanMH(
        prior=pl.variables.TruncatedNormal(
            loc=pl.Constant(None, G=GRAPH),
            scale2=pl.Constant(None, G=GRAPH), G=GRAPH), 
        child_name=STRNAMES.GROWTH_VALUE, G=GRAPH)
    prior_growth = pl.variables.TruncatedNormal(
        loc=mean_growth, scale2=var_growth,
        name='prior_{}'.format(STRNAMES.GROWTH_VALUE), G=GRAPH)
    growth = posterior.Growth(prior=prior_growth, G=GRAPH)

    # Self-Interactions
    var_si = posterior.PriorVarMH(
        prior=pl.variables.SICS(
            dof=pl.Constant(None, G=GRAPH),
            scale=pl.Constant(None, G=GRAPH), G=GRAPH),
        child_name=STRNAMES.SELF_INTERACTION_VALUE, G=GRAPH)
    mean_si = posterior.PriorMeanMH(
        prior=pl.variables.TruncatedNormal(
            loc=pl.Constant(None, G=GRAPH),
            scale2=pl.Constant(None, G=GRAPH), G=GRAPH), 
        child_name=STRNAMES.SELF_INTERACTION_VALUE, G=GRAPH)
    prior_si = pl.variables.TruncatedNormal(
        loc=mean_si, scale2=var_si,
        name='prior_{}'.format(STRNAMES.SELF_INTERACTION_VALUE), G=GRAPH)
    self_interactions = posterior.SelfInteractions(prior=prior_si, G=GRAPH)

    # Process Variance
    prior_processvar = pl.variables.SICS( 
        dof=pl.Constant(None, G=GRAPH),
        scale=pl.Constant(None, G=GRAPH), G=GRAPH)
    processvar = posterior.ProcessVarGlobal(G=GRAPH, prior=prior_processvar)

    # Clustering
    prior_concentration = pl.variables.Gamma(
        shape=pl.Constant(None, G=GRAPH),
        scale=pl.Constant(None, G=GRAPH),
        G=GRAPH)
    concentration = posterior.Concentration(
        prior=prior_concentration, G=GRAPH)
    cluster_assignments = posterior.ClusterAssignments(
        clustering=clustering, concentration=concentration,
        G=GRAPH, mp=params.MP_CLUSTERING)

    # Filtering and zero inflation
    filtering = posterior.FilteringLogMP(G=GRAPH, mp=params.MP_FILTERING, 
        zero_inflation_transition_policy=params.ZERO_INFLATION_TRANSITION_POLICY)
    zero_inflation = posterior.ZeroInflation(G=GRAPH)

    # Perturbations
    if subjset.perturbations is not None:
        for pidx, subj_pert in enumerate(subjset.perturbations):
            name = subj_pert.name
            perturbation = pl.ClusterPerturbationEffect(
                starts=subj_pert.starts, ends=subj_pert.ends, 
                probability=pl.variables.Beta(
                    name=name + '_probability', G=GRAPH, value=None, a=None, b=None),
                clustering=clustering, G=GRAPH, name=name,
                signal_when_clusters_change=False, signal_when_item_assignment_changes=False)

            magnitude_var = posterior.PriorVarPerturbationSingle(
                prior=pl.variables.SICS(
                    dof=pl.Constant(None, G=GRAPH),
                    scale=pl.Constant(None, G=GRAPH), G=GRAPH), 
                perturbation=perturbation, G=GRAPH)
            magnitude_mean = posterior.PriorMeanPerturbationSingle(
                prior=pl.variables.Normal(
                    loc=pl.Constant(None, G=GRAPH),
                    scale2=pl.Constant(None, G=GRAPH),
                    G=GRAPH), 
                perturbation=perturbation, G=GRAPH)
            prior_magnitude = pl.variables.Normal(G=GRAPH, loc=magnitude_mean, scale2=magnitude_var)
            perturbation.magnitude.add_prior(prior_magnitude)

            prior_prob = pl.variables.Beta(
                a=pl.Constant(None, G=GRAPH),
                b=pl.Constant(None, G=GRAPH),
                G=GRAPH)
            perturbation.probability.add_prior(prior_prob)
        
        magnitude_var_perts = posterior.PriorVarPerturbations(G=GRAPH)
        magnitude_mean_perts = posterior.PriorMeanPerturbations(G=GRAPH)
        magnitude_perts = posterior.PerturbationMagnitudes(G=GRAPH)
        indicator_perts = posterior.PerturbationIndicators(G=GRAPH, need_to_trace=False, relative=True)
        indicator_prob_perts = posterior.PerturbationProbabilities(G=GRAPH)
    else:
        magnitude_perts = None
        pert_ind = None
        pert_ind_prob = None

    beta = posterior.GLVParameters(
        growth=growth, self_interactions=self_interactions,
        interactions=interactions, pert_mag=magnitude_perts, G=GRAPH)

    # Set qPCR variance priors and hyper priors
    qpcr_variances = posterior.qPCRVariances(G=GRAPH, L=params.N_QPCR_BUCKETS)
    qpcr_dofs = posterior.qPCRDegsOfFreedoms(G=GRAPH, L=params.N_QPCR_BUCKETS)
    qpcr_scales = posterior.qPCRScales(G=GRAPH, L=params.N_QPCR_BUCKETS)

    for l in range(params.N_QPCR_BUCKETS):
        qpcr_scale_prior = pl.variables.SICS( 
            dof=pl.Constant(None, G=GRAPH),
            scale=pl.Constant(None, G=GRAPH),
            name='prior_' + STRNAMES.QPCR_SCALES + '_{}'.format(l), G=GRAPH)
        qpcr_dof_prior = pl.variables.Uniform(
            low=pl.Constant(None, G=GRAPH),
            high=pl.Constant(None, G=GRAPH),
            name='prior_' + STRNAMES.QPCR_DOFS + '_{}'.format(l), G=GRAPH)
        
        # add priors
        qpcr_dofs.value[l].add_prior(qpcr_dof_prior)
        qpcr_scales.value[l].add_prior(qpcr_scale_prior)

    # Allocate qpcr measurements into buckets
    mean_log_measurements = []
    indices = []
    for ridx in range(d.n_replicates):
        for tidx,t in enumerate(d.given_timepoints[ridx]):
            mean_log_measurements.append(np.mean(d.qpcr[ridx][t].log_data))
            indices.append((ridx, tidx))

    idxs = np.argsort(mean_log_measurements)
    l_len = int(len(mean_log_measurements)/params.N_QPCR_BUCKETS)
    logger.info('There are {} qPCR measurements for {} buckets. Each bucket is' \
        ' {} measurements long'.format(len(indices), params.N_QPCR_BUCKETS, l_len))

    iii = 0
    for bucket_idx in range(params.N_QPCR_BUCKETS):
        # If it is the last bucket, assign the rest of the elements to it
        if bucket_idx == params.N_QPCR_BUCKETS - 1:
            l_len = len(mean_log_measurements) - iii
        for i in range(l_len):
            idx = idxs[iii]
            ridx,tidx = indices[idx]
            qpcr_variances.add_qpcr_measurement(ridx=ridx, tidx=tidx, sidx=bucket_idx)
            qpcr_dofs.add_qpcr_measurement(ridx=ridx, tidx=tidx, sidx=bucket_idx)
            qpcr_scales.add_qpcr_measurement(ridx=ridx, tidx=tidx, sidx=bucket_idx)
            iii += 1
    qpcr_dofs.set_shape()
    qpcr_scales.set_shape()

    # Set up inference and the inference order.
    # -----------------------------------------
    mcmc = pl.BaseMCMC(burnin=params.BURNIN, n_samples=params.N_SAMPLES, graph=GRAPH)
    order = []
    for name in params.INFERENCE_ORDER:
        if params.LEARN[name]:
            if not STRNAMES.is_perturbation_param(name):
                order.append(name)
            elif subjset.perturbations is not None:
                order.append(name)
    mcmc.set_inference_order(order)

    if intermediate_validation is not None:
        mcmc.set_intermediate_validation(**intermediate_validation)
    
    # Initialize the posterior and instantiate the design matrices
    # ------------------------------------------------------------
    for name in params.INITIALIZATION_ORDER:
        logger.info('Initializing {}'.format(name))
        if STRNAMES.is_perturbation_param(name) and subjset.perturbations is None:
            logger.info('Skipping over {} because it is a perturbation parameter ' \
                'and there are no perturbations'.format(name))
            continue
        
        # Call `initialize`
        try:
            GRAPH[name].initialize(**params.INITIALIZATION_KWARGS[name])
        except Exception as error:
            logger.critical('Initialization in `{}` failed with the parameters: {}'.format(
                name, params.INITIALIZATION_KWARGS[name]) + ' with the follwing error:\n{}'.format(
                    error))
            for a in GRAPH._persistent_pntr:
                a.kill()
            raise

        # Initialize data matrices if necessary
        if name == STRNAMES.ZERO_INFLATION:
            # Initialize the basic data matrices after initializing filtering
            lhs = design_matrices.LHSVector(G=GRAPH, name='lhs_vector')
            lhs.build()
            growthDM = design_matrices.GrowthDesignMatrix(G=GRAPH)
            growthDM.build_without_perturbations()
            selfinteractionsDM = design_matrices.SelfInteractionDesignMatrix(G=GRAPH)
            selfinteractionsDM.build()
        if name == STRNAMES.CLUSTER_INTERACTION_INDICATOR:
            # Initialize the interactions data matrices after initializing the interactions
            interactionsDM = design_matrices.InteractionsDesignMatrix(G=GRAPH)
            interactionsDM.build()
        if name == STRNAMES.PERT_INDICATOR and subjset.perturbations is not None:
            # Initialize the perturbation data matrices after initializing the perturbations
            perturbationsDM = design_matrices.PerturbationDesignMatrix(G=GRAPH)
            perturbationsDM.base.build()
            perturbationsDM.M.build()
        if name == STRNAMES.PERT_VALUE and subjset.perturbations is not None:
            d.design_matrices[STRNAMES.GROWTH_VALUE].build_with_perturbations()

    logger.info('\n\n\n')
    logger.info('Initialization Values:')
    logger.info('Growth')
    logger.info('\tprior.loc: {}'.format(GRAPH[STRNAMES.GROWTH_VALUE].prior.loc.value))
    logger.info('\tprior.scale2: {}'.format(GRAPH[STRNAMES.GROWTH_VALUE].prior.scale2.value))
    logger.info('\tvalue: {}'.format(GRAPH[STRNAMES.GROWTH_VALUE].value.flatten()))

    logger.info('Self-Interactions')
    logger.info('\tprior.loc: {}'.format(GRAPH[STRNAMES.SELF_INTERACTION_VALUE].prior.loc.value))
    logger.info('\tprior.scale2: {}'.format(GRAPH[STRNAMES.SELF_INTERACTION_VALUE].prior.scale2.value))
    logger.info('\tvalue: {}'.format(GRAPH[STRNAMES.SELF_INTERACTION_VALUE].value.flatten()))

    logger.info('Prior Variance Growth')
    logger.info('\tprior.dof: {}'.format(GRAPH[STRNAMES.PRIOR_VAR_GROWTH].prior.dof.value))
    logger.info('\tprior.scale: {}'.format(GRAPH[STRNAMES.PRIOR_VAR_GROWTH].prior.scale.value))
    logger.info('\tvalue: {}'.format(GRAPH[STRNAMES.PRIOR_VAR_GROWTH].value))

    logger.info('Prior Variance Self-Interactions')
    logger.info('\tprior.dof: {}'.format(GRAPH[STRNAMES.PRIOR_VAR_SELF_INTERACTIONS].prior.dof.value))
    logger.info('\tprior.scale: {}'.format(GRAPH[STRNAMES.PRIOR_VAR_SELF_INTERACTIONS].prior.scale.value))
    logger.info('\tvalue: {}'.format(GRAPH[STRNAMES.PRIOR_VAR_SELF_INTERACTIONS].value))

    logger.info('Prior Variance Interactions')
    logger.info('\tprior.dof: {}'.format(GRAPH[STRNAMES.PRIOR_VAR_INTERACTIONS].prior.dof.value))
    logger.info('\tprior.scale: {}'.format(GRAPH[STRNAMES.PRIOR_VAR_INTERACTIONS].prior.scale.value))
    logger.info('\tvalue: {}'.format(GRAPH[STRNAMES.PRIOR_VAR_INTERACTIONS].value))

    logger.info('Process Variance')
    logger.info('\tprior.dof: {}'.format(GRAPH[STRNAMES.PROCESSVAR].prior.dof.value))
    logger.info('\tprior.scale: {}'.format(GRAPH[STRNAMES.PROCESSVAR].prior.scale.value))
    logger.info('\tprior mean: {}'.format(GRAPH[STRNAMES.PROCESSVAR].prior.mean()))

    logger.info('Concentration')
    logger.info('\tprior.shape: {}'.format(GRAPH[STRNAMES.CONCENTRATION].prior.shape.value))
    logger.info('\tprior.scale: {}'.format(GRAPH[STRNAMES.CONCENTRATION].prior.scale.value))
    logger.info('\tvalue: {}'.format(GRAPH[STRNAMES.CONCENTRATION].value))

    logger.info('Indicator probability')
    logger.info('\tprior.a: {}'.format(GRAPH[STRNAMES.CLUSTER_INTERACTION_INDICATOR_PROB].prior.a.value))
    logger.info('\tprior.b: {}'.format(GRAPH[STRNAMES.CLUSTER_INTERACTION_INDICATOR_PROB].prior.b.value))
    logger.info('\tvalue: {}'.format(GRAPH[STRNAMES.CLUSTER_INTERACTION_INDICATOR_PROB].value))

    if subjset.perturbations is not None:
        logger.info('Perturbation values:')
        for perturbation in GRAPH.perturbations:
            logger.info('\tperturbation {}'.format(perturbation.name))
            logger.info('\t\tvalue: {}'.format(perturbation.magnitude.value))
            logger.info('\t\tprior.loc: {}'.format(perturbation.magnitude.prior.loc.value))
        logger.info('Perturbation prior variances:')
        for perturbation in GRAPH.perturbations:
            logger.info('\t\tdof: {}'.format(perturbation.magnitude.prior.scale2.prior.dof.value))
            logger.info('\t\tscale: {}'.format(perturbation.magnitude.prior.scale2.prior.scale.value))
            logger.info('\t\tvalue: {}'.format(perturbation.magnitude.prior.scale2.value))
        logger.info('Perturbation indicators:')
        for perturbation in GRAPH.perturbations:
            logger.info('\tperturbation {}: {}'.format(perturbation.name,
                perturbation.indicator.cluster_array()))
        logger.info('Perturbation indicator probability:')
        for perturbation in GRAPH.perturbations:
            logger.info('\tperturbation {}'.format(perturbation.name))
            logger.info('\t\tvalue: {}'.format(perturbation.probability.value))
            logger.info('\t\tprior.a: {}'.format(perturbation.probability.prior.a.value))
            logger.info('\t\tprior.b: {}'.format(perturbation.probability.prior.b.value))

    # Setup filenames
    # ---------------
    hdf5_filename = os.path.join(basepath, config.HDF5_FILENAME)
    mcmc_filename = os.path.join(basepath, config.MCMC_FILENAME)
    mcmc.set_tracer(filename=hdf5_filename, checkpoint=params.CHECKPOINT)
    mcmc.set_save_location(mcmc_filename)
    params.save(os.path.join(basepath, config.PARAMS_FILENAME))
    pl.seed(params.SEED)

    return mcmc

def run_graph(mcmc: BaseMCMC, crash_if_error: bool=True, log_every: int=100) -> BaseMCMC:
    '''Run the MCMC chain `mcmc`. Initialize the MCMC chain with `build_graph`

    Parameters
    ----------
    mcmc : mdsine2.BaseMCMC
        Inference object that is already built and initialized
    crash_if_error : bool
        If True, throws an error if there is an exception during inference. Otherwise
        it continues out of inference.

    Returns
    -------
    mdsine2.BaseMCMC
    '''
    try:
        mcmc.run(log_every=log_every)
    except Exception as e:
        logger.critical('CHAIN `{}` CRASHED'.format(mcmc.graph.name))
        logger.exception(e)
        if crash_if_error:
            raise
    mcmc.graph[STRNAMES.FILTERING].kill()
    mcmc.graph[STRNAMES.CLUSTERING].kill()

    if mcmc.graph.data.subjects.qpcr_normalization_factor is not None:
        mcmc, mcmc.graph.data.subjects = denormalize_parameters(mcmc)
    mcmc.graph.data.design_matrices = None
    return mcmc

def normalize_parameters(mcmc: BaseMCMC, subjset: Study) -> Tuple[BaseMCMC, Study]:
    '''Normalize the abundance of the parameters by the normalization factor
    in the subject set

    Parameters
    ----------
    mcmc : mdsine2.BaseMCMC
        This is the inference object that has all of the parameters in it
    subjset : mdsine2.Study
        This is the data object that contains all of the trajectories

    Returns
    -------
    mdsine2.BaseMCMC, mdsine2.Study
    '''
    GRAPH = mcmc.graph
    if subjset.qpcr_normalization_factor is not None:
        f = h5py.File(GRAPH.tracer.filename, 'r+', libver='latest')
        checkpoint = GRAPH.tracer.checkpoint

        try:
            GRAPH[STRNAMES.FILTERING].v2 *= subjset.qpcr_normalization_factor
        except:
            logger.info('v2 not applicable')

        # Adjust the self interactions if necessary
        if STRNAMES.SELF_INTERACTION_VALUE in mcmc.tracer.being_traced:
            dset = f[STRNAMES.SELF_INTERACTION_VALUE]
            dset[:,:] = dset[:,:] / subjset.qpcr_normalization_factor

        if STRNAMES.PRIOR_MEAN_SELF_INTERACTIONS in mcmc.tracer.being_traced:
            dset = f[STRNAMES.PRIOR_MEAN_SELF_INTERACTIONS]
            dset[:] = dset[:] / subjset.qpcr_normalization_factor

        if mcmc.is_in_inference_order(STRNAMES.PRIOR_VAR_SELF_INTERACTIONS):
            dset = f[STRNAMES.PRIOR_VAR_SELF_INTERACTIONS]
            dset[:] = dset[:] / (subjset.qpcr_normalization_factor**2)

        if mcmc.is_in_inference_order(STRNAMES.QPCR_VARIANCES):
            vs = GRAPH[STRNAMES.QPCR_VARIANCES]
            for l in range(vs.L):
                dset = f[STRNAMES.QPCR_VARIANCES + '_{}'.format(l)]
                dset[:] = dset[:] / (subjset.qpcr_normalization_factor**2)

        if mcmc.is_in_inference_order(STRNAMES.QPCR_SCALES):
            vs = GRAPH[STRNAMES.QPCR_SCALES]
            for l in range(vs.L):
                dset = f[STRNAMES.QPCR_SCALES + '_{}'.format(l)]
                dset[:] = dset[:] / (subjset.qpcr_normalization_factor**2)

        # Adjust the interactions if necessary
        if mcmc.tracer.is_being_traced(STRNAMES.INTERACTIONS_OBJ):
            dset = f[STRNAMES.INTERACTIONS_OBJ]
            total_samples = dset.attrs['end_iter']
            i = 0
            while (i * checkpoint) < total_samples:
                start_idx = int(i * checkpoint)
                end_idx = int((i+1) * checkpoint)

                if end_idx > total_samples:
                    end_idx = total_samples
                dset[start_idx: end_idx] = dset[start_idx: end_idx] / subjset.qpcr_normalization_factor
                i += 1
        
        if mcmc.is_in_inference_order(STRNAMES.PRIOR_MEAN_INTERACTIONS):
            dset = f[STRNAMES.PRIOR_MEAN_INTERACTIONS]
            dset[:] = dset[:] / subjset.qpcr_normalization_factor

        if mcmc.is_in_inference_order(STRNAMES.PRIOR_VAR_INTERACTIONS):
            dset = f[STRNAMES.PRIOR_VAR_INTERACTIONS]
            dset[:] = dset[:] / (subjset.qpcr_normalization_factor**2)

        if mcmc.is_in_inference_order(STRNAMES.FILTERING):
            for subj in subjset:
                name = STRNAMES.LATENT_TRAJECTORY + '_{}'.format(subj.name)
                if name not in f:
                    continue
                dset = f[name]
                total_samples = dset.attrs['end_iter']
                i = 0
                while (i * checkpoint) < total_samples:
                    start_idx = int(i * checkpoint)
                    end_idx = int((i+1) * checkpoint)

                    if end_idx > total_samples:
                        end_idx = total_samples
                    dset[start_idx: end_idx] = dset[start_idx: end_idx]*subjset.qpcr_normalization_factor
                    i += 1

        f.close()
    else:
        logger.info('Objects are already normalized')
    
    return mcmc, subjset

def denormalize_parameters(mcmc: BaseMCMC) -> Tuple[BaseMCMC, Study]:
    '''Denormalize the abundance of the parameters by the normalization factor
    in the subject set

    Parameters
    ----------
    mcmc : mdsine2.BaseMCMC
        This is the inference object that has all of the parameters in it
    subjset : mdsine2.Study
        This is the data object that contains all of the trajectories

    Returns
    -------
    mdsine2.BaseMCMC, mdsine2.Study
    '''
    GRAPH = mcmc.graph
    subjset = mcmc.graph.data.subjects
    if subjset.qpcr_normalization_factor is not None:
        logger.info('Denormalizing the parameters')

        f = h5py.File(GRAPH.tracer.filename, 'r+', libver='latest')
        checkpoint = GRAPH.tracer.checkpoint

        try:
            GRAPH[STRNAMES.FILTERING].v2 /= subjset.qpcr_normalization_factor
        except:
            logger.info('v2 not applicable')

        # Adjust the self interactions if necessary
        if STRNAMES.SELF_INTERACTION_VALUE in mcmc.tracer.being_traced:
            dset = f[STRNAMES.SELF_INTERACTION_VALUE]
            dset[:,:] = dset[:,:] * subjset.qpcr_normalization_factor

        if STRNAMES.PRIOR_MEAN_SELF_INTERACTIONS in mcmc.tracer.being_traced:
            dset = f[STRNAMES.PRIOR_MEAN_SELF_INTERACTIONS]
            dset[:] = dset[:] * subjset.qpcr_normalization_factor

        if mcmc.is_in_inference_order(STRNAMES.PRIOR_VAR_SELF_INTERACTIONS):
            dset = f[STRNAMES.PRIOR_VAR_SELF_INTERACTIONS]
            dset[:] = dset[:] * (subjset.qpcr_normalization_factor**2)

        if mcmc.is_in_inference_order(STRNAMES.QPCR_VARIANCES):
            vs = GRAPH[STRNAMES.QPCR_VARIANCES]
            for l in range(vs.L):
                dset = f[STRNAMES.QPCR_VARIANCES + '_{}'.format(l)]
                dset[:] = dset[:] * (subjset.qpcr_normalization_factor**2)

        if mcmc.is_in_inference_order(STRNAMES.QPCR_SCALES):
            vs = GRAPH[STRNAMES.QPCR_SCALES]
            for l in range(vs.L):
                dset = f[STRNAMES.QPCR_SCALES + '_{}'.format(l)]
                dset[:] = dset[:] * (subjset.qpcr_normalization_factor**2)

        # Adjust the interactions if necessary
        if mcmc.tracer.is_being_traced(STRNAMES.INTERACTIONS_OBJ):
            dset = f[STRNAMES.INTERACTIONS_OBJ]
            total_samples = dset.attrs['end_iter']
            i = 0
            while (i * checkpoint) < total_samples:
                start_idx = int(i * checkpoint)
                end_idx = int((i+1) * checkpoint)

                if end_idx > total_samples:
                    end_idx = total_samples
                dset[start_idx: end_idx] = dset[start_idx: end_idx] * subjset.qpcr_normalization_factor
                i += 1
        
        if mcmc.is_in_inference_order(STRNAMES.PRIOR_MEAN_INTERACTIONS):
            dset = f[STRNAMES.PRIOR_MEAN_INTERACTIONS]
            dset[:] = dset[:] * subjset.qpcr_normalization_factor

        if mcmc.is_in_inference_order(STRNAMES.PRIOR_VAR_INTERACTIONS):
            dset = f[STRNAMES.PRIOR_VAR_INTERACTIONS]
            dset[:] = dset[:] * (subjset.qpcr_normalization_factor**2)

        if mcmc.is_in_inference_order(STRNAMES.FILTERING):
            for subj in subjset:
                name = STRNAMES.LATENT_TRAJECTORY + '_{}'.format(subj.name)
                if name not in f:
                    continue
                dset = f[name]
                total_samples = dset.attrs['end_iter']
                i = 0
                while (i * checkpoint) < total_samples:
                    start_idx = int(i * checkpoint)
                    end_idx = int((i+1) * checkpoint)

                    if end_idx > total_samples:
                        end_idx = total_samples
                    dset[start_idx: end_idx] = dset[start_idx: end_idx]/subjset.qpcr_normalization_factor
                    i += 1

        f.close()
        subjset.denormalize_qpcr()
        mcmc.save()
    else:
        logger.info('Data already denormalized')
    return mcmc, subjset

def calculate_stability_over_gibbs(mcmc: BaseMCMC, section: str='auto', log_every: int=1000) -> np.ndarray:
    '''Calculate the stability over each of the Gibb steps in the chain `mcmc`

    stability = diag(r) @ A, where
        r : growth rates
        A : interaction matrix

    Note that if the growth, self_interactions, or interactions are fixed during inference,
    we use the fixed values

    Parameters
    ----------
    mcmc : mdsine2.BaseMCMC
        Inference object that contains the traces
    section : str
        This is the section of the trace we are calculating over. Options:
            'entire' : burn-in and the posterior
            'burnin' : just the burn-in samples
            'posterior' : just the posterior samples
            'auto' : most robust choice. If the chain did not do the total number of
                Gibb steps, it will only calcualte the stability over the Gibb steps that
                it has samples for. If the number is less than the burn-in, it will return the
                stability over the burn-in. If the number of Gibb steps is greater than the 
                number of burn-in. It will only return the stability of the samples in the posterior.
                If the inference has done the total number of gibb samples, then it is done over the
                entire posterior.
    log_every : int, None
        Print out the Gibb step the stability calcualtion is currently doing ever `log_every` Gibb
        steps. If None then there is not display.
    
    Returns
    -------
    np.ndarray (n_gibb, n_taxa, n_taxa)
    '''
    # Type check
    # ----------
    if not pl.isMCMC(mcmc):
        raise TypeError('`mcmc` ({}) must be a mdsine2.BaseMCMC object'.format(type(mcmc)))
    if log_every is not None:
        if not pl.isint(log_every):
            raise TypeError('`log_every` ({}) must be an int'.format(type(log_every)))
        if log_every <= 0:
            raise ValueError('`log_every` ({}) must be > 0'.format(log_every))

    smaller_arr = False
    if not pl.isstr(section):
        raise TypeError('`section` ({}) must be a str'.format(type(section)))
    if section == 'burnin':
        if mcmc.sample_iter >= mcmc.burnin:
            LEN_ARR = mcmc.burnin
        else:
            raise ValueError('chain `{}` only has {} Gibb steps but you chose to `burnin` ({}) section.'.format(
                mcmc.graph.name, mcmc.sample_iter, mcmc.burnin))
    elif section == 'posterior':
        if mcmc.ran:
            LEN_ARR = mcmc.n_samples - mcmc.burnin
        else:
            raise ValueError('chain `{}` only has {} Gibb steps but you chose to `posterior` ({}) section.'.format(
                mcmc.graph.name, mcmc.sample_iter, mcmc.n_samples))
    elif section == 'entire':
        if mcmc.ran:
            LEN_ARR = mcmc.n_samples
        else:
            raise ValueError('chain `{}` only has {} Gibb steps but you chose to `entire` ({}) section.'.format(
                mcmc.graph.name, mcmc.sample_iter, mcmc.n_samples))
    elif section == 'auto':
        if mcmc.ran:
            LEN_ARR = mcmc.n_samples
            section='posterior'
        elif mcmc.sample_iter > mcmc.burnin:
            section = 'posterior'
            LEN_ARR = mcmc.sample_iter + 1 - mcmc.burnin
            smaller_arr = True
        else:
            section = 'burnin'
            LEN_ARR = mcmc.sample_iter + 1
            smaller_arr = True
    else:
        raise ValueError('`section` ({}) not recognized'.format(section))

    # Load data
    # ---------
    N_TAXA = len(mcmc.graph.data.taxa)
    logger.info('Loading data for stability')
    growth = mcmc.graph[STRNAMES.GROWTH_VALUE]
    if mcmc.tracer.is_being_traced(STRNAMES.GROWTH_VALUE):
        growth = growth.get_trace_from_disk(section=section)
        if smaller_arr:
            growth = growth[:LEN_ARR, ...]
    else:
        growth = growth.value
        growth = growth.reshape(-1,1) + np.zeros(shape=(LEN_ARR, N_TAXA))

    si = mcmc.graph[STRNAMES.SELF_INTERACTION_VALUE]
    if mcmc.tracer.is_being_traced(STRNAMES.SELF_INTERACTION_VALUE):
        si = si.get_trace_from_disk(section=section)
        if smaller_arr:
            si = si[:LEN_ARR, ...]
    else:
        si = si.value
        si = si.reshape(-1,1) + np.zeros(shape=(LEN_ARR, N_TAXA))

    interactions = mcmc.graph[STRNAMES.GROWTH_VALUE]
    if mcmc.tracer.is_being_traced(STRNAMES.GROWTH_VALUE):
        interactions = interactions.get_trace_from_disk(section=section)
        interactions[np.isnan(interactions)] = 0
        if smaller_arr:
            interactions = interactions[:LEN_ARR, ...]
    else:
        interactions = interactions.get_datalevel_value_matrix(set_neg_indicators_to_nan=False)
        interactions = interactions.reshape(-1,1) + np.zeros(shape=(LEN_ARR, N_TAXA, N_TAXA))

    # Set the self-interactions as the diagonal
    for i in range(N_TAXA):
        interactions[:,i,i] = - np.absolute(si[:, i])

    # Calculate stability
    # -------------------
    if log_every is None:
        log_every = LEN_ARR + 1
    ret = np.zeros(shape=interactions.shape)
    for i in range(ret.shape[0]):
        if i % log_every == 0:
            if i == 0:
                continue
            else:
                logger.info('{}/{}'.foramt(i, LEN_ARR))
        
        ret[i] = np.diag(growth[i]) @ interactions[i]
    return ret
