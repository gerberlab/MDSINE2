'''These are the names, IDs, and Latex representations of the 
parameters that we are learning.
'''
class _BaseNameClass:
    def __iter__(self):
        a = vars(self).values()
        for b in a:
            yield b

    def __contains__(self, key):
        return key in vars(self).values()

    def is_perturbation_param(self, name):
        return name in [
            self.PERT_VALUE, 
            self.PERT_INDICATOR, 
            self.PERT_INDICATOR_PROB,
            self.PRIOR_MEAN_PERT,
            self.PRIOR_VAR_PERT]


class StrNamesClass(_BaseNameClass):
    '''String representation of each of the variables

    Names
    -----
    - CLUSTERING_OBJ: str
        - This is the name for the `mdsine2.pylab.cluster.Clustering` object.
    - CLUSTERING: str
        - This is the name for the `mdsine2.posterior.ClusterAssignments` object.
    - CONCENTRATION: str
        - This is the name for the `mdsine2.posterior.Concentration` object.
    - LATENT_TRAJECTORY: str
        - This is the name for the `mdsine2.posterior.TrajectorySet` object.
    - FILTERING: str
        - This is the name for the `mdsine2.posterior.FilteringLogMP` object.
    - ZERO_INFLATION: str
        - This is the name for the `mdsine2.posterior.ZeroInflation` object.
    - PROCESSVAR: str
        - This is the name for the `mdsine2.posterior.ProcessVarGlobal` object.
    - NEGBIN_A0: str
        - This is the name for the `mdsine2.negbin.NegBinDispersionParam` object.
    - NEGBIN_A1: str
        - This is the name for the `mdsine2.negbin.NegBinDispersionParam` object.
    - GLV_PARAMETERS: str
        - This is the name for the `mdsine2.posterior.GLVParameters` object.
    - GROWTH_VALUE: str
        - This is the name for the `mdsine2.posterior.Growth` object.
    - PRIOR_VAR_GROWTH: str
        - This is the name for one of the `mdsine2.posterior.PriorVarMH` object instantiations.
    - PRIOR_MEAN_GROWTH: str
        - This is the name for one of the `mdsine2.posterior.PriorMeanMH` object instantiations.
    - SELF_INTERACTION_VALUE: str
        - This is the name for the `mdsine2.posterior.SelfInteractions` object.
    - PRIOR_VAR_SELF_INTERACTIONS: str
        - This is the name for one of the `mdsine2.posterior.PriorVarMH` object instantiations.
    - PRIOR_MEAN_SELF_INTERACTIONS: str
        - This is the name for one of the `mdsine2.posterior.PriorMeanMH` object instantiations.
    - INTERACTIONS_OBJ: str
        - This is the name for the `mdsine2.pylab.contrib.Interactions` object.
    - CLUSTER_INTERACTION_VALUE: str
        - This is the name for the `mdsine2.posterior.ClusterInteractionValue` object.
    - CLUSTER_INTERACTION_INDICATOR: str
        - This is the name for the `mdsine2.posterior.ClusterInteractionIndicators` object.
    - CLUSTER_INTERACTION_INDICATOR_PROB: str
        - This is the name for the `mdsine2.posterior.ClusterInteractionProbability` object.
    - PRIOR_VAR_INTERACTIONS: str
        - This is the name for the `mdsine2.posterior.PriorVarInteractions` object.
    - PRIOR_MEAN_INTERACTIONS: str
        - This is the name for the `mdsine2.posterior.PriorMeanInteractions` object.
    - PERT_VALUE: str
        - This is the name for the `mdsine2.posterior.PerturbationMagnitudes` object.
    - PERT_INDICATOR: str
        - This is the name for the `mdsine2.posterior.PerturbationIndicators` object.
    - PERT_INDICATOR_PROB: str
        - This is the name for the `mdsine2.posterior.PerturbationProbabilities` object.
    - PRIOR_VAR_PERT: str
        - This is the name for the `mdsine2.posterior.PriorVarPerturbations` object.
    - PRIOR_MEAN_PERT: str
        - This is the name for the `mdsine2.posterior.PriorMeanPerturbations` object.
    - QPCR_VARIANCES: str
        - This is the name for the `mdsine2.posterior.qPCRVariances` object.
    - QPCR_DOFS: str
        - This is the name for the `mdsine2.posterior.qPCRDegreesOfFreedoms` object.
    - QPCR_SCALES: str
        - This is the name for the `mdsine2.posterior.qPCRScales` object.
    '''
    def __init__(self):
        #  GARY MODIFIED ***
        # temporarily added to pass path to zero-inflation data-file; this should be 
        # done in a better way, probably as option to the 'study' object or option to
        # set during runtime based on abundance values
        self.ZERO_INFLATION_DATA_PATH = None

        self.CLUSTERING_OBJ = 'Clustering object (contains ability to change the cluster assignments)'
        self.CLUSTERING = 'Clustering parameter'
        self.CONCENTRATION = 'Clustering concentration parameter'

        self.LATENT_TRAJECTORY = 'Latent trajectory parameter'
        self.FILTERING = 'Filtering'
        self.ZERO_INFLATION = 'Zero inflation'
        self.PROCESSVAR = 'Process Variance parameter'

        self.NEGBIN_A0 = 'Negative binomial dispersion a0'
        self.NEGBIN_A1 = 'Negative binomial dispersion a1'

        self.GLV_PARAMETERS = 'Logistic growth parameters (growth, self-interactions, interactions/perturbations)'

        self.GROWTH_VALUE = 'Growth parameter'
        self.PRIOR_VAR_GROWTH = 'Variance parameter for the truncated normal prior of the growth parameter'
        self.PRIOR_MEAN_GROWTH = 'Mean parameter for the truncated normal prior of the growth parameter'

        self.SELF_INTERACTION_VALUE = 'Self interaction parameter'
        self.PRIOR_VAR_SELF_INTERACTIONS = 'Variance parameter for the truncated normal prior of the self-interaction parameter'
        self.PRIOR_MEAN_SELF_INTERACTIONS = 'Mean parameter for the truncated normal prior of the self-interaction parameter'
        
        self.INTERACTIONS_OBJ = 'Interactions object'
        self.CLUSTER_INTERACTION_VALUE = 'Cluster interaction value parameter'
        self.CLUSTER_INTERACTION_INDICATOR = 'Cluster interaction indicator parameter'
        self.CLUSTER_INTERACTION_INDICATOR_PROB = 'Cluster interaction probability'
        self.PRIOR_VAR_INTERACTIONS = 'Variance parameter for the normal prior of the interaction parameter'
        self.PRIOR_MEAN_INTERACTIONS = 'Mean parameter for the normal prior of the interaction parameter'

        self.PERT_VALUE = 'Perturbation value parameter'
        self.PERT_INDICATOR = 'Perturbation indicator parameter'
        self.PERT_INDICATOR_PROB = 'Probability parameter for the beta prior of the perturbation indicator parameter'
        self.PRIOR_VAR_PERT = 'Variance parameter for the normal prior of the perturbation parameter'
        self.PRIOR_MEAN_PERT = 'Mean parameter for the normal prior of the perturbation parameter'

        self.QPCR_VARIANCES = 'qPCR variances'
        self.QPCR_DOFS = 'qPCR hyperprior degrees of freedom'
        self.QPCR_SCALES = 'qPCR hyperprior scales'

        
class LatexNamesClass(_BaseNameClass):
    '''Latex representation of the variables

    Make sure to do a double \\ for latex names orelse the matplotlib
    string parser will throw an error'''

    # Use the pylab.taxaname_formatter method to replace each index
    def __init__(self):
        self.DATA = 'data'
        self.GROWTH_VALUE = '$a_{%(index)s,1}$'
        self.SELF_INTERACTION_VALUE = '$a_{%(index)s,2}$'
        self.CLUSTER_INTERACTION_VALUE = '$b_{(c_i, c_j)}$'

        self.CLUSTER_INTERACTION_INDICATOR = '$z^{(b)}_{(c_i, c_j)}$'
        self.CLUSTER_INTERACTION_INDICATOR_PROB = '$\\pi_z$'
        self.CLUSTERING = '$c_i$'
        self.LATENT_TRAJECTORY = '$x$'
        self.CONCENTRATION = '$\\alpha$'
        #necessary for homoscedastic
        self.PROCESSVAR = '$\\sigma^2_{w}$'

        self.NEGBIN_A0 = '$a_0$'
        self.NEGBIN_A1 = '$a_1$'

        self.ZERO_INFLATION = '$w$'

        self.PRIOR_VAR_GROWTH = '$\\sigma^2_{a_1}$'
        self.PRIOR_VAR_SELF_INTERACTIONS = '$\\sigma^2_{a_2}$'
        self.PRIOR_VAR_INTERACTIONS = '$\\sigma^2_{b}$'
        self.PRIOR_VAR_PERT = '$\\sigma^2_{\\gamma}$'

        self.PRIOR_MEAN_GROWTH = '$\\mu_{a_1}$'
        self.PRIOR_MEAN_SELF_INTERACTIONS = '$\\mu_{a_2}$'
        self.PRIOR_MEAN_INTERACTIONS = '$\\mu_{b}$'
        self.PRIOR_MEAN_PERT = '$\\mu_{\\gamma}$'

        self.PERT_VALUE = '$\\gamma$'
        self.PERT_INDICATOR = '$z^{(\\gamma)}_{(c_i, c_j)}$'
        self.PERT_INDICATOR_PROB = '$\\pi_(\\gamma)$'

        self.QPCR_VARIANCES = '$\\sigma^2_{Q_s (k)}$'
        self.QPCR_DOFS = '$\\nu_Q ( \\omega_l )$'
        self.QPCR_SCALES = '$\\tau^2_Q ( \\omega_l )$'


STRNAMES = StrNamesClass()
LATEXNAMES = LatexNamesClass()
