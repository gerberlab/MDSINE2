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


class ReprNamesClass(_BaseNameClass):
    '''These are the IDs of each of the objects. You need to call the function
    `set` with the given graph in order for them to be valid.
    '''
    def __init__(self):
        self.REGRESSCOEFF = None
        self.GROWTH_VALUE = None
        self.SELF_INTERACTION_VALUE = None
        self.CLUSTER_INTERACTION_VALUE = None

        self.CLUSTERING_OBJ = None
        self.INTERACTIONS_OBJ = None

        self.CLUSTER_INTERACTION_INDICATOR = None
        self.INDICATOR_PROB = None

        self.CLUSTERING = None
        self.CONCENTRATION = None

        self.AUX_TRAJECTORY = None
        self.LATENT_TRAJECTORY = None
        self.FILTERING = None
        self.ZERO_INFLATION = None

        self.PROCESSVAR = None

        self.NEGBIN_A0 = None
        self.NEGBIN_A1 = None

        self.PRIOR_VAR_GROWTH = None
        self.PRIOR_VAR_SELF_INTERACTIONS = None
        self.PRIOR_VAR_INTERACTIONS = None
        self.PRIOR_VAR_PERT = None

        self.PRIOR_MEAN_GROWTH = None
        self.PRIOR_MEAN_SELF_INTERACTIONS = None
        self.PRIOR_MEAN_INTERACTIONS = None
        self.PRIOR_MEAN_PERT = None

        self.PERT_VALUE = None
        self.PERT_INDICATOR = None
        self.PERT_INDICATOR_PROB = None

        self.QPCR_VARIANCES = None
        self.QPCR_DOFS = None
        self.QPCR_SCALES = None

    def set(self, G):
        '''Sets the IDs of the above variables using the string representations.
        If these names are not initialized in the namespace then we do not add them
        '''
        for key in vars(self):
            strname = STRNAMES.__getattribute__(key)
            try:
                node = G[strname]
            except:
                # Node does not exist, continue to next
                continue
            setattr(self, key, node.id)


class StrNamesClass(_BaseNameClass):
    '''String representation of each of the variables
    '''
    def __init__(self):
        self.REGRESSCOEFF = 'beta'
        self.GROWTH_VALUE = 'growth'
        self.SELF_INTERACTION_VALUE = 'self_interactions'
        self.CLUSTER_INTERACTION_VALUE = 'cluster_interaction_values'

        self.CLUSTERING_OBJ = 'clustering_object'
        self.INTERACTIONS_OBJ = 'interactions_object'

        self.CLUSTER_INTERACTION_INDICATOR = 'cluster_interaction_indicators'
        self.INDICATOR_PROB = 'pi_z'

        self.CLUSTERING = 'ClusterAssignments'
        self.CONCENTRATION = 'clustering concentration'

        self.AUX_TRAJECTORY = 'q'
        self.LATENT_TRAJECTORY = 'x'
        self.FILTERING = 'filtering'
        self.ZERO_INFLATION = 'zero_inflation'

        self.PROCESSVAR = 'process_var'

        self.NEGBIN_A0 = 'a0'
        self.NEGBIN_A1 = 'a1'

        self.PRIOR_VAR_GROWTH = 'prior_var_growth'
        self.PRIOR_VAR_SELF_INTERACTIONS = 'prior_var_self_interactions'
        self.PRIOR_VAR_INTERACTIONS = 'prior_var_interactions'

        self.PRIOR_MEAN_GROWTH = 'prior_mean_growth'
        self.PRIOR_MEAN_SELF_INTERACTIONS = 'prior_mean_self_interactions'
        self.PRIOR_MEAN_INTERACTIONS = 'prior_mean_interactions'

        self.PERTURBATIONS = 'pert'
        self.PERT_VALUE = 'pert_value'
        self.PERT_INDICATOR = 'Z_pert'
        self.PERT_INDICATOR_PROB = 'pi_z_pert'
        self.PRIOR_VAR_PERT = 'prior_var_pert'
        self.PRIOR_MEAN_PERT = 'prior_mean_pert'

        self.QPCR_VARIANCES = 'qpcr_variance'
        self.QPCR_DOFS = 'qpcr_variance_dof'
        self.QPCR_SCALES = 'qpcr_variance_scale'

        
class LatexNamesClass(_BaseNameClass):
    '''Latex representation of the variables

    Make sure to do a double \\ for latex names orelse the matplotlib
    string parser will throw an error'''

    # Use the pylab.asvname_formatter method to replace each index
    def __init__(self):
        self.GROWTH_VALUE = '$a_{%(index)s,1}$'
        self.SELF_INTERACTION_VALUE = '$a_{%(index)s,2}$'
        self.CLUSTER_INTERACTION_VALUE = '$b_{(c_i, c_j)}$'

        self.CLUSTER_INTERACTION_INDICATOR = '$z^{(b)}_{(c_i, c_j)}$'
        self.INDICATOR_PROB = '$\\pi_z$'
        self.CLUSTERING = '$c_i$'
        self.AUX_TRAJECTORY = '$q$'
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
REPRNAMES = ReprNamesClass()
