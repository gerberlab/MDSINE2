'''qPCR variance parameters for the posterior
'''
import time
import numpy as np

from .util import build_prior_covariance, build_prior_mean
from .names import STRNAMES, REPRNAMES

from . import pylab as pl

class _qPCRBase(pl.Variable):
    '''Base class for qPCR measurements
    '''
    def __init__(self, L, **kwargs):

        pl.Variable.__init__(self, **kwargs)
        self._sample_iter = 0
        self.L = L
        self.n_replicates = self.G.data.n_replicates
        self.value = []

    def __getitem__(self, key):
        return self.value[key]

    def __str__(self):
        # Make them into an array?
        try:
            s = ''
            for a in self.value:
                s += str(a) + '\n'
        except:
            s = 'not set'
        return s

    def update(self):
        '''Update each of the qPCR variances
        '''
        for a in self.value:
            a.update()
        self._sample_iter += 1

    def initialize(self, **kwargs):
        '''Every variance gets the same initialization
        '''
        for a in self.value:
            a.initialize(**kwargs)

    def add_trace(self, *args, **kwargs):
        for a in self.value:
            a.add_trace(*args, **kwargs)

    def set_trace(self, *args, **kwargs):
        for a in self.value:
            a.set_trace(*args, **kwargs)

    def add_init_value(self):
        for a in self.value:
            a.add_init_value()


class _qPCRPriorAggVar(_qPCRBase):
    '''Base class for `qPCRDegsOfFreedoms` and `qPCRScales`
    '''
    def __init__(self, L, child,**kwargs):
        _qPCRBase.__init__(self, L, **kwargs)
        for l in range(self.L):
            self.value.append(
                child(L=L, l=l, **kwargs))

    def add_qpcr_measurement(self, ridx, tidx, l):
        '''Add a qPCR measurement for subject `ridx` at time index
        `tidx` to qPCR set `l`

        Parameters
        ----------
        ridx : int
            Subject index
        tidx : int
            Time index
        l : int
            qPCR set index
        '''
        if not pl.isint(ridx):
            raise TypeError('`ridx` ({}) must be an int'.format(type(ridx)))
        if ridx >= self.G.data.n_replicates:
            raise ValueError('`ridx` ({}) out of range ({})'.format(ridx, 
                self.G.data.n_replicates))
        if not pl.isint(tidx):
            raise TypeError('`tidx` ({}) must be an int'.format(type(tidx)))
        if tidx >= len(self.G.data.given_timepoints[ridx]):
            raise ValueError('`tidx` ({}) out of range ({})'.format(tidx, 
                len(self.G.data.given_timepoints[ridx])))
        if not pl.isint(l):
            raise TypeError('`l` ({}) must be an int'.format(type(l)))
        if l >= self.L:
            raise ValueError('`l` ({}) out of range ({})'.format(tidx, 
                self.L))
        self.value[l].add_qpcr_measurement(ridx=ridx, tidx=tidx)

    def set_shape(self):
        for a in self.value:
            a.set_shape()


class qPCRVariances(_qPCRBase):
    '''Aggregation class for qPCR variance for a set of qPCR variances. 
    The qPCR variances are 

    Parameters
    ----------
    L : int
        How many qPCR variance groupings there are
    '''
    def __init__(self, **kwargs):
        kwargs['name'] = STRNAMES.QPCR_VARIANCES
        _qPCRBase.__init__(self, **kwargs)
        self.G.data.qpcr_variances = self
        
        for ridx in range(self.n_replicates):
            self.value.append( 
                qPCRVarianceReplicate(ridx=ridx, **kwargs))

    def add_qpcr_measurement(self, ridx, tidx, l):
        '''Add a qPCR measurement for subject `ridx` at time index
        `tidx` to qPCR set `l`

        Parameters
        ----------
        ridx : int
            Subject index
        tidx : int
            Time index
        l : int
            qPCR set index
        '''
        if not pl.isint(ridx):
            raise TypeError('`ridx` ({}) must be an int'.format(type(ridx)))
        if ridx >= self.G.data.n_replicates:
            raise ValueError('`ridx` ({}) out of range ({})'.format(ridx, 
                self.G.data.n_replicates))
        if not pl.isint(tidx):
            raise TypeError('`tidx` ({}) must be an int'.format(type(tidx)))
        if tidx >= len(self.G.data.given_timepoints[ridx]):
            raise ValueError('`tidx` ({}) out of range ({})'.format(tidx, 
                len(self.G.data.given_timepoints[ridx])))
        if not pl.isint(l):
            raise TypeError('`l` ({}) must be an int'.format(type(l)))
        if l >= self.L:
            raise ValueError('`l` ({}) out of range ({})'.format(tidx, 
                self.L))
        self.value[ridx].add_qpcr_measurement(tidx=tidx, l=l)        


class qPCRVarianceReplicate(pl.variables.SICS):
    '''Posterior for a set of single qPCR variances for replicate `ridx`

    Parameters
    ----------
    ridx : int
        Which subject replicate index this set of qPCR variances belongs
        to.
    L : int
        How many qPCR variance groupings there are
    '''
    def __init__(self, ridx, L, **kwargs):
        self.ridx = ridx
        self.L = L
        kwargs['name'] = STRNAMES.QPCR_VARIANCES + '_{}'.format(ridx)
        pl.variables.SICS.__init__(self, **kwargs)
        self.priors_idx = np.full(len(self.G.data.given_timepoints[ridx]), -1, dtype=int)
        self.set_value_shape(shape=(len(self.G.data.given_timepoints[ridx]),))

    def initialize(self, value_option, value=None, inflated=None):
        '''Initialize the values. We do not set any hyperparameters because those are
        set in their own classes.

        Parameters
        ----------
        value_option : str
            How to initialize the variances
                'empirical', 'auto'
                    Set to the empirical variance of the respective measurements
                'inflated'
                    Set to an inflated value of the empirical variance.
                'manual'
                    Set the values manually
        value : float, np.ndarray(float)
            If float, set all the values to the same number. If array then set the 
            values to each of the parameters
        inflated : float, None
            Necessary if `value_option` == 'inflated'
        '''
        if not pl.isstr(value_option):
            raise TypeError('`value_option` ({}) must be a str'.format(type(value_option)))
        if value_option in ['empirical', 'auto']:
            self.value = np.zeros(len(self.G.data.qpcr[self.ridx]), dtype=float)
            for idx, t in enumerate(self.G.data.qpcr[self.ridx]):
                self.value[idx] = np.var(self.G.data.qpcr[self.ridx][t].log_data)

        elif value_option == 'inflated':
            if not pl.isnumeric(inflated):
                raise TypeError('`inflated` ({}) must be a numeric'.format(type(inflated)))
            if inflated < 0:
                raise ValueError('`inflated` ({}) must be positive'.format(inflated))
            # Set each variance by the empirical variance * inflated
            self.value = np.zeros(len(self.G.data.qpcr[self.ridx]), dtype=float)
            for idx, t in enumerate(self.G.data.qpcr[self.ridx]):
                self.value[idx] = np.var(self.G.data.qpcr[self.ridx][t].log_data) * inflated

        elif value_option == 'manual':
            raise NotImplementedError('Need to implement')
        else:
            raise ValueError('`value_option` ({}) not recognized'.format(value_option))

        # Set the qPCR measurements
        self.qpcr_measurements = []
        for tidx, t in enumerate(self.G.data.given_timepoints[self.ridx]):
            self.qpcr_measurements.append(self.G.data.qpcr[self.ridx][t].log_data)

    def update(self):

        prior_dofs = []
        prior_scales = []
        for l in range(self.L):
            prior_dofs.append(self.G[REPRNAMES.QPCR_DOFS].value[l].value)
            prior_scales.append(self.G[REPRNAMES.QPCR_SCALES].value[l].value)

        for tidx in range(len(self.priors_idx)):
            t = self.G.data.given_timepoints[self.ridx][tidx]
            l = self.priors_idx[tidx]
            prior_dof = prior_dofs[l]
            prior_scale = prior_scales[l]

            # qPCR measurements (these are already in log space)
            values = self.qpcr_measurements[tidx]

            # Current mean is the log of the sum of latent abundance
            tidx_in_arr = self.G.data.timepoint2index[self.ridx][t]
            mean = np.log(np.sum(self.G.data.data[self.ridx][:, tidx_in_arr]))

            # Calculate the residual sum
            resid_sum = np.sum(np.square(values - mean))

            # posterior
            dof = prior_dof + len(values)
            scale = ((prior_scale * prior_dof) + resid_sum)/dof
            self.value[tidx] = pl.random.sics.sample(dof, scale)

    def add_qpcr_measurement(self, tidx, l):
        '''Add qPCR measurement for subject index `ridx` and time index `tidx`

        Parameters
        ----------
        ridx : int
            Subject index
        tidx : int
            Time index
        '''
        if not pl.isint(tidx):
            raise TypeError('`tidx` ({}) must be an int'.format(type(tidx)))
        if tidx >= len(self.G.data.given_timepoints[self.ridx]):
            raise ValueError('`tidx` ({}) out of range ({})'.format(tidx, 
                len(self.G.data.given_timepoints[self.ridx])))
        if not pl.isint(l):
            raise TypeError('`l` ({}) must be an int'.format(type(l)))
        self.priors_idx[tidx] = l


class qPCRDegsOfFreedoms(_qPCRPriorAggVar):
    '''Aggregation class for a degree of freedom parameter of qPCR variance

    Parameters
    ----------
    L : int
        How many qPCR variance groupings there are
    '''
    def __init__(self, L, **kwargs):
        kwargs['name'] = STRNAMES.QPCR_DOFS
        _qPCRPriorAggVar.__init__(self, L=L, child=qPCRDegsOfFreedomL, **kwargs)


class qPCRDegsOfFreedomL(pl.variables.Uniform):
    '''Posterior for a single qPCR degrees of freedom parameter for a SICS set
    
    Parameters
    ----------
    L : int
        How many qPCR variance groupings there are
    l : int
        Which specific grouping this hyperprior is
    '''
    def __init__(self, L, l, **kwargs):

        self.L = L
        self.l = l
        kwargs['name'] = STRNAMES.QPCR_DOFS + '_{}'.format(l)
        pl.variables.Uniform.__init__(self, **kwargs)

        self.data_locs = []
        self.proposal = pl.variables.TruncatedNormal(mean=None, var=None, value=None)

    def __str__(self):
        # If this fails, it is because we are dividing by 0 sampler_iter
        # If which case we just return the value 
        try:
            s = 'Value: {}, Acceptance rate: {}'.format(
                self.value, np.mean(self.acceptances[
                    np.max([self.sample_iter-50, 0]):self.sample_iter]))
        except:
            s = str(self.value)
        return s

    def set_shape(self):
        '''Set the shape of the array (how many qPCR variances this is a prior for)
        '''
        self.set_value_shape(shape=(len(self.data_locs), ))

    def add_qpcr_measurement(self, ridx, tidx):
        '''Add the qPCR measurement for subject index `ridx` and time index
        `tidx` to 
        '''
        self.data_locs.append((ridx, tidx))

    def initialize(self, value_option, low_option, high_option, proposal_option, 
        target_acceptance_rate, tune, end_tune, value=None, low=None, high=None, 
        proposal_var=None, delay=0):
        '''Initialize the values and hyperparameters. The proposal truncation is 
        always set to the same as the parameterization of the prior.

        Parameters
        ----------
        value_option : str
            How to initialize the value. Options:
                'auto', 'diffuse'
                    Set the value to 2.5
                'strong'
                    Set to be 50% of the data
                'manual'
                    `value` must also be specified
        low_option : str
            How to set the low parameter of the prior
            'auto', 'valid'
                Set to 2 so that the prior stays proper during inference
            'zero'
                Set to 0
            'manual'
                Specify the value with the parameter `low`
        high_option : str
            How to set the high parameter of the prior
            'auto', 'med'
                Set to 10 X the maximum in the set
            'high'
                Set to 100 X the maximum in the set
            'low'
                Set to 1 X the maximum in the set
            'manual'
                Set the value with the parameter `high`
        proposal_option : str
            How to initialize the proposal variance:
                'auto'
                    mean**2 / 100
                'manual'
                    `proposal_var` must also be supplied
        '''
        if not pl.isint(delay):
            raise TypeError('`delay` ({}) must be an int'.format(type(delay)))
        if delay < 0:
            raise ValueError('`delay` ({}) must be >= 0'.format(delay))
        self.delay = delay

        self.qpcr_data = []
        for ridx, tidx in self.data_locs:
            t = self.G.data.given_timepoints[ridx][tidx]
            self.qpcr_data = np.append(self.qpcr_data, 
                self.G.data.qpcr[ridx][t].log_data)
        
        # Set the prior low
        if not pl.isstr(low_option):
            raise TypeError('`low_option` ({}) must be a str'.format(type(low_option)))
        if low_option == 'manual':
            if not pl.isnumeric(low):
                raise TypeError('`low` ({}) must be a numeric'.format(type(low)))
            if low < 2:
                raise ValueError('`low` ({}) must be >= 2'.format(low))
        elif low_option in ['valid', 'auto']:
            low = 2
        elif low_option == 'zero':
            low = 0
        else:
            raise ValueError('`low_option` ({}) not recognized'.format(low_option))
        self.prior.low.override_value(low)  

        # Set the prior high
        if not pl.isstr(high_option):
            raise TypeError('`high_option` ({}) must be a str'.format(type(high_option)))
        if high_option == 'manual':
            if not pl.isnumeric(high):
                raise TypeError('`high` ({}) must be a numeric'.format(type(high)))
        elif high_option in ['med', 'auto']:
            high = 10 * len(self.data_locs)
        elif high_option == 'low':
            high = len(self.data_locs)
        elif high_option == 'high':
            high = 100 * len(self.data_locs)
        else:
            raise ValueError('`high_option` ({}) not recognized'.format(high_option))
        if high < self.prior.low.value:
            raise ValueError('`high` ({}) must be >= low ({})'.format(high, 
                self.prior.low.value))
        self.prior.high.override_value(high)

        # Set the value
        if not pl.isstr(value_option):
            raise TypeError('`value_option` ({}) must be a str'.format(type(value_option)))
        if value_option == 'manual':
            if not pl.isnumeric(value):
                raise TypeError('`value` ({}) must be a numeric'.format(type(value)))
        elif value_option in ['auto', 'diffuse']:
            value = 2.5
        elif value_option == 'strong':
            value = len(self.data_locs)
        else:
            raise ValueError('`value_option` ({}) not recognized'.format(value_option))
        self.value = value
        if self.value <= self.prior.low.value or self.value >= self.prior.high.value:
            raise ValueError('`value` ({}) out of range ({})'.format(self.value))

        # Set the propsal parameters
        if pl.isstr(target_acceptance_rate):
            if target_acceptance_rate in ['optimal', 'auto']:
                target_acceptance_rate = 0.44
            else:
                raise ValueError('`target_acceptance_rate` ({}) not recognized'.format(
                    target_acceptance_rate))
        elif pl.isfloat(target_acceptance_rate):
            if target_acceptance_rate < 0 or target_acceptance_rate > 1:
                raise ValueError('`target_acceptance_rate` ({}) out of range'.format(
                    target_acceptance_rate))
        else:
            raise TypeError('`target_acceptance_rate` ({}) type not recognized'.format(
                type(target_acceptance_rate)))
        self.target_acceptance_rate = target_acceptance_rate
        
        if pl.isstr(tune):
            if tune in ['auto']:
                tune = 50
            else:
                raise ValueError('`tune` ({}) not recognized'.format(tune))
        elif pl.isint(tune):
            if tune < 0:
                raise ValueError('`tune` ({}) must be > 0'.format(
                    tune))
        else:
            raise TypeError('`tune` ({}) type not recognized'.format(type(tune)))
        self.tune = tune

        if pl.isstr(end_tune):
            if end_tune in ['auto', 'half-burnin']:
                end_tune = int(self.G.inference.burnin/2)
            else:
                raise ValueError('`tune` ({}) not recognized'.format(end_tune))
        elif pl.isint(end_tune):
            if end_tune < 0 or end_tune > self.G.inference.burnin:
                raise ValueError('`end_tune` ({}) out of range (0, {})'.format(
                    end_tune, self.G.inference.burnin))
        else:
            raise TypeError('`end_tune` ({}) type not recognized'.format(type(end_tune)))
        self.end_tune = end_tune

        # Set the proposal variance
        if not pl.isstr(proposal_option):
            raise TypeError('`proposal_option` ({}) must be a str'.format(
                type(proposal_option)))
        elif proposal_option == 'manual':
            if not pl.isnumeric(proposal_var):
                raise TypeError('`proposal_var` ({}) must be a numeric'.format(
                    type(proposal_var)))
            if proposal_var <= 0:
                raise ValueError('`proposal_var` ({}) not proper'.format(proposal_var))
        elif proposal_option in ['auto']:
            proposal_var = (self.value ** 2)/10
        else:
            raise ValueError('`proposal_option` ({}) not recognized'.format(
                proposal_option))
        self.proposal.var.value = proposal_var
        self.proposal.low = self.prior.low.value
        self.proposal.high = self.prior.high.value

    def update_var(self):
        '''Update the variance of the proposal
        '''
        if self.sample_iter == 0:
            self.temp_acceptances = 0
            self.acceptances = np.zeros(self.G.inference.n_samples, dtype=bool)
        
        elif self.sample_iter > self.end_tune:
            # Don't do any more updates
            return
        
        elif self.sample_iter % self.tune == 0:
            # Update var
            acceptance_rate = self.temp_acceptances / self.tune
            if acceptance_rate > self.target_acceptance_rate:
                self.proposal.var.value *= 1.5
            else:
                self.proposal.var.value /= 1.5
            self.temp_acceptances = 0

    def update(self):
        '''First we update the proposal (if necessary) and then we do a MH step
        '''
        self.update_var()
        proposal_std = np.sqrt(self.proposal.var.value)

        # Get the data
        xs = []
        for ridx, tidx in self.data_locs:
            xs.append(self.G[REPRNAMES.QPCR_VARIANCES].value[ridx].value[tidx])

        # Get the scale
        scale = self.G[REPRNAMES.QPCR_SCALES].value[self.l].value

        # Propose a new value for the dof
        prev_dof = self.value
        self.proposal.mean.value = self.value
        new_dof = self.proposal.sample()

        if new_dof < self.prior.low.value or new_dof > self.prior.high.value:
            # Automatic reject
            self.value = prev_dof
            return

        # Calculate the target distribution log likelihood
        prev_target_ll = 0
        for x in xs:
            prev_target_ll += pl.random.sics.logpdf(value=x,
                scale=scale, dof=prev_dof)
        new_target_ll = 0
        for x in xs:
            new_target_ll += pl.random.sics.logpdf(value=x,
                scale=scale, dof=new_dof)

        # Normalize by the loglikelihood of the proposal
        prev_prop_ll = pl.random.truncnormal.logpdf(
            value=prev_dof, mean=new_dof, std=proposal_std,
            low=self.proposal.low, high=self.proposal.high)
        new_prop_ll = pl.random.truncnormal.logpdf(
            value=new_dof, mean=prev_dof, std=proposal_std,
            low=self.proposal.low, high=self.proposal.high)

        # Accept or reject
        r = (new_target_ll - prev_prop_ll) - \
            (prev_target_ll - new_prop_ll)
        u = np.log(pl.random.misc.fast_sample_standard_uniform())

        # print('\n\n\n{} prior_mean\n----------'.format(self.child_name))
        # print('x', x)
        # print('prev_dof', prev_dof)
        # print('prev_target_ll', prev_target_ll)
        # print('prev_prop_ll', prev_prop_ll)
        # print('new dof', new_dof)
        # print('new_target_ll', new_target_ll)
        # print('new_prop_ll', new_prop_ll)
        # print('\nr', r, u)
            
        if r >= u:
            self.acceptances[self.sample_iter] = True
            self.value = new_dof
            self.temp_acceptances += 1
        else:
            self.value = prev_dof


class qPCRScales(_qPCRPriorAggVar):
    '''Aggregation class for a scale parameter of qPCR variance

    Parameters
    ----------
    L : int
        How many qPCR variance groupings there are
    '''
    def __init__(self, L, **kwargs):
        kwargs['name'] = STRNAMES.QPCR_SCALES
        _qPCRPriorAggVar.__init__(self, L=L, child=qPCRScaleL, **kwargs)


class qPCRScaleL(pl.variables.SICS):
    '''Posterior for a single qPCR scale set
    '''
    def __init__(self, L, l, **kwargs):

        self.L = L
        self.l = l
        kwargs['name'] = STRNAMES.QPCR_SCALES + '_{}'.format(l)
        pl.variables.Uniform.__init__(self, **kwargs)

        self.data_locs = []
        self.proposal = pl.variables.TruncatedNormal(mean=None, var=None, value=None)

    def __str__(self):
        # If this fails, it is because we are dividing by 0 sampler_iter
        # If which case we just return the value 
        try:
            s = 'Value: {}, Acceptance rate: {}'.format(
                self.value, np.mean(self.acceptances[
                    np.max([self.sample_iter-50, 0]):self.sample_iter]))
        except:
            s = str(self.value)
        return s

    def set_shape(self):
        '''Set the shape of the array (how many qPCR variances this is a prior for)
        '''
        self.set_value_shape(shape=(len(self.data_locs), ))

    def add_qpcr_measurement(self, ridx, tidx):
        '''Add the qPCR measurement for subject index `ridx` and time index
        `tidx` to 
        '''
        self.data_locs.append((ridx, tidx))

    def initialize(self, value_option, scale_option, dof_option, proposal_option, 
        target_acceptance_rate, tune, end_tune, value=None, dof=None, scale=None,
        proposal_var=None, delay=0):
        '''Initialize the values and hyperparameters

        Parameters
        ----------
        value_option : str
            How to initialize the value. Options:
                'auto', 'prior-mean'
                    Set to the prior mean
                'manual'
                    `value` must also be specified
        dof_option : str
            How to set the prior dof
                'auto', 'diffuse'
                    2.5
                'manual':
                    set with `dof`
        scale_option : str
            How to set the prior scale
                'empirical', 'auto'
                    Set to the variance of the data assigned to the set
                'manual'
                    Set with the parameter `scale`
        proposal_option : str
            How to initialize the proposal variance:
                'auto'
                    mean**2 / 100
                'manual'
                    `proposal_var` must also be supplied     
        '''
        if not pl.isint(delay):
            raise TypeError('`delay` ({}) must be an int'.format(type(delay)))
        if delay < 0:
            raise ValueError('`delay` ({}) must be >= 0'.format(delay))
        self.delay = delay

        self.qpcr_data = []
        for ridx, tidx in self.data_locs:
            t = self.G.data.given_timepoints[ridx][tidx]
            self.qpcr_data = np.append(self.qpcr_data, 
                self.G.data.qpcr[ridx][t].log_data)

        # Set the prior dof
        if not pl.isstr(dof_option):
            raise TypeError('`dof_option` ({}) must be a str'.format(type(dof_option)))
        if dof_option == 'manual':
            if not pl.isnumeric(dof):
                raise TypeError('`dof` ({}) must be a numeric'.format(type(dof)))
            if dof < 2:
                raise ValueError('`dof` ({}) must be >= 2'.format(dof))
        elif dof_option in ['diffuse', 'auto']:
            dof = 2.5
        else:
            raise ValueError('`dof_option` ({}) not recognized'.format(dof_option))
        if dof < 2:
            raise ValueError('`dof` ({}) must be strictly larger than 2 to be a proper' \
                ' prior'.format(dof))
        self.prior.dof.override_value(dof)

        # Set the prior scale
        if not pl.isstr(scale_option):
            raise TypeError('`scale_option` ({}) must be a str'.format(type(scale_option)))
        if scale_option == 'manual':
            if not pl.isnumeric(scale):
                raise TypeError('`scale` ({}) must be a numeric'.format(type(scale)))
            if scale <= 0:
                raise ValueError('`scale` ({}) must be positive'.format(scale))
        elif scale_option in ['auto', 'empirical']:
            v = np.var(self.qpcr_data)
            scale = v * (self.prior.dof.value - 2) / self.prior.dof.value
        else:
            raise ValueError('`scale_option` ({}) not recognized'.format(scale_option))
        self.prior.scale.override_value(scale)

        # Set the value
        if not pl.isstr(value_option):
            raise TypeError('`value_option` ({}) must be a str'.format(type(value_option)))
        if value_option == 'manual':
            if not pl.isnumeric(value):
                raise TypeError('`value` ({}) must be a numeric'.format(type(value)))
        elif value_option in ['auto', 'prior-mean']:
            value = self.prior.mean()
        else:
            raise ValueError('`value_option` ({}) not recognized'.format(value_option))
        self.value = value

        # Set the propsal parameters
        if pl.isstr(target_acceptance_rate):
            if target_acceptance_rate in ['optimal', 'auto']:
                target_acceptance_rate = 0.44
            else:
                raise ValueError('`target_acceptance_rate` ({}) not recognized'.format(
                    target_acceptance_rate))
        elif pl.isfloat(target_acceptance_rate):
            if target_acceptance_rate < 0 or target_acceptance_rate > 1:
                raise ValueError('`target_acceptance_rate` ({}) out of range'.format(
                    target_acceptance_rate))
        else:
            raise TypeError('`target_acceptance_rate` ({}) type not recognized'.format(
                type(target_acceptance_rate)))
        self.target_acceptance_rate = target_acceptance_rate

        if pl.isstr(tune):
            if tune in ['auto']:
                tune = 50
            else:
                raise ValueError('`tune` ({}) not recognized'.format(tune))
        elif pl.isint(tune):
            if tune < 0:
                raise ValueError('`tune` ({}) must be > 0'.format(
                    tune))
        else:
            raise TypeError('`tune` ({}) type not recognized'.format(type(tune)))
        self.tune = tune

        if pl.isstr(end_tune):
            if end_tune in ['auto', 'half-burnin']:
                end_tune = int(self.G.inference.burnin/2)
            else:
                raise ValueError('`tune` ({}) not recognized'.format(end_tune))
        elif pl.isint(end_tune):
            if end_tune < 0 or end_tune > self.G.inference.burnin:
                raise ValueError('`end_tune` ({}) out of range (0, {})'.format(
                    end_tune, self.G.inference.burnin))
        else:
            raise TypeError('`end_tune` ({}) type not recognized'.format(type(end_tune)))
        self.end_tune = end_tune

        # Set the proposal variance
        if not pl.isstr(proposal_option):
            raise TypeError('`proposal_option` ({}) must be a str'.format(
                type(proposal_option)))
        elif proposal_option == 'manual':
            if not pl.isnumeric(proposal_var):
                raise TypeError('`proposal_var` ({}) must be a numeric'.format(
                    type(proposal_var)))
            if proposal_var <= 0:
                raise ValueError('`proposal_var` ({}) not proper'.format(proposal_var))
        elif proposal_option in ['auto']:
            proposal_var = (self.value ** 2)/10
        else:
            raise ValueError('`proposal_option` ({}) not recognized'.format(
                proposal_option))
        self.proposal.var.value = proposal_var
        self.proposal.low = 0
        self.proposal.high = float('inf')

    def update_var(self):
        '''Update the variance of the proposal
        '''
        if self.sample_iter == 0:
            self.temp_acceptances = 0
            self.acceptances = np.zeros(self.G.inference.n_samples, dtype=bool)
        
        elif self.sample_iter > self.end_tune:
            # Don't do any more updates
            return
        
        elif self.sample_iter % self.tune == 0:
            # Update var
            acceptance_rate = self.temp_acceptances / self.tune
            if acceptance_rate > self.target_acceptance_rate:
                self.proposal.var.value *= 1.5
            else:
                self.proposal.var.value /= 1.5
            self.temp_acceptances = 0

    def update(self):
        '''First we update the proposal (if necessary) and then we do a MH step
        '''
        self.update_var()
        proposal_std = np.sqrt(self.proposal.var.value)

        # Get the data
        xs = []
        for ridx, tidx in self.data_locs:
            xs.append(self.G[REPRNAMES.QPCR_VARIANCES].value[ridx].value[tidx])

        # Get the dof
        dof = self.G[REPRNAMES.QPCR_DOFS].value[self.l].value

        # Propose a new value for the scale
        prev_scale = self.value
        self.proposal.mean.value = self.value
        new_scale = self.proposal.sample()

        # Calculate the target distribution log likelihood
        prev_target_ll = pl.random.sics.logpdf(value=prev_scale, 
            dof=self.prior.dof.value, scale=self.prior.scale.value)
        for x in xs:
            prev_target_ll += pl.random.sics.logpdf(value=x,
                scale=prev_scale, dof=dof)
        new_target_ll = pl.random.sics.logpdf(value=new_scale, 
            dof=self.prior.dof.value, scale=self.prior.scale.value)
        for x in xs:
            new_target_ll += pl.random.sics.logpdf(value=x,
                scale=new_scale, dof=dof)

        # Normalize by the loglikelihood of the proposal
        prev_prop_ll = pl.random.truncnormal.logpdf(
            value=prev_scale, mean=new_scale, std=proposal_std,
            low=self.proposal.low, high=self.proposal.high)
        new_prop_ll = pl.random.truncnormal.logpdf(
            value=new_scale, mean=prev_scale, std=proposal_std,
            low=self.proposal.low, high=self.proposal.high)

        # Accept or reject
        r = (new_target_ll - prev_prop_ll) - \
            (prev_target_ll - new_prop_ll)
        u = np.log(pl.random.misc.fast_sample_standard_uniform())

        # print('\n\n\n{} prior_mean\n----------'.format(self.child_name))
        # print('x', x)
        # print('prev_scale', prev_scale)
        # print('prev_target_ll', prev_target_ll)
        # print('prev_prop_ll', prev_prop_ll)
        # print('new scale', new_scale)
        # print('new_target_ll', new_target_ll)
        # print('new_prop_ll', new_prop_ll)
        # print('\nr', r, u)
            
        if r >= u:
            self.acceptances[self.sample_iter] = True
            self.value = new_scale
            self.temp_acceptances += 1
        else:
            self.value = prev_scale

