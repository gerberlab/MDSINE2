'''Logistic growth parameters for the posterior
'''
import logging
import time
import numpy as np
import os

from .util import expected_n_clusters, build_prior_covariance, build_prior_mean, sample_categorical_log, \
    log_det, pinv
from .perturbations import PerturbationMagnitudes
from .interactions import ClusterInteractionValue
from .names import STRNAMES, REPRNAMES

from . import pylab as pl

from . import visualization
import matplotlib.pyplot as plt

class PriorVarMH(pl.variables.SICS):
    '''This is the posterior for the prior variance of either the growth
    or self-interaction parameter. We update with a MH update since this 
    prior is not conjugate.

    Parameters
    ----------
    prior : pl.variables.SICS
        This is the prior of this distribution - which is a Squared
        Inverse Chi Squared (SICS) distribution
    child_name : str
        This is the name of the variable that this is a prior variance
        for. This is either the name of the growth parameter or the 
        self-interactions parameter
    kwargs : dict
        These are the other parameters for the initialization.
    '''

    def __init__(self, prior, child_name, **kwargs):
        if child_name == STRNAMES.GROWTH_VALUE:
            kwargs['name'] = STRNAMES.PRIOR_VAR_GROWTH
        elif child_name == STRNAMES.SELF_INTERACTION_VALUE:
            kwargs['name'] = STRNAMES.PRIOR_VAR_SELF_INTERACTIONS
        else:
            raise ValueError('`child_name` ({}) not recognized'.format(child_name))
        pl.variables.SICS.__init__(self, dtype=float, **kwargs)
        self.child_name = child_name
        self.add_prior(prior)
        self.proposal = pl.variables.SICS(dof=None, scale=None, value=None)

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

    def initialize(self, value_option, dof_option, scale_option, 
        proposal_option, target_acceptance_rate, tune, end_tune,
        value=None, dof=None, scale=None, proposal_dof=None, delay=0):
        '''Initialize the parameters of the distribution and the 
        proposal distribution

        Parameters
        ----------
        value_option : str
            Different ways to initialize the values
            Options
                'manual'
                    Set the value manually, `value` must also be specified
                'unregularized'
                    Do unregularized regression and the value is set to the
                    variance of the growth values
                'prior-mean', 'auto'
                    Set the value to the prior of the mean
        scale_option : str
            Different ways to initialize the scale of the prior
            Options
                'manual'
                    Set the value manually, `scale` must also be specified
                'auto', 'inflated-median'
                    We set the scale such that the mean of the prior is
                    equal to the median growth values calculated
                    with linear regression squared and inflated by 100.
        dof_option : str
            How informative the prior should be (setting the dof)
                'diffuse': set to the mimumum value (2)
                'weak': set so that 10% of the posterior comes from the prior
                'strong': set so that 50% of the posterior comes from the prior
                'manual': set to the value provided in the parameter `shape`
                'auto': Set to 'weak'
        proposal_option : str
            How to set the initial dof of the proposal - this will get adjusted with
            tuning
                'tight', 'auto'
                    Set the dof to be 15, relatively strong initially
                'diffuse'
                    Set the dof to be 2.5, relatively diffuse initially
                'manual'
                    Set the dof with the parameter `proposal_dof'
        target_acceptance_rate : float, str
            This is the target_acceptance rate. Options:
                'auto', 'optimal'
                    Set to 0.44
                float
                    This is the value you want
        tune : str, int
            This is how often you want to update the proposal dof
                int
                'auto'
                    Set to every 50 iterations
        end_tune : str, int
            This is when to stop the tuning
                'half-burnin', 'auto'
                    Half of burnin, rounded down
                int
        delay : int
            How many iterations to delay updating the value of the variance
        '''
        if not pl.isint(delay):
            raise TypeError('`delay` ({}) must be an int'.format(type(delay)))
        if delay < 0:
            raise ValueError('`delay` ({}) must be >= 0'.format(delay))
        self.delay = delay

        # Set the proposal dof
        if not pl.isstr(proposal_option):
            raise TypeError('`proposal_option` ({}) must be a str'.format(
                type(proposal_option)))
        elif proposal_option == 'manual':
            if not pl.isnumeric(proposal_dof):
                raise TypeError('`proposal_dof` ({}) must be a numeric'.format(
                    type(proposal_dof)))
            if proposal_dof < 2:
                raise ValueError('`proposal_dof` ({}) not proper'.format(proposal_dof))
        elif proposal_option in ['tight', 'auto']:
            proposal_dof = 15
        elif proposal_option == 'diffuse':
            proposal_dof = 2.5
        else:
            raise ValueError('`proposal_option` ({}) not recognized'.format(
                proposal_option))
        self.proposal.dof.value = proposal_dof

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

        # Set the prior dof
        if not pl.isstr(dof_option):
            raise TypeError('`dof_option` ({}) must be a str'.format(type(dof_option)))
        if dof_option == 'manual':
            if not pl.isnumeric(dof):
                raise TypeError('`dof` ({}) must be a numeric'.format(type(dof)))
            if dof < 2:
                raise ValueError('`dof` ({}) must be >= 2'.format(dof))
        elif dof_option == 'diffuse':
            dof = 2.5
        elif dof_option in ['weak', 'auto']:
            dof = len(self.G.data.asvs)/9
        elif dof_option == 'strong':
            dof = len(self.G.data.asvs)/2
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
        elif scale_option in ['auto', 'inflated-median']:
            # Perform linear regression
            rhs = [REPRNAMES.GROWTH_VALUE, REPRNAMES.SELF_INTERACTION_VALUE]
            X = self.G.data.construct_rhs(keys=rhs,
                kwargs_dict={REPRNAMES.GROWTH_VALUE:{'with_perturbations':False}},
                index_out_perturbations=True)
            y = self.G.data.construct_lhs(index_out_perturbations=True)

            prec = X.T @ X
            cov = pinv(prec, self)
            mean = cov @ X.T @ y
            if self.child_name == STRNAMES.GROWTH_VALUE:
                mean = 1e4*(np.median(mean[:self.G.data.n_asvs]) ** 2)
            else:
                mean = 1e4*(np.median(mean[self.G.data.n_asvs:]) ** 2)

            # Calculate the scale
            scale = mean * (self.prior.dof.value - 2) / self.prior.dof.value
        else:
            raise ValueError('`scale_option` ({}) not recognized'.format(scale_option))
        self.prior.scale.override_value(scale)

        # Set the initial value of the prior
        if not pl.isstr(value_option):
            raise TypeError('`value_option` ({}) must be a str'.format(type(value_option)))
        if value_option == 'manual':
            if not pl.isnumeric(value):
                raise ValueError('If `value_option` == "manual", value ({}) ' \
                    'must be a numeric (float, int)'.format(value.__class__))
        elif value_option in ['inflated-median']:
            # No interactions
            rhs = [
                REPRNAMES.GROWTH_VALUE,
                REPRNAMES.SELF_INTERACTION_VALUE]
            X = self.G.data.construct_rhs(keys=rhs,
                kwargs_dict={REPRNAMES.GROWTH_VALUE:{'with_perturbations':False}},
                index_out_perturbations=True)
            y = self.G.data.construct_lhs(index_out_perturbations=True)

            prec = X.T @ X
            cov = pinv(prec, self)
            mean = cov @ X.T @ y
            if self.child_name == STRNAMES.GROWTH_VALUE:
                value = 1e4*(np.median(mean[:self.G.data.n_asvs]) ** 2)
            else:
                value = 1e4*(np.median(mean[self.G.data.n_asvs:]) ** 2)
        elif value_option in ['prior-mean', 'auto']:
            value = self.prior.mean()
        else:
            raise ValueError('`value_option` "{}" not recognized'.format(value_option))
        self.value = value

    def update_dof(self):
        '''Updat the `dof` parameter so that we adjust the acceptance
        rate to `target_acceptance_rate`
        '''
        if self.sample_iter == 0:
            self.temp_acceptances = 0
            self.acceptances = np.zeros(self.G.inference.n_samples, dtype=bool)
        
        elif self.sample_iter > self.end_tune:
            # Don't do any more updates
            return
        
        elif self.sample_iter % self.tune == 0:
            # Update dof
            acceptance_rate = self.temp_acceptances / self.tune
            if acceptance_rate > self.target_acceptance_rate:
                self.proposal.dof.value = self.proposal.dof.value * 1.5
            else:
                self.proposal.dof.value = self.proposal.dof.value / 1.5
            self.temp_acceptances = 0

    def update(self):
        '''First we check if we need to tune the dof, which we do during
        the first half of burnin. We calculate the likelihoods in logspace
        '''
        if self.sample_iter < self.delay:
            return
        self.update_dof()

        # Get necessary data of the respective parameter
        var = self.G[self.child_name]
        x = var.value.ravel()
        mu = var.prior.mean.value
        low = var.low
        high = var.high

        # propose a new value
        prev_value = self.value
        prev_value_std = math.sqrt(prev_value)
        self.proposal.scale.value = self.value
        new_value = self.proposal.sample() # Sample a new value
        new_value_std = math.sqrt(new_value)

        # Calculate the target distribution ll
        prev_target_ll = 0
        for i in range(len(x)):
            prev_target_ll += pl.random.truncnormal.logpdf(
                value=x[i], mean=mu, std=prev_value_std,
                low=low, high=high)
        new_target_ll = 0
        for i in range(len(x)):
            new_target_ll += pl.random.truncnormal.logpdf(
                value=x[i], mean=mu, std=new_value_std,
                low=low, high=high)

        # Normalize by the ll of the proposal
        prev_prop_ll = self.proposal.logpdf(value=prev_value)
        new_prop_ll = self.proposal.logpdf(value=new_value)

        # Accept or reject
        r = (new_target_ll - prev_prop_ll) - \
            (prev_target_ll - new_prop_ll)
        u = np.log(pl.random.misc.fast_sample_standard_uniform())

        # print('\n\n\n{} prior_var\n----------'.format(self.child_name))
        # print('prev_value', prev_value)
        # print('prev_target_ll', prev_target_ll)
        # print('prev_prop_ll', prev_prop_ll)
        # print('new value', new_value)
        # print('new_target_ll', new_target_ll)
        # print('new_prop_ll', new_prop_ll)
        # print('mu', mu)
        # print('prev_value_std', prev_value_std)
        # print('new_value_std', new_value_std)
        # print('\nr', r, u)

        if r >= u:
            self.acceptances[self.sample_iter] = True
            self.value = new_value
            self.temp_acceptances += 1
        else:
            self.value = prev_value

    def visualize_posterior(self, path, f, section='posterior'):
        '''Render the traces in the folder `basepath` and write the 
        learned values to the file `f`.

        Parameters
        ----------
        path : str
            This is the path to write the files to
        f : _io.TextIOWrapper
            File that we are writing the values to
        section : str
            Section of the trace to compute on. Options:
                'posterior' : posterior samples
                'burnin' : burn-in samples
                'entire' : both burn-in and posterior samples

        Returns
        -------
        _io.TextIOWrapper
        '''
        f.write('\n\n###################################\n')
        f.write(self.name)
        f.write('\n###################################\n')
        if not self.G.inference.is_being_traced(self):
            f.write('`{}` not learned\n\tValue: {}\n'.format(self.name, self.value))
            return f
        summ = pl.summary(self, section=section)
        for k,v in summ.items():
            f.write('\t{}: {}\n'.format(k,v))

        # Plot the traces
        ax1, ax2 = visualization.render_trace(var=self, plt_type='both', section=section,
            include_burnin=True, log_scale=True, rasterized=True)

        # Plot the prior over the posterior
        l,h = ax1.get_xlim()
        xs = np.arange(l,h,step=(h-l)/100) 
        ys = []
        for x in xs:
            ys.append(pl.random.sics.pdf(value=x, 
                dof=self.prior.dof.value,
                scale=self.prior.scale.value))
        ax1.plot(xs, ys, label='prior', alpha=0.5, color='red', rasterized=True)
        ax1.legend()

        # Plot the acceptance rate over the trace
        ax3 = ax2.twinx()
        ax3 = visualization.render_acceptance_rate_trace(var=self, ax=ax3, 
            label='Acceptance Rate', color='red', scatter=False, rasterized=True)
        ax3.legend()
        fig = plt.gcf()
        fig.tight_layout()
        fig.suptitle(self.name)
        plt.savefig(path)
        plt.close()

        return f


class PriorMeanMH(pl.variables.TruncatedNormal):
    '''This implements the posterior for the prior mean of the either
    the growths or the self-interactions

    Parameters
    ----------
    '''
    def __init__(self, prior, child_name, **kwargs):
        if child_name == STRNAMES.GROWTH_VALUE:
            kwargs['name'] = STRNAMES.PRIOR_MEAN_GROWTH
        elif child_name == STRNAMES.SELF_INTERACTION_VALUE:
            kwargs['name'] = STRNAMES.PRIOR_MEAN_SELF_INTERACTIONS
        else:
            raise ValueError('`child_name` ({}) not recognized'.format(child_name))
        pl.variables.TruncatedNormal.__init__(self, mean=None, var=None, dtype=float, **kwargs)
        self.child_name = child_name
        self.add_prior(prior)
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

    def initialize(self, value_option, mean_option, var_option,
        truncation_settings, proposal_option, target_acceptance_rate,
        tune, end_tune, value=None, mean=None, var=None, proposal_var=None,
        delay=0):
        '''These are the parameters to initialize the parameters
        of the class. Depending whether it is a self-interaction
        or a growth, it does it differently.

        Parameters
        ----------
        value_option : str
            How to initialize the value. Options:
                'auto', 'prior-mean'
                    Set to the prior mean
                'linear-regression'
                    Set the values from an unregularized linear regression
                'manual'
                    `value` must also be specified
        truncation_settings: str, tuple
            How to set the truncation parameters. The proposal trucation will
            be set the same way.
                tuple - (low,high)
                    These are the truncation parameters
                'auto'
                    If self-interactions, 'negative'. If growths, 'positive'
                'positive'
                    (0, \infty)
                'negative'
                    (-\infty, 0)
                'in-vivo'
                    Not implemented
        mean_option : str
            How to set the mean
                'auto', 'median-linear-regression'
                    Set the mean to the median of the values from an
                    unregularized linear-regression
                'manual'
                    `mean` must also be specified
        var_option : str
            How to set the var
                'auto', 'diffuse-linear-regression'
                    Set the var to 10^4 * median(a_l)
                'manaul'
                    `var` must also be specified.
        proposal_option : str
            How to initialize the proposal variance:
                'auto'
                    mean**2 / 100
                'manual'
                    `proposal_var` must also be supplied
        target_acceptance_rate : str, float
            If float, this is the target acceptance rate
            If str: 
                'optimal', 'auto': 0.44
        tune : str, int
            How often to tune the proposal. If str:
                'auto': 50
        end_tune : str, int
            When to stop tuning the proposal. If str:
                'auto', 'half-burnin': Half of burnin
        '''
        self._there_are_perturbations = self.G.perturbations is not None
        if not pl.isint(delay):
            raise TypeError('`delay` ({}) must be an int'.format(type(delay)))
        if delay < 0:
            raise ValueError('`delay` ({}) must be >= 0'.format(delay))
        self.delay = delay

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

        # Set the truncation settings
        if truncation_settings is None:
            truncation_settings = 'positive'
        if pl.isstr(truncation_settings):
            if truncation_settings == 'positive':
                self.low = 0.
                self.high = float('inf')
            # elif truncation_settings == 'negative':
            #     self.low = float('-inf')
            #     self.high = 0
            elif truncation_settings == 'in-vivo':
                self.low = 0.1
                self.high = np.log(10)
            else:
                raise ValueError('`truncation_settings` ({}) not recognized'.format(
                    truncation_settings))
        elif pl.istuple(truncation_settings):
            if len(truncation_settings) != 2:
                raise ValueError('If `truncation_settings` is a tuple, it must have a ' \
                    'length of 2 ({})'.format(len(truncation_settings)))
            l,h = truncation_settings

            if (not pl.isnumeric(l)) or (not pl.isnumeric(h)):
                raise TypeError('`low` ({}) and `high` ({}) must be numerics'.format(
                    type(l), type(h)))
            if l < 0 or h < 0:
                raise ValueError('`low` ({}) and `high` ({}) must be >= 0'.format(l,h))
            if h <= l:
                raise ValueError('`low` ({}) must be strictly less than high ({})'.format(l,h))
            self.high = h
            self.low = l
        else:
            raise TypeError('`truncation_settings` ({}) type not recognized')
        self.proposal.high = self.high
        self.proposal.low = self.low

        # Set the mean
        if not pl.isstr(mean_option):
            raise TypeError('`mean_option` ({}) must be a str'.format(type(mean_option)))
        if mean_option == 'manual':
            if not pl.isnumeric(mean):
                raise TypeError('`mean` ({}) must be a numeric'.format(type(mean)))
        elif mean_option in ['auto', 'median-linear-regression']:
            # Perform linear regression
            if self.child_name == STRNAMES.GROWTH_VALUE:
                rhs = [REPRNAMES.GROWTH_VALUE, REPRNAMES.SELF_INTERACTION_VALUE]
                lhs = []
            else:
                rhs = [REPRNAMES.SELF_INTERACTION_VALUE]
                lhs = [REPRNAMES.GROWTH_VALUE]
            X = self.G.data.construct_rhs(keys=rhs,
                kwargs_dict={REPRNAMES.GROWTH_VALUE:{'with_perturbations':False}},
                index_out_perturbations=True)
            y = self.G.data.construct_lhs(keys=lhs, 
                kwargs_dict={REPRNAMES.GROWTH_VALUE:{'with_perturbations':False}},
                index_out_perturbations=True)

            prec = X.T @ X
            cov = pinv(prec, self)
            mean = cov @ X.T @ y

            if self.child_name == STRNAMES.GROWTH_VALUE:
                mean = np.median(mean[:self.G.data.n_asvs])
            else:
                mean = np.median(mean)
        else:
            raise ValueError('`mean_option` ({}) not recognized'.format(mean_option))
        self.prior.mean.override_value(mean)

        # Set the var
        if not pl.isstr(var_option):
            raise TypeError('`var_option` ({}) must be a str'.format(type(var_option)))
        if var_option == 'manual':
            if not pl.isnumeric(var):
                raise TypeError('`var` ({}) must be a numeric'.format(type(var)))
        elif var_option in ['auto', 'diffuse-linear-regression']:
            # Perform linear regression
            if self.child_name == STRNAMES.GROWTH_VALUE:
                rhs = [REPRNAMES.GROWTH_VALUE, REPRNAMES.SELF_INTERACTION_VALUE]
                lhs = []
            else:
                rhs = [REPRNAMES.SELF_INTERACTION_VALUE]
                lhs = [REPRNAMES.GROWTH_VALUE]
            X = self.G.data.construct_rhs(keys=rhs,
                kwargs_dict={REPRNAMES.GROWTH_VALUE:{'with_perturbations':False}},
                index_out_perturbations=True)
            y = self.G.data.construct_lhs(keys=lhs, 
                kwargs_dict={REPRNAMES.GROWTH_VALUE:{'with_perturbations':False}},
                index_out_perturbations=True)

            prec = X.T @ X
            cov = pinv(prec, self)

            logging.critical('here')
            print(cov.shape)
            print(X.T.shape)
            print(y.shape)

            mean = cov @ X.T @ y

            if self.child_name == STRNAMES.GROWTH_VALUE:
                mean = np.median(mean[:self.G.data.n_asvs])
            else:
                mean = np.median(mean)
            var = 1e4 * (mean**2)
        else:
            raise ValueError('`var_option` ({}) not recognized'.format(var_option))
        self.prior.var.override_value(var)

        # Set the value
        if not pl.isstr(value_option):
            raise TypeError('`value_option` ({}) must be a str'.format(type(value_option)))
        if value_option == 'manual':
            if not pl.isnumeric(value):
                raise TypeError('`value` ({}) must be a numeric'.format(type(value)))
        elif value_option in ['linear-regression']:
            # Perform linear regression
            if self.child_name == STRNAMES.GROWTH_VALUE:
                rhs = [REPRNAMES.GROWTH_VALUE, REPRNAMES.SELF_INTERACTION_VALUE]
                lhs = []
            else:
                rhs = [REPRNAMES.SELF_INTERACTION_VALUE]
                lhs = [REPRNAMES.GROWTH_VALUE]
            X = self.G.data.construct_rhs(keys=rhs,
                kwargs_dict={REPRNAMES.GROWTH_VALUE:{'with_perturbations':False}},
                index_out_perturbations=True)
            y = self.G.data.construct_lhs(keys=lhs, 
                kwargs_dict={REPRNAMES.GROWTH_VALUE:{'with_perturbations':False}},
                index_out_perturbations=True)

            prec = X.T @ X
            cov = pinv(prec, self)
            mean = cov @ X.T @ y

            if self.child_name == STRNAMES.GROWTH_VALUE:
                value = mean[:self.G.data.n_asvs]
            else:
                value = mean
        elif value_option in ['auto', 'prior-mean']:
            value = self.prior.mean.value
        else:
            raise ValueError('`value_option` ({}) not recognized'.format(value_option))
        self.value = value

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

    def update_var(self):
        '''Update the `var` parameter so that we adjust the acceptance
        rate to `target_acceptance_rate`
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
        '''First we check if we need to tune the var, which we do during
        the first half of burnin. We calculate the likelihoods in logspace
        '''
        if self.sample_iter < self.delay:
            return
        self.update_var()
        proposal_std = np.sqrt(self.proposal.var.value)

        # Get necessary data of the respective parameter
        variable = self.G[self.child_name]
        x = variable.value.ravel()
        std = np.sqrt(variable.prior.var.value)

        low = variable.low
        high = variable.high

        # propose a new value for the mean
        prev_mean = self.value
        self.proposal.mean.value = self.value
        new_mean = self.proposal.sample() # Sample a new value

        # Calculate the target distribution ll
        prev_target_ll = pl.random.truncnormal.logpdf( 
            value=prev_mean, mean=self.prior.mean.value, 
            std=np.sqrt(self.prior.var.value), low=self.low,
            high=self.high)
        for i in range(len(x)):
            prev_target_ll += pl.random.truncnormal.logpdf(
                value=x[i], mean=prev_mean, std=std,
                low=low, high=high)
        new_target_ll = pl.random.truncnormal.logpdf( 
            value=new_mean, mean=self.prior.mean.value, 
            std=np.sqrt(self.prior.var.value), low=self.low,
            high=self.high)
        for i in range(len(x)):
            new_target_ll += pl.random.truncnormal.logpdf(
                value=x[i], mean=new_mean, std=std,
                low=low, high=high)

        # Normalize by the ll of the proposal
        prev_prop_ll = pl.random.truncnormal.logpdf(
            value=prev_mean, mean=new_mean, std=proposal_std,
            low=low, high=high)
        
        new_prop_ll = pl.random.truncnormal.logpdf(
            value=new_mean, mean=prev_mean, std=proposal_std,
            low=low, high=high)

        # Accept or reject
        r = (new_target_ll - prev_prop_ll) - \
            (prev_target_ll - new_prop_ll)
        u = np.log(pl.random.misc.fast_sample_standard_uniform())

        # print('\n\n\n{} prior_mean\n----------'.format(self.child_name))
        # print('x', x)
        # print('prev_mean', prev_mean)
        # print('prev_target_ll', prev_target_ll)
        # print('prev_prop_ll', prev_prop_ll)
        # print('new mean', new_mean)
        # print('new_target_ll', new_target_ll)
        # print('new_prop_ll', new_prop_ll)
        # print('\nr', r, u)

        if r >= u:
            self.acceptances[self.sample_iter] = True
            self.value = new_mean
            self.temp_acceptances += 1
        else:
            self.value = prev_mean

    def visualize_posterior(self, path, f, section='posterior'):
        '''Render the traces in the folder `basepath` and write the 
        learned values to the file `f`.

        Parameters
        ----------
        path : str
            This is the path to write the files to
        f : _io.TextIOWrapper
            File that we are writing the values to
        section : str
            Section of the trace to compute on. Options:
                'posterior' : posterior samples
                'burnin' : burn-in samples
                'entire' : both burn-in and posterior samples

        Returns
        -------
        _io.TextIOWrapper
        '''
        f.write('\n\n###################################\n')
        f.write(self.name)
        f.write('\n###################################\n')
        if not self.G.inference.is_being_traced(self):
            f.write('`{}` not learned\n\tValue: {}\n'.format(self.name, self.value))
            return f
        summ = pl.summary(self, section=section)
        for k,v in summ.items():
            f.write('\t{}: {}\n'.format(k,v))

        # Plot the traces
        ax1, ax2 = visualization.render_trace(var=self, plt_type='both', section=section,
            include_burnin=True, log_scale=True, rasterized=True)

        # Plot the prior over the posterior
        l,h = ax1.get_xlim()
        xs = np.arange(l,h,step=(h-l)/100) 
        ys = []
        for x in xs:
            ys.append(pl.random.sics.pdf(value=x, 
                dof=self.prior.dof.value,
                scale=self.prior.scale.value))
        ax1.plot(xs, ys, label='prior', alpha=0.5, color='red', rasterized=True)
        ax1.legend()

        # Plot the acceptance rate over the trace
        ax3 = ax2.twinx()
        ax3 = visualization.render_acceptance_rate_trace(var=self, ax=ax3, 
            label='Acceptance Rate', color='red', scatter=False, rasterized=True)
        ax3.legend()
        fig = plt.gcf()
        fig.tight_layout()
        fig.suptitle(self.name)
        plt.savefig(path)
        plt.close()

        return f


class Growth(pl.variables.TruncatedNormal):
    '''Growth values of Lotka-Voltera
    '''
    def __init__(self, prior, **kwargs):
        kwargs['name'] = STRNAMES.GROWTH_VALUE
        pl.variables.TruncatedNormal.__init__(self, mean=None, var=None, low=0.,
            high=float('inf'), dtype=float, **kwargs)
        self.set_value_shape(shape=(len(self.G.data.asvs),))
        self.add_prior(prior)
        self.delay = 0
        self._initialized = False

    def __str__(self):
        return str(self.value)

    def update_str(self):
        return

    def initialize(self, value_option, truncation_settings,
        value=None, delay=0, mean=None):
        '''Initialize the growth values and hyperparamters

        Parameters
        ----------
        value_option : str
            How to initialize the values.
            Options:
                'manual'
                    Set the values manually. `value` must also be specified.
                'linear regression'
                    Set the values of the growth using linear regression
                'ones'
                    Set all of the values to 1.
                'auto'
                    Alias for 'ones'
                'prior-mean'
                    Set to the mean of the prior
        value : array
            Only necessary if `value_option` is 'manual'
        delay : int
            How many MCMC iterations to delay starting to update
        truncation_settings : str, tuple, None
            These are the settings of how you set the upper and lower limit of the
            truncated distribution. If it is None, it will default to 'standard'.
            Options
                'positive', None
                    Only constrains the values to being positive
                    low=0., high=float('inf')
                'in-vivo', 'auto'
                    Tighter constraint on the growth values.
                    low=0.1, high=ln(10)
                    These values have the following meaning
                        The slowest growing microbe will grow an order of magnitude in ~10 days
                        The fastest growing microbe will grow an order of magnitude in 1 day
                tuple(low, high)
                    These are manually specified values for the low and high
        '''
        self._initialized = True
        self._there_are_perturbations = self.G.perturbations is not None
        if not pl.isint(delay):
            raise TypeError('`delay` ({}) must be an int'.format(type(delay)))
        if delay < 0:
            raise ValueError('`delay` ({}) must be >= 0'.format(delay))
        self.delay = delay

        # Truncation settings
        if truncation_settings is None:
            truncation_settings = 'positive'
        if pl.isstr(truncation_settings):
            if truncation_settings == 'positive':
                self.low = 0.
                self.high = float('inf')
            elif truncation_settings in ['in-vivo', 'auto']:
                self.low = 0.1
                self.high = math.log(10)
            else:
                raise ValueError('`truncation_settings` ({}) not recognized'.format(
                    truncation_settings))
        elif pl.istuple(truncation_settings):
            if len(truncation_settings) != 2:
                raise ValueError('If `truncation_settings` is a tuple, it must have a ' \
                    'length of 2 ({})'.format(len(truncation_settings)))
            l,h = truncation_settings

            if (not pl.isnumeric(l)) or (not pl.isnumeric(h)):
                raise TypeError('`low` ({}) and `high` ({}) must be numerics'.format(
                    type(l), type(h)))
            if l < 0 or h < 0:
                raise ValueError('`low` ({}) and `high` ({}) must be >= 0'.format(l,h))
            if h <= l:
                raise ValueError('`low` ({}) must be strictly less than high ({})'.format(l,h))
            self.high = h
            self.low = l
        else:
            raise TypeError('`truncation_settings` ({}) type not recognized')

        # Setting the value
        if not pl.isstr(value_option):
            raise TypeError('`value_option` ({}) must be a str'.format(type(value_option)))
        if value_option == 'manual':
            if not pl.isarray(value):
                value = np.ones(len(self.G.data.asvs))*value
            if len(value) != self.G.data.n_asvs:
                raise ValueError('`value` ({}) must be ({}) long'.format(
                    len(value), len(self.G.data.asvs)))
            self.value = value
        elif value_option == 'linear-regression':
            rhs = [
                REPRNAMES.GROWTH_VALUE,
                REPRNAMES.SELF_INTERACTION_VALUE
            ]
            lhs = []
            X = self.G.data.construct_rhs(
                keys=rhs, kwargs_dict={REPRNAMES.GROWTH_VALUE:{'with_perturbations':False}},
                index_out_perturbations=True)
            y = self.G.data.construct_lhs(keys=lhs, index_out_perturbations=True)

            prec = X.T @ X
            cov = pinv(prec, self)
            mean = (cov @ X.transpose().dot(y)).ravel()
            self.value = np.absolute(mean[:len(self.G.data.asvs)])
        elif value_option in ['auto', 'ones']:
            self.value = np.ones(len(self.G.data.asvs), dtype=float)
        elif value_option == 'prior-mean':
            self.value = self.prior.mean.value * np.ones(self.G.data.n_asvs)
        else:
            raise ValueError('`value_option` ({}) not recognized'.format(value_option))

        logging.info('Growth value initialization: {}'.format(self.value))
        logging.info('Growth prior mean: {}'.format(self.prior.mean.value))
        logging.info('Growth truncation settings: {}'.format((self.low, self.high)))

    def update(self):
        '''Update the values using a truncated normal
        '''
        if self.sample_iter < self.delay:
            return

        self.calculate_posterior()
        self.sample()

        if not pl.isarray(self.value):
            # This will happen if there is 1 ASV
            self.value = np.array([self.value])

        if np.any(np.isnan(self.value)):
            logging.critical('mean: {}'.format(self.mean.value))
            logging.critical('var: {}'.format(self.var.value))
            logging.critical('value: {}'.format(self.value))
            raise ValueError('`Values in {} are nan: {}'.format(self.name, self.value))

        if self._there_are_perturbations:
            # If there are perturbations then we need to update their
            # matrix because the growths changed
            self.G.data.design_matrices[REPRNAMES.PERT_VALUE].update_values()

    def calculate_posterior(self):
        rhs = [REPRNAMES.GROWTH_VALUE]
        if self._there_are_perturbations:
            lhs = [
                REPRNAMES.SELF_INTERACTION_VALUE,
                REPRNAMES.CLUSTER_INTERACTION_VALUE]
        else:
            lhs = [
                REPRNAMES.SELF_INTERACTION_VALUE,
                REPRNAMES.CLUSTER_INTERACTION_VALUE]
        X = self.G.data.construct_rhs(keys=rhs,
            kwargs_dict={REPRNAMES.GROWTH_VALUE:{
                'with_perturbations':self._there_are_perturbations}})
        y = self.G.data.construct_lhs(keys=lhs)
        # X = X.toarray()

        process_prec = self.G[REPRNAMES.PROCESSVAR].build_matrix(
            cov=False, sparse=True)

        prior_prec = build_prior_covariance(G=self.G, cov=False,
            order=rhs, sparse=True)
        prior_mean = build_prior_mean(G=self.G, order=rhs).reshape(-1,1)
        pm = prior_prec @ prior_mean

        prec = X.T @ process_prec @ X + prior_prec
        cov = pinv(prec, self)

        self.mean.value = np.asarray(cov @ (X.T @ process_prec.dot(y) + pm)).ravel()
        self.var.value = np.diag(cov)

    def visualize_posterior(self, basepath, f, section='posterior', asv_formatter='%(name)s', 
        true_value=None):
        '''Render the traces in the folder `basepath` and write the 
        learned values to the file `f`.

        Parameters
        ----------
        basepath : str
            This is the loction to write the files to
        f : _io.TextIOWrapper
            File that we are writing the values to
        section : str
            Section of the trace to compute on. Options:
                'posterior' : posterior samples
                'burnin' : burn-in samples
                'entire' : both burn-in and posterior samples
        true_value : np.ndarray
            Ground truth values of the variable

        Returns
        -------
        _io.TextIOWrapper
        '''
        f.write('\n\n###################################\n')
        f.write(self.name)
        f.write('\n###################################\n')
        if not self.G.inference.is_being_traced(self):
            f.write('`{}` not learned\n\tValue: {}\n'.format(self.name, self.value))
            return f

        asvs = self.G.data.subjects.asvs
        summ = pl.summary(self, section=section)
        for key,arr in summ.items():
            f.write('{}\n'.format(key))
            for idx,ele in enumerate(arr):
                prefix = ''
                if asv_formatter is not None:
                    prefix = pl.asvname_formatter(format=asv_formatter, asv=asvs[idx], asvs=asvs)
                f.write('\t' + prefix + '{}\n'.format(ele)) 

        if section == 'posterior':
            len_posterior = self.G.inference.sample_iter + 1 - self.G.inference.burnin
        elif section == 'burnin':
            len_posterior = self.G.inference.burnin
        else:
            len_posterior = self.G.inference.sample_iter + 1

        # Plot the prior on top of the posterior
        if self.G.tracer.is_being_traced(STRNAMES.PRIOR_MEAN_GROWTH):
            prior_mean_trace = self.G[STRNAMES.PRIOR_MEAN_GROWTH].get_trace_from_disk(
                    section=section)
        else:
            prior_mean_trace = self.prior.mean.value * np.ones(len_posterior, dtype=float)
        if self.G.tracer.is_being_traced(STRNAMES.PRIOR_VAR_GROWTH):
            prior_std_trace = np.sqrt(
                self.G[STRNAMES.PRIOR_VAR_GROWTH].get_trace_from_disk(section=section))
        else:
            prior_std_trace = np.sqrt(self.prior.var.value) * np.ones(len_posterior, dtype=float)

        for idx in range(len(asvs)):
            fig = plt.figure()
            ax_posterior = fig.add_subplot(1,2,1)
            visualization.render_trace(var=self, idx=idx, plt_type='hist',
                label=section, color='blue', ax=ax_posterior, section=section,
                include_burnin=True, rasterized=True)

            # Get the limits and only look at the posterior within 20% range +- of
            # this number
            low_x, high_x = ax_posterior.get_xlim()

            arr = np.zeros(len(prior_std_trace), dtype=float)
            for i in range(len(prior_std_trace)):
                arr[i] = pl.random.truncnormal.sample(mean=prior_mean_trace[i], std=prior_std_trace[i], 
                    low=self.low, high=self.high)
            visualization.render_trace(var=arr, plt_type='hist', 
                label='prior', color='red', ax=ax_posterior, rasterized=True)

            if true_value is not None:
                ax_posterior.axvline(x=true_value[idx], color='red', alpha=0.65, 
                    label='True Value')

            ax_posterior.legend()
            ax_posterior.set_xlim(left=low_x*.8, right=high_x*1.2)

            # plot the trace
            ax_trace = fig.add_subplot(1,2,2)
            visualization.render_trace(var=self, idx=idx, plt_type='trace', 
                ax=ax_trace, section=section, include_burnin=True, rasterized=True)

            if true_value is not None:
                ax_trace.axhline(y=true_value[idx], color='red', alpha=0.65, 
                    label='True Value')
                ax_trace.legend()

            if asv_formatter is not None:
                asvname = pl.asvname_formatter(
                    format=asv_formatter,
                    asv=asvs[idx],
                    asvs=asvs)
            else:
                asvname = asvs[idx].name
            asvname = asvname.replace('/', '_').replace(' ', '_')

            fig.suptitle('Growth {}'.format(asvname))
            fig.tight_layout()
            fig.subplots_adjust(top=0.85)
            plt.savefig(basepath + '{}.pdf'.format(asvs[idx].name))
            plt.close()

        return f


class SelfInteractions(pl.variables.TruncatedNormal):
    '''self-interactions of Lotka-Voltera

    Since our dynamics subtract this parameter, this parameter must be positive
    '''
    def __init__(self, prior, **kwargs):
        kwargs['name'] = STRNAMES.SELF_INTERACTION_VALUE
        pl.variables.TruncatedNormal.__init__(self, mean=None, var=None, low=0.,
            high=float('inf'), dtype=float, **kwargs)
        self.set_value_shape(shape=(len(self.G.data.asvs),))
        self.add_prior(prior)

    def __str__(self):
        return str(self.value)

    def update_str(self):
        return

    def initialize(self, value_option, truncation_settings,
        value=None, delay=0, mean=None, q=None, rescale_value=None):
        '''Initialize the self-interactions values and hyperparamters

        Parameters
        ----------
        value_option : str
            How to initialize the values.
            Options:
               'manual'
                    Set the values manually. `value` must also be specified.
                'fixed-growth'
                    Fix the growth values and then sample the self-interactions
                'strict-enforcement-partial'
                    Do an unregularized regression then take the absolute value of the numbers.
                    We assume there are no interactions and we index out the time points that have
                    perturbations in them. We assume that we do not know the growths (the growths
                    are being regressed as well).
                'strict-enforcement-full'
                    Do an unregularized regression then take the absolute value of the numbers.
                    We assume there are no interactions and we index out the time points that have
                    perturbations in them. We assume that we know the growths (the growths are on
                    the lhs)
                'steady-state', 'auto'
                    Set to the steady state values. Must also provide the quantile with the
                    parameter `q`. In here we assume that the steady state is the `q`th quantile
                    of the off perturbation data
                'prior-mean'
                    Set the value to the mean of the prior
        truncation_settings : str, 2-tuple
            How to set the truncations for the normal distribution
            (low,high)
                These are the low and high values
            'negative'
                Truncated (-inf, 0)
            'positive', 'auto'
                Truncated (0, inf)
            'human'
                This assumes that the range of the steady state abundances of the human gut
                fluctuate between 1e2 and  1e13. We requie that the growth value be initialized first.
                We set the vlaues to be (growth.high/1e14, growth.low/1e2)
            'mouse'
                This assumes that the range of the steady state abundances of the mouse gut
                fluctuate between 1e2 and  1e12. We requie that the growth value be initialized first.
                We set the vlaues to be (growth.high/1e13, growth.low/1e2)
        value : array
            Only necessary if `value_option` is 'manual'
        mean : array
            Only necessary if `mean_option` is 'manual'
        delay : int
            How many MCMC iterations to delay starting to update
        rescale_value : None, float
            This is the rescale value of the qPCR. This will rescale the truncation settings.
            This is only used for either the 'mouse' or 'human' settings
        '''
        self._there_are_perturbations = self.G.perturbations is not None
        if not pl.isint(delay):
            raise TypeError('`delay` ({}) must be an int'.format(type(delay)))
        if delay < 0:
            raise ValueError('`delay` ({}) must be >= 0'.format(delay))
        self.delay = delay

        # Set truncation settings
        if pl.isstr(truncation_settings):
            # if truncation_settings == 'negative':
            #     self.low = float('-inf')
            #     self.high= 0
            if truncation_settings in ['mouse', 'human']:
                growth = self.G[STRNAMES.GROWTH_VALUE]
                if not growth._initialized:
                    raise ValueError('Growth values `{}` must be initialized first'.format(
                        STRNAMES.GROWTH_VALUE))
                if truncation_settings == 'mouse':
                    high = 1e13
                else:
                    high = 1e14
                low = 1e2
                if rescale_value is not None:
                    if not pl.isnumeric(rescale_value):
                        raise TypeError('`rescale_value` ({}) must be a numeric'.format(
                            type(rescale_value)))
                    if rescale_value <= 0:
                        raise ValueError('`rescale_value` ({}) must be > 0'.format(rescale_value))
                    high *= rescale_value
                    low *= rescale_value
                self.low = growth.high/low
                self.high = growth.low/high
            elif truncation_settings in ['auto', 'positive']:
                self.low = 0
                self.high = float('inf')
            else:
                raise ValueError('`truncation_settings) ({}) not recognized'.format(
                    truncation_settings))
        elif pl.istuple(truncation_settings):
            if len(truncation_settings) != 2:
                raise ValueError('If `truncation_settings` is a tuple, it must have a ' \
                    'length of 2 ({})'.format(len(truncation_settings)))
            l,h = truncation_settings

            if (not pl.isnumeric(l)) or (not pl.isnumeric(h)):
                raise TypeError('`low` ({}) and `high` ({}) must be numerics'.format(
                    type(l), type(h)))
            if l < 0 or h < 0:
                raise ValueError('`low` ({}) and `high` ({}) must be >= 0'.format(l,h))
            if h <= l:
                raise ValueError('`low` ({}) must be strictly less than high ({})'.format(l,h))
            self.high = h
            self.low = l
        else:
            raise TypeError('`truncation_settings` ({}) must be a tuple or str'.format(
                type(truncation_settings)))

        # Set value option
        if not pl.isstr(value_option):
            raise TypeError('`value_option` ({}) must be a str'.format(type(value_option)))
        if value_option == 'manual':
            if not pl.isarray(value):
                value = np.ones(len(self.G.data.asvs))*value
            if len(value) != self.G.data.n_asvs:
                raise ValueError('`value` ({}) must be ({}) long'.format(
                    len(value), len(self.G.data.asvs)))
            self.value = value
        elif value_option == 'fixed-growth':
            X = self.G.data.construct_rhs(keys=[REPRNAMES.SELF_INTERACTION_VALUE],
                index_out_perturbations=True)
            y = self.G.data.construct_lhs(keys=[REPRNAMES.GROWTH_VALUE], kwargs_dict={
                REPRNAMES.GROWTH_VALUE:{'with_perturbations':False}},
                index_out_perturbations=True)
            prec = X.T @ X
            cov = pinv(prec, self)
            self.value = np.absolute((cov @ X.transpose().dot(y)).ravel())
        elif 'strict-enforcement' in value_option:
            if 'full' in value_option:
                rhs = [REPRNAMES.SELF_INTERACTION_VALUE]
                lhs = [REPRNAMES.GROWTH_VALUE]
            elif 'partial' in value_option:
                lhs = []
                rhs = [REPRNAMES.GROWTH_VALUE, REPRNAMES.SELF_INTERACTION_VALUE]
            else:
                raise ValueError('`value_option` ({}) not recognized'.format(value_option))
            X = self.G.data.construct_rhs(
                keys=rhs, kwargs_dict={REPRNAMES.GROWTH_VALUE:{'with_perturbations':False}},
                index_out_perturbations=True)
            y = self.G.data.construct_lhs(keys=lhs, index_out_perturbations=True)

            prec = X.T @ X
            cov = pinv(prec, self)
            mean = (cov @ X.transpose().dot(y)).ravel()
            self.value = np.absolute(mean[len(self.G.data.asvs):])
        elif value_option == 'prior-mean':
            self.value = self.prior.mean.value * np.ones(self.G.data.n_asvs)
        elif value_option in ['steady-state', 'auto']:
            # check quantile
            if not pl.isnumeric(q):
                raise TypeError('`q` ({}) must be numeric'.format(type(q)))
            if q < 0 or q > 1:
                raise ValueError('`q` ({}) must be [0,1]'.format(q))

            # Get the data off perturbation
            datas = None
            for ridx in range(self.G.data.n_replicates):
                if self._there_are_perturbations:
                    # Exclude the data thats in a perturbation
                    base_idx = 0
                    for start,end in self.G.data.tidxs_in_perturbation[ridx]:
                        if datas is None:
                            datas = self.G.data.data[ridx][:,base_idx:start]
                        else:
                            datas = np.hstack((datas, self.G.data.data[ridx][:,base_idx:start]))
                        base_idx = end
                    if end != self.G.data.data[ridx].shape[1]:
                        datas = np.hstack((datas, self.G.data.data[ridx][:,base_idx:]))
                else:
                    if datas is None:
                        datas = self.G.data.data[ridx]
                    else:
                        datas = np.hstack((datas, self.G.data.data[ridx]))

            # Set the steady-state for each ASV
            ss = np.quantile(datas, q=q, axis=1)

            # Get the self-interactions by using the values of the growth terms
            self.value = 1/ss
        elif value_option == 'linear-regression':
            
            rhs = [REPRNAMES.SELF_INTERACTION_VALUE]
            lhs = [REPRNAMES.GROWTH_VALUE]
            X = self.G.data.construct_rhs(keys=rhs,
                index_out_perturbations=True)
            y = self.G.data.construct_lhs(keys=lhs, index_out_perturbations=True,
                kwargs_dict={REPRNAMES.GROWTH_VALUE:{'with_perturbations':False}})

            prec = X.T @ X
            cov = pinv(prec, self)
            mean = cov @ X.T @ y
            self.value = np.asarray(mean).ravel()
        else:
            raise ValueError('`value_option` ({}) not recognized'.format(value_option))

        logging.info('Self-interactions value initialization: {}'.format(self.value))
        logging.info('Self-interactions truncation settings: {}'.format((self.low, self.high)))

    def update(self):
        if self.sample_iter < self.delay:
            return

        self.calculate_posterior()
        self.sample()

        if not pl.isarray(self.value):
            # This will happen if there is 1 ASV
            self.value = np.array([self.value])

        if np.any(np.isnan(self.value)):
            logging.critical('mean: {}'.format(self.mean.value))
            logging.critical('var: {}'.format(self.var.value))
            logging.critical('value: {}'.format(self.value))
            raise ValueError('`Values in {} are nan: {}'.format(self.name, self.value))

    def calculate_posterior(self):

        rhs = [REPRNAMES.SELF_INTERACTION_VALUE]
        if self._there_are_perturbations:
            lhs = [
                REPRNAMES.GROWTH_VALUE,
                REPRNAMES.CLUSTER_INTERACTION_VALUE]
        else:
            lhs = [
                REPRNAMES.GROWTH_VALUE,
                REPRNAMES.CLUSTER_INTERACTION_VALUE]
        X = self.G.data.construct_rhs(keys=rhs)
        y = self.G.data.construct_lhs(keys=lhs, kwargs_dict={REPRNAMES.GROWTH_VALUE:{
                'with_perturbations':self._there_are_perturbations}})
        process_prec = self.G[REPRNAMES.PROCESSVAR].build_matrix(
            cov=False, sparse=True)
        prior_prec = build_prior_covariance(G=self.G, cov=False,
            order=rhs, sparse=True)

        pm = prior_prec @ (self.prior.mean.value * np.ones(self.G.data.n_asvs).reshape(-1,1))

        prec = X.T @ process_prec @ X + prior_prec
        cov = pinv(prec, self)
        self.mean.value = np.asarray(cov @ (X.T @ process_prec.dot(y) + pm)).ravel()
        self.var.value = np.diag(cov)

    def visualize_posterior(self, basepath, f, section='posterior', asv_formatter='%(name)s', 
        true_value=None):
        '''Render the traces in the folder `basepath` and write the 
        learned values to the file `f`.

        Parameters
        ----------
        basepath : str
            This is the loction to write the files to
        f : _io.TextIOWrapper
            File that we are writing the values to
        section : str
            Section of the trace to compute on. Options:
                'posterior' : posterior samples
                'burnin' : burn-in samples
                'entire' : both burn-in and posterior samples
        true_value : np.ndarray
            Ground truth values of the variable

        Returns
        -------
        _io.TextIOWrapper
        '''
        f.write('\n\n###################################\n')
        f.write(self.name)
        f.write('\n###################################\n')
        if not self.G.inference.is_being_traced(self):
            f.write('`{}` not learned\n\tValue: {}\n'.format(self.name, self.value))
            return f

        asvs = self.G.data.subjects.asvs
        summ = pl.summary(self, section=section)
        for key,arr in summ.items():
            f.write('{}\n'.format(key))
            for idx,ele in enumerate(arr):
                prefix = ''
                if asv_formatter is not None:
                    prefix = pl.asvname_formatter(format=asv_formatter, asv=asvs[idx], asvs=asvs)
                f.write('\t' + prefix + '{}\n'.format(ele)) 

        if section == 'posterior':
            len_posterior = self.G.inference.sample_iter + 1 - self.G.inference.burnin
        elif section == 'burnin':
            len_posterior = self.G.inference.burnin
        else:
            len_posterior = self.G.inference.sample_iter + 1

        # Plot the prior on top of the posterior
        if self.G.tracer.is_being_traced(STRNAMES.PRIOR_MEAN_SELF_INTERACTIONS):
            prior_mean_trace = self.G[STRNAMES.PRIOR_MEAN_SELF_INTERACTIONS].get_trace_from_disk(
                    section=section)
        else:
            prior_mean_trace = self.prior.mean.value * np.ones(len_posterior, dtype=float)
        if self.G.tracer.is_being_traced(STRNAMES.PRIOR_VAR_SELF_INTERACTIONS):
            prior_std_trace = np.sqrt(
                self.G[STRNAMES.PRIOR_VAR_SELF_INTERACTIONS].get_trace_from_disk(section=section))
        else:
            prior_std_trace = np.sqrt(self.prior.var.value) * np.ones(len_posterior, dtype=float)

        for idx in range(len(asvs)):
            fig = plt.figure()
            ax_posterior = fig.add_subplot(1,2,1)
            visualization.render_trace(var=self, idx=idx, plt_type='hist',
                label=section, color='blue', ax=ax_posterior, section=section,
                include_burnin=True, rasterized=True, log_scale=True)

            # Get the limits and only look at the posterior within 20% range +- of
            # this number
            low_x, high_x = ax_posterior.get_xlim()

            arr = np.zeros(len(prior_std_trace), dtype=float)
            for i in range(len(prior_std_trace)):
                arr[i] = pl.random.truncnormal.sample(mean=prior_mean_trace[i], std=prior_std_trace[i], 
                    low=self.low, high=self.high)
            visualization.render_trace(var=arr, plt_type='hist', log_scale=True,
                label='prior', color='red', ax=ax_posterior, rasterized=True)

            if true_value is not None:
                ax_posterior.axvline(x=true_value[idx], color='red', alpha=0.65, 
                    label='True Value')

            ax_posterior.legend()
            ax_posterior.set_xlim(left=low_x*.8, right=high_x*1.2)

            # plot the trace
            ax_trace = fig.add_subplot(1,2,2)
            visualization.render_trace(var=self, idx=idx, plt_type='trace', 
                ax=ax_trace, section=section, include_burnin=True, rasterized=True,
                log_scale=True)

            if true_value is not None:
                ax_trace.axhline(y=true_value[idx], color='red', alpha=0.65, 
                    label='True Value')
                ax_trace.legend()

            if asv_formatter is not None:
                asvname = pl.asvname_formatter(
                    format=asv_formatter,
                    asv=asvs[idx],
                    asvs=asvs)
            else:
                asvname = asvs[idx].name
            asvname = asvname.replace('/', '_').replace(' ', '_')

            fig.suptitle('Self-Interactions {}'.format(asvname))
            fig.tight_layout()
            fig.subplots_adjust(top=0.85)
            plt.savefig(basepath + '{}.pdf'.format(asvs[idx].name))
            plt.close()

        return f
    

class RegressCoeff(pl.variables.MVN):
    '''This is the posterior of the regression coefficients.
    The current posterior assumes a prior mean of 0.

    This class samples the growth, self-interactions, and cluster
    interactions jointly.

    Parameters
    ----------
    growth : posterior.Growth
        This is the class that has the growth variables
    self_interactions : posterior.SelfInteractions
        The self interaction terms for the ASVs
    interactions : ClusterInteractionValue
        These are the cluster interaction values
    pert_mag : PerturbationMagnitudes, None
        These are the magnitudes of the perturbation parameters (per clsuter)
        Set to None if there are no perturbations
    '''
    def __init__(self, growth, self_interactions, interactions,
        pert_mag, **kwargs):

        if not issubclass(growth.__class__, Growth):
            raise ValueError('`growth` ({}) must be a subclass of the Growth ' \
                'class'.format(type(growth)))
        if not issubclass(self_interactions.__class__, SelfInteractions):
            raise ValueError('`self_interactions` ({}) must be a subclass of the SelfInteractions ' \
                'class'.format(type(self_interactions)))
        if not issubclass(interactions.__class__, ClusterInteractionValue):
            raise ValueError('`interactions` ({}) must be a subclass of the Interactions ' \
                'class'.format(type(interactions)))
        if pert_mag is not None:
            if not issubclass(pert_mag.__class__, PerturbationMagnitudes):
                raise ValueError('`pert_mag` ({}) must be a subclass of the PerturbationMagnitudes ' \
                    'class'.format(type(pert_mag)))


        kwargs['name'] = STRNAMES.REGRESSCOEFF
        pl.variables.MVN.__init__(self, mean=None, cov=None, dtype=float, **kwargs)

        self.n_asvs = self.G.data.n_asvs
        self.growth = growth
        self.self_interactions = self_interactions
        self.interactions = interactions
        self.pert_mag = pert_mag
        self.clustering = interactions.clustering

        # These serve no functional purpose but we do it so that they are
        # connected in the graph structure. Each of these should have their
        # prior already initialized
        self.add_parent(self.growth)
        self.add_parent(self.self_interactions)
        self.add_parent(self.interactions)

    def __str__(self):
        '''Make it more readable
        '''
        try:
            a = 'Growth:\n{}\nSelf Interactions:\n{}\nInteractions:\n{}\nPerturbations:\n{}\n' \
                'Acceptances:\n{}'.format(
                self.growth.value, self.self_interactions.value,
                str(self.G[REPRNAMES.CLUSTER_INTERACTION_VALUE]),
                str(self.pert_mag), np.mean(
                    self.acceptances[ np.max([self.sample_iter-50, 0]):self.sample_iter], axis=0))
        except:
            a = 'Growth:\n{}\nSelf Interactions:\n{}\nInteractions:\n{}\nPerturbations:\n{}'.format(
                self.growth.value, self.self_interactions.value,
                str(self.G[REPRNAMES.CLUSTER_INTERACTION_VALUE]),
                str(self.pert_mag))
        return a

    def initialize(self, update_jointly_pert_inter, update_jointly_growth_si, 
        tune=None, end_tune=None):
        '''The interior objects are initialized by themselves. Define which variables
        get updated together.

        Note that the interactions and perturbations will always be updated before the
        growth rates and the self-interactions

        Interactions and perturbations
        ------------------------------
        These are conjugate and have a normal prior. If these are said to be updated
        jointly then we can sample directly with Gibbs sampling.

        Growths and self-interactions
        -----------------------------
        These are conjugate and have a truncated normal prior. If they are set to be 
        updated together, then we must do MH because we cannot sample from a truncated
        multivariate gaussian.

        Parameters
        ----------
        update_jointly_pert_inter : bool
            If True, update the interactions and the perturbations jointly.
            If False, update the interactions and perturbations separately - you
            randomly choose which one to update first.
        update_jointly_growth_si : bool
            If True, update the interactions and the perturbations jointly.
            If False, update the interactions and perturbations separately - you
            randomly choose which one to update first.
        '''
        self._there_are_perturbations = self.G.perturbations is not None
        if not pl.isbool(update_jointly_growth_si):
            raise TypeError('`update_jointly_growth_si` ({}) must be a bool'.format(
                type(update_jointly_growth_si)))
        if not pl.isbool(update_jointly_pert_inter):
            raise TypeError('`update_jointly_pert_inter` ({}) must be a bool'.format(
                type(update_jointly_pert_inter)))

        self.update_jointly_growth_si = update_jointly_growth_si
        self.update_jointly_pert_inter = update_jointly_pert_inter
        self.sample_iter = 0

        if self.update_jointly_growth_si:
            raise NotImplementedError('Not Implemented')
                
    # @profile
    def asarray(self):
        '''
        Builds the full regression coefficient vector. If `asv_id` and
        `cid` are None, build the entire thing. Else build it for
        the ASV or cluster specifically.

        Parameters
        ----------
        '''
        # build the entire thing
        a = np.append(self.growth.value, self.self_interactions.value)
        a = np.append(a, self.interactions.obj.get_values(use_indicators=True))
        return a

    def update(self):
        '''Either updated jointly using multivariate normal or update independently
        using truncated normal distributions for growth and self-interactions.

        Always update the one that the interactions is in first
        '''
        self._update_perts_and_inter()
        if self._there_are_perturbations:
            self.G.data.design_matrices[REPRNAMES.GROWTH_VALUE].build_with_perturbations()
        
        self._update_growth_and_self_interactions()
        self.sample_iter += 1

        if self._there_are_perturbations:
            # If there are perturbations then we need to update their
            # matrix because the growths changed
            self.G.data.design_matrices[REPRNAMES.PERT_VALUE].update_values()

    # @profile
    def _update_perts_and_inter(self):
        '''Update the with Gibbs sampling of a multivariate normal.

        Parameters
        ----------
        args : tuple
            This is a tuple of length > 1 that holds the variables on what to update
            together
        '''
        if not self.update_jointly_pert_inter:
            # Update separately
            if pl.random.misc.fast_sample_standard_uniform() < 0.5:
                self.G[REPRNAMES.CLUSTER_INTERACTION_VALUE].update()
                if self._there_are_perturbations:
                    self.G[REPRNAMES.PERT_VALUE].update()
            else:
                if self._there_are_perturbations:
                    self.G[REPRNAMES.PERT_VALUE].update()
                self.G[REPRNAMES.CLUSTER_INTERACTION_VALUE].update()
        else:
            # Update jointly
            rhs = []
            lhs = []
            if self.interactions.obj.sample_iter >= \
                self.interactions.delay:
                rhs.append(REPRNAMES.CLUSTER_INTERACTION_VALUE)
            else:
                lhs.append(REPRNAMES.CLUSTER_INTERACTION_VALUE)
            if self._there_are_perturbations:
                if self.pert_mag.sample_iter >= self.pert_mag.delay:
                    rhs.append(REPRNAMES.PERT_VALUE)
                else:
                    lhs.append(REPRNAMES.PERT_VALUE)

            if len(rhs) == 0:
                return

            lhs += [REPRNAMES.GROWTH_VALUE, REPRNAMES.SELF_INTERACTION_VALUE]
            X = self.G.data.construct_rhs(keys=rhs)
            if X.shape[1] == 0:
                logging.info('No columns, skipping')
                return
            y = self.G.data.construct_lhs(keys=lhs,
                kwargs_dict={REPRNAMES.GROWTH_VALUE:{'with_perturbations': False}})

            process_prec = self.G[REPRNAMES.PROCESSVAR].build_matrix(
                cov=False, sparse=True)
            prior_prec = build_prior_covariance(G=self.G, cov=False,
                order=rhs, sparse=True)
            prior_means = build_prior_mean(G=self.G,order=rhs).reshape(-1,1)

            # Make the prior covariance matrix and process varaince
            prec = X.T @ process_prec @ X + prior_prec
            self.cov.value = pinv(prec, self)
            self.mean.value = np.asarray(self.cov.value @ (X.T @ process_prec.dot(y) + \
                prior_prec @ prior_means)).ravel()

            # sample posterior jointly and then assign the values to each coefficient
            # type, respectfully
            try:
                value = self.sample()
            except:
                logging.critical('failed here, updating separately')
                self.pert_mag.update()
                self.interactions.update()
                return

            i = 0
            if REPRNAMES.CLUSTER_INTERACTION_VALUE in rhs:
                l = self.interactions.obj.num_pos_indicators()
                self.interactions.value = value[:l]
                self.interactions.set_values(arr=value[:l], use_indicators=True)
                self.interactions.update_str()
                i += l
            if self._there_are_perturbations:
                if REPRNAMES.PERT_VALUE in rhs:
                    self.pert_mag.value = value[i:]
                    self.pert_mag.set_values(arr=value[i:], use_indicators=True)
                    self.pert_mag.update_str()
                    self.G.data.design_matrices[REPRNAMES.GROWTH_VALUE].update_value()
                    # self.G.data.design_matrices[REPRNAMES.PERT_VALUE].build()

    def _update_acceptances(self):
        if self.growth.sample_iter == 0:
            self.temp_acceptances= np.zeros(len(self.G.data.asvs), dtype=int)
            self.acceptances = np.zeros(shape=(self.G.inference.n_samples, 
                len(self.G.data.asvs)), dtype=bool)
        elif self.growth.sample_iter > self.end_tune:
            return
        elif self.growth.sample_iter % self.tune == 0:
            self.temp_acceptances = np.zeros(len(self.G.data.asvs), dtype=int)

    def _update_growth_and_self_interactions(self):
        '''Update the growth and self-interactions
        Our proposal is the posterior distribution.
        '''
        if not self.update_jointly_growth_si:
            # Update separately
            self.growth.update()
            self.self_interactions.update()
        else:
            # Update together
            raise NotImplementedError('Not Implemented')

    def add_trace(self):
        '''Trace values for growth, self-interactions, and cluster interaction values
        '''
        self.growth.add_trace()
        self.self_interactions.add_trace()
        self.interactions.add_trace()
        if self._there_are_perturbations:
            self.pert_mag.add_trace()

    def set_trace(self):
        self.growth.set_trace()
        self.self_interactions.set_trace()
        self.interactions.set_trace()
        if self._there_are_perturbations:
            self.pert_mag.set_trace()

    def add_init_value(self):
        self.growth.add_init_value()
        self.self_interactions.add_init_value()
        self.interactions.add_init_value()
        if self._there_are_perturbations:
            self.pert_mag.add_init_value()
