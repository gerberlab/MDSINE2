'''Posterior Objects used for learning the negative binomial dispersion parameters.

This contains all of the data structures used for inference: design matrices, posterior
classes, auxiliary functions, building the graph
'''
import numpy as np
from mdsine2.logger import logger
import time
import os
import os.path
import math

import numpy.random as npr
import numba

from typing import Union, Tuple, IO

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from . import visualization

from . import pylab as pl
from .pylab import BaseMCMC
from .base import *
from .names import STRNAMES
from . import config

@numba.jit(nopython=True, fastmath=True, cache=True)
def negbin_loglikelihood(k: float, m: float, dispersion: float) -> float:
    '''Loglikelihood - with parameterization in [1]
    
    Parameters
    ----------
    k : numeric
        Observed counts
    m : numeric
        Mean
    phi : float
        Dispersion
    
    Returns
    -------
    float
        Negative Binomial Log Likelihood
    
    References
    ----------
    [1] TE Gibson, GK Gerber. Robust and Scalable Models of Microbiome Dynamics. ICML (2018)
    '''
    r = 1/dispersion
    return math.lgamma(k+r) - math.lgamma(k+1) - math.lgamma(r) \
            + r * (math.log(r) - math.log(r+m)) + k * (math.log(m) - math.log(r+m))


class Data(pl.graph.DataNode):
    '''This is the raw data that we are regressing over

    Parameters
    ----------
    subjects : pl.base.SubjectSet
        These are a list of the subjects that we are going to get data from
    '''

    def __init__(self, subjects: Study, **kwargs):
        kwargs['name'] = 'Data'
        pl.graph.DataNode.__init__(self, **kwargs)

        self.taxa = subjects.taxa # mdsine2.pylab.base.TaxaSet
        self.subjects = subjects # mdsine2.pylab.base.Study
        self.n_taxa = len(self.taxa) # int

        self.data = []  # list(np.ndarray)
        self.read_depths = [] # list(np.ndarray)
        self.qpcr = []  # qPCR measurement for each value
        logger.debug("Available subjects: {}".format(
            ",".join(
                subj.name for subj in self.subjects
            )
        ))
        for subject in self.subjects:
            logger.debug("Subject {}, available qpcr: {}".format(
                subject.name,
                ",".join(str(i) for i in subject.qpcr.keys())
            ))

            d = subject.matrix()['raw']
            self.data.append(d)
            self.read_depths.append(np.sum(d, axis=0))
            self.qpcr.append(subject.qpcr[0])

        self.n_replicates = len(self.data)

    def __len__(self) -> int:
        return self.n_replicates


class NegBinDispersionParam(pl.variables.Uniform):
    '''These are for learning the a0 and a1 parameters - updated with 
    Metropolis-Hastings

    We assume these are uniform and have a uniform prior [1]

    Proposal distribution is a truncated normal distribution with truncation
    set to the same high and lows as the prior.

    References
    ----------
    [1] Bucci, Vanni, et al. "MDSINE: Microbial Dynamical Systems INference 
        Engine for microbiome time-series analyses." Genome biology 17.1 (2016): 121.
    '''

    def __init__(self, name: str, **kwargs):
        pl.variables.Uniform.__init__(
            self, dtype=float, name=name, **kwargs)

    def __str__(self) -> str:
        try:
            s = 'Value: {}, Acceptance rate: {}'.format(
                self.value, np.mean(self.acceptances[
                    np.max([self.sample_iter-50, 0]):self.sample_iter]))
        except:
            s = str(self.value)
        return s

    def initialize(self, value: Union[float, int], truncation_settings: Union[str, Tuple[float, float]], 
        proposal_option: str, target_acceptance_rate: Union[str, float], tune: Union[str, int], 
        end_tune: Union[str, int], proposal_var: float=None, delay: int=0):
        '''Initialize the negative binomial dispersion parameter

        Parameters
        ----------
        value : numeric
            This is the initial value
        truncation_settings: str, tuple
            How to set the truncation parameters. The proposal trucation will
            be set the same way.
                tuple - (low,high)
                    These are the truncation parameters
                'auto'
                    (0, 1e5)
        proposal_option : str
            How to initialize the proposal variance:
                'auto'
                    initial_value**2 / 100
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
        if pl.isstr(truncation_settings):
            if truncation_settings == 'auto':
                self.low = 0.
                self.high = 1e5
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
            self.high.value = h
            self.low.value = l
        else:
            raise TypeError('`truncation_settings` ({}) type not recognized')

        # Set the value
        if not pl.isnumeric(value):
            raise TypeError('`value` ({}) must be a numeric'.format(type(value)))
        if value <= self.low or value >= self.high:
            raise ValueError('`value` ({}) out of range ({})'.format(
                value, (self.low, self.high)))
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
            proposal_var = (self.value ** 2)/100
        else:
            raise ValueError('`proposal_option` ({}) not recognized'.format(
                proposal_option))
        self.proposal_var = proposal_var

    def _update_proposal_variance(self):
        '''Update the proposal variance
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
                self.proposal_var *= 1.5
            else:
                self.proposal_var /= 1.5
            self.temp_acceptances = 0

    def update(self):
        '''Do a metropolis update
        '''
        # Update proposal variance if necessary
        if self.sample_iter < self.delay:
            return
        self._update_proposal_variance()
        proposal_std = np.sqrt(self.proposal_var)

        # Get the current likelihood
        old_loglik = self.data_likelihood()
        prev_value = self.value

        # Propose a new value and get the likelihood
        self.value = pl.random.truncnormal.sample(
            loc=self.value, scale=proposal_std,
            low=self.low, high=self.high)
        new_loglik = self.data_likelihood()

        # reverse jump probabilities
        jump_to_new = pl.random.truncnormal.logpdf(value=self.value, 
            loc=prev_value, scale=proposal_std, 
            low=self.low, high=self.high)
        jump_to_old = pl.random.truncnormal.logpdf(value=prev_value, 
            loc=self.value, scale=proposal_std, 
            low=self.low, high=self.high)
        

        r = (new_loglik + jump_to_old) - (old_loglik + jump_to_new)
        u = np.log(pl.random.misc.fast_sample_standard_uniform())
        if r > u:
            self.acceptances[self.sample_iter] = True
            self.temp_acceptances += 1
        else:
            self.value = prev_value

    def data_likelihood(self) -> float:
        '''Calculate the current log likelihood
        '''
        a0 = self.G[STRNAMES.NEGBIN_A0].value
        a1 = self.G[STRNAMES.NEGBIN_A1].value
        latents = [v.value for v in self.G[STRNAMES.FILTERING].value]
        datas = [v.data for v in self.G[STRNAMES.FILTERING].value]
        read_depths = [v.read_depths for v in self.G[STRNAMES.FILTERING].value]
        
        cumm = 0
        for ridx in range(len(latents)):
            data=datas[ridx]
            latent = latents[ridx]
            read_depth = read_depths[ridx]
            total_abund = np.sum(latent)
            rel_abund = latent / total_abund

            cumm += NegBinDispersionParam._data_likelihood(a0=a0, a1=a1,
                data=data, read_depth=read_depth, rel_abund=rel_abund)
        return cumm
    
    @staticmethod
    @numba.jit(nopython=True)
    def _data_likelihood(a0: float, a1: float, data: np.ndarray, read_depth: np.ndarray,
        rel_abund: np.ndarray) -> float:
        cumm = 0

        # For each taxon
        for oidx in range(data.shape[0]):
            # For each replicate
            for ridx in range(data.shape[1]):
                y = data[oidx, ridx]
                mean = read_depth[ridx] * rel_abund[oidx]
                dispersion = a0/rel_abund[oidx] + a1

                # This is the negative binomial loglikelihood
                r = 1/dispersion
                # try:
                cumm += math.lgamma(y+r) - math.lgamma(y+1) - math.lgamma(r) \
                    + r * (math.log(r) - math.log(r+mean)) + y * (math.log(mean) - math.log(r+mean))

                #     raise
        return cumm

    def visualize(self, path: str, f: IO, section: str='posterior') -> IO:
        '''Visualize the posterior of the negative binomial dispersion parameter

        Parameters
        ----------
        path : str
            This is the path to save the posterior trace plots
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
        f.write('\n\n###################################\n{}'.format(self.name))
        f.write('\n###################################\n')
        if not self.G.inference.tracer.is_being_traced(self):
            f.write('`{}` not learned\n\tValue: {}\n'.format(self.name, self.value))
            return f
        
        summ = pl.summary(self, section=section)
        for k,v in summ.items():
            f.write('\t{}: {}\n'.format(k,v))

        axleft, axright = visualization.render_trace(self, plt_type='both', 
            include_burnin=True, rasterized=True, log_scale=self.name==STRNAMES.NEGBIN_A0)

        # Plot the acceptance rate on the right hand side
        ax2 = axright.twinx()
        ax2 = visualization.render_acceptance_rate_trace(self, ax=ax2, 
            label='Acceptance Rate', color='red', scatter=False, rasterized=True)

        ax2.legend()
        fig = plt.gcf()
        fig.suptitle(self.name)
        fig.tight_layout()
        plt.savefig(path)
        plt.close()
        return f


class TrajectorySet(pl.variables.Variable):
    '''This aggregates a set of trajectories from each Replicate
    '''
    def __init__(self, ridx: int, subjname: str, **kwargs):
        kwargs['name'] = STRNAMES.LATENT_TRAJECTORY + '_{}'.format(subjname)
        pl.variables.Variable.__init__(self, **kwargs)
        n_taxa = len(self.G.data.taxa)
        self.set_value_shape(shape=(n_taxa,))
        self.ridx = ridx
        self.value = np.zeros(n_taxa, dtype=float)
        self.data = self.G.data.data[self.ridx] # np.ndarray
        self.read_depths = self.G.data.read_depths[self.ridx] # np.ndarray
        self.qpcr_measurement = self.G.data.qpcr[self.ridx] # mdsine2.pylab.base.qPCRData
    
        prior = pl.variables.Normal(
            loc=pl.variables.Constant(name=self.name+'_prior_loc', value=None, G=self.G),
            scale2=pl.variables.Constant(name=self.name+'_prior_scale2', value=None, G=self.G),
            name=self.name+'_prior', G=self.G)
        self.add_prior(prior)

    def __getitem__(self, idx: int) -> float:
        return self.value[idx]

    def initialize(self):
        '''Initialize the value
        '''
        # Get the mean relative abundance
        rel = np.sum(self.data, axis=1)
        rel = rel / np.sum(rel)
        value = rel * self.qpcr_measurement.mean()

        self.value = np.zeros(len(value))
        for i in range(len(value)):
            self.value[i] = pl.random.truncnormal.sample(loc=value[i], scale=1e-2, 
                low=0, high=float('inf'))

        self.prior.loc.override_value(self.value)
        self.prior.scale2.override_value(100 * np.var(self.value))


class FilteringMP(pl.graph.Node):
    '''This handles multiprocessing of the latent state

    Parallelization Modes
    ---------------------
    'debug'
        If this is selected, then we dont actually parallelize, but we go in
        order of the objects in sequential order. We would do this if we want
        to benchmark within each processor or do easier print statements
    'full'
        This is where each subject gets their own process

    This assumes that we are using the log model for the dynamics
    '''
    def __init__(self, mp: str, **kwargs):
        kwargs['name'] = STRNAMES.FILTERING
        pl.graph.Node.__init__(self, **kwargs)
        self.value = []
        for ridx, subj in enumerate(self.G.data.subjects):
            self.value.append(TrajectorySet(G=self.G, ridx=ridx, subjname=subj.name))
        
        self.print_vals = False
        self._strr = 'NA'
        self.mp = mp

    def __str__(self) -> str:
        return self._strr

    @property
    def sample_iter(self) -> int:
        # It doesnt matter if we chose q or x because they are both the same
        return self.value[0].sample_iter

    def initialize(self, tune: Union[int, str], end_tune: Union[int, str], target_acceptance_rate: Union[float, str],  
        qpcr_variance_inflation: Union[float, int], delay: int=0):
        '''Initialize the latent state

        Parameters
        ----------
        value_option : str
            'tight-coupling'
                Set the value to the empirical mean of the trajectory with a small variance
            'small-bias'
                Add 1e-10 to all of the latent states
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
        proposal_option : str
            How to initialize the proposal variance:
                'auto'
                    initial_value**2 / 100
                'manual'
                    `proposal_var` must also be supplied
        qpcr_variance_inflation : float
            This is the factor to inflate the qPCR variance
        delay : int
            How many Gibb stepps to delay
        '''
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

        # Initialize the trajectory sets
        for ridx in range(self.G.data.n_replicates):
            self.value[ridx].initialize()

        if self.mp == 'full':
            self.pool = pl.multiprocessing.PersistentPool(G=self.G, ptype='sadw')
        elif self.mp == 'debug':
            self.pool = []
        else:
            raise ValueError('Filtering mutliprocessing argument ({}) not recognized'.format(
                self.mp))

        for ridx in range(len(self.value)):
            worker = _LatentWorker()
            worker.initialize(
                reads=self.value[ridx].data,
                qpcr_loc=self.value[ridx].qpcr_measurement.loc,
                qpcr_scale=np.sqrt(qpcr_variance_inflation) *self.value[ridx].qpcr_measurement.scale,
                proposal_std=np.log(1.5),
                prior_loc=self.value[ridx].prior.loc.value,
                prior_scale=np.sqrt(self.value[ridx].prior.scale2.value),
                tune=tune, end_tune=end_tune,
                target_acceptance_rate=target_acceptance_rate,
                value=self.value[ridx].value,
                delay=delay,
                ridx=ridx)
            if self.mp == 'full':
                self.pool.add_worker(worker)
            else:
                self.pool.append(worker)

        self.total_n_datapoints = len(self.G.data.taxa) * len(self.G.data)

    def update(self):
        '''Gibb step
        '''
        start_time = time.time()
        a0 = self.G[STRNAMES.NEGBIN_A0].value
        a1 = self.G[STRNAMES.NEGBIN_A1].value

        kwargs={'a0': a0, 'a1':a1}
        str_acc = [None]*self.G.data.n_replicates
        mpstr = None
        if self.mp == 'debug':
            for ridx in range(len(self.value)):
                _, x, acc_rate = self.pool[ridx].update(**kwargs)
                self.value[ridx].value = x
                str_acc[ridx] = '{:.3f}'.format(acc_rate)
                mpstr = 'no-mp'
        else:
            ret = self.pool.map(func='update', args=kwargs)
            for ridx, x, acc_rate in ret:
                self.value[ridx].value = x
                str_acc[ridx] = '{:.3f}'.format(acc_rate)
            mpstr = 'mp'

        t = time.time() - start_time
        try:
            self._strr = '{} : Time: {:.4f}, Acc: {}, data/sec: {:.2f}'.format(mpstr, t,
                str(str_acc).replace("'",''), self.total_n_datapoints/t)
        except:
            self._strr = 'NA'

    def add_trace(self):
        for x in self.value:
            x.add_trace()

    def set_trace(self, *args, **kwargs):
        for x in self.value:
            x.set_trace(*args, **kwargs)
    
    def kill(self):
        if pl.ispersistentpool(self.pool):
            self.pool.kill()

    def visualize(self, basepath: str, section: str='posterior', taxa_formatter: str='%(paperformat)s'):
        '''Render the latent trajectories in the base folder and write the statistics.

        Parameters
        ----------
        basepath : str
            This is the loction to write the files to
        section : str
            Section of the trace to compute on. Options:
                'posterior' : posterior samples
                'burnin' : burn-in samples
                'entire' : both burn-in and posterior samples
        taxa_formatter : str
            This is the format to write taxonomy of the Taxa
        '''
        chain = self.G.inference
        taxa  = chain.graph.data.taxa
        os.makedirs(basepath, exist_ok=True)

        if chain.is_in_inference_order(STRNAMES.FILTERING):
            taxanames = taxa.names.order

            for ridx, subj in enumerate(self.G.data.subjects):
                subj_basepath = os.path.join(basepath, subj.name)
                os.makedirs(subj_basepath, exist_ok=True)

                M_subj = subj.matrix()['abs']
                latent_name = STRNAMES.LATENT_TRAJECTORY + '_' + subj.name
                latent = self.G[latent_name]

                fname_subj = os.path.join(subj_basepath, 'output.txt')
                f = open(fname_subj, 'w')
                f.write('Subject {} output\n'.format(subj.name))
                f.write('---------------------\n')

                # Get from disk only once
                latent_trace = latent.get_trace_from_disk(section=section)
                summ = pl.summary(latent_trace)
                for aidx, aname in enumerate(taxanames):
                    f.write('\n\nTaxa {}: {}\n'.format(
                        aidx, taxaname_formatter(taxa_formatter,
                        taxon=aname, taxa=taxa)))
                    f.write('-------------------\n')

                    # Write what the data is
                    f.write('Data: ')
                    row = M_subj[aidx, :]
                    for ele in row:
                        f.write('{:.4E}  '.format(ele))
                    f.write('\n')

                    f.write('Learned Values:\n')
                    for k,v in summ.items():
                        f.write('\t{}: {:.4E}\n'.format(k,v[aidx]))

                    # plot the variable
                    axpost, axtrace = visualization.render_trace(latent_trace[:,aidx], plt_type='both', 
                        rasterized=True, log_scale=True)
                    for idx in range(M_subj.shape[1]):
                        if idx == 0:
                            label = 'data'
                        else:
                            label = None
                        axpost.axvline(x=M_subj[aidx, idx], color='green', label=label)
                        axtrace.axhline(y=M_subj[aidx, idx], color='green', label=label)

                    fig = plt.gcf()
                    fig.suptitle(taxaname_formatter(format=taxa_formatter,
                        taxon=aname, taxa=taxa))
                    plotpath = os.path.join(subj_basepath, '{}.pdf'.format(aname))
                    plt.savefig(plotpath)
                    plt.close()
                f.close()


class _LatentWorker(pl.multiprocessing.PersistentWorker):
    '''Worker class for multiprocessing. Everything is in log scale
    '''
    def __init__(self):
        return

    def initialize(self, reads: np.ndarray, qpcr_loc: float, qpcr_scale: float, prior_loc: float, prior_scale: float,
        proposal_std: float, tune: int, end_tune: int, target_acceptance_rate: float, value: np.ndarray, 
        delay: int, ridx: int):
        '''Initialize the values

        reads : np.ndarray((n_taxa x n_reps))
        qpcr_mean : float
        qpcr_std : float
        prior_mean : float
        prior_std : float
        proposal_std : float
        tune : int
        end_tune : int
        target_acceptance_rate :float
        value : np.ndarray((n_taxa,))
        ridx : int
        '''
        self.reads = reads
        self.read_depths = np.sum(self.reads, axis=0)
        self.qpcr_loc = qpcr_loc
        self.qpcr_scale = qpcr_scale
        self.prior_loc = prior_loc
        self.prior_scale = prior_scale
        self.proposal_std = proposal_std
        self.tune = tune
        self.end_tune = end_tune
        self.target_acceptance_rate = target_acceptance_rate
        self.value = value
        self.ridx = ridx

        self.sumq = np.sum(self.value)
        self.log_sumq = np.log(self.sumq)

        self.sample_iter = 0
        self.acceptances = 0
        self.total_acceptances = 0

    def update_proposal_std(self):
        if self.sample_iter > self.end_tune:
            return
        if self.sample_iter == 0:
            return
        if self.sample_iter % self.tune == 0:
            # Adjust
            acc_rate = self.acceptances/self.n_props_total
            if acc_rate < 0.1:
                logger.debug('Very low acceptance rate, scaling down past covariance')
                self.proposal_std *= 0.01
            elif acc_rate < self.target_acceptance_rate:
                self.proposal_std /= np.sqrt(1.5)
            else:
                self.proposal_std *= np.sqrt(1.5)
            
            self.acceptances = 0
            self.n_props_local = 0

    def update(self, a0: float, a1: float):
        '''Update the latent state with the updated negative binomial
        dispersion parameters
        '''
        self.a0 = a0
        self.a1 = a1

        self.update_proposal_std()
        self.n_props_local = 0
        self.n_props_total = 0
        self.n_accepted_iter = 0

        oidxs = npr.permutation(self.reads.shape[0])
        for oidx in oidxs:
            self.update_single(oidx=oidx)

        return self.ridx, self.value, self.n_accepted_iter/len(self.value)

    def update_single(self, oidx: int):
        '''Update the latent state for the Taxa index `oidx` in this replicate
        '''
        old_log_value = np.log(self.value[oidx])
        old_value = self.value[oidx]
        self.oidx = oidx
        self.curr_log_val = old_log_value

        aaa = self.prior_ll()
        bbb = self.qpcr_ll()
        ccc = self.negbin_ll()

        old_ll = aaa + bbb + ccc

        # propose new value
        log_new = pl.random.normal.sample(loc=old_log_value, scale=self.proposal_std)
        self.value[oidx] = np.exp(log_new)

        self.sumq = self.sumq - old_value + self.value[oidx]
        self.log_sumq = np.log(self.sumq)

        aaa = self.prior_ll()
        bbb = self.qpcr_ll()
        ccc = self.negbin_ll()

        new_ll = aaa + bbb + ccc

        r_accept = new_ll - old_ll
        r = pl.random.misc.fast_sample_standard_uniform()
        if math.log(r) > r_accept:
            # Reject
            self.sumq = self.sumq + old_value - self.value[oidx]
            self.log_sumq = np.log(self.sumq)
            self.value[self.oidx] = old_value
        else:
            self.acceptances += 1
            self.total_acceptances += 1
            self.n_accepted_iter += 1

        self.n_props_local += 1
        self.n_props_total += 1

    def prior_ll(self) -> float:
        '''Prior loglikelihood
        '''
        return pl.random.normal.logpdf(value=self.curr_log_val,
            loc=self.prior_loc[self.oidx], scale=self.prior_scale)

    def qpcr_ll(self) -> float:
        '''qPCR loglikelihood
        '''
        return pl.random.normal.logpdf(value=self.log_sumq, 
            loc=self.qpcr_loc, scale=self.qpcr_scale)

    def negbin_ll(self) -> float:
        '''Negative binomial loglikelihood
        '''
        cumm = 0
        rel = self.value[self.oidx]/self.sumq
        for k in range(self.reads.shape[1]):
            cumm += negbin_loglikelihood(
                k=self.reads[self.oidx, k],
                m=self.read_depths[k] * rel,
                dispersion=self.a0/rel + self.a1)
        return cumm


@numba.jit(nopython=True, fastmath=True, cache=True)
def _single_calc_mean_var(means: np.ndarray, variances: np.ndarray, a0: float, a1: float, 
    rels: np.ndarray, read_depths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    i = 0
    for col in range(rels.shape[1]):
        for oidx in range(rels.shape[0]):
            mean = rels[oidx, col] * read_depths[col]
            disp = a0 / mean + a1
            variances[i] = mean + disp * (mean**2)
            means[i] = mean

            i += 1
    return means, variances

def visualize_learned_negative_binomial_model(mcmc: BaseMCMC, section: str='posterior') -> matplotlib.pyplot.figure:
    '''Visualize the negative binomial dispersion model.

    Plot variance on y-axis, mean on x-axis. both in logscale.

    Parameters
    ----------
    mcmc : mdsine2.BaseMCMC
        This is the inference object with the negative binomial posteriors
        and the data it was learned on
    section : str
        Section of the trace to compute on. Options:
            'posterior' : posterior samples
            'burnin' : burn-in samples
            'entire' : both burn-in and posterior samples
    
    Returns
    -------
    matplotlib.pyplot.Figure
    '''
    # Get the data
    # ------------
    subjset = mcmc.graph.data.subjects
    reads = []
    for subj in subjset:
        reads.append(subj.matrix()['raw'])
    reads = np.hstack(reads)
    read_depths = np.sum(reads, axis=0)
    rels = reads / read_depths + 1e-20

    # Get the traces of a0 and a1
    # ---------------------------
    a0 = mcmc.graph[STRNAMES.NEGBIN_A0]
    a1 = mcmc.graph[STRNAMES.NEGBIN_A1]

    if mcmc.tracer.is_being_traced(STRNAMES.NEGBIN_A0):
        a0s = a0.get_trace_from_disk(section=section)
    else:
        a0s = a0.value * np.ones(mcmc.n_samples - mcmc.burnin)
    if mcmc.tracer.is_being_traced(STRNAMES.NEGBIN_A0):
        a1s = a1.get_trace_from_disk(section=section)
    else:
        a1s = a1.value * np.ones(mcmc.n_samples - mcmc.burnin)

    means = np.zeros(shape=(a0s.shape[0], rels.size), dtype=float)
    variances = np.zeros(shape=(a0s.shape[0], rels.size), dtype=float)

    for i in range(len(a0s)):
        _single_calc_mean_var(
            means=means[i,:],
            variances=variances[i,:],
            a0=a0s[i], a1=a1s[i], rels=rels, 
            read_depths=read_depths)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # plot the data
    colors = sns.color_palette(n_colors=len(subjset))
    for sidx, subj in enumerate(subjset):
        reads_subj = subj.matrix()['raw']

        x = np.mean(reads_subj, axis=1)
        y = np.var(reads_subj, axis=1)

        idxs = x > 0
        x = x[idxs]
        y = y[idxs]

        ax.scatter(
            x=x, y=y, alpha=0.5,
            color=colors[sidx], rasterized=False, 
            label='Subject {}'.format(subj.name))

    # Still need to get the 2.5th percentile, 97.5th percentile and the median
    summ_m = pl.summary(means)
    summ_v = pl.summary(variances)

    med_m = summ_m['median']
    med_v = summ_v['median']
    low_v = np.nanpercentile(variances, 2.5, axis=0)
    high_v = np.nanpercentile(variances, 97.5, axis=0)

    idxs = np.argsort(med_m)
    med_m = med_m[idxs]
    med_v = med_v[idxs]
    low_v = low_v[idxs]
    high_v = high_v[idxs]

    ax.plot(med_m, med_v, color='black', label='Fitted NegBin Model', rasterized=False)
    ax.fill_between(x=med_m, y1=low_v, y2=high_v, color='black', alpha=0.3, label='95th percentile')

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Mean (counts)')
    ax.set_ylabel('Variance (counts)')
    ax.set_title('Empirical mean vs variance of counts')
    ax.set_xlim(left=0.5)
    ax.set_ylim(bottom=0.5)
    ax.legend()

    return fig

def build_graph(params: config.NegBinConfig, graph_name: str, subjset: Study) -> BaseMCMC:
    '''Builds the graph used for posterior inference of the negative binomial
    dispersion parameters

    Parameters
    ----------
    params : mdsine2.config.NegBinConfig
        This specfies the parameters to run the model
    graph_name : str
        This is what we label the graph with
    subjset : mdsine2.Study
        This is the MDSINE2 object that contains all of the data and the Taxas
    '''
    if not config.isModelConfig(params):
        raise TypeError('`params` ({}) needs to be a config.ModelConfig object'.format(type(params)))

    # Initialize the graph and make the save location
    # -----------------------------------------------
    GRAPH = pl.Graph(name=graph_name, seed=params.SEED)
    GRAPH.as_default()

    basepath = params.MODEL_PATH
    os.makedirs(basepath, exist_ok=True)

    # Initialize the inference objects
    # --------------------------------
    d = Data(subjset, G=GRAPH)

    x = FilteringMP(mp=params.MP_FILTERING, G=GRAPH, name=STRNAMES.FILTERING)
    a0 = NegBinDispersionParam(name=STRNAMES.NEGBIN_A0, G=GRAPH, low=0, high=1e5)
    a1 = NegBinDispersionParam(name=STRNAMES.NEGBIN_A1, G=GRAPH, low=0, high=1e5)

    mcmc = pl.BaseMCMC(burnin=params.BURNIN, n_samples=params.N_SAMPLES, graph=GRAPH)

    # Set the inference order
    # -----------------------
    inference_order = []
    for name in params.INFERENCE_ORDER:
        if params.LEARN[name]:
            inference_order.append(name)
    mcmc.set_inference_order(inference_order)

    # Initialize the parameters
    # -------------------------
    for name in params.INITIALIZATION_ORDER:
        try:
            GRAPH[name].initialize(**params.INITIALIZATION_KWARGS[name])
        except:
            logger.critical('Failed in {}'.format(name))
            raise

    # Set tracing object
    # ------------------
    hdf5_filename = os.path.join(basepath, config.HDF5_FILENAME)
    mcmc_filename = os.path.join(basepath, config.MCMC_FILENAME)
    param_filename = os.path.join(basepath, config.PARAMS_FILENAME)
    mcmc.set_tracer(filename=hdf5_filename, checkpoint=params.CHECKPOINT)
    mcmc.set_save_location(mcmc_filename)
    params.save(param_filename)

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
        logger.critical('Error: {}'.format(e))
        if crash_if_error:
            raise
    mcmc.graph[STRNAMES.FILTERING].kill()
    return mcmc