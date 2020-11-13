'''Filtering parameters for the posterior
'''
import logging
import time

import numpy as np
import numpy.random as npr
import math

import matplotlib.pyplot as plt

from .util import negbin_loglikelihood_MH_condensed, \
    negbin_loglikelihood, build_prior_covariance, build_prior_mean, \
    prod_gaussians, negbin_loglikelihood_MH_condensed_not_fast, Loess
from .names import STRNAMES, REPRNAMES

from . import pylab as pl

_LOG_INV_SQRT_2PI = np.log(1/np.sqrt(2*math.pi))
def _normal_logpdf(value, mean, std):
    '''We use this function if `pylab.random.normal.logpdf` fails to compile,
    which can happen when running jobs on the cluster.
    '''
    return _LOG_INV_SQRT_2PI + (-0.5*((value-mean)/std)**2) - np.log(std)

class TrajectorySet(pl.graph.Node):
    '''This aggregates a set of trajectories from each set

    Parameters
    ----------
    name : str
        Name of the object
    G : pylab.graph.Graph
        Graph object to attach it to
    '''
    def __init__(self, name, G, **kwargs):
        pl.graph.Node.__init__(self, name=name, G=G)
        self.value = []
        n_asvs = self.G.data.n_asvs

        for ridx in range(self.G.data.n_replicates):
            n_timepoints = self.G.data.n_timepoints_for_replicate[ridx]

            # initialize values to zeros for initialization
            self.value.append(pl.variables.Variable(
                name=name+'_ridx{}'.format(ridx), G=G, shape=(n_asvs, n_timepoints),
                value=np.zeros((n_asvs, n_timepoints), dtype=float), **kwargs))
        prior = pl.variables.Normal(
            mean=pl.variables.Constant(name=self.name+'_prior_mean', value=0, G=self.G),
            var=pl.variables.Constant(name=self.name+'_prior_var', value=1, G=self.G),
            name=self.name+'_prior', G=self.G)
        self.add_prior(prior)

    def __getitem__(self, ridx):
        return self.value[ridx]

    @property
    def sample_iter(self):
        return self.value[0].sample_iter

    def reset_value_size(self):
        '''Change the size of the trajectory when we set the intermediate timepoints
        '''
        n_asvs = self.G.data.n_asvs
        for ridx in range(len(self.value)):
            n_timepoints = self.G.data.n_timepoints_for_replicate[ridx]
            self.value[ridx].value = np.zeros((n_asvs, n_timepoints),dtype=float)
            self.value[ridx].set_value_shape(self.value[ridx].value.shape)

    def _vectorize(self):
        '''Get all the data in vector form
        '''
        vals = np.array([])
        for data in self.value:
            vals = np.append(vals, data.value)
        return vals

    def mean(self):
        return np.mean(self._vectorize())

    def var(self):
        return np.var(self._vectorize())

    def iter_indices(self):
        '''Iterate through the indices and the values
        '''
        for ridx in range(self.G.data.n_replicates):
            for tidx in range(self.G.data.n_timepoints_for_replicate[ridx]):
                for oidx in range(self.G.data.asvs.n_asvs):
                    yield (ridx, tidx, oidx)

    def set_trace(self, *args, **kwargs):
        for ridx in range(len(self.value)):
            self.value[ridx].set_trace(*args, **kwargs)

    def add_trace(self):
        for ridx in range(len(self.value)):
            # Set the zero inflation values to nans
            self.value[ridx].value[~self.G[REPRNAMES.ZERO_INFLATION].value[ridx]] = np.nan
            self.value[ridx].add_trace()

    def add_init_value(self):
        for ridx in range(len(self.value)):
            self.value[ridx].add_init_value()


class FilteringLogMP(pl.graph.Node):
    '''This is the posterior for the latent trajectory that are
    sampled using a standard normal Metropolis-Hastings proposal.

    This is the multiprocessing version of the class. All of the computation is
    done on the subject level in parallel.

    Parallelization Modes
    ---------------------
    'debug'
        If this is selected, then we dont actually parallelize, but we go in
        order of the objects in sequential order. We would do this if we want
        to benchmark within each processor or do easier print statements
    'full'
        This is where each subject gets their own process

    This assumes that we are using the log model
    '''
    def __init__(self, mp, zero_inflation_transition_policy,**kwargs):
        '''
        Parameters
        ----------
        mp : str
            'debug'
                Does not actually parallelize, does it serially - we do this in case we
                want to debug and/or benchmark
            'full'
                Send each replicate to a processor each
        zero_inflation_transition_policy : None, str
            Type of zero inflation to do. If None then there is no zero inflation
        '''
        kwargs['name'] = STRNAMES.FILTERING
        pl.graph.Node.__init__(self, **kwargs)
        self.x = TrajectorySet(name=STRNAMES.LATENT_TRAJECTORY, G=self.G)
        self.mp = mp
        self.zero_inflation_transition_policy = zero_inflation_transition_policy

        self.print_vals = False
        self._strr = 'parallel'

    def __str__(self):
        return self._strr

    @property
    def sample_iter(self):
        # It doesnt matter if we chose q or x because they are both the same
        return self.x.sample_iter

    def initialize(self, x_value_option, a0, a1, v1, v2, essential_timepoints, tune, 
        proposal_init_scale, intermediate_step, h5py_filename,
        intermediate_interpolation=None, delay=0, bandwidth=None, window=None,
        target_acceptance_rate=0.44, plot_initial=False,
        calculate_qpcr_loglik=True):
        '''Initialize the values of the error model (values for the
        latent and the auxiliary trajectory). Additionally this sets
        the intermediate time points

        Initialize the values of the prior.

        Parameters
        ----------
        x_value_option : str
            Option to initialize the value of the latent trajectory.
            Options
                'coupling'
                    Sample the values around the data with extremely low variance.
                    This also truncates the data so that it stays > 0.
                'moving-avg'
                    Initialize the values using a moving average around the points.
                    The bandwidth of the filter is by number of days, not the order
                    of timepoints. You must also provide the argument `bandwidth`.
                'loess', 'auto'
                    Implements the initialization of the values using LOESS (Locally
                    Estimated Scatterplot Smoothing) algorithm. You must also provide
                    the `window` parameter
        tune : tuple(int, int)
            This is how often to tune the individual covariances
            The first element indicates which MCMC sample to stop the tuning
            The second element is how often to update the proposal covariance
        a0, a1 : float, str
            These are the hyperparameters to calculate the dispersion of the
            negative binomial.
        v1, v2 : float, int, str
            These are the values used to calulcate the coupling variance between
            x and q
        intermediate_step : tuple(str, args), array, None
            This is the type of interemediate timestep to intialize and the arguments
            for them. If this is None, then we do no intermediate timesteps.
            Options:
                'step'
                    args: (stride (numeric), eps (numeric))
                    We simulate at each timepoint every `stride` days.
                    We do not set an intermediate time point if it is within `eps`
                    days of a given data point.
                'preserve-density'
                    args: (n (int), eps (numeric))
                    We preserve the denisty of the given data by only simulating data
                    `n` times between each essential datapoint. If a timepoint is within
                    `eps` days of a given timepoint then we do not make an intermediate
                    point there.
                'manual;
                    args: np.ndarray
                    These are the points that we want to set. If these are not given times
                    then we set them as timepoints
        intermediate_interpolation : str
            This is the type of interpolation to perform on the intermediate timepoints.
            Options:
                'linear-interpolation', 'auto'
                    Perform linear interpolation between the two closest given timepoints
        essential_timepoints : np.ndarray, str, None
            These are the timepoints that must be included in each subject. If one of the
            subjects has a missing timepoint there then we use an intermediate time point
            that this timepoint. It is initialized with linear interpolation. If all of the
            timepoints specified in this vector are included in a subject then nothing is
            done. If it is a str:
                'union', 'auto'
                    We take a union of all the timepoints in each subject and make sure
                    that all of the subjects have all those points.
        bandwidth : float
            This is the day bandwidth of the filter if the initialization method is
            done with 'moving-avg'.
        window : int
            This is the window term for the LOESS initialization scheme. This is
            only used if value_initialization is done with 'loess'
        target_acceptance_rate : numeric
            This is the target acceptance rate for each time point individually
        calculate_qpcr_loglik : bool
            If True, calculate the loglikelihood of the qPCR measurements during the 
            proposal
        '''
        if not pl.isint(delay):
            raise TypeError('`delay` ({}) must be an int'.format(type(delay)))
        if delay < 0:
            raise ValueError('`delay` ({}) must be >= 0'.format(delay))
        self.delay = delay
        self._there_are_perturbations = self.G.perturbations is not None

        # Set the hyperparameters
        if not pl.isfloat(target_acceptance_rate):
            raise TypeError('`target_acceptance_rate` must be a float'.format(
                type(target_acceptance_rate)))
        if target_acceptance_rate < 0 or target_acceptance_rate > 1:
            raise ValueError('`target_acceptance_rate` ({}) must be in (0,1)'.format(
                target_acceptance_rate))
        if not pl.istuple(tune):
            raise TypeError('`tune` ({}) must be a tuple'.format(type(tune)))
        if len(tune) != 2:
            raise ValueError('`tune` ({}) must have 2 elements'.format(len(tune)))
        if not pl.isint(tune[0]):
            raise TypeError('`tune` ({}) 1st parameter must be an int'.format(type(tune[0])))
        if tune[0] < 0:
            raise ValueError('`tune` ({}) 1st parameter must be > 0'.format(tune[0]))
        if not pl.isint(tune[1]):
            raise TypeError('`tune` ({}) 2nd parameter must be an int'.format(type(tune[1])))
        if tune[1] < 0:
            raise ValueError('`tune` ({}) 2nd parameter must be > 0'.format(tune[1]))
        
        if not pl.isnumeric(a0):
            raise TypeError('`a0` ({}) must be a numeric type'.format(type(a0)))
        elif a0 <= 0:
            raise ValueError('`a0` ({}) must be > 0'.format(a0))
        if not pl.isnumeric(a1):
            raise TypeError('`a1` ({}) must be a numeric type'.format(type(a1)))
        elif a1 <= 0:
            raise ValueError('`a1` ({}) must be > 0'.format(a1))

        if not pl.isnumeric(proposal_init_scale):
            raise TypeError('`proposal_init_scale` ({}) must be a numeric type (int, float)'.format(
                type(proposal_init_scale)))
        if proposal_init_scale < 0:
            raise ValueError('`proposal_init_scale` ({}) must be positive'.format(
                proposal_init_scale))

        self.tune = tune
        self.a0 = a0
        self.a1 = a1
        self.target_acceptance_rate = target_acceptance_rate
        self.proposal_init_scale = proposal_init_scale
        self.v1 = v1
        self.v2 = v2

        # Set the essential timepoints (check to see if there is any missing data)
        if essential_timepoints is not None:
            logging.info('Setting up the essential timepoints')
            if pl.isstr(essential_timepoints):
                if essential_timepoints in ['auto', 'union']:
                    essential_timepoints = set()
                    for ts in self.G.data.times:
                        essential_timepoints = essential_timepoints.union(set(list(ts)))
                    essential_timepoints = np.sort(list(essential_timepoints))
                else:
                    raise ValueError('`essential_timepoints` ({}) not recognized'.format(
                        essential_timepoints))
            elif not pl.isarray(essential_timepoints):
                raise TypeError('`essential_timepoints` ({}) must be a str or an array'.format(
                    type(essential_timepoints)))
            logging.info('Essential timepoints: {}'.format(essential_timepoints))
            self.G.data.set_timepoints(times=essential_timepoints, eps=None, reset_timepoints=True)
            self.x.reset_value_size()

        # Set the intermediate timepoints if necessary
        if intermediate_step is not None:
            # Set the intermediate timepoints in the data
            if not pl.istuple(intermediate_step):
                raise TypeError('`intermediate_step` ({}) must be a tuple'.format(
                    type(intermediate_step)))
            if len(intermediate_step) != 2:
                raise ValueError('`intermediate_step` ({}) must be length 2'.format(
                    len(intermediate_step)))
            f, args = intermediate_step
            if not pl.isstr(f):
                raise TypeError('intermediate_step type ({}) must be a str'.format(type(f)))
            if f == 'step':
                if not pl.istuple(args):
                    raise TypeError('`args` ({}) must be a tuple'.format(type(args)))
                if len(args) != 2:
                    raise TypeError('`args` ({}) must have 2 arguments'.format(len(args)))
                step, eps = args
                self.G.data.set_timepoints(timestep=step, eps=eps, reset_timepoints=False)
            elif f == 'preserve-density':
                if not pl.istuple(args):
                    raise TypeError('`args` ({}) must be a tuple'.format(type(args)))
                if len(args) != 2:
                    raise TypeError('`args` ({}) must have 2 arguments'.format(len(args)))
                n, eps = args
                if not pl.isint(n):
                    raise TypeError('`n` ({}) must be an int'.format(type(n)))

                # For each timepoint, add `n` intermediate timepoints
                for ridx in range(self.G.data.n_replicates):
                    times = []
                    for i in range(len(self.G.data.times[ridx])-1):
                        t0 = self.G.data.times[ridx][i]
                        t1 = self.G.data.times[ridx][i+1]
                        step = (t1-t0)/(n+1)
                        times = np.append(times, np.arange(t0,t1,step=step))
                    times = np.sort(np.unique(times))
                    # print('\n\ntimes to put in', times)
                    self.G.data.set_timepoints(times=times, eps=eps, ridx=ridx, reset_timepoints=False)
                    # print('times for ridx {}'.format(self.G.data.times[ridx]))
                    # print('len times', len(self.G.data.times[ridx]))
                    # print('data shape', self.G.data.data[ridx].shape)

                # sys.exit()
            elif f == 'manual':
                raise NotImplementedError('Not Implemented')
            else:
                raise ValueError('`intermediate_step type ({}) not recognized'.format(f))
            self.x.reset_value_size()

        if intermediate_interpolation is not None:
            if intermediate_interpolation in ['linear-interpolation', 'auto']:
                for ridx in range(self.G.data.n_replicates):
                    for tidx in range(self.G.data.n_timepoints_for_replicate[ridx]):
                        if tidx not in self.G.data.given_timeindices[ridx]:
                            # We need to interpolate this time point
                            # get the previous given and next given timepoint
                            prev_tidx = None
                            for ii in range(tidx-1,-1,-1):
                                if ii in self.G.data.given_timeindices[ridx]:
                                    prev_tidx = ii
                                    break
                            if prev_tidx is None:
                                # Set to the same as the closest forward timepoint then continue
                                next_idx = None
                                for ii in range(tidx+1, self.G.data.n_timepoints_for_replicate[ridx]):
                                    if ii in self.G.data.given_timeindices[ridx]:
                                        next_idx = ii
                                        break
                                self.G.data.data[ridx][:,tidx] = self.G.data.data[ridx][:,next_idx]
                                continue

                            next_tidx = None
                            for ii in range(tidx+1, self.G.data.n_timepoints_for_replicate[ridx]):
                                if ii in self.G.data.given_timeindices[ridx]:
                                    next_tidx = ii
                                    break
                            if next_tidx is None:
                                # Set to the previous timepoint then continue
                                self.G.data.data[ridx][:,tidx] = self.G.data.data[ridx][:,prev_tidx]
                                continue

                            # Interpolate from prev_tidx to next_tidx
                            x = self.G.data.times[ridx][tidx]
                            x0 = self.G.data.times[ridx][prev_tidx]
                            y0 = self.G.data.data[ridx][:,prev_tidx]
                            x1 = self.G.data.times[ridx][next_tidx]
                            y1 = self.G.data.data[ridx][:,next_tidx]
                            self.G.data.data[ridx][:,tidx] = y0 * (1-((x-x0)/(x1-x0))) + y1 * (1-((x1-x)/(x1-x0)))
            else:
                raise ValueError('`intermediate_interpolation` ({}) not recognized'.format(intermediate_interpolation))

        # Initialize the latent trajectory
        if not pl.isstr(x_value_option):
            raise TypeError('`x_value_option` ({}) is not a str'.format(type(x_value_option)))
        if x_value_option == 'coupling':
            self._init_coupling()
        elif x_value_option == 'moving-avg':
            if not pl.isnumeric(bandwidth):
                raise TypeError('`bandwidth` ({}) must be a numeric'.format(type(bandwidth)))
            if bandwidth <= 0:
                raise ValueError('`bandwidth` ({}) must be positive'.format(bandwidth))
            self.bandwidth = bandwidth
            self._init_moving_avg()
        elif x_value_option in ['loess', 'auto']:
            if window is None:
                raise TypeError('If `value_option` is loess, then `window` must be specified')
            if not pl.isint(window):
                raise TypeError('`window` ({}) must be an int'.format(type(window)))
            if window <= 0:
                raise ValueError('`window` ({}) must be > 0'.format(window))
            self.window = window
            self._init_loess()
        else:
            raise ValueError('`x_value_option` ({}) not recognized'.format(x_value_option))

        # Get necessary data and set the parallel objects
        if self._there_are_perturbations:
            pert_starts = []
            pert_ends = []
            for perturbation in self.G.perturbations:
                pert_starts.append(perturbation.start)
                pert_ends.append(perturbation.end)
        else:
            pert_starts = None
            pert_ends = None

        if self.mp is None:
            self.mp = 'debug'
        if not pl.isstr(self.mp):
            raise TypeError('`mp` ({}) must either be a string or None'.format(type(self.mp)))
        if self.mp == 'debug':
            self.pool = []
        elif self.mp == 'full':
            self.pool = pl.multiprocessing.PersistentPool(G=self.G, ptype='sadw')
            self.worker_pids = []
        else:
            raise ValueError('`mp` ({}) not recognized'.format(self.mp))

        for ridx in range(self.G.data.n_replicates):
            # Set up qPCR measurements and reads to send
            qpcr_log_measurements = {}
            for t in self.G.data.given_timepoints[ridx]:
                qpcr_log_measurements[t] = self.G.data.qpcr[ridx][t].log_data
            reads = self.G.data.subjects.iloc(ridx).reads

            worker = SubjectLogTrajectorySetMP()
            worker.initialize(
                zero_inflation_transition_policy=self.zero_inflation_transition_policy,
                times=self.G.data.times[ridx],
                qpcr_log_measurements=qpcr_log_measurements,
                reads=reads,
                there_are_intermediate_timepoints=True,
                there_are_perturbations=self._there_are_perturbations,
                pv_global=self.G[REPRNAMES.PROCESSVAR].global_variance,
                x_prior_mean=np.log(1e7),
                x_prior_std=1e10,
                tune=tune[1],
                delay=delay,
                end_iter=tune[0],
                proposal_init_scale=proposal_init_scale,
                a0=a0,
                a1=a1,
                x=self.x[ridx].value,
                pert_starts=np.asarray(pert_starts),
                pert_ends=np.asarray(pert_ends),
                ridx=ridx,
                calculate_qpcr_loglik=calculate_qpcr_loglik,
                h5py_filename=h5py_filename,
                h5py_xname=self.x[ridx].name,
                target_acceptance_rate=self.target_acceptance_rate)
            if self.mp == 'debug':
                self.pool.append(worker)
            elif self.mp == 'full':
                pid = self.pool.add_worker(worker)
                self.worker_pids.append(pid)

        # Set the data to the latent values
        self.set_latent_as_data(update_values=False)

        self.total_n_datapoints = 0
        for ridx in range(self.G.data.n_replicates):
            self.total_n_datapoints += self.x[ridx].value.shape[0] * self.x[ridx].value.shape[1]

        if plot_initial:
            oidxs = np.arange(self.G.data.n_asvs)
            latent = self.x.value[0].value
            for i in range(len(oidxs)):
                fig = plt.figure()
                ax = fig.add_subplot(111)
                times = self.G.data.times
                ax.plot(self.G.data.given_timepoints[0], self.G.data.abs_data[0][oidxs[i], :], label='Given', color='r', marker='o', alpha=0.5)
                # ax.plot(self.G.data.given_timepoints[0],
                #     syndata.data[0][oidxs[i], :] * self.G.data.subjects.qpcr_normalization_factor,
                #     label='Truth', color='black', marker='o', alpha=0.5)
                ax.plot(times[0], latent[i,:], label='before latent', color='b', linestyle=':', 
                    marker='o')
                ax.set_title('oidx {}'.format(oidxs[i]))
                ax.legend()
                ax.set_yscale('log')
                plt.savefig('oidx{}.pdf'.format(oidxs[i]))
                plt.close()

    def _init_coupling(self):
        '''Initialize `x` by sampling around the data using a small
        variance using a truncated normal distribution
        '''
        for ridx, tidx, oidx in self.x.iter_indices():
            val = self.G.data.data[ridx][oidx,tidx]
            self.x[ridx][oidx,tidx] = pl.random.truncnormal.sample(
                mean=val,
                std=math.sqrt(self.v1 * (val ** 2) + self.v2),
                low=0, high=float('inf'))

    def _init_moving_avg(self):
        '''Initializes `x` by using a moving
        average over the data - using `self.bandwidth` as the bandwidth
        of number of days - it then samples around that point using the
        coupling variance.

        If there are no other points within the bandwidth around the point,
        then it just samples around the current timepoint with the coupling
        variance.
        '''
        for ridx in range(self.G.data.n_replicates):
            for tidx in range(self.G.data.n_timepoints_for_replicate[ridx]):
                tidx_low = np.searchsorted(
                    self.G.data.times[ridx], self.G.data.times[ridx][tidx]-self.bandwidth)
                tidx_high = np.searchsorted(
                    self.G.data.times[ridx], self.G.data.times[ridx][tidx]+self.bandwidth)

                for oidx in range(len(self.G.data.asvs)):
                    val = np.mean(self.G.data.data[ridx][oidx, tidx_low: tidx_high])
                    self.x[ridx][oidx,tidx] = pl.random.truncnormal.sample(
                        mean=val,
                        std=math.sqrt(self.v1 * (val ** 2) + self.v2),
                        low=0, high=float('inf'))

    def _init_loess(self):
        '''Initialize the data using LOESS algorithm and then samples around that
        the coupling variance we implement the LOESS algorithm in the module
        `fit_loess.py`
        '''
        for ridx in range(self.G.data.n_replicates):
            xx = self.G.data.times[ridx]
            for oidx in range(len(self.G.data.asvs)):
                yy = self.G.data.data[ridx][oidx, :]
                loess = Loess(xx, yy)

                for tidx, t in enumerate(self.G.data.times[ridx]):
                    val = loess.estimate(t, window=self.window)
                    self.x[ridx][oidx,tidx] = pl.random.truncnormal.sample(
                        mean=val,
                        std=math.sqrt(self.v1 * (val ** 2) + self.v2),
                        low=0, high=float('inf'))

                    if np.isnan(self.x[ridx][oidx, tidx]):
                        print('crashed here', ridx, tidx, oidx)
                        print('mean', val)
                        print('t', t)
                        print('yy', yy)
                        print('std', math.sqrt(self.v1 * (val ** 2) + self.v2))
                        raise ValueError('')

    def set_latent_as_data(self, update_values=True):
        '''Change the values in the data matrix so that it is the latent variables
        '''
        data = []
        for obj in self.x.value:
            data.append(obj.value)
        self.G.data.data = data
        if update_values:
            self.G.data.update_values()

    def add_trace(self):
        self.x.add_trace()

    def add_init_value(self):
        self.x.add_init_value()

    def set_trace(self, *args, **kwargs):
        self.x.set_trace(*args, **kwargs)

    def kill(self):
        if self.mp == 'full':
            self.pool.kill()

    def update(self):
        '''Send out to each parallel object
        '''
        if self.sample_iter < self.delay:
            return
        start_time = time.time()

        growth = self.G[REPRNAMES.GROWTH_VALUE].value.ravel()
        self_interactions = self.G[REPRNAMES.SELF_INTERACTION_VALUE].value.ravel()
        pv = self.G[REPRNAMES.PROCESSVAR].value
        interactions = self.G[REPRNAMES.INTERACTIONS_OBJ].get_datalevel_value_matrix(
            set_neg_indicators_to_nan=False)
        perts = None
        if self._there_are_perturbations:
            perts = []
            for perturbation in self.G.perturbations:
                perts.append(perturbation.item_array().reshape(-1,1))
            perts = np.hstack(perts)

        # zero_inflation = [self.G[REPRNAMES.ZERO_INFLATION].value[ridx] for ridx in range(self.G.data.n_replicates)]
        qpcr_vars = []
        for aaa in self.G[REPRNAMES.QPCR_VARIANCES].value:
            qpcr_vars.append(aaa.value)
        
        
        kwargs = {'growth':growth, 'self_interactions':self_interactions,
            'pv':pv, 'interactions':interactions, 'perturbations':perts, 
            'zero_inflation_data': None, 'qpcr_variances':qpcr_vars}

        str_acc = [None]*self.G.data.n_replicates
        if self.mp == 'debug':

            for ridx in range(self.G.data.n_replicates):
                _, x, acc_rate = self.pool[ridx].persistent_run(**kwargs)
                self.x[ridx].value = x
                str_acc[ridx] = '{:.3f}'.format(acc_rate)

        else:
            # raise NotImplementedError('Multiprocessing for filtering with zero inflation ' \
            #     'is not implemented')
            ret = self.pool.map(func='persistent_run', args=kwargs)
            for ridx, x, acc_rate in ret:
                self.x[ridx].value = x
                str_acc[ridx] = '{:.3f}'.format(acc_rate)

        self.set_latent_as_data()

        t = time.time() - start_time
        try:
            self._strr = 'Time: {:.4f}, Acc: {}, data/sec: {:.2f}'.format(t,
                str(str_acc).replace("'",''), self.total_n_datapoints/t)
        except:
            self._strr = 'NA'

    

class SubjectLogTrajectorySetMP(pl.multiprocessing.PersistentWorker):
    '''This performs filtering on a multiprocessing level. We send the
    other parameters of the model and return the filtered `x` and values.
    With multiprocessing, this class has ~91% efficiency. Additionally, the code
    in this class is optimized to be ~20X faster than the code in `Filtering`. This
    assumes we have the log model.

    It might seem unneccessary to have so many local attributes, but it speeds up
    the inference considerably if we index a value from an array once and store it
    as a float instead of repeatedly indexing the array - the difference in
    reality is super small but we do this so often that it adds up to ~40% speedup
    as supposed to not doing it - so we do this as often as possible - This speedup
    is even greater for indexing keys of dictionaries and getting parameters of objects.

    General efficiency speedups
    ---------------------------
    All of these are done relative to a non-optimized filtering implementation
        - Specialized sampling and logpdf functions. About 95% faster than
          scipy or numpy functions. All of these add up to a ~35% speed up
        - Explicit function definitions:
          instead of doing `self.prior.logpdf(...)`, we do
          `pl.random.normal.logpdf(...)`, about a ~10% overall speedup
        - Precomputation of values so that the least amout of computation
          is done on a data level - All of these add up to a ~25% speed up
    Benchmarked on a MacPro

    Non/trivial efficiency speedups w.r.t. non-multiprocessed filtering
    -------------------------------------------------------------------
    All of the speed ups are done relative to the current implementation
    in non-multiprocessed filtering.
        - Whenever possible we replace a 2D variable like `self.x` with a
          `curr_x`, which is 1D, because indexing a 1D array is 10-20% faster
          than a 2D array. All of these add up to a ~8% speed up
        - We only every compute the forward dynamics (not the reverse), because
          we can use the forward of the previous timepoint as the reverse for
          the next timepoint. This is about 45% faster and adds up to a ~40%
          speedup.
        - Whenever possible, we replace a dictionary like `read_depths`
          with a float because indexing a dict is 12-20% slower than a 1D
          array. All these add up to a ~7% speed up
        - We precompute AS MUCH AS POSSIBLE in `update` and in `initialize`,
          even simple this as `self.curr_tidx_minus_1`: all of these add up
          to about ~5% speedup
        - If an attribute of a class is being referenced more than once in
          a subroutine, we "get" it by making it a local variable. Example:
          `tidx = self.tidx`. This has about a 5% speed up PER ADDITIONAL
          CALL within the subroutine. All of these add up to ~2.5% speed up.
        - If an indexed value gets indexed more than once within a subroutine,
          we "get" the value by making it a local variable. All of these
          add up to ~4% speed up.
        - We "get" all of the means and stds of the qPCR data-structures so we
          do not reference an object. This is about 22% faster and adds up
          to a ~3% speed up.
    Benchmarked on a MacPro
    '''
    def __init__(self):
        '''Set all local variables to None
        '''
        return

    def initialize(self, times, qpcr_log_measurements, reads, there_are_intermediate_timepoints,
        there_are_perturbations, pv_global, x_prior_mean,
        x_prior_std, tune, delay, end_iter, proposal_init_scale, a0, a1, x, calculate_qpcr_loglik,
        pert_starts, pert_ends, ridx, h5py_filename, h5py_xname, target_acceptance_rate,
        zero_inflation_transition_policy):
        '''Initialize the object at the beginning of the inference

        n_o = Number of ASVs
        n_gT = Number of given time points
        n_T = Total number of time points, including intermediate
        n_P = Number of Perturbations

        Parameters
        ----------
        times : np.array((n_T, ))
            Times for each of the time points
        qpcr_log_measurements : dict(t -> np.ndarray(float))
            These are the qPCR observations for every timepoint in log space.
        reads : dict (float -> np.ndarray((n_o, )))
            The counts for each of the given timepoints. Each value is an
            array for the counts for each of the ASVs
        there_are_intermediate_timepoints : bool
            If True, then there are intermediate timepoints, else there are only
            given timepoints
        there_are_perturbations : bool
            If True, that means there are perturbations, else there are no
            perturbations
        pv_global : bool
            If True, it means the process variance is is global for each ASV. If
            False it means that there is a separate `pv` for each ASV
        pv : float, np.ndarray
            This is the process variance value. This is a float if `pv_global` is True
            and it is an array if `pv_global` is False.
        x_prior_mean, x_prior_std : numeric
            This is the prior mean and std for `x` used when sampling the reverse for
            the first timepoint
        tune : int
            How often we should update the proposal for each ASV
        delay : int
            How many MCMC iterations we should delay the start of updating
        end_iter : int
            What iteration we should stop updating the proposal
        proposal_init_scale : float
            Scale to multiply the initial covariance of the poposal
        a0, a1 : floats
            These are the negative binomial dispersion parameters that specify how
            much noise there is in the counts
        x : np.ndarray((n_o, n_T))
            This is the x initialization
        pert_starts, pert_ends : np.ndarray((n_P, ))
            The starts and ends for each one of the perturbations
        ridx : int
            This is the replicate index that this object corresponds to
        h5py_filename : str
            Name of the h5py object that stores the values
        h5py_xname : str
            This is the name for the x in the h5py object
        target_acceptance_rate : float
            This is the target acceptance rate for each point
        calculate_qpcr_loglik : bool
            If True, calculate the loglikelihood of the qPCR measurements during the proposal
        '''
        self.h5py_filename = h5py_filename
        self.h5py_xname = h5py_xname
        self.target_acceptance_rate = target_acceptance_rate
        self.zero_inflation_transition_policy = zero_inflation_transition_policy

        self.times = times
        self.qpcr_log_measurements = qpcr_log_measurements
        self.reads = reads
        self.there_are_intermediate_timepoints = there_are_intermediate_timepoints
        self.there_are_perturbations = there_are_perturbations
        self.pv_global = pv_global
        if not pv_global:
            raise TypeError('Filtering with MP not implemented for non global process variance')
        self.x_prior_mean = x_prior_mean
        self.x_prior_std = x_prior_std
        self.tune = tune
        self.delay = 0
        self.end_iter = end_iter
        self.proposal_init_scale = proposal_init_scale
        self.a0 = a0
        self.a1 = a1
        self.n_asvs = x.shape[0]
        self.n_timepoints = len(times)
        self.n_timepoints_minus_1 = len(times)-1
        self.logx = np.log(x)
        self.x = x
        self.pert_starts = pert_starts
        self.pert_ends = pert_ends
        self.total_n_points = self.x.shape[0] * self.x.shape[1]
        self.ridx = ridx
        self.calculate_qpcr_loglik = calculate_qpcr_loglik

        self.sample_iter = 0
        self.n_data_points = self.x.shape[0] * self.x.shape[1]

        # latent state
        self.sum_q = np.sum(self.x, axis=0)
        shape = (self.tune, ) + self.x.shape
        self.trace_iter = 0

        # proposal
        self.proposal_std = np.log(1.5) #np.log(3)
        self.acceptances = 0
        self.n_props_total = 0
        self.n_props_local = 0
        self.total_acceptances = 0
        self.add_trace = True

        # Intermediate timepoints
        if self.there_are_intermediate_timepoints:
            self.is_intermediate_timepoint = {}
            self.data_loglik = self.data_loglik_w_intermediates
            for t in self.times:
                self.is_intermediate_timepoint[t] = t not in self.reads
        else:
            self.data_loglik = self.data_loglik_wo_intermediates

        # Reads
        self.read_depths = {}
        for t in self.reads:
            self.read_depths[t] = float(np.sum(self.reads[t]))

        # t
        self.dts = np.zeros(self.n_timepoints_minus_1)
        self.sqrt_dts = np.zeros(self.n_timepoints_minus_1)
        for k in range(self.n_timepoints_minus_1):
            self.dts[k] = self.times[k+1] - self.times[k]
            self.sqrt_dts[k] = np.sqrt(self.dts[k])
        self.t2tidx = {}
        for tidx, t in enumerate(self.times):
            self.t2tidx[t] = tidx

        self.cnt_accepted_times = np.zeros(len(self.times))

        # Perturbations
        # -------------
        # in_pert_transition : np.ndarray(dtype=bool)
        #   This is a bool array where if it is a true it means that the
        #   forward and reverse growth rates are different
        # fully_in_pert : np.ndarray(dtype=int)
        #   This is an int-array where it tells you which perturbation you are fully in
        #   (the forward and reverse growth rates are the same but not the default).
        #   If there is no perturbation then the value is -1. If it is not -1, then the
        #   number corresponds to what perturbation index you are in.
        #
        # Edge cases
        # ----------
        #   * missing data for start
        #       There could be a situation where there was no sample collection
        #       on the day that they started a perturbation. In this case we
        #       assume that the next time point is the `start` of the perturbation.
        #       i.e. the next time point is the perturbation transition.
        #   * missing data for end
        #       There could be a situation where no sample was collected when the
        #       perturbation ended. In this case we assume that the pervious time
        #       point was the end of the perturbation.
        if self.there_are_perturbations:
            self.in_pert_transition = np.zeros(self.n_timepoints, dtype=bool)
            self.fully_in_pert = np.ones(self.n_timepoints, dtype=int) * -1
            for pidx, t in enumerate(self.pert_starts):
                if t == self.times[-1] or t == self.times[0]:
                    raise ValueError('The code right now does not support either a perturbation that ' \
                        'started on the first day or ended on the last day. The code where this is ' \
                        'incompatible is when we checking if we are in a perturbation transition')
                if t > np.max(self.times):
                    continue
                if t not in self.t2tidx:
                    # Use the next time point
                    tidx = np.searchsorted(self.times, t)
                else:
                    tidx = self.t2tidx[t]
                self.in_pert_transition[tidx] = True
            for pidx, t in enumerate(self.pert_ends):
                if t == self.times[-1] or t == self.times[0]:
                    raise ValueError('The code right now does not support either a perturbation that ' \
                        'started on the first day or ended on the last day. The code where this is ' \
                        'incompatible is when we checking if we are in a perturbation transition')
                if t < np.min(self.times) or t > np.max(self.times):
                    continue
                if t not in self.t2tidx:
                    # Use the previous time point
                    tidx = np.searchsorted(self.times, t) - 1
                else:
                    tidx = self.t2tidx[t]
                self.in_pert_transition[tidx] = True

            # check if anything is weird
            if np.sum(self.in_pert_transition) % 2 != 0:
                raise ValueError('The number of in_pert_transition periods must be even ({})' \
                    '. There is either something wrong with the data (start and end day are ' \
                    'the same) or with the algorithm ({})'.format(
                        np.sum(self.in_pert_transition),
                        self.in_pert_transition))

            # Make the fully in perturbation times
            for pidx in range(len(self.pert_ends)):
                try:
                    start_tidx = self.t2tidx[self.pert_starts[pidx]] + 1
                    end_tidx = self.t2tidx[self.pert_ends[pidx]]
                except:
                    # This means there is a missing datapoint at either the
                    # start or end of the perturbation
                    start_t = self.pert_starts[pidx]
                    end_t = self.pert_ends[pidx]
                    start_tidx = np.searchsorted(self.times, start_t)
                    end_tidx = np.searchsorted(self.times, end_t) - 1

                self.fully_in_pert[start_tidx:end_tidx] = pidx
    
    # @profile
    def persistent_run(self, growth, self_interactions, pv, interactions,
        perturbations, qpcr_variances, zero_inflation_data):
        '''Run an update of the values for a single gibbs step for all of the data points
        in this replicate

        Parameters
        ----------
        growth : np.ndarray((n_asvs, ))
            Growth rates for each ASV
        self_interactions : np.ndarray((n_asvs, ))
            Self-interactions for each ASV
        pv : numeric, np.ndarray
            This is the process variance
        interactions : np.ndarray((n_asvs, n_asvs))
            These are the ASV-ASV interactions
        perturbations : np.ndarray((n_perturbations, n_asvs))
            Perturbation values in the right perturbation order, per ASV
        zero_inflation : np.ndarray
            These are the points that are delibertly pushed down to zero
        qpcr_variances : np.ndarray
            These are the sampled qPCR variances as an array - they are in
            time order

        Returns
        -------
        (int, np.ndarray, float)
            1 This is the replicate index
            2 This is the updated latent state for logx
            3 This is the acceptance rate for this past update.
        '''
        self.master_growth_rate = growth

        if self.sample_iter < self.delay:
            self.sample_iter += 1
            return self.ridx, self.x, np.nan

        self.update_proposals()
        self.n_accepted_iter = 0
        self.pv = pv
        self.pv_std = np.sqrt(pv)
        self.qpcr_stds = np.sqrt(qpcr_variances[self.ridx])
        self.qpcr_stds_d = {}
        # self.zero_inflation_data = zero_inflation_data[self.ridx]
        self.zero_inflation_data = None

        for tidx,t in enumerate(self.qpcr_log_measurements):
            self.qpcr_stds_d[t] = self.qpcr_stds[tidx]

        if self.there_are_perturbations:
            self.growth_rate_non_pert = growth.ravel()
            self.growth_rate_on_pert = growth.reshape(-1,1) * (1 + perturbations)
                
        # Go through each randomly ASV and go in time order
        oidxs = npr.permutation(self.n_asvs)
        # print('===============================')
        # print('===============================')
        # print('ridx', self.ridx)
        for oidx in oidxs:

            # Set the necessary global parameters
            self.oidx = oidx
            self.curr_x = self.x[oidx, :]
            self.curr_logx = self.logx[oidx, :]
            self.curr_interactions = interactions[oidx, :]
            self.curr_self_interaction = self_interactions[oidx]
            # self.curr_zero_inflation = self.zero_inflation[oidx, :]

            if self.pv_global:
                self.curr_pv_std = self.pv_std
            else:
                self.curr_pv_std = self.pv_std[oidx]

            # Set for first time point
            self.tidx = 0
            self.set_attrs_for_timepoint()
            self.forward_loglik = self.default_forward_loglik
            self.reverse_loglik = self.first_timepoint_reverse
            # Calculate A matrix for forward
            self.forward_interaction_vals = np.nansum(self.x[:, self.tidx] * self.curr_interactions)
            self.update_single()
            self.reverse_loglik = self.default_reverse_loglik
            # Set for middle timepoints
            for tidx in range(1, self.n_timepoints-1):
                # Check if it needs to be zero inflated
                # if not self.curr_zero_inflation[tidx]:
                #     raise NotImplementedError('Zero inflation not implemented for logmodel')

                self.tidx = tidx
                self.set_attrs_for_timepoint()

                # Calculate A matrix for forward and reverse
                # Set the reverse of the current time step to the forward of the previous
                self.reverse_interaction_vals = self.forward_interaction_vals #np.sum(self.x[:, self.prev_tidx] * self.curr_interactions)
                self.forward_interaction_vals = np.nansum(self.x[:, self.tidx] * self.curr_interactions)

                # Run single update
                self.update_single()

            # Set for last timepoint
            self.tidx = self.n_timepoints_minus_1
            self.set_attrs_for_timepoint()
            self.forward_loglik = self.last_timepoint_forward
            # Calculate A matrix for reverse
            # Set the reverse of the current time step to the forward of the previous
            self.reverse_interaction_vals = self.forward_interaction_vals # np.sum(self.x[:, self.prev_tidx] * self.curr_interactions)
            self.update_single()

            # if self.sample_iter == 4:
            # sys.exit()

        self.sample_iter += 1
        if self.add_trace:
            self.trace_iter += 1

        # print(self.cnt_accepted_times/self.sample_iter)

        return self.ridx, self.x, self.n_accepted_iter/self.n_data_points

    def set_attrs_for_timepoint(self):
        self.prev_tidx = self.tidx-1
        self.next_tidx = self.tidx+1
        self.forward_growth_rate = self.master_growth_rate[self.oidx]
        self.reverse_growth_rate = self.master_growth_rate[self.oidx]

        if self.there_are_intermediate_timepoints:
            if not self.is_intermediate_timepoint[self.times[self.tidx]]:
                # It is not intermediate timepoints - we need to get the data
                t = self.times[self.tidx]
                self.curr_reads = self.reads[t][self.oidx]
                self.curr_read_depth = self.read_depths[t]
                self.curr_qpcr_log_measurements = self.qpcr_log_measurements[t]
                self.curr_qpcr_std = self.qpcr_stds_d[t]
        else:
            t = self.times[self.tidx]
            self.curr_reads = self.reads[t][self.oidx]
            self.curr_read_depth = self.read_depths[t]
            self.curr_qpcr_log_measurements = self.qpcr_log_measurements[t]
            self.curr_qpcr_std = self.qpcr_stds_d[t]

        # Set perturbation growth rates
        if self.there_are_perturbations:
            if self.in_pert_transition[self.tidx]:
                if self.fully_in_pert[self.tidx-1] != -1:
                    # If the previous time point is in the perturbation, that means
                    # we are going out of the perturbation
                    # self.forward_growth_rate = self.master_growth_rate[self.oidx]
                    pidx = self.fully_in_pert[self.tidx-1]
                    self.reverse_growth_rate = self.growth_rate_on_pert[self.oidx,pidx]
                else:
                    # Else we are going into a perturbation
                    # self.reverse_growth_rate = self.master_growth_rate[self.oidx]
                    pidx = self.fully_in_pert[self.tidx+1]
                    self.forward_growth_rate = self.growth_rate_on_pert[self.oidx,pidx]
            elif self.fully_in_pert[self.tidx] != -1:
                pidx = self.fully_in_pert[self.tidx]
                self.forward_growth_rate = self.growth_rate_on_pert[self.oidx,pidx]
                self.reverse_growth_rate = self.forward_growth_rate

    # @profile
    def update_single(self):
        '''Update a single oidx, tidx
        '''
        tidx = self.tidx
        oidx = self.oidx

        # Check if we should update the zero inflation policy
        if self.zero_inflation_transition_policy is not None:
            if self.zero_inflation_transition_policy == 'ignore':
                if not self.zero_inflation_data[oidx,tidx]:
                    self.x[oidx, tidx] = np.nan
                    self.logx[oidx, tidx] = np.nan
                    return
                else:
                    if tidx < self.zero_inflation_data.shape[1]-1:
                        do_forward = self.zero_inflation_data[oidx, tidx+1]
                    else:
                        do_forward = True
                    if tidx > 0:
                        do_reverse = self.zero_inflation_data[oidx, tidx-1]
                    else:
                        do_reverse = True
            else:
                raise NotImplementedError('Not Implemented')
        else:
            do_forward = True
            do_reverse = True

        # t = self.times[self.tidx]
        # # proposal
        # mu1 = self.curr_logx[tidx]
        # rel = self.reads[t][oidx]/self.read_depths[t]
        # if rel == 0:
        #     rel = 1e-5
        # mu2 = np.log(rel*np.exp(self.curr_qpcr_loc + (self.curr_qpcr_scale/2)))

        # var1 = self.proposal_std[(tidx, oidx)]**2
        # var2 = (self.curr_qpcr_scale)**2
        # mu,var = prod_gaussians(means=[mu1,mu2], variances=[var1,var2])

        try:
            logx_new = pl.random.misc.fast_sample_normal(
                self.curr_logx[tidx],
                self.proposal_std)
        except:
            print('mu', self.curr_logx[tidx])
            print('std', self.proposal_std)
            raise
        # try:
        #     logx_new = pl.random.misc.fast_sample_normal(
        #         mu, np.sqrt(var))
        # except:
        #     print('mu', mu)
        #     print('std', np.sqrt(var))
        #     raise

        x_new = np.exp(logx_new)
        prev_logx_value = self.curr_logx[tidx]
        prev_x_value = self.curr_x[tidx]

        # print('prex_x', prev_x_value, np.exp(prev_logx_value))
        # print('prev_logx', prev_logx_value)

        # if tidx == 5:
        #     print('\ntidx', tidx)
        #     print('oidx', oidx)
        #     print('t', self.times[self.tidx])
        #     print('curr_logx', self.curr_logx[tidx])
        #     # print('curr_logx', self.curr_logx[tidx])
        #     # print('mu1, mu2', mu1, mu2)
        #     # print('mu', mu)
        #     # print('var1, var2', var1,var2)
        #     # print('var',var)
        #     print('prop_logx', logx_new)
        #     # print('start perts', self.pert_starts)
        #     # print('end perts', self.pert_ends)
        #     # print('in perturbation transition?', self.in_pert_transition[tidx])
        #     # print('fully in pert?', self.fully_in_pert[self.tidx])
        #     print('forward growth', self.forward_growth_rate)

        if do_forward:
            prev_aaa = self.forward_loglik()
        else:
            prev_aaa = 0
        if do_reverse:
            prev_bbb = self.reverse_loglik()
        else:
            prev_bbb = 0
        prev_ddd = self.data_loglik()

        # if tidx == 5:
        #     print('\nold')
        #     print('forward ll', aaa)
        #     print('reverse ll', bbb)
        #     print('data ll', ddd)

        l_old = prev_aaa + prev_bbb + prev_ddd

        self.curr_x[tidx] = x_new
        self.curr_logx[tidx] = logx_new
        self.sum_q[tidx] = self.sum_q[tidx] - prev_x_value + x_new

        if do_forward:
            new_aaa = self.forward_loglik()
        else:
            new_aaa = 0
        if do_reverse:
            new_bbb = self.reverse_loglik()
        else:
            new_bbb = 0
        new_ddd = self.data_loglik()

        # if tidx == 5:
        #     print('\nnew')
        #     print('forward ll', aaa)
        #     print('reverse ll', bbb)
        #     print('data ll', ddd)
        #     print('\nold x value', prev_x_value)
        #     print('old logx value', prev_logx_value)
        #     print('proposal std', self.proposal_std[(tidx, oidx)])
        #     print('new x value', x_new)
        #     print('new logx value', logx_new)

        l_new = new_aaa + new_bbb + new_ddd
        r_accept = l_new - l_old

        # if tidx == 0:
        #     print('\n\noidx {} diff lls:'.format(oidx), r_accept)
        #     print('\tforward', new_aaa - prev_aaa)
        #     print('\treverse', new_bbb - prev_bbb)
        #     print('\tdata', new_ddd - prev_ddd)

        # if tidx == 5:
        #     print('r_accept', r_accept)
        r = pl.random.misc.fast_sample_standard_uniform()
        if math.log(r) > r_accept:
            # print('reject')
            # reject
            self.sum_q[tidx] = self.sum_q[tidx] + prev_x_value - x_new
            self.curr_x[tidx] = prev_x_value
            self.curr_logx[tidx] = prev_logx_value
        else:
            # print('accept')
            self.x[oidx, tidx] = x_new
            self.logx[oidx, tidx] = logx_new
            self.acceptances += 1
            self.total_acceptances += 1
            self.n_accepted_iter += 1

        self.n_props_local += 1
        self.n_props_total += 1

    def update_proposals(self):
        '''Update the proposal if necessary
        '''
        if self.sample_iter > self.end_iter:
            self.add_trace = False
            return
        if self.sample_iter == 0:
            return
        if self.trace_iter - self.delay == self.tune  and self.sample_iter - self.delay > 0:

            # Adjust
            acc_rate = self.acceptances/self.n_props_total
            if acc_rate < 0.1:
                logging.debug('Very low acceptance rate, scaling down past covariance')
                self.proposal_std *= 0.01
            elif acc_rate < self.target_acceptance_rate:
                self.proposal_std /= np.sqrt(1.5)
            else:
                self.proposal_std *= np.sqrt(1.5)
            
            self.acceptances = 0
            self.n_props_local = 0

    def last_timepoint_forward(self):
        return 0

    # @profile
    def default_forward_loglik(self):
        '''From the current timepoint (tidx) to the next timepoint (tidx+1)
        '''
        logmu = self.compute_dynamics(
            tidx=self.tidx,
            Axj=self.forward_interaction_vals,
            a1=self.forward_growth_rate)

        try:
            return pl.random.normal.logpdf(
                value=self.curr_logx[self.next_tidx], 
                mean=logmu, std=self.curr_pv_std*self.sqrt_dts[self.tidx])
        except:
            return _normal_logpdf(
                value=self.curr_logx[self.next_tidx], 
                mean=logmu, std=self.curr_pv_std*self.sqrt_dts[self.tidx])

    def first_timepoint_reverse(self):
        # sample from the prior
        try:
            return pl.random.normal.logpdf(value=self.curr_logx[self.tidx],
                mean=self.x_prior_mean, std=self.x_prior_std)
        except:
            return _normal_logpdf(value=self.curr_logx[self.tidx],
                mean=self.x_prior_mean, std=self.x_prior_std)

    # @profile
    def default_reverse_loglik(self):
        '''From the previous timepoint (tidx-1) to the current time point (tidx)
        '''
        logmu = self.compute_dynamics(
            tidx=self.prev_tidx,
            Axj=self.reverse_interaction_vals,
            a1=self.reverse_growth_rate)

        try:
            return pl.random.normal.logpdf(value=self.curr_logx[self.tidx], 
                mean=logmu, std=self.curr_pv_std*self.sqrt_dts[self.prev_tidx])
        except:
            return _normal_logpdf(value=self.curr_logx[self.tidx], 
                mean=logmu, std=self.curr_pv_std*self.sqrt_dts[self.prev_tidx])

    def data_loglik_w_intermediates(self):
        '''data loglikelihood w/ intermediate timepoints
        '''
        if self.is_intermediate_timepoint[self.times[self.tidx]]:
            return 0
        else:
            return self.data_loglik_wo_intermediates()

    # @profile
    def data_loglik_wo_intermediates(self):
        '''data loglikelihood with intermediate timepoints
        '''
        sum_q = self.sum_q[self.tidx]
        log_sum_q = math.log(sum_q)
        rel = self.curr_x[self.tidx] / sum_q

        try:
            negbin = negbin_loglikelihood_MH_condensed(
                k=self.curr_reads,
                m=self.curr_read_depth * rel,
                dispersion=self.a0/rel + self.a1)
        except:
            negbin = negbin_loglikelihood_MH_condensed_not_fast(
                k=self.curr_reads,
                m=self.curr_read_depth * rel,
                dispersion=self.a0/rel + self.a1)

        qpcr = 0
        if self.calculate_qpcr_loglik:
            for qpcr_val in self.curr_qpcr_log_measurements:
                a = pl.random.normal.logpdf(value=qpcr_val, mean=log_sum_q, std=self.curr_qpcr_std)
                qpcr += a

        # tidx = self.tidx
        # if True: #tidx in [3,6,7]:
        #     print('\n\nData, tidx', tidx)
        #     print('sum_q:', sum_q)
        #     print('rel:', rel)
        #     print('qpcr: {}\n\tvalue: {}\n\tmean: {}\n\tstd: {}'.format(
        #         qpcr, self.curr_qpcr_loc,
        #         sum_q,
        #         self.curr_qpcr_scale))
        #     print('neg_bin: {}\n\tk: {}\n\tm: {}\n\tdispersion: {}'.format( 
        #         negbin, self.curr_reads, self.curr_read_depth * rel,
        #         self.a0/rel + self.a1))
            
        #     print('data\n\tcurr_x: {}, {}\n\tcurr_logx: {}'.format(
        #         self.curr_x[self.tidx], 
        #         np.exp(self.curr_logx[self.tidx]),
        #         self.curr_logx[self.tidx]))
        return negbin + qpcr

    def compute_dynamics(self, tidx, Axj, a1):
        '''Compute dynamics going into tidx+1

        a1 : growth rates and perturbations (if necessary)
        Axj : cluster interactions with the other abundances already multiplied
        tidx : time index

        Zero-inflation
        --------------
        When we get here, the current asv at `tidx` is not a structural zero, but 
        there might be other bugs in the system that do have a structural zero there.
        Thus we do nan adds
        '''
        logxi = self.curr_logx[tidx]
        xi = self.curr_x[tidx]

        # print('dynamics')
        # print('xi*a1', xi*a1* self.dts[tidx])
        # print('xi*xi*self.curr_self_interaction', xi*xi*self.curr_self_interaction* self.dts[tidx])
        # print('xi*Axj', xi*Axj* self.dts[tidx])

        # compute dynamics
        return logxi + (a1 - xi*self.curr_self_interaction + Axj) * self.dts[tidx]


class ZeroInflation(pl.graph.Node):
    '''This is the posterior distribution for the zero inflation model. These are used
    to learn when the model should use use the data and when it should not. We do not need
    to trace this object because we set the structural zeros to nans in the trace for 
    filtering.

    TODO: Parallel version of the class
    '''

    def __init__(self, mp, **kwargs):
        '''
        Parameters
        ----------
        mp : str
            This is the type of parallelization to use. This is not implemented yet.
        '''
        kwargs['name'] = STRNAMES.ZERO_INFLATION
        pl.graph.Node.__init__(self, **kwargs)
        self.value = []
        self.mp = mp
        self._strr = 'NA'

        for ridx in range(self.G.data.n_replicates):
            n_timepoints = self.G.data.n_timepoints_for_replicate[ridx]
            self.value.append(np.ones(shape=(len(self.G.data.asvs), n_timepoints), dtype=bool))

    def reset_value_size(self):
        '''Change the size of the trajectory when we set the intermediate timepoints
        '''
        n_asvs = self.G.data.n_asvs
        for ridx in range(len(self.value)):
            n_timepoints = self.G.data.n_timepoints_for_replicate[ridx]
            self.value[ridx] = np.ones((n_asvs, n_timepoints), dtype=bool)

    def __str__(self):
        return self._strr

    def initialize(self, value_option, delay=0):
        '''Initialize the values. Right now this is static and we are not learning this so
        do not do anything fancy

        Parameters
        ----------

        delay : None, int
            How much to delay starting the sampling
        '''
        if delay is None:
            delay = 0
        if not pl.isint(delay):
            raise TypeError('`delay` ({}) must be an int'.format(type(delay)))
        self.delay = delay

        if value_option in [None, 'auto']:
            # Set everything to on
            self.value = []
            for ridx in range(self.G.data.n_replicates):
                n_timepoints = self.G.data.n_timepoints_for_replicate[ridx]
                self.value.append(np.ones(
                    shape=(len(self.G.data.asvs), n_timepoints), dtype=bool))
            turn_on = None
            turn_off = None

        elif value_option == 'mdsine-cdiff':
            # Set everything to on except for cdiff before day 28 for every subject
            self.value = []
            for ridx in range(self.G.data.n_replicates):
                n_timepoints = self.G.data.n_timepoints_for_replicate[ridx]
                self.value.append(np.ones(
                    shape=(len(self.G.data.asvs), n_timepoints), dtype=bool))

            # Get cdiff
            cdiff_idx = self.G.data.asvs['Clostridium-difficile'].idx
            turn_off = []
            turn_on = []
            for ridx in range(self.G.data.n_replicates):
                for tidx, t in enumerate(self.G.data.times[ridx]):
                    for oidx in range(len(self.G.data.asvs)):
                        if t < 28 and oidx == cdiff_idx:
                            self.value[ridx][cdiff_idx, tidx] = False
                            turn_off.append((ridx, tidx, cdiff_idx))
                        else:
                            turn_on.append((ridx, tidx, oidx))

        else:
            raise ValueError('`value_option` ({}) not recognized'.format(value_option))

        self.G.data.set_zero_inflation(turn_on=turn_on, turn_off=turn_off)
                
