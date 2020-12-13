'''These are classes for constructing the design matrices.

Main classes
------------
`Data`
    This class is what `mdsine2.Graph.data` points to. It keeps all the design matrices for
    each class consistent as well as provides functionality and core functions that are used
    in inference. The main job of this class is to act as a pointer to the individual design
    matrices that are used in inference. You can access these classes with the pointer
    `Data.design_matrices[name_of_class]`.
`LHSVector`
    This is the Left Hand Side (LHS) of the MDSINE2 model:
        (log(x[k+1]) - log(x[k+1]))/(t[k+1] - t[k])
    where
        x is the latent abundance at point k
        t is the time at point k
`DesignMatrix` subclasses
    These build the right hand side of the MDSINE2 model and is broken up into individual
    classes:
        Growth : The RHS for the growth parameter is in `GrowthDesignMatrix`
        Self-interactions : The RHS for the self-interaction parameter is in `SelfInteractionDesignMatrix`
        Perturbations : The RHS for the perturbation parameter is in `PerturbationDesignMatrix` which is
                        composed of `PerturbationBaseDesignMatrix` and `PerturbationMixingDesignMatrix`. For
                        more details, see `PerturbationDesignMatrix`.
        Interactions : The RHS for the perturbation parameter is in `InteractionsDesignMatrix` which is
                       composed of `InteractionsBaseDesignMatrix` and `InteractionsMixingDesignMatrix`. For
                       more details, see `InteractionsDesignMatrix`.
'''

import numpy as np
import logging
import numba
import time
import itertools
import scipy.sparse
from orderedset import OrderedSet

from .pylab.graph import DataNode, Node
from .names import STRNAMES

from . import pylab as pl

class Data(DataNode):
    '''Acts as a collection for the Observation object and a collection of Covariate objects.

    Description of internal objects
    -------------------------------
    self.data : list(np.ndarray(n_taxas, n_times)) 
        These are the data matrices that are used to build the design matrices. Index
        the replicate by the index of the list. 
    
    self.dt[ridx][k] : list(np.ndarray((n_times, ))) 
        This is the change in time from time index k to time index k+1 for replicate index `ridx`.
    self.dt_vec : np.ndarray 
        These have the same values as `self.dt` excpet that the arrays are flattened so that the 
        index of dt corresponds to the row in the design matrix. IE (x[k+1] - x[k])/self.dt[k] can be thought
        of as: 
            (x[ridx][aidx, k+1] - x2[ridx][aidx, k])/(times[ridx][k+1] - times[ridx][k])
        for each replicate index `ridx` and Taxa index `aidx`
    Difference between `self.times` and `self.given_timepoints`
        `self.times` are all the timepoints that we have data for the latent trajectory whereas
        `self.given_timepoints` are all the timepoints where we actually have data for as specified in 
        `self.subjects`. These are going to be different if we have intermediate datapoints

    We assume that once this object gets initialized there is not more deletions of
    Taxas or replicates for the inference, i.e. `subjects` needs to stay fixed during inference
    for this object to stay consistent.


    Parameters
    ----------
    subjects : pl.base.Study
        These are a list of the subjects that we are going to get data from
    zero_inflation_transition_policy : str
        How we handle the transitions from a structural zero to a non-structural zero.
        If None then we do not assume any zero-inflation. Options:
        'sample'
            Intermediate timepoint is uniformly sampled between the structural zero and 
            non-strctural zero and set at that point.
        'half-way'
            Intermediate timepoint is set to half-way between the structural-zero and
            non-structural zero
        'ignore'
            We ignore the change from not being there to being there and vice versa.
    **kwargs
        - These are the extra arguments for DataNode
    '''
    def __init__(self, subjects, zero_inflation_transition_policy=None, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'data matrix'
        DataNode.__init__(self, **kwargs)
        if not pl.isstudy(subjects):
            raise ValueError('`subjects` ({}) must be a pylab Study'.format(
                type(subjects)))

        self.taxas = subjects.taxas
        self.subjects = subjects
        self.zero_inflation_transition_policy = zero_inflation_transition_policy

        self.raw_data = []
        self.rel_data = []
        self.abs_data = []
        self.qpcr = []
        self.qpcr_variances = None
        self.read_depths = []
        self.given_timepoints = []
        self._given_timepoints_set = []
        self.given_timeindices = []
        self.data_timeindex2given_timeindex = {}
        self.n_timepoints_for_replicate = []
        self.n_dts_for_replicate = []
        self.times = []
        self.data = []
        self.dt = []
        self.timepoint2index = []
        self.toupdate = OrderedSet()
        self.n_replicates = len(self.subjects)
        self.n_taxas = len(self.taxas)

        for ridx, s in enumerate(self.subjects):
            d = s.matrix()
            self.raw_data.append(d['raw'])
            self.rel_data.append(d['rel'])
            self.abs_data.append(d['abs'])
            temp_data = np.array(d['abs']) # to copy the array
            self.data.append(temp_data)
            self.qpcr.append(s.qpcr)
            self.n_timepoints_for_replicate.append(d['raw'].shape[1])
            self.n_dts_for_replicate.append(d['raw'].shape[1]-1)
            self.given_timepoints.append(np.asarray(s.times))
            self._given_timepoints_set.append(OrderedSet())
            self.given_timeindices.append(OrderedSet())
            self.times.append(np.asarray(s.times))
            for tidx,t in enumerate(self.times[ridx]):
                self._given_timepoints_set[ridx].add(t)
                self.given_timeindices[ridx].add(tidx)
                self.data_timeindex2given_timeindex[(ridx,tidx)] = tidx

            self.timepoint2index.append({})
            for tidx,t in enumerate(self.times[ridx]):
                self.timepoint2index[ridx][t] = tidx

            curr_read_depths = np.zeros(self.n_timepoints_for_replicate[ridx])
            for tidx,t in enumerate(s.times):
                curr_read_depths[tidx] = s.read_depth(t)
            self.read_depths.append(curr_read_depths)

            self.dt.append(np.zeros(self.n_timepoints_for_replicate[ridx] - 1))
            for k in range(len(self.dt[ridx])):
                self.dt[ridx][k] = self.times[ridx][k+1] - self.times[ridx][k]

        self.total_n_timepoints_per_taxa = 0
        self.total_n_dts_per_taxa = 0
        for nt in self.n_timepoints_for_replicate:
            self.total_n_timepoints_per_taxa += nt
            self.total_n_dts_per_taxa += (nt-1)

        # make dt_vec
        n = 0
        for ridx in range(self.n_replicates):
            n += len(self.dt[ridx])
        n *= self.n_taxas
        n_taxas = self.n_taxas
        self.dt_vec = np.zeros(n)
        i = 0
        for ridx in range(self.n_replicates):
            for t in self.dt[ridx]:
                self.dt_vec[i:i+n_taxas] = t
                i += n_taxas
        self.sqrt_dt_vec = np.sqrt(self.dt_vec)

        self.design_matrices = {}
        self.lhs = None

        # Make tidx arrays for perturbations if necessary
        self.tidxs_in_perturbation = None
        if self.subjects.perturbations is not None:
            self.tidxs_in_perturbation = []
            for ridx, subj in enumerate(self.subjects):
                self.tidxs_in_perturbation.append([])
                for perturbation in self.subjects.perturbations:
                    start = perturbation.starts[subj.name]
                    end = perturbation.ends[subj.name]

                    if start in self.timepoint2index[ridx]:
                        # There is a measurement at the start of the perturbation
                        start_idx = self.timepoint2index[ridx][start]
                    else:
                        # There is no measurement at the start of the perturbation.
                        # Get the next timepoint
                        start_idx = np.searchsorted(self.times[ridx], start)
                    
                    if end in self.timepoint2index[ridx]:
                        # There is a measurement at the end of the perturbation
                        end_idx = self.timepoint2index[ridx][end]
                    else:
                        # There is no measurement at the end of the perturbation
                        # Get the previous timepoint
                        end_idx = np.searchsorted(self.times[ridx], end) - 1

                    # Check if anything is weird
                    start_idx = int(start_idx)
                    end_idx = int(end_idx)
                    if start_idx > end_idx:
                        # raise ValueError('end time index ({}) of a perturbation less' \
                        #     ' than the start index ({})?'.format(end_idx, start_idx))
                        self.tidxs_in_perturbation[ridx].append((None, None))
                    if start_idx == end_idx:
                        self.tidxs_in_perturbation[ridx].append((start_idx, start_idx+1))
                    else:
                        self.tidxs_in_perturbation[ridx].append((start_idx, end_idx))
        
        # Set structural zeros - everything is set to non-structural zero initially
        if self.zero_inflation_transition_policy is not None:
            self._structural_zeros = []
            for ridx in range(self.n_replicates):
                self._structural_zeros.append(np.zeros(
                    shape=(len(self.taxas), self.n_timepoints_for_replicate[ridx]), dtype=bool))
            self._setrows_to_include_zero_inflation()

    def iter_for_building(self):
        for ridx in range(self.n_replicates):
            for tidx in range(self.n_dts_for_replicate[ridx]):
                for oidx in range(len(self.taxas)):
                    yield oidx, tidx, ridx

    def make_delta_t(self, sqrt=False):
        '''Returns the whole delta_t vector for each time point.

        Parameters
        ----------
        sqrt : bool
            If True, return the square root of the vector. Default is False
        '''
        if sqrt:
            return self.sqrt_dt_vec
        else:
            return self.dt_vec

    def is_intermediate_timepoint(self, ridx, t):
        '''Checks if the given time `t` for subject index `ridx` is
        an intermediate time point or not

        Parameters
        ----------
        ridx : int
            Replicate index
        t : numeric
            Time point

        Returns
        -------
        bool
        '''
        return t not in self._given_timepoints_set[ridx]

    def is_intermediate_timeindex(self, ridx, tidx):
        '''Checks if the given time index `tidx` for subject index `ridx` is
        an intermediate time point or not

        Parameters
        ----------
        ridx : int
            Replicate index
        tidx : int
            Time index

        Returns
        -------
        bool
        '''
        return tidx not in self.given_timeindices[ridx]

    def set_timepoints(self, times=None, timestep=None, ridx=None, eps=None,
        reset_timepoints=False):
        '''Set times if you want intermediate timepoints.
        
        These are the time points that you want to generate the latent state at.
        If there is a time in `times` that is not in the given data, then we set
        it as an intermediate time point. You can specify the times that you 
        want either as a vector with `times` or with a constant time-step
        with the interval `timestep`. Only one of them is necessary. It will crash
        if both are supplied.

        `eps` is a radius around each of the given timepoints already supplied by
        the data in `subjects` where no intermediate timepoints can be set.
        If a timepoint that is set is within `eps` days of a timepoint already there,
        then we skip adding that timepoint. If `eps` is None then we assume
        that there is no constraint. 

        If `reset_timepoints` is True, then we delete the intermediate timepoints
        that may have been added at a different time and use only these. If False
        then the set of timepoints added at this call is adde to the set of 
        timepoints that were added at a previous call. 

        Parameters
        ----------
        times : array
            These are the time points to set
            If there are any time points not included in here but are included
            with the given data then we automatically add them in. If there
            are any time points in times that are already in the given data,
            it is automatically set as a given time point
            You need to provide this argument if you are not providing `timestep`
        timestep : float, int
            This is the time steps to generate the times at until the last time
            at every replicate.
            A data point will only be added if there is not already a datapoint
            at that time. This is only necessary if you did not pass in `times`
        ridx : int, Optional
            If this is given, then we only set these times for the given
            replicate index. If nothing is specified then we set the time points
            for all of the replicates.
        eps : numeric, None
            If an intermediate timestep is within `eps` days of a given timepoint
            then we do not add it. If None then there is no restriction to how close
            the intermediate timepoint can be.
        reset_timepoints : bool
            If this is True then we delete the previous added intermediate timepoints.
            If False then we add the set of timepoints added at this call with the
            intermediate timepoints from a previous call.
        '''
        if (times is None and timestep is None) or (times is not None and timestep is not None):
            raise ValueError('Either `times` or `timestep` must be provided')
        if not pl.isbool(reset_timepoints):
            raise TypeError('`reset_timepoints` ({}) must be a bool'.format(
                type(reset_timepoints)))
        if ridx is not None:
            if not pl.isint(ridx):
                raise ValueError('`ridx` ({}) must be an int'.format(type(ridx)))
            ridxs = [ridx]
        else:
            ridxs = np.arange(self.n_replicates)
        
        if timestep is not None:
            if not pl.isnumeric(timestep):
                raise ValueError('`timestep` ({}) must be a numeric'.format(type(timestep)))
            # get the earliest start and the latest end
            start = float('inf')
            end = -1
            for ts in self.times:
                if start > ts[0]:
                    start = ts[0]
                if end < ts[-1]:
                    end = ts[-1]
            times = np.arange(start,end,timestep)
        if not pl.isarray(times):
            raise ValueError('`times` ({}) must be an array'.format(type(times)))
        
        if eps is not None:
            if not pl.isnumeric(eps):
                raise TypeError('`eps` ({}) must be a numeric'.format(type(eps)))
            if eps < 0:
                raise ValueError('`eps` ({}) must be >= 0'.format(eps))
            
        for ridx in ridxs:
            if reset_timepoints:
                new_times = np.array(self.given_timepoints[ridx])
            else:
                new_times = np.array(self.times[ridx])

            n_added = 0
            for t in times:
                if t not in new_times:
                    # Check if the datapoint is within `eps` of real data
                    # Get the surrounding timepoints
                    if eps is None:
                        n_added += 1
                        new_times = np.append(new_times, t)
                    else:
                        smallest_big = float('-inf')
                        largest_small = float('inf')
                        for tt in new_times:
                            if tt < t and tt > largest_small:
                                largest_small = tt
                            elif tt > t and tt < smallest_big:
                                smallest_big = tt
                        if np.min([t-largest_small, smallest_big-t]) < eps:
                            n_added += 1
                            new_times = np.append(new_times, t)
                        
            sorted_tidxs = np.argsort(new_times)
            self.times[ridx] = new_times[sorted_tidxs]
            if reset_timepoints:
                self.data[ridx] = np.hstack((
                    self.abs_data[ridx], 
                    np.zeros(shape=(self.n_taxas, n_added))*np.nan))
            else:
                self.data[ridx] = np.hstack((
                    self.data[ridx],
                    np.zeros(shape=(self.n_taxas, n_added)) * np.nan))
            self.data[ridx] = self.data[ridx][:,sorted_tidxs]
            self.n_timepoints_for_replicate[ridx] = len(self.times[ridx])
            self.n_dts_for_replicate[ridx] = len(self.times[ridx])-1

            # redo `given_timeindices`
            self.given_timeindices[ridx] = OrderedSet()
            for tidx in range(len(self.times[ridx])):
                if self.times[ridx][tidx] in self._given_timepoints_set[ridx]:
                    self.given_timeindices[ridx].add(tidx)

            # redo `data_timeindex2given_timeindex` - first delete all ones that
            # have that as the replicate and then add the new oens
            new_d = {}
            for aaa,bbb in self.data_timeindex2given_timeindex:
                if aaa == ridx:
                    continue
                new_d[(aaa,bbb)] = self.data_timeindex2given_timeindex[(aaa,bbb)]
            self.data_timeindex2given_timeindex = new_d

            given_times = self.given_timepoints[ridx]
            for tidx in range(self.n_timepoints_for_replicate[ridx]):
                if tidx in self.given_timeindices[ridx]:
                    # Get index where the time occurs in the given times
                    t = self.times[ridx][tidx]
                    found = False
                    for i in range(len(given_times)):
                        if t == given_times[i]:
                            found = True
                            self.data_timeindex2given_timeindex[(ridx,tidx)] = i
                            break
                    if not found:
                        raise ValueError('Not found - something is wrong')

                else:
                    self.data_timeindex2given_timeindex[(ridx,tidx)] = np.nan

            # Redo the reverse indexing
            self.timepoint2index[ridx] = {}
            for tidx,t in enumerate(self.times[ridx]):
                self.timepoint2index[ridx][t] = tidx

            # redo `dt`
            self.dt[ridx] = np.zeros(self.n_timepoints_for_replicate[ridx]-1)
            for k in range(len(self.dt[ridx])):
                self.dt[ridx][k] = self.times[ridx][k+1] - self.times[ridx][k]

        # redo dt_vec
        n = 0
        for ridx in range(self.n_replicates):
            n += len(self.dt[ridx])
        n *= self.n_taxas
        n_taxas = self.n_taxas
        self.dt_vec = np.zeros(n)
        i = 0
        for ridx in range(self.n_replicates):
            for t in self.dt[ridx]:
                self.dt_vec[i:i+n_taxas] = t
                i += n_taxas
        self.sqrt_dt_vec = np.sqrt(self.dt_vec)

        self.total_n_timepoints_per_taxa = 0
        self.total_n_dts_per_taxa = 0
        for nt in self.n_timepoints_for_replicate:
            self.total_n_timepoints_per_taxa += nt
            self.total_n_dts_per_taxa += (nt - 1)


        # redo tidx arrays for perturbations if necessary
        if self.tidxs_in_perturbation is not None:
            self.tidxs_in_perturbation = []
            for ridx, subj in enumerate(self.subjects):
                self.tidxs_in_perturbation.append([])
                for perturbation in self.subjects.perturbations:
                    start = perturbation.starts[subj.name]
                    end = perturbation.ends[subj.name]

                    if start in self.timepoint2index[ridx]:
                        # There is a measurement at the start of the perturbation
                        start_idx = self.timepoint2index[ridx][start]
                    else:
                        # There is no measurement at the start of the perturbation.
                        # Get the next timepoint
                        start_idx = np.searchsorted(self.times[ridx], start)
                    
                    if end in self.timepoint2index[ridx]:
                        # There is a measurement at the end of the perturbation
                        end_idx = self.timepoint2index[ridx][end]
                    else:
                        # There is no measurement at the end of the perturbation
                        # Get the previous timepoint
                        end_idx = np.searchsorted(self.times[ridx], end) - 1

                    # Check if anything is weird
                    start_idx = int(start_idx)
                    end_idx = int(end_idx)
                    if start_idx > end_idx:
                        # raise ValueError('end time index ({}) of a perturbation less' \
                        #     ' than the start index ({})?'.format(end_idx, start_idx))
                        self.tidxs_in_perturbation[ridx].append((None, None))
                    if start_idx == end_idx:
                        self.tidxs_in_perturbation[ridx].append((start_idx, start_idx + 1))
                    else:
                        self.tidxs_in_perturbation[ridx].append((start_idx, end_idx))

    def set_zero_inflation(self, turn_on=None, turn_off=None):
        '''Set which timepoints taxas are set to be turned off. Any taxa, timepoints tuple
        not in `d` is assumed to be "present" (a nonn-structural zero). `d` is an array
        of 3-tuples, where:
            (ridx, tidx, aidx)
                ridx: replicate index (not the same as replicate name)
                tidx: timepoint index (not the same as timepoint)
                aidx: Taxa index (not the same as Taxa name)
        
        Parameters
        ----------
        turn_on : list(3-tuple)
            A list of ridx, tidx, aidx to set to being present
        turn_on : list(3-tuple)
            A list of ridx, tidx, aidx to set to a structural zero
        '''
        if self.zero_inflation_transition_policy is None:
            # raise ValueError('Cannot set set the zero infation if `zero_inflation_transition_policy` ' \
            #     'is not set during initialization')
            logging.warning('`zero_inflation_transition_policy` is None so we are not doing anything')
            return
        if turn_on is not None:
            for i, (ridx,tidx,aidx) in enumerate(turn_on):
                if ridx > self.n_replicates or ridx < 0:
                    raise ValueError('ridx ({}) in index `{}` ({}) is out of range. Only {} replicates'.format(
                        ridx, i, (ridx,tidx,aidx), self.n_replicates))
                if tidx > self.n_timepoints_for_replicate[ridx] or tidx < 0:
                    raise ValueError('tidx ({}) in index `{}` ({}) is out of range. Only {} timepoints in replicate {}'.format(
                        tidx, i, (ridx,tidx,aidx), self.n_timepoints_for_replicate[ridx], ridx))
                if aidx > len(self.taxas) or aidx < 0:
                    raise ValueError('aidx ({}) in index `{}` ({}) is out of range. Only {} s'.format(
                        aidx, i, (ridx,tidx,aidx), len(self.taxas)))

                self._structural_zeros[ridx][aidx,tidx] = False

        if turn_off is not None:
            for i, (ridx,tidx,aidx) in enumerate(turn_off):
                if ridx > self.n_replicates or ridx < 0:
                    raise ValueError('ridx ({}) in index `{}` ({}) is out of range. Only {} replicates'.format(
                        ridx, i, (ridx,tidx,aidx), self.n_replicates))
                if tidx > self.n_timepoints_for_replicate[ridx] or tidx < 0:
                    raise ValueError('tidx ({}) in index `{}` ({}) is out of range. Only {} timepoints in replicate {}'.format(
                        tidx, i, (ridx,tidx,aidx), self.n_timepoints_for_replicate[ridx], ridx))
                if aidx > len(self.taxas) or aidx < 0:
                    raise ValueError('aidx ({}) in index `{}` ({}) is out of range. Only {} s'.format(
                        aidx, i, (ridx,tidx,aidx), len(self.taxas)))

                self._structural_zeros[ridx][aidx,tidx] = True
        self._setrows_to_include_zero_inflation()

    def is_timepoint_structural_zero(self, ridx, tidx, aidx):
        '''Returns True if the replicate index `ridx`, timepoint index `tidx`, and
        Taxa index `aidx` is a structural zero or not
        '''
        if self.zero_inflation_transition_policy is None:
            raise ValueError('Cannot set set the zero infation if `zero_inflation_transition_policy` ' \
                'is not set during initialization')
        return self._structural_zeros[ridx][aidx, tidx]

    def _setrows_to_include_zero_inflation(self):
        '''Make a rows matrix for what to include based on `self._structural_zeros`
        '''
        if self.zero_inflation_transition_policy is None:
            raise ValueError('Cannot set set the zero infation if `zero_inflation_transition_policy` ' \
                'is not set during initialization')

        iii = 0

        l = len(self.taxas) * self.total_n_dts_per_taxa
        self.rows_to_include_zero_inflation = np.zeros(l, dtype=bool)
        iii = 0
        for ridx in range(self.n_replicates):
            curr_structural_zero = self._structural_zeros[ridx]
            for dtidx in range(self.n_dts_for_replicate[ridx]):
                # we look at timepoint indices `dtidx` and `dtidx+1`

                tidxstart = dtidx
                tidxend = dtidx

                for aidx in range(len(self.taxas)):

                    structzero_start = curr_structural_zero[aidx, tidxstart]
                    structzero_end = curr_structural_zero[aidx, tidxend]
                    
                    if structzero_start and structzero_end:
                        # If both are not there, exclude
                        self.rows_to_include_zero_inflation[iii] = False
                    
                    elif (not structzero_end) and (not structzero_start):
                        # If both are there, include
                        self.rows_to_include_zero_inflation[iii] = True

                    else:
                        # Else we are in a transition and we must use a policy
                        if self.zero_inflation_transition_policy == 'ignore':
                            # don't include it
                            self.rows_to_include_zero_inflation[iii] = False
                        elif self.zero_inflation_transition_policy == 'half-way':
                            raise NotImplementedError('Not implemented')
                        elif self.zero_inflation_transition_policy == 'sample':
                            raise NotImplementedError('Not implemented')
                        else:
                            raise ValueError('`zero_inflation_transition_policy` ({}) not recognized'.format(
                                self.zero_inflation_transition_policy))

                    iii += 1

        self.off_previously_arr_zero_inflation = np.zeros(l, dtype=int)
        n_off_prev = 0
        for i in range(1, l):
            if not self.rows_to_include_zero_inflation[i-1]:
                n_off_prev += 1
            self.off_previously_arr_zero_inflation[i] = n_off_prev

    def _get_non_pert_rows_of_regress_matrices(self):
        '''This will get the rows where there are no perturbations in the
        regressor matrices
        '''
        if self.G.perturbations is None:
            return None

        replicate_offset = 0
        ridxs = np.array([], dtype=int)
        n_taxas = self.n_taxas
        for ridx in range(self.n_replicates):
            # For each replicate, get the time indices where ther are 
            # perturbations and then index them out
            for pidx in range(len(self.G.perturbations)):
                start_tidx, end_tidx = self.G.data.tidxs_in_perturbation[ridx][pidx]
                if start_tidx is None:
                    continue

                ridxs = np.append(ridxs, replicate_offset + np.arange(
                    start_tidx*n_taxas, (end_tidx-1)*n_taxas, dtype=int))
            replicate_offset += n_taxas * self.n_dts_for_replicate[ridx]

        ret = np.ones(len(self.lhs), dtype=bool)
        ret[ridxs] = False

        if self.zero_inflation_transition_policy is not None:
            ret = ret[self.rows_to_include_zero_inflation]

        return ret

    def construct_lhs(self, keys=[], kwargs_dict={}, index_out_perturbations=False):
        '''Does the stacking and subtracting necessary to make the observation vector

        Parameters
        ----------
        keys(list(keys))
            These are the keys of the matrices to go on the left-hand-side (lhs)
        kwargs_dict (dict(dict))
            This is a dict of dicts:
                str -> (str -> val)
            The first level dictionary is one of the names in the keys
            The second level dictionary are the additional arguements that are
            send to that keys `construct_lhs` function.
        index_out_perturbations (bool, Optional)
            If this is True, it will index out the rows that are in the perturbation.
            This would be used for times that you want to initialize the data not
            with any perturbation periods
        Returns
        -------
        np.ndarray
        '''
        y = self.lhs.vector
        valid_indices = None
        if index_out_perturbations and self.G.perturbations is not None:
            valid_indices = self._get_non_pert_rows_of_regress_matrices()
            y = y[valid_indices]
        for x in keys:
            if x in kwargs_dict:
                kwargs = kwargs_dict[x]
            else:
                kwargs = {}
            try:
                b = self.design_matrices[x].set_to_lhs(**kwargs)
            except:
                logging.critical('Crash in `construct_lhs` making the matrix. Key: {}. {}'.format(x,
                    keys))
                raise
            if valid_indices is not None:
                b = b[valid_indices]
            try:
                y = y - b
            except:
                logging.critical('Crash in `construct_lhs` subtracting the matrix.' \
                    ' Key: {}, y.shape: {}, b.shape: {}'.format(x, y.shape, b.shape))
                raise
        return y.reshape(-1,1)

    # @profile
    def construct_rhs(self, keys, kwargs_dict={}, index_out_perturbations=False, 
        toarray=False):
        '''Does the stacking and subtracting necessary to make the covariate matrix.
        Default setting for this matrix is a `scipy.sparse` matrix unless you
        explicitly convert it with `toarray`.

        Parameters
        ----------
        keys : list(keys)
            - These are the keys of the matrices to go on the right-hand-side (rhs)
        kwargs_dict : dict(dict)
            - This is a dict of dicts:
                str -> (str -> val)
            - The first level dictionary is one of the names in the keys
            - The second level dictionary are the additional arguements that are
              send to that keys `construct_lhs` function.
        index_out_perturbations : bool, Optional
            - If this is True, it will index out the rows that are in the perturbation.
              This would be used for times that you want to initialize the data not
              with any perturbation periods
        toarray : bool
            If True, converts the input into a numpy C_CONTIGUOUS array

        Returns
        -------
        scipy.sparse.csc_matrix or np.ndarray
        '''
        v = []
        valid_indices = None
        if index_out_perturbations and self.G.perturbations is not None:
            valid_indices = self._get_non_pert_rows_of_regress_matrices() 
        for x in keys:
            if x in kwargs_dict:
                kwargs = kwargs_dict[x]
            else:
                kwargs = {}
            if x not in self.design_matrices:
                raise KeyError('Key `{}` not found. Valid keys: {}'.format(
                    x, list(self.design_matrices.keys())))
            X = self.design_matrices[x].set_to_rhs(**kwargs)
            if valid_indices is not None:
                X = X[valid_indices, :]
            v.append(X)
        if len(keys) == 0:
            X =  v[0]
        else:
            try:
                X = scipy.sparse.hstack(v)
            except:
                # try:
                #     X = torch.cat(v, 1)
                #     return X
                # except:
                t = [type(a) for a in v]
                s = [a.shape for a in v]
                logging.critical('shapes: {}, types: {}'.format(s,t))
                logging.critical('keys: {}'.format([self.G[a].name for a in keys]))
                raise
        if toarray:
            X_ = np.zeros(shape=X.shape)
            X.toarray(out=X_)
            return X_
        return X

    def update_values(self):
        '''Update the values of the data (because the latent state changed)
        '''
        self.lhs.update_value()
        for key in self.toupdate:
            self.design_matrices[key].update_value()


class ObservationVector(Node):
    '''This is the left hand side (lhs) vector
    '''
    def __init__(self, name, G):
        self.G = G
        self.name = name
        self.G.data.lhs = self
        self.vector = None

    def build(self):
        raise NotImplementedError('You must implement this function')

    def update_value(self):
        '''This updates the data for each design matrix
        '''
        raise NotImplementedError('You must implement this function')


class DesignMatrix:
    '''This is a covariate class
    '''
    def __init__(self, varname, G, update=False, add_to_dict=True):
        '''Parameters

        varname (str)
            - This is the name of the variable we are building it for
        update (bool)
            - If True, this matrix gets updated when Data.update_values() is
              called.
        add_to_dict (bool)
            - If True, it adds to the design_matrices dictionary.
            - Else it does not
        '''

        name = varname + '_design_matrix'
        self.name = name
        self.G = G
        self.varname = varname
        if add_to_dict:
            self.G.data.design_matrices[varname] = self
        self.matrix = None
        if update:
            self.G.data.toupdate.add(varname)

    def build(self):
        raise NotImplementedError('You must implement this function')

    def set_to_lhs(self):
        '''Multiply the current value of the var with the current matrix
        '''
        raise NotImplementedError('You must implement this function')

    def update_value(self):
        '''This updates the data for each design matrix
        '''
        raise NotImplementedError('You must implement this function')

    def toarray(self, dest=None, T=False):
        '''Converts `self.matrix` into a C_CONTIGUOUS numpy matrix if 
        the matrix is sparse. If it is not sparse then it just returns
        the matrix.

        Parameters
        ----------
        dest : np.ndarray
            If this is specified, send the array into this array. Assumes
            the shapes are compatible. Else create a new array
        T : bool
            If True, set the transpose

        Returns
        -------
        np.ndarray
        '''
        return pl.toarray(self.matrix, dest=dest, T=T)


################################################################################
################################################################################
# Design matrices
################################################################################
################################################################################
class LHSVector(ObservationVector):
    '''This builds the Left-Hand-Side (LHS) vector
    '''
    def __init__(self, **kwargs):
        ObservationVector.__init__(self, **kwargs)
        logging.info('Initializing LHS vector')

    def build(self, subjects='all'):
        '''Build the observation vector

        (log(x_{k+1}) - log(x_{k}))/dt

        Parameters
        ----------
        subjects : str, array(int), int
            These are the subjects to build the vector for. If 'all', we build it
            for all the subjects. If you want to pass in an array, we do it with 
            the subject index
        '''
        if pl.isint(subjects):
            subjects = [subjects]
        if subjects == 'all':
            subjects = np.arange(self.G.data.n_replicates)
            n_dts = self.G.data.total_n_dts_per_taxa
        else:
            n_dts = 0
            for sidx in subjects:
                n_dts += self.G.dta.n_dts_for_replicate[sidx]
        self.vector = np.zeros(self.G.data.n_taxas * n_dts, dtype=float)
        i = 0
        for ridx in range(self.G.data.n_replicates):
            if ridx not in subjects:
                # skip subject
                continue
            l = self.G.data.n_dts_for_replicate[ridx] * self.G.data.n_taxas
            LHSVector._fast_build_log(
                ret=self.vector[i:i+l],
                data=self.G.data.data[ridx],
                dt=self.G.data.dt[ridx],
                n_ts=self.G.data.n_dts_for_replicate[ridx],
                n_taxas=self.G.data.n_taxas)
            i += l

        if self.G.data.zero_inflation_transition_policy is not None:
            self.vector = self.vector[self.G.data.rows_to_include_zero_inflation]
        self.vector = self.vector.reshape(-1,1)

    def __len__(self):
        return len(self.vector)

    @staticmethod
    @numba.jit(nopython=True, cache=True, fastmath=True)
    def _fast_build_log(ret, data, dt, n_ts, n_taxas):
        '''About 99.4% faster than regular python looping
        '''
        i = 0
        for tidx in range(n_ts):
            for oidx in range(n_taxas):
                ret[i] = (np.log(data[oidx, tidx+1]) - np.log(data[oidx,tidx]))/dt[tidx]
                i += 1

    def update_value(self):
        self.build()

    def print_mer(self):
        i = 0
        for ridx in range(self.G.data.n_replicates):
            l = self.G.data.n_dts_for_replicate[ridx] * self.G.data.n_taxas
            data = self.G.data.data[ridx]
            dt = self.G.data.dt[ridx]
            print('\n\n\n\n\n\n ridx', ridx)
            for tidx in range(self.G.data.n_dts_for_replicate[ridx]):
                print('-------\ntidx',tidx)
                for oidx in range(self.G.data.n_taxas):
                    print('oidx', oidx)
                    print('\tnext',np.log(data[oidx, tidx+1]), data[oidx, tidx+1])
                    print('\tcurr', np.log(data[oidx,tidx]), data[oidx, tidx])
                    print('\tdt', dt[tidx])
                    print('\tvector', self.vector[i])
                    i += 1
            

class SelfInteractionDesignMatrix(DesignMatrix):
    '''Base matrix class for growth and self interactions
    Since the dynamics subtract the self-interaction parameter, we set the
    parameter to positive, which means our data is negative.
    '''
    def __init__(self, **kwargs):
        DesignMatrix.__init__(self,
            varname=STRNAMES.SELF_INTERACTION_VALUE,
            update=True, **kwargs)
        self.n_cols_master = self.G.data.n_taxas
        total_n_dts = self.G.data.total_n_dts_per_taxa
        self.n_rows_master = self.n_cols_master * total_n_dts
        self.master_rows = np.arange(self.n_rows_master, dtype=int)
        self.master_cols = np.kron(
                np.ones(total_n_dts, dtype=int),
                np.arange(self.G.data.n_taxas,dtype=int))
        logging.info('Initializing self-interactions design matrix')

    def build(self):
        '''Builds the matrix. Flatten Fortran style
        '''
        self.rows = self.master_rows
        self.cols = self.master_cols

        self.data = np.zeros(self.n_rows_master, dtype=float)
        data = self.G.data.data
        i = 0
        for ridx in range(self.G.data.n_replicates):
            l = (self.G.data.n_dts_for_replicate[ridx]) * self.G.data.n_taxas
            self.data[i:i+l] = -data[ridx][:,:-1].ravel('F')
            i += l
            
        shape = (self.n_rows_master, self.n_cols_master)
        self.matrix = scipy.sparse.coo_matrix(
            (self.data,(self.rows,self.cols)), shape=shape).tocsc()
        if self.G.data.zero_inflation_transition_policy is not None:
            self.matrix = self.matrix[self.G.data.rows_to_include_zero_inflation, :]


    def set_to_lhs(self):
        '''Multiply self.matrix by the current value of
        growth/self interaction
        '''
        b = self.G[self.varname].value.reshape(-1,1)
        return self.matrix.dot(b)

    def set_to_rhs(self):
        '''Add in perturbations
        '''
        return self.matrix

    def _fast_build(self, ret, data):
        ret = np.square(data.ravel('F'))

    def update_value(self):
        self.build()


class GrowthDesignMatrix(DesignMatrix):
    '''Builds the design matrix for the growth

    We need two different matrices for growth, depending on if we are conditioning on the
    perturbations or not. If we are setting the growth to the RHS, that means
    that we are trying to learn the growth values and we have to keep the
    perturbation parameters fixed, so:
        a_1 * (1 + \\gamma)
    we need to put the perturbation parameters on the rhs matrix factored into the data matrix

    If we are putting the growth on the LHS, that means that we are either
    trying to learn the perturbations or we are marginalizing over the parameters
    dependent on the cluster assignments:
        a_1 + \\gamma * a_1
        ---
    The underlined part goes to the LHS.

    If there are no perturbations, the rhs and the lhs are equal and are set to
    the parameterization of the LHS
    '''
    def __init__(self, **kwargs):
        DesignMatrix.__init__(self,
            varname=STRNAMES.GROWTH_VALUE, update=True, **kwargs)
        self.n_cols_master = self.G.data.n_taxas
        total_n_dts = self.G.data.total_n_dts_per_taxa
        self.n_rows_master = self.n_cols_master * total_n_dts
        self.master_rows = np.arange(self.n_rows_master, dtype=int)
        self.master_cols = np.kron(
                np.ones(total_n_dts, dtype=int),
                np.arange(self.G.data.n_taxas,dtype=int))
        logging.info('Initializing growth design matrix')

    def build(self):
        '''Build RHS matrices with perturbations multiplied in and not multiplied in.
        '''
        self.build_without_perturbations()
        self.build_with_perturbations()

    def build_without_perturbations(self):
        '''Builds the matrix without perturbations factored in.
        '''
        self.cols = self.master_cols
        self.rows = self.master_rows
        self.data = np.ones(self.n_rows_master, dtype=float)

        shape = (self.n_rows_master, self.n_cols_master)

        self.matrix_without_perturbations = scipy.sparse.coo_matrix(
            (self.data,(self.rows,self.cols)), shape=shape).tocsc()
        
        if self.G.data.zero_inflation_transition_policy is not None:
            self.matrix_without_perturbations = self.matrix_without_perturbations[self.G.data.rows_to_include_zero_inflation, :]

    def build_with_perturbations(self):
        '''Incorporate perturbation factors while building the data structure.

        a_1 * (1 + \\gamma) * x_k

        How perturbations are switched on/off:
        ------------------------------------------------------------------------
        'The time ahead prediction must be included in the perturbation' - Travis

        Example: Pertubtion period (2,5) - this is **3** doses

                           |-->|-->|-->
        perturbation on    #############
        Days           1   2   3   4   5   6

        `d1` indicates the perturbation parameter that gets added for the day that it
        should be included in.

        x2 = x1 + ...
        x3 = x2 + ... + d1
        x4 = x3 + ... + d1
        x5 = x4 + ... + d1
        x6 = x5 + ...

        The perturbation periods that are given are in the format (start, end).
        For the above example our perturbation period would be (2, 5). Thus, we should do
        inclusion/exclusion brackets such that:

        (start, end]
            - The first day is inclusive
            - Last day is exclusive
        '''
        self.cols = self.master_cols
        self.rows = self.master_rows

        if self.G.perturbations is None:
            self.matrix_with_perturbations = None
            return

        self.data_w_perts = np.zeros(self.n_rows_master, dtype=float)
        d = []
        for ridx in range(self.G.data.n_replicates):
            d.append(np.ones(shape=self.G.data.data[ridx].shape))

        for pidx, pert in enumerate(self.G.perturbations):
            val = (pert.item_array(only_pos_ind=True) + 1).reshape(-1,1)
            oidxs = pert.indicator.item_arg_array()

            for ridx in range(self.G.data.n_replicates):
                start,end = self.G.data.tidxs_in_perturbation[ridx][pidx]
                if len(oidxs) > 0:
                    d[ridx][oidxs, start:end] *= val
        i = 0
        for ridx in range(self.G.data.n_replicates):
            l = (self.G.data.n_dts_for_replicate[ridx]) * self.G.data.n_taxas
            self.data_w_perts[i:i+l] = d[ridx][:,:-1].ravel('F')
            i += l

        shape = (self.n_rows_master, self.n_cols_master)
        self.matrix_with_perturbations = scipy.sparse.coo_matrix(
            (self.data_w_perts,(self.rows,self.cols)), shape=shape).tocsc()
        if self.G.data.zero_inflation_transition_policy is not None:
            self.matrix_with_perturbations = \
                self.matrix_with_perturbations[self.G.data.rows_to_include_zero_inflation, :]
            self.matrix_with_perturbations = self.matrix_with_perturbations.tocsc()

    def set_to_lhs(self, with_perturbations):
        '''Multiply the design matrix by the current value of
        growth
        '''
        if not pl.isbool(with_perturbations):
            raise ValueError('`with_perturbations` ({}) must be a bool'.format(
                type(with_perturbations)))
        if with_perturbations:
            matrix = self.matrix_with_perturbations
        else:
            matrix = self.matrix_without_perturbations
        b = self.G[self.varname].value.reshape(-1,1)
        return matrix.dot(b)

    def set_to_rhs(self, with_perturbations=None):
        '''If `with_perturbations` is True, we return the matrix with the
        perturbation factors incorporated. If False we return the matrix
        without perturbations
        '''
        if not pl.isbool(with_perturbations):
            raise ValueError('`with_perturbations` ({}) must be a bool'.format(
                type(with_perturbations)))
        if with_perturbations:
            matrix = self.matrix_with_perturbations
        else:
            matrix = self.matrix_without_perturbations
        return matrix

    def update_value(self):
        self.build()


class PerturbationBaseDesignMatrix(DesignMatrix):
    '''This is the base data for the perturbations.

    This creates the baseline perturbation effects for each Taxa, for every indicator.
    This class used in conjungtion with `PerturbationMixingDesignMatrix` and should
    be accessed through `PerturbationDesignMatrix`.

    Parameterization
    ----------------
    We parameterize the MDSINE2 model with multiplicative perturbations
    .. math::
        \frac {log(x_{i,k+1}) - log(x_{i,k})} {t_{k+1} - t_{k}} =
            a_{1,i} (1 + \sum_{p=1}^P u_p(k) \gamma_{i,p} ) + \sum_{j} b_{ij} x{j,k} 
    where:
        :math:`x_{i,k}` : abundance for Taxa :math:`i` at time :math:`t_k`
        :math:`t_k` : time at k
        :math:`a_{1,i}` : growth for Taxa :math:`i`
        :math:`b_{ij}` : interactions from :math:`i` to :math:`j`
        :math:`b_{ii}` : self interactions for :math:`i`
        :math:`u_p(k)` : step function for perturbation :math:`p` at time :math:`t_k`
        :math:`\gamma_{i,p}` : perturbation value for perturbation :math:`p` for taxa :math:`i`
        Here the perturbation has an "effect" on the growth rates

    '''
    def __init__(self, **kwargs):
        name = STRNAMES.PERT_VALUE+'_base_data'
        DesignMatrix.__init__(self, varname=name, **kwargs)
        if self.G.data.zero_inflation_transition_policy is not None:
            raise NotImplementedError('Not Implemented')

        self.perturbations = self.G.perturbations
        self.n_perturbations = len(self.perturbations)
        self.n_replicates = self.G.data.n_replicates
        self.n_taxas = len(self.G.data.taxas)
        self.growths = self.G[STRNAMES.GROWTH_VALUE]

        self.starts = []
        self.ends = []
        self.tidxs_in_pert_per_replicate = []

        self.tidxs_in_perturbation = np.zeros(shape=(self.n_replicates,
            self.n_perturbations, 2), dtype=int) - 1

        # Get the total number of timepoints in perturbations
        total_tidxs = 0
        for ridx, subj in enumerate(self.G.data.subjects):
            self.starts.append([])
            self.ends.append([])
            i = 0
            for pidx in range(self.n_perturbations):
                start, end = self.G.data.tidxs_in_perturbation[ridx][pidx]
                if start is None:
                    continue
                self.tidxs_in_perturbation[ridx,pidx,0] = start
                self.tidxs_in_perturbation[ridx,pidx,1] = end
                self.starts[-1].append(start)
                self.ends[-1].append(end)
                i += end-start
                total_tidxs += i

            self.tidxs_in_pert_per_replicate.append(i)
            self.starts[-1] = np.asarray(self.starts[-1], dtype=int)
            self.ends[-1] = np.asarray(self.ends[-1], dtype=int)

        # Set rows and cols
        self.total_len = int(total_tidxs * self.n_taxas)
        self.tidxs_in_pert_per_replicate = np.asarray(self.tidxs_in_pert_per_replicate, dtype=int)
        self.rows = np.zeros(self.total_len, dtype=int)
        self.cols = np.zeros(self.total_len, dtype=int)
        self.data = np.zeros(self.total_len)

        PerturbationBaseDesignMatrix.init(
            rows=self.rows, 
            cols=self.cols, 
            n_perturbations=self.n_perturbations, 
            n_taxas=self.n_taxas, 
            n_replicates=self.n_replicates, 
            tidxs_in_perturbation=self.tidxs_in_perturbation, 
            n_dts_for_replicate=self.G.data.n_dts_for_replicate)

        # Initialize the rows and cols for the data
        self.n_rows = self.G.data.total_n_dts_per_taxa * self.G.data.n_taxas
        self.n_cols = self.n_taxas * self.n_perturbations
        self.shape = (self.n_rows, self.n_cols)

    @staticmethod
    @numba.jit(nopython=True, cache=True, fastmath=True)
    def init(rows, cols, n_perturbations, n_taxas, n_replicates, tidxs_in_perturbation, 
        n_dts_for_replicate):

        i = 0
        base_row_idx = 0
        for ridx in range(n_replicates):
            for pidx in range(n_perturbations):
                start = tidxs_in_perturbation[ridx,pidx,0]
                end = tidxs_in_perturbation[ridx,pidx,1]
                if start == -1:
                    continue
                base_col_idx = pidx * n_taxas
                for oidx in range(n_taxas):
                    col = oidx + base_col_idx
                    for tidx in range(start, end):
                        rows[i] = oidx + tidx * n_taxas + base_row_idx
                        cols[i] = col
                        i += 1

            base_row_idx = base_row_idx + n_taxas * n_dts_for_replicate[ridx]

    @staticmethod
    @numba.jit(nopython=True, cache=True, fastmath=True)
    def fast_build(ret, n_perturbations, n_taxas, n_replicates, tidxs_in_perturbation, 
        growths, data):

        i = 0
        for pidx in range(n_perturbations):
            start, end = tidxs_in_perturbation[pidx]
            if start == -1:
                continue
            for oidx in range(n_taxas):
                growth = growths[oidx]
                ret[i:(i+end-start)] = growth
                i += end-start

    def build(self):
        growths = self.growths.value
        i = 0
        for ridx in range(self.n_replicates):
            l = self.tidxs_in_pert_per_replicate[ridx] * self.n_taxas
            PerturbationBaseDesignMatrix.fast_build(
                ret=self.data[i:i+l], 
                n_perturbations=self.n_perturbations, 
                n_taxas=self.n_taxas, 
                n_replicates=self.n_replicates, 
                tidxs_in_perturbation=self.tidxs_in_perturbation[ridx], 
                growths=growths, 
                data=self.G.data.data[ridx])
            i += l

        self.matrix = scipy.sparse.coo_matrix(
            (self.data,(self.rows,self.cols)),shape=self.shape).tocsc()

    def update_value(self):
        self.build()

    def set_to_rhs(self):
        return self.matrix


class PerturbationMixingDesignMatrix(DesignMatrix):
    '''This class creates the permutation matrix required for mixing the base matrix into
    cluster effects
    '''
    def __init__(self, parent, **kwargs):
        DesignMatrix.__init__(self,
            varname=STRNAMES.PERT_VALUE+'mixing_matrix',
            **kwargs)

        self.parent = parent
        self.perturbations = self.G.perturbations
        self.n_perturbations = len(self.perturbations)
        self.n_taxas = len(self.G.data.taxas)
        self.n_rows = self.n_perturbations * self.n_taxas

        # Maps an  and a perturbation to the column it corresponds to in `base`
        self.keypair2col = np.zeros(shape=(self.n_taxas, self.n_perturbations), dtype=int)
        i = 0
        for pidx in range(self.n_perturbations):
            for oidx in range(self.n_taxas):
                self.keypair2col[oidx, pidx] = i
                i += 1
        self.build(build=False)

    # @profile
    def build(self, build=True, build_for_neg_ind=False, only_cids=None):
        '''Build the matrix

        Parameters
        ----------
        build : bool
            If True, the parent will re-build as well
        build_for_neg_ind : bool
            If True, it will build for negative indicators as well
        only_cids : list, None
            If specified, it will only build for the cids specified.
        '''
        if only_cids is not None:
            oc = OrderedSet(list(only_cids))
        else:
            oc = None
        keypair2col = self.keypair2col
        rows = []
        cols = []
        col = 0
        for pidx, perturbation in enumerate(self.G.perturbations):
            ind = perturbation.indicator.value
            order = perturbation.clustering.order
            for cid in order:
                if oc is not None:
                    if cid not in oc:
                        continue
                if ind[cid] or build_for_neg_ind:
                    for oidx in perturbation.clustering[cid].members:
                        rows.append(keypair2col[oidx, pidx])
                        cols.append(col)
                    col += 1
        self._make_matrix(rows=rows, cols=cols, n_cols=col, build=build)

    # @profile
    def _make_matrix(self, rows, cols, n_cols, build):
        '''Builds the mixing matrix from the specified rows and columns
        (data is always going to be 1 because it is a mixing matrix)

        Rebuild after we have changed the mixing matrix if `build` is True
        '''
        data = np.ones(len(rows), dtype=np.float64)
        self.matrix = scipy.sparse.coo_matrix((data,(rows,cols)),
            shape=(self.n_rows, n_cols)).tocsc()
        self.shape = self.matrix.shape
        self.n_rows = self.shape[0]
        self.n_cols = self.shape[1]
        if build:
            self.parent.build()
  

class PerturbationDesignMatrix(DesignMatrix):
    '''Builds the design matrix for the perturbations.

    This matrix is composed of two, individual design matrices, `Base` and `M`.
    To make the matrix that we use during inference, we matrix multiply `Base`@`M`,
    which is what this class is for. It wraps these two base classes so that it
    is more streamlined in the inference code.

    `Base` : mdsine2.design_matrices.PerturbationBaseDesignMatrix
        This is an object that builds the perturbation matrix as if there was no
        clustering or indicators. It builds the data for all the Taxas and
        as if every perturbation indicator was on. This is actually faster than
        just building it for individual indicators for a few different reasons:
            1) We only need to update `Base` when we do filtering or update the 
               values of the growth matrix because these are the only two things
               that `Base` is dependent on.
            2) Because we don't have to check indicators or have different shapes
               when building the matrix, it is much easier to build this matrix
               with Numba, which is nearly as fast as C.
    `Mixing` : mdsine2.design_matrices.PerturbationMixingDesignMatrix
        This is the object that selects for indicators and groups taxas together
        into clusters. When we change the indicators of the perturbations or 
        the cluster assignments of the Taxas, we only need to change this matrix,
        which is a lot faster than changing everything.
    '''
    def __init__(self, **kwargs):
        DesignMatrix.__init__(self, varname=STRNAMES.PERT_VALUE, **kwargs)

        self.n_rows = self.G.data.total_n_dts_per_taxa * self.G.data.n_taxas
        self.n_cols = None

        self.base = PerturbationBaseDesignMatrix(add_to_dict=False, **kwargs)
        self.M = PerturbationMixingDesignMatrix(add_to_dict=False, parent=self, **kwargs)

    def set_to_lhs(self):
        # Make the perturbation vector
        b = self.G[STRNAMES.PERT_VALUE].toarray().reshape(-1,1)

        return self.matrix.dot(b)

    def set_to_rhs(self):
        return self.matrix

    def update_values(self):
        # self.build()
        self.base.update_value()
        self.build()

    def build(self):
        self.matrix = self.base.matrix @ self.M.matrix
        self.n_cols = self.matrix.shape[1]
        self.shape = self.matrix.shape


class InteractionsBaseDesignMatrix(DesignMatrix):
    '''This is the base data for the design matrix of the interactions.

    This builds the interaction matrix for each Taxa-Taxa interaction as if there
    were no indicators.
    '''
    def __init__(self, **kwargs):
        name = STRNAMES.CLUSTER_INTERACTION_VALUE+'_base_data'
        DesignMatrix.__init__(self,varname=name, **kwargs)

        # Initialize and set up rows and cols for base matrix
        total_n_dts = self.G.data.total_n_dts_per_taxa

        n_taxas = self.G.data.n_taxas
        self.n_rows = int(n_taxas * total_n_dts)
        self.n_cols = int(n_taxas * (n_taxas - 1))
        self.shape = (self.n_rows, self.n_cols)

        self.master_rows = np.kron(
                np.arange(self.n_rows, dtype=int),
                np.ones(n_taxas-1, dtype=int))
        self.master_cols = np.kron(
            np.ones(total_n_dts, dtype=int),
            np.arange(self.n_cols, dtype=int))

        self.master_data = np.zeros(len(self.master_cols))
        logging.info('Initializing interactions base design matrix')

    # @profile
    def build(self):
        '''Build the base matrix
        '''
        n_taxas = self.G.data.n_taxas
        data = self.G.data.data

        self.rows = self.master_rows
        self.cols = self.master_cols
        self.data = self.master_data

        i = 0
        for ridx in range(self.G.data.n_replicates):
            i = InteractionsBaseDesignMatrix._fast_build(
                ret=self.master_data, data=data[ridx], n_taxas=n_taxas,
                n_dts=self.G.data.n_dts_for_replicate[ridx], i=i)
            
        if self.G.data.zero_inflation_transition_policy is not None:
            # All of the rows that need to be taken out will be taken out. All of the 
            # remaining nans in the matrix are effects of a structural zero on a non-structural
            # zero - making a nan. We can set this to zero because if this is the case we can say 
            # this means "no effect"
            self.master_data[np.isnan(self.master_data)] = 0
        else:
            if np.any(np.isnan(self.master_data)):
                raise ValueError('nans in matrix, this should not happen. check the values')
        self.matrix = scipy.sparse.coo_matrix(
            (self.master_data,(self.master_rows,self.master_cols)),shape=self.shape).tocsc()
        
        if self.G.data.zero_inflation_transition_policy is not None:
            self.matrix = self.matrix[self.G.data.rows_to_include_zero_inflation, :]

    @staticmethod
    @numba.jit(nopython=True, cache=True, fastmath=True)
    def _fast_build(ret, data, n_dts, n_taxas, i):
        '''About 99.5% faster than regular python looping
        '''
        for tidx in range(n_dts):
            for toidx in range(n_taxas):
                for soidx in range(n_taxas):
                    if toidx == soidx:
                        continue
                    ret[i] = data[soidx, tidx]
                    i = i + 1
        return i

    @staticmethod
    @numba.jit(nopython=True, cache=True, fastmath=True)
    def _fast_build_zi_ignore(ret, data, n_dts, n_taxas, i, zero_inflation, zi_mask):
        '''About 99.5% faster than regular python looping

        Only set to include if target taxa current timepoint and future timepoint are there and
        if the source taxa timepoint is there
        '''
        for tidx in range(n_dts):
            for toidx in range(n_taxas):
                for soidx in range(n_taxas):
                    if toidx == soidx:
                        continue

                    if not (zero_inflation[soidx, tidx] and zero_inflation[toidx, tidx] and \
                        zero_inflation[toidx, tidx+1]):
                        zi_mask[i] = False
                    else:
                        ret[i] = data[soidx, tidx]
                        zi_mask[i] = True
                    
                    i = i + 1
        return i

    def update_value(self):
        self.build()

    def set_to_rhs(self):
        return self.matrix


class InteractionsMixingDesignMatrix(DesignMatrix):
    '''This is the mixing matrix that is used along with
    `InteractionsBaseDesignMatrix` for the  `InteractionsDesignMatrix`
    class

    The `keypair2col` dictionary maps a tuple of (target_taxa_idx,source_taxa_idx)
    to the column they belong to in the full, non-clustered matrix.

    This class has a few different options on how to build M, dependiing on where
    it is being called in inference, some are faster than others, or build very
    specific parts, or build the matrix given some specific data. 
    '''
    def __init__(self, parent, **kwargs):
        DesignMatrix.__init__(self,
            varname=STRNAMES.CLUSTER_INTERACTION_VALUE+'mixing_matrix',
            **kwargs)

        self.parent = parent
        n_taxas = self.G.data.n_taxas
        self.clustering = self.G[STRNAMES.CLUSTERING_OBJ]
        self.interactions = self.G[STRNAMES.INTERACTIONS_OBJ]
        self.n_rows = int(n_taxas * (n_taxas - 1))

        # Build the keypair2col dictionary
        self.keypair2col = np.zeros(
            shape=(len(self.G.data.taxas), len(self.G.data.taxas)), dtype=int)
        i = 0
        for tidx in range(len(self.G.data.taxas)):
            for sidx in range(len(self.G.data.taxas)):
                if tidx == sidx:
                    continue
                self.keypair2col[tidx, sidx] = i
                i += 1

        self.build(build=False)
        logging.info('Initialized interactions mixing design matrix')

        # get _get_rows in cache
        a = np.zeros(1, int)
        tmems = np.asarray([1], dtype=int)
        smems = np.asarray([2], dtype=int)
        InteractionsMixingDesignMatrix.get_indices(a, self.keypair2col, tmems, smems)

    # @profile
    def build(self, build=True, build_for_neg_ind=False):
        '''This makes the rows, cols, and data vectors for the mixing matrix
        from scratch slowly - it does not take advantage of the clus2clus dictionary.

        This will only build for positive indicators unless `build_for_neg_ind` is
        True, where it will also build for negative indicators as well.

        We save the castings of the sets of Taxa ids into numpy arrays so we 
        dont have to do it each iteration. This saves ~45% computation time

        Parameters
        ----------
        build : bool
            This is a flag whether we should build the parent matrix in the
            function `make_matrix`
        build_for_neg_ind : bool
            If True, builds for all the interactions and not just the positively
            indicated
        '''
        rows = []
        cols = []

        # interaction terms
        # Cluster 2 Cluster Interaction InDeX (c2ciidx)
        c2ciidx = 0
        d = {}

        if build_for_neg_ind:
            for interaction in self.interactions:
                d, rows, cols = self.inner(rows, cols, d, interaction, c2ciidx)
                c2ciidx += 1
        else:
            for tcid, scid in self.interactions.iter_valid_pairs():
                d, rows, cols = self.inner_faster(rows, cols, d, tcid, scid, c2ciidx)
                c2ciidx += 1

        rows = np.asarray(list(itertools.chain.from_iterable(rows))) #, dtype=int)
        cols = np.asarray(list(itertools.chain.from_iterable(cols))) #, dtype=int)
        self._make_matrix(rows=rows, cols=cols, n_cols=c2ciidx, build=build)

        self.rows = rows
        self.cols = cols

    # @profile
    def build_clustering(self, build=True):
        '''This makes the rows, cols, and data vectors for the mixing matrix
        from scratch slowly - it does not take advantage of the clus2clus dictionary.

        This will only build for positive indicators unless `build_for_neg_ind` is
        True, where it will also build for negative indicators as well.

        We save the castings of the sets of Taxa ids into numpy arrays so we 
        dont have to do it each iteration. This saves ~45% computation time

        Parameters
        ----------
        build : bool
            This is a flag whether we should build the parent matrix in the
            function `make_matrix`
        '''
        self.cols = np.zeros(400, dtype=int)
        self.rows = np.zeros(400, dtype=int)
        self.baseidx = 0


        # interaction terms
        # Cluster 2 Cluster Interaction InDeX (c2ciidx)
        c2ciidx = 0
        self.d = {}

        # print('this bitch')

        for tcid, scid in self.interactions.iter_valid_pairs():
            self.inner_faster_faster(tcid, scid, c2ciidx)
            c2ciidx += 1

        self.rows = self.rows[:self.baseidx]
        self.cols = self.cols[:self.baseidx]
        self._make_matrix(rows=self.rows, cols=self.cols, n_cols=c2ciidx, build=build)

    # @profile
    def build_for_cols(self, build, cols):
        '''This does the same as `build` but it only builds for the 
        column indices specified, in order

        NOTE - these are the indicies for the on interactions only, that means we skip 
        over the interactions that are false for enumerating them
        '''
        input_cols = OrderedSet(list(cols))

        rows = []
        cols = []
        d = {}

        iidx = 0
        i = 0
        for tcid,scid in self.interactions.iter_valid_pairs():
            if iidx in input_cols:
                d, rows, cols = self.inner_faster(rows, cols, d, tcid, scid, i)
                i += 1
            iidx += 1
        
        rows = np.asarray(list(itertools.chain.from_iterable(rows)))
        cols = np.asarray(list(itertools.chain.from_iterable(cols)))
        self._make_matrix(rows=rows, cols=cols, n_cols=len(input_cols), build=build)

    # @profile
    def build_for_specified(self, build, idxs, tcids, scids):
        '''This does the same as `build` but it only builds for the 
        pair of tcids and scids at the same index passed in
        '''
        # input_cols = OrderedSet(list(idxs))

        rows = []
        cols = []
        d = {}

        for i in range(len(idxs)):
            d, rows, cols = self.inner_faster(rows, cols, d, tcids[idxs[i]], scids[idxs[i]], i)

        
        rows = np.asarray(list(itertools.chain.from_iterable(rows)))
        cols = np.asarray(list(itertools.chain.from_iterable(cols)))
        self._make_matrix(rows=rows, cols=cols, n_cols=len(idxs), build=build)

    # @profile
    def build_to_and_from(self, cids, build):
        '''This makes the rows, cols, and data vectors for the mixing matrix
        from scratch for only positive interactions going to an from each 
        of the clusters in `cids`.

        Parameters
        ----------
        cids : list
            This is a list of cids that we are getting the M matrix for
        '''
        rows = []
        cols = []
        d = {}
        c2ciidx = 0

        cids = OrderedSet(cids)
        # print('cidxs\n', cidxs)
        # print('from data\n', self.interactions.clustering.order)

        for interaction in self.interactions:
            if not interaction.indicator:
                continue
            if interaction.target_cid in cids:
                d, rows, cols = self.inner(rows, cols, d, interaction, c2ciidx)
            elif interaction.source_cid in cids:
                d, rows, cols = self.inner(rows, cols, d, interaction, c2ciidx)
            c2ciidx += 1

        # print('interactions_used', interactions_used)
            
        rows = np.asarray(list(itertools.chain.from_iterable(rows))) #, dtype=int)
        cols = np.asarray(list(itertools.chain.from_iterable(cols))) #, dtype=int)
        self._make_matrix(rows=rows, cols=cols, n_cols=c2ciidx, build=build)

    # @profile
    def inner(self, rows, cols, d, interaction, c2ciidx):
        tcid = interaction.target_cid
        scid = interaction.source_cid

        if tcid not in d:
            tmems = np.asarray(list(self.clustering.clusters[tcid].members))
            d[tcid] = tmems
        else:
            tmems = d[tcid]

        if scid not in d:
            smems = np.asarray(list(self.clustering.clusters[scid].members))
            d[scid] = smems
        else:
            smems = d[scid]

        a = np.zeros(len(tmems)*len(smems), int)
        rows.append(InteractionsMixingDesignMatrix.get_indices(
            a, self.keypair2col, tmems, smems))
        cols.append(np.full(len(tmems)*len(smems), fill_value=c2ciidx))

        return d, rows, cols

    # @profile
    def inner_faster(self, rows, cols, d, tcid, scid, c2ciidx):
        if tcid not in d:
            tmems = np.asarray(list(self.clustering.clusters[tcid].members))
            d[tcid] = tmems
        else:
            tmems = d[tcid]

        if scid not in d:
            smems = np.asarray(list(self.clustering.clusters[scid].members))
            d[scid] = smems
        else:
            smems = d[scid]

        a = np.zeros(len(tmems)*len(smems), int)
        rows.append(InteractionsMixingDesignMatrix.get_indices(
            a, self.keypair2col, tmems, smems))
        cols.append(np.full(len(tmems)*len(smems), fill_value=c2ciidx))

        return d, rows, cols

    # @profile
    def inner_faster_faster(self, tcid, scid, c2ciidx):
        if tcid not in self.d:
            tmems = np.asarray(list(self.clustering.clusters[tcid].members))
            self.d[tcid] = tmems
        else:
            tmems = self.d[tcid]

        if scid not in self.d:
            smems = np.asarray(list(self.clustering.clusters[scid].members))
            self.d[scid] = smems
        else:
            smems = self.d[scid]

        end = self.baseidx + len(tmems)*len(smems)
        if end > len(self.cols):
            # pad 400 to the length
            self.rows = np.append(self.rows, np.zeros(400, dtype=int))
            self.cols = np.append(self.cols, np.zeros(400, dtype=int))

        InteractionsMixingDesignMatrix.get_indices(
            self.rows[self.baseidx:end], self.keypair2col, tmems, smems)
        self.cols[self.baseidx:end] = c2ciidx
        self.baseidx = end

    # @profile
    def _make_matrix(self, rows, cols, n_cols, build):
        '''Builds the mixing matrix from the specified rows and columns
        (data is always going to be 1 because it is a mixing matrix)

        Rebuild after we have changed the mixing matrix if `build` is True.
        '''
        self.n_cols = n_cols
        # else:
        data = np.ones(len(rows), dtype=int)
        self.matrix = scipy.sparse.coo_matrix((data,(rows,cols)),
            shape=(self.n_rows, n_cols)).tocsc()
        self.shape = self.matrix.shape
        if build:
            self.parent.build()

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def get_indices(a, keypair2col, tmems, smems):
        '''Use Just in Time compilation to reduce the 'getting' time
        by about 95%

        Parameters
        ----------
        a : np.ndarray
            Return array
        keypair2col : np.ndarray
            Maps (target_oidx, source_oidx) pair to the place the interaction
            index would be on a full interactio design matrix on the Taxa level
        tmems, smems : np.ndarray
            These are the Taxa indices in the target cluster and the source cluster
            respectively
        '''
        i = 0
        for tidx in tmems:
            for sidx in smems:
                a[i] = keypair2col[tidx, sidx]
                i += 1
        return a


class InteractionsDesignMatrix(DesignMatrix):
    '''Builds the design matrix for the interactions: 
        A_{i,j} @ M = A_{c_i,c_j}, where
            A_{i,j} is the Taxa-Taxa interaction matrix (self.base)
            M is the mixing matrix (self.M)
            A_{c_i,c_j} is the cluster-cluster interaction matrix

    This matrix is composed of two, individual design matrices, `Base` and `M`.
    To make the matrix that we use during inference, we matrix multiply `Base`@`M`,
    which is what this class is for. It wraps these two base classes so that it
    is more streamlined in the inference code.

    `Base` : mdsine2.design_matrices.InteractionsBaseDesignMatrix
        This is an object that builds the interaction matrix as if there was no
        clustering or indicators. It builds the data for all the Taxas/OTUs and
        as if every interaction indicator was on. This is actually faster than
        just building it for individual indicators for a few different reasons:
            1) We only need to update `Base` when we do filtering or update the 
               values of the growth matrix because these are the only two things
               that `Base` is dependent on.
            2) Because we don't have to check indicators or have different shapes
               when building the matrix, it is much easier to build this matrix
               with Numba, which is nearly as fast as C. This speeds up building
               time by ~97%.
    `Mixing` : mdsine2.design_matrices.InteractionsMixingDesignMatrix
        This is the object that selects for indicators and groups taxas together
        into clusters. When we change the indicators of the perturbations or 
        the cluster assignments of the Taxas, we only need to change this matrix,
        which is a lot faster than changing everything. Because both matrices are
        sparse matrices and this matrix is 98% zeros, this is a very fast operation.
    '''
    def __init__(self, **kwargs):
        DesignMatrix.__init__(self,
            varname=STRNAMES.CLUSTER_INTERACTION_VALUE,
            update=True, **kwargs)

        # Initialize and set up rows and cols for base matrix
        total_n_dts = self.G.data.total_n_dts_per_taxa
        n_taxas = self.G.data.n_taxas

        self.n_rows = int(n_taxas * total_n_dts)
        self.clustering = self.G[STRNAMES.CLUSTER_INTERACTION_VALUE].clustering

        self.base = InteractionsBaseDesignMatrix(add_to_dict=False, **kwargs)
        self.base.build()
        self.M = InteractionsMixingDesignMatrix(add_to_dict=False, parent=self,**kwargs)
        self.build()
        self.interactions = self.M.interactions
        logging.info('Initializing interactions matrix')

    def build(self):
        self.matrix = self.base.matrix @ self.M.matrix
        self.n_cols = self.shape[1]

    @property
    def shape(self):
        return self.matrix.shape

    def set_to_lhs(self):
        b_cicj = self.interactions.get_values(
            use_indicators=True).reshape(-1,1)
        return self.matrix.dot(b_cicj)

    def set_to_rhs(self):
        return self.matrix

    def update_value(self):
        self.base.update_value()
        self.build()
