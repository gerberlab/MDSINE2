import os
import copy
import sys
from collections import defaultdict
from contextlib import contextmanager

import h5py
from mdsine2.logger import logger
import time
import shutil
import inspect
import pickle
import numpy as np

# Typing
from typing import Any, Union, Dict, Iterator, Tuple, \
    Callable, Type, List

from .base import *
from .variables import isVariable, Variable
from . import util

from tqdm import tqdm

# Constants
DEFAULT_LOG_EVERY = 5
REQUIRED_ATTRS = ['update', 'initialize', 'set_trace', 'add_trace']
GLOBAL_PARAMETER = -1


class TraceNotFoundException(BaseException):
    pass


def isMCMC(x: Any) -> bool:
    '''Checks if the input array is an MCMC inference object

    Parameters
    ----------
    x : any
        Instance we are checking
    
    Returns
    -------
    bool
        True if `x` is a an MCMC inference object
    '''
    return x is not None and issubclass(x.__class__, BaseMCMC)

def ismodel(x: Any) -> bool:
    '''Checks if the input array is a model object

    Parameters
    ----------
    x : any
        Instance we are checking
    
    Returns
    -------
    bool
        True if `x` is a model object
    '''
    return x is not None and issubclass(x.__class__, BaseModel)


class time_counter:
    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.e = time.time()

    def __float__(self):
        return float(self.e - self.t)

    def __coerce__(self, other):
        return (float(self), other)

    def __str__(self):
        return str(float(self))

    def __repr__(self):
        return str(float(self))


class BaseModel(Saveable):
    '''Base class for a model

    Parameters
    ----------
    graph : pylab.graph.Graph, Optional
        The graph we want to do the inference over
        If nothing is provided, it grabs the default graph
    '''
    def __init__(self, graph: Graph=None):
        if graph is None:
            graph = get_default_graph()
        if not isgraph(graph):
            raise TypeError('`graph` ({}) must be None or a pylab.graph.Graph object' \
                ''.format(type(graph)))
        self.graph = graph
        self.ran = False
        self.graph.inference = self


class BaseMCMC(BaseModel):
    '''Base MCMC over a graph. This only runs 1 chain.

    Typical use
    -----------
    ```
    >>> # You first initialize the object with the graph and parameters you want
    >>> inf = BaseMCMC(burnin=1000, n_samples=2000, graph=G)
    >>> # Then you set the inference order
    >>> inf.set_inference_order(['a', 'b', 'c'])
    >>> # Then set some diagnostic variables (Optional)
    >>> inf.set_diagnostic_variables(['e'])
    >>> # Then we can run the inference
    >>> inf.run(log_every=5, checkpoint=100, tracer_filename='./output/tracer.hdf5')
    ```

    Inference order
    ---------------
    This datastructure will perform the inference specified in the inference 
    order, which is set using the `set_inference_order` method. Each element
    in `inf_order` must be an ID in `graph` that implements the functions: `update` 
    (how we sample the variable during inference), `set_trace` (how the 
    tracing gets set up), `add_trace` (how the current value gets added to the 
    trace (this is called immediately after `update`)), and `initialize` (
    how the values get initialized before inference).

    If your posterior class directly inherits a class that is a subclass of 
    `pylab.variables.Variable` then the functions `set_trace` and 
    `add_trace` are already implemented for you.

    Sometimes we want to randomize the order that we update variables. For
    example, if I am updating the growth, self_interactions, and interactions,
    I may want to randomize the order that I update them so that there is no
    unintentional bias during inference. In that case I can make a new object 
    called `gLVParams` where the `update` function randomizes the order that
    growth.update(), self_interactions.update(), and interactions.update() 
    functions are called. 

    Parameters
    ----------
    burnin : int
        Number of initial samples to throw away
    n_samples : int
        Total number of samples of the posterior
        Number of posterior samples = n_samples-burnin
    '''

    def __init__(self,
                 burnin: int,
                 n_samples: int,
                 *args,
                 **kwargs):
        BaseModel.__init__(self, *args, **kwargs)
        if not util.isint(burnin):
            raise TypeError('`burnin` ({}) must be an int'.format(type(burnin)))
        if not util.isint(n_samples):
            raise TypeError('`n_samples` ({}) must be an int'.format(type(n_samples)))
        if n_samples < burnin:
            raise TypeError('The total number of sample (n_samples) must be'\
                'larger than the burn in (burnin)')

        self.burnin = burnin
        self.n_samples = n_samples
        self.sample_iter = 0 # This holds the current sample we are at
        self.inf_order = None # This is a list of the inference order
        self.diagnostic_variables = None # DEPRECIATED
        self.tracer = None # This is a pointer to the pylab.inference.Tracer object

        # This is the sample iteration to start on. Unless we are loading from a 
        # saved MCMC object, then this will be 0
        self.start_step = None 

        # Functions to save intermediately, this is used for semisynthetic
        self._intermediate_func = None # callable
        self._intermediate_t = None # float
        self._intermediate_kwargs = None # dict

    def num_iters(self, section: str):
        if section == 'burnin':
            return self.burnin
        elif section == 'full':
            return self.n_samples
        elif section == 'posterior':
            return self.n_samples - self.burnin
        else:
            raise ValueError("Unexpected section {}".format(section))

    @classmethod
    def load(cls, filename: str):
        '''Override base Saveable to redo the filename of the
        tracer object if it has one
        
        Paramters
        ---------
        filename : str
            This is the location of the file to unpickle
        '''
        with open(str(filename), 'rb') as handle:
            b = pickle.load(handle)
        
        # redo the filename to the new path if it has a save location
        if hasattr(b, '_save_loc'):
            filename = os.path.abspath(filename)
            b._save_loc = filename

            # Redo the filename of the tracer object if necessart
            if b.tracer is not None:
                if b.tracer.filename is not None:
                    _, tracer_fname = os.path.split(b.tracer.filename)
                    currpath, _ = os.path.split(b._save_loc)

                    new_loc = os.path.join(currpath, tracer_fname)

                    if os.path.isfile(new_loc):
                        b.tracer.filename = new_loc
                    else:
                        raise ValueError('Looking for tracer hdf5 object in {}, could not find it ' \
                            'in the local path even though inference says it contains the object. On ' \
                            'file it shows the location {}'.format(
                                new_loc, b.tracer.filename))

        return b

    @property
    def ckpt(self) -> int:
        '''For backwards compatibility
        '''
        return self.checkpoint

    @ckpt.setter
    def ckpt(self, a: int):
        '''Set ckpt
        '''
        self.ckpt = a

    def names(self) -> Iterator[str]:
        '''Get the names of the nodes in the inference object

        Returns
        -------
        list(str)
        '''
        return list(self.graph.name2id.keys())

    def ids(self) -> Iterator[int]:
        '''Get the IDs of the nodes in the inference object

        Returns
        -------
        list(int)
        '''
        return list(self.graph.nodes.keys())

    def set_inference_order(self, order: Iterator[Union[str, int]]):
        '''`order` is an array of nodes we want to sample in the order
        that we want. Check that they are all in the graph. Can put in the 
        ID or the name

        Parameters
        ----------
        order : array_like(int or str)
            Order to do the inference. This must be the IDs of the nodes and can
            be an ID (int) or the name (str)
        '''
        if not util.isarray(order):
            raise TypeError('order ({}) must be an array'.format(order))

        ret = []
        for nid in order:
            if nid not in self.graph:
                raise ValueError('Node ({}) not found in graph ({})'.format(
                    nid, self.names()))
            node = self.graph[nid]

            for attr in REQUIRED_ATTRS:
                if hasattr(node, attr):
                    if not callable(getattr(node, attr)):
                        raise UndefinedError('node ({}) must have `{}` be callable'.format(
                            node.name, attr))
                else:
                    raise UndefinedError('node ({}) must have the function `{}`'.format(
                        node.name, attr))
            ret.append(node.id)
        self.inf_order = ret

    def is_in_inference_order(self, var: Union[int, str, Variable]) -> bool:
        '''Checks if the variable is in the inference order

        Parameters
        ----------
        var : int, str, pylab.variables.Variable
            Identifier for the variable
            If it is an int - it is assumed this is the id of the variable
            If is is a str - it is assumed this is the name of the variable

        Returns
        -------
        bool
        '''
        if not isnode(var):
            var = self.graph[var].id
        else:
            var = var.id
        return var in self.inf_order

    def set_diagnostic_variables(self, vars):
        '''DEPRECIATED
        
        A list of variables that you want to track over time. These variables
        do not necessarily have to be variables that we are tracing, rather they
        can be any variable that changes over time in the inference. However,
        the variables that we are tracing need to be a subclass of
        pylab.variables.Variable

        We manually override the tracing for each of the variables.

        Parameters
        ----------
        vars : array_like, set, 1-dim
            - A list of the variables you want to track (the actual object)
            - The elements in the list should be of type pylab.variables.Variable
        '''
        # wrap vars as an iterable if it is not
        if not hasattr(vars, '__iter__'):
            vars = [vars]

        if len(vars) == 0:
            raise ValueError('No values in `vars`')

        # Check parameters passed in
        for var in vars:
            if not isVariable(var):
                raise InheritanceError('Each variable passed in should be of a subclass of' \
                    ' `pylab.variables.Variable`. This is of type `{}`'.format(
                        var.__class__.__name__))

        # Add variables to dictionary
        self.diagnostic_variables = {}
        for var in vars:
            if var.name in self.diagnostic_variables:
                raise ValueError('Two different diagnostic variables cannot have ' \
                    'the same name.')
            self.diagnostic_variables[var.name] = var

    def set_tracer(self, filename: str, checkpoint: int=100):
        '''Sets up the tracing object

        Parameters
        ----------
        filename : str, None
            File location to save the hdf5 object
        checkpoint : int, None
            Saves the current progress of the inference chain every `checkpoint` iterations
            If None then there is no intermediate checkpointing
        '''
        if checkpoint is None:
            checkpoint = self.n_samples
        self.tracer_filename = filename
        self.checkpoint = checkpoint
        self.tracer = Tracer(mcmc=self, filename=filename)

    def set_intermediate_validation(self, t: float, func: Callable, kwargs: Dict[str, Any]=None):
        '''Every `t`, run the function `func` during validation in a new process during inference.

        Parameters
        ----------
        t : int
            How often to run in number of seconds
        func : callable
            This is the function the intermediate must call. It assumes the
            only parameters are:
                chain : this is the chain object (self)
                burnin : number of burnin
                n_samples : number of total samples
                sample_iter : current iteration number
        kwargs : dict
            Extra arguments
        '''
        if not util.isint(t):
            raise TypeError('`t` ({}) must be an int'.format(type(t)))
        if t < 0:
            raise ValueError('`t` ({}) must be > 0'.format(t))
        if kwargs is None:
            kwargs = {}
        for k,v in kwargs.items():
            if not util.isstr(k):
                raise ValueError('Keys in kwargs ({}) must be strings'.format(type(k)))
        if not callable(func):
            raise TypeError('`func` ({}) must be callable'.format(type(str)))

        # Check if the function has the right arguments
        kwarg_keys = list(kwargs.keys())
        valid_args = set(['burnin', 'n_samples', 'chain', 'sample_iter'] + kwarg_keys)
        args = inspect.getargspec(func).args
        for arg in args:
            if arg not in valid_args:
                raise ValueError('Function `{}` does not have the correct arguments. ' \
                    'It has the argument `{}` when it should only have the arguments {}.'.format(
                        func.__name__, arg, list(valid_args)))
        for arg in valid_args:
            if arg not in args:
                raise ValueError('Function `{}` does not have the correct arguments. ' \
                    'It is excluding the argument `{}`, which must be included'.format(
                        func.__name__, arg))

        self._intermediate_t = t
        self._intermediate_func = func
        self._intermediate_kwargs = kwargs

    def run(self, log_every: int=1, benchmarking: bool = False) -> "BaseMCMC":
        '''Run the inference.

        Parameters
        ----------
        log_every : int, None
            Logs the values of the variables you are learning every `log_every` iterations.
            If the logger is set to DEBUG then override to log every iteration.
        
        Returns
        -------
        pylab.inference.BaseMCMC
            Output from the inference, self
        '''

        update_runtimes: Dict[str, List[float]] = defaultdict(list)

        seed(self.graph.seed)
        if self.start_step is None:
            self.start_step = 0
        try:
            if self.inf_order is None:
                raise UndefinedError('Cannot run mcmc until you have set the inference order.' \
                    ' Set with the function `self.set_inference_order`.')
            if self.tracer is None:
                logger.warning('No tracer set - assume you do not want to write to disk')
                self.tracer_filename = None
                self.tracer = Tracer(mcmc=self, filename=None)
                self.checkpoint = None

                logger.info('Setting the trace of learned parameters')
                logger.info('#######################################')
                for nid in self.inf_order:
                    logger.info('Setting the trace of {}'.format(self.graph[nid].name))
                    self.graph[nid].set_trace()
                logger.info('Setting the trace for diagnostic variables')
                logger.info('##########################################')
                if self.diagnostic_variables is not None:
                    for nid in self.diagnostic_variables:
                        logger.info('Setting the trace of {}'.format(self.graph[nid].name))
                        self.graph[nid].set_trace()

            total_time = time.time()
            if log_every is None:
                log_every = DEFAULT_LOG_EVERY

            start = time.time()
            intermediate_time = time.time()

            for i in tqdm(range(self.start_step, self.n_samples), file=sys.stdout):
                self.sample_iter = i

                # Check if we need to run the intermediate script
                try:
                    if self._intermediate_t is not None:
                        if time.time() - intermediate_time > self._intermediate_t:
                            logger.info('Running intermediate script {}'.format(
                                self._intermediate_func.__name__))
                            kwargs = {
                                'chain': self, 'burnin': self.burnin, 
                                'n_samples': self.n_samples, 'sample_iter': self.sample_iter}
                            for k,v in self._intermediate_kwargs.items():
                                kwargs[k] = v
                            try:
                                self._intermediate_func(**kwargs)
                            except:
                                raise ValueError('failed in intermediate function')
                            intermediate_time = time.time() 
                except AttributeError:
                    logger.info('Pre Pylab 3.0.0 implementation. Ignore')
                except:
                    logger.error('unknown error')
                    raise

                # Log where necessary
                if i % log_every == 0 and i > 0:
                    logger.debug('Inference iteration {}/{}, time: {}'.format(
                        i,
                        self.n_samples,
                        time.time() - start)
                    )
                    start = time.time()
                    for id in self.inf_order:
                        if type(id) == list:
                            for id_ in id:
                                logger.debug('{}: {}'.format(self.graph.nodes[id_].name,
                                    str(self.graph.nodes[id_])))
                        else:
                            logger.debug('{}: {}'.format(self.graph.nodes[id].name,
                                str(self.graph.nodes[id])))

                    if benchmarking:
                        # print benchmarking results.
                        logger.info("Runtime summary:")
                        for node_name, _durations in update_runtimes.items():
                            _durations = np.array(_durations)
                            logger.info("{}: [min = {}, max = {}, median = {}, mean = {}]".format(
                                node_name,
                                np.min(_durations),
                                np.max(_durations),
                                np.median(_durations),
                                np.mean(_durations)
                            ))

                # Sample posterior in the order indicated and add the trace
                for _id in self.inf_order:
                    node = self.graph.nodes[_id]
                    try:
                        with time_counter() as t:
                            node.update()
                            node.add_trace()

                        if benchmarking:
                            update_runtimes[node.name].append(float(t))
                    except:
                        logger.error('Crashed in `{}`'.format(self.graph[_id].name))
                        # self.graph.tracer.finish_tracing()
                        raise

                # Save diagnostic variables where necessary
                if self.diagnostic_variables is not None:
                    for name in self.diagnostic_variables:
                        if self.diagnostic_variables[name].sample_iter == self.sample_iter:
                            self.diagnostic_variables[name].add_trace()

                # If we just saved the traces to disk, save the MCMC object
                if self.sample_iter % self.checkpoint == 0 and self.sample_iter > 0:
                    try:
                        self.save()
                    except:
                        logger.critical('If you want to checkpoint, you must set the save location ' \
                            'of tracer, graph, and mcmc using the function self.set_save_location()')
                        print(self._save_loc)
                        raise

            # Finish the tracing
            self.graph.tracer.finish_tracing()
            self.ran = True
            logger.info('Inference total time: {}/Gibb step'.format(
                (time.time() - total_time)/self.n_samples))
            
            # Remove the local traces
            if self.tracer is not None:
                logger.info('remove local traces')
                for node in self.graph:
                    if isVariable(node):
                        node.remove_local_trace()
            self.save()

            return self
        except:
            for a in self.graph._persistent_pntr:
                a.kill()
            raise

    def continue_inference(self, gibb_step_start: int):
        '''Resume inference at the gibb step number `gibb_step_start`. Note that
        we do not resume the random seed where it was at that point of the inference.

        Parameters
        ----------
        gibb_step_start : int
            Gibb step to start
        '''
        self.tracer.continue_inference(gibb_step_start=gibb_step_start)
        self.start_step = gibb_step_start


class Tracer(Saveable): 
    '''This sets up the graph to be traced using h5py.

    Able to checkpoint the values of the graph through inference.
    Write the data to disk. The only variables that are traced are the 
    variables in the set `being_traced`.

    There might be other processes/threads that are reading the current 
    chain to plot it intermittently, so we create the file in SWMR 
    (Single Writer, Multiple Reader) mode which will make the 
    file always readable (never ina  corrupt state from writing).

    Parameters
    ----------
    mcmc : pylab.inference.BaseMCMC
        This is the inference object that we are tracing
    filename : str
        This is where to save it
    '''
    def __init__(self, mcmc: BaseMCMC, filename: str):
        # Check parameters and set up the attributes
        self.mcmc = mcmc # This is a pointer to the BaseMCMC object
        self.graph = self.mcmc.graph # This is a pointer to the pylab.graph.Graph object
        self.graph.tracer = self # Sets up internal pointers
        self.mcmc.tracer = self # Sets up internal pointers

        self.filename  = filename
        if self.filename is not None:
            if not util.isstr(filename):
                raise TypeError('filename ({}) must be a str'.format(type(filename)))
            self.filename = os.path.abspath(filename)
            self.f = h5py.File(self.filename, 'w', libver='latest') #This is the h5py file
            self.being_traced = set()

            self.f.attrs['burnin'] = self.mcmc.burnin
            self.f.attrs['n_samples'] = self.mcmc.n_samples
            self.f.attrs['ckpt'] = self.mcmc.checkpoint

            self.burnin = self.mcmc.burnin
            self.n_samples = self.mcmc.n_samples
            self.checkpoint = self.mcmc.checkpoint

            # Get the inference order and diagnostic variables
            ret = []
            for nid in self.mcmc.inf_order:
                ret.append(self.mcmc.graph[nid].name)
            self.f.attrs['inf_order'] = ret
            if self.mcmc.diagnostic_variables is not None:
                a = list(self.mcmc.diagnostic_variables.keys())
            else:
                a = []
            self.f.attrs['diagnostic_variables'] = a
            logger.info('Setting Single Write, Multiple Read Mode')
            self.f.swmr_mode = True # single writer, multiple reader mode
            self.close()

            # Add all of the variables being traced to the file
            logger.info('Setting the trace of learned parameters')
            logger.info('#######################################')
            for nid in self.mcmc.inf_order:
                logger.info('Setting the trace of {}'.format(self.graph[nid].name))
                self.graph[nid].set_trace()
            logger.info('Setting the trace for diagnostic variables')
            logger.info('##########################################')
            if self.mcmc.diagnostic_variables is not None:
                for nid in self.mcmc.diagnostic_variables:
                    logger.info('Setting the trace of {}'.format(self.graph[nid].name))
                    self.graph[nid].set_trace()

    # def __enter__(self):
    #     self.f = h5py.File(self.filename, 'r+')
    #     return self.f

    # def __exit__(self, type, value, traceback):
    #     self.close()

    @property
    def ckpt(self) -> int:
        '''For backwards compatitbility
        '''
        return self.ckpt

    @ckpt.setter
    def ckpt(self, a: int):
        '''Set ckpt
        '''
        self.ckpt = a

    def close(self):
        '''Close the file
        '''
        self.f.close()
        self.f = None

    def open(self, mode='r+'):
        '''Open the file
        '''
        self.f = h5py.File(self.filename, mode=mode, libver='latest')

    def copy(self) -> "Tracer":
        '''Return a copy of the object but do not copy the underlying hdf5 object
        '''
        new_obj = type(self)(mcmc=self.mcmc, filename=self.filename)
        new_obj.__dict__.update(self.__dict__)
        return new_obj

    def deepcopy(self, hdf5_dst: str=None) -> "Tracer":
        '''Return a deepcopy of the object and copy the underlying hdf5 object as well

        Parameters
        ----------
        hdf5_dst : str, None
            Destination to copy the hdf5 object. If Nothing is provided then we 
            will just append '_copy' to the name of the current filename
        '''
        if hdf5_dst is None:
            dst = self.filename.replace('.hdf5', '_copy.hdf5')
        else:
            if not util.isstr(hdf5_dst):
                raise TypeError('`hdf5_dst` ({}) must be a str')
            dst = hdf5_dst


        new_obj = copy.deepcopy(self)
        new_obj.filename = dst
        shutil.copyfile(src=self.filename, dst=new_obj.filename)
        return new_obj

    def is_being_traced(self, var: Union[int, str, Node]) -> bool:
        '''Checks if the variable is being traced

        Parameters
        ----------
        var : int, str, pylab.graph.Node
            An ID of a graph object or the object

        Returns
        -------
        bool
        '''
        if isnode(var):
            var = var.name
        else:
            if var in self.graph:
                var = self.graph[var].name
            else:
                # raise IndexError('`var` ({}) ({}) not recognized in graph'.format(type(var), var))
                return False
        return var in self.being_traced

    def set_trace(self, name: str, shape: Tuple[int], dtype: Type):
        '''Set up a dataset for the variable. If a group is specified, it will 
        add it to the group

        Parameters
        ----------
        name : str
            This is the name of the variable.        
        shape : tuple, None
            This is the shape of the variable. This should not include the 
            trace length, that is added in this function. If it is a scalar,
            then the shape is None
        dtype : Type
            This is the type of the trace
        group : str, h5py.Group
            This is the group you want it added to 
        '''
        if name in self.being_traced:
            logger.info('Skipping adding the trace of `{}` because it is already being' \
                ' traced ({})'.format(name, list(self.being_traced)))
            return
        if not util.isstr(name):
            raise TypeError('`name` ({}) must be a str'.format(type(name)))
        if not (util.istuple(shape) or shape is None):
            raise TypeError('`shape` ({}) must be a tuple or None'.format(type(shape)))
        if not util.istype(dtype):
            raise TypeError('`dtype` ({}) ({}) must be a Type'.format(dtype, type(dtype)))
        self.open()
        
        if shape is not None:
            shape = (self.n_samples, ) + shape
        else:
            shape = (self.n_samples, )
        dset = self.f.create_dataset(name=name, shape=shape, dtype=dtype, chunks=True)
        dset.attrs['end_iter'] = 0
        self.being_traced.add(name)
        self.close()

    def write_to_disk(self, name: str):
        '''Append the RAM trace of the variable `name` into disk. Copies the data
        from RAM (self.graph[name]/trace) into disk memory.

        Parameters
        ----------
        name : str
            Name of the variable we are writing 
        '''
        if self.filename is None:
            raise ValueError('Tracing to disk not setup')
        self.open()
        dset = self.f[name]
        i = dset.attrs['end_iter']
        node = self.graph[name]
        l = node.ckpt_iter
        
        # print('\nwriting to disk,', name)
        # print('ckpt_iter', l)
        # print('trace.shape', node.trace.shape)
        # print(dset[i:i+l].shape)

        dset[i:i+l] = node.trace
        dset.attrs['end_iter'] = i + l
        self.close()

    def overwrite_entire_trace_on_disk(self, name: str, data: np.ndarray, dtype: Type=None):
        '''Overwrite all the data we have on disk for the variable 
        with name `name` and data `data`. Blanks everything out and 
        sets the end variable to the end of data
        
        Parameters
        ----------
        name : str
            Name of the variable we are overwriting
        data : np.ndarray
            Array we are overwriting the data with
        '''
        if not util.isarray(data):
            raise TypeError('`data` ({}) must be an array'.format(type(data)))
        data = np.asarray(data)

        self.open()
        dset = self.f[name]
        shape = dset.shape
        if dtype is None:
            dtype = data.dtype

        # Delete the old dataset and make a new one
        del self.f[name]
        self.close()
        self.being_traced.remove(name)

        self.set_trace(name=name, shape=shape, dtype=dtype)
        
        self.open()
        dset = self.f[name]
        dset[:data.shape[0]] = data
        dset.attrs['end_iter'] = data.shape[0]
        self.close()

    def finish_tracing(self):
        '''Append the rest of the buffers to the dataset. Delete the local trace
        '''
        if self.filename is None:
            return
        self.open()
        for name in self.being_traced:
            dset = self.f[name]
            i = dset.attrs['end_iter']
            node = self.graph[name]
            l = node.ckpt_iter
            dset[i:i+l] = node.trace
            dset.attrs['end_iter'] = i+l

            self.graph[name].trace = None
        self.close()

    def get_disk_trace_iteration(self) -> int:
        '''Returns the last iteration the disk is saved to

        Returns
        -------
        int
        '''
        self.open()
        name = list(self.f.keys())[0]
        dset = self.f[name]
        n = dset.attrs['end_iter']
        self.close()
        return n

    def get_trace(self, name: str, section: str='posterior', slices: slice=None) -> np.ndarray:
        '''Return the trace that corresponds with the name.

        Depdending on the parameter `section`, it will return different parts of
        the trace. Options:
            'posterior': 
                Returns the trace after the burnin
            'burnin'
                Returns the samples that were in the burnin
            'entire'
                Returns all the samples
            'slice'
                All of the arguments for slicing are in slices
            
        Parameters
        ----------
        name : str
            Name of the variable
        section : str
            Which part of the trace to return - description above
        slices : list(slice), slice
            A list of slicing objects or a slice object.

            slice(start, stop, step)
            Example, single dimension:
                slice(None) == :
                slice(5) == :5
                slice(4, None, None) == 4:
                slice(9, 22,None) == 9:22
            Example, multiple dimensions:
                [slice(None), slice(4, None, None)] == :, 4:
                [slice(None), 4, 5] == :, 4, 5

        Returns
        -------
        np.ndarray
        '''
        if not util.isstr(section):
            raise TypeError('`section` ({}) must be a str'.format(type(str)))
        
        self.open(mode='r')
        try:
            dset = self.f[name]
        except KeyError:
            raise TraceNotFoundException("Trace of {} not found.".format(name))

        if section == 'posterior':
            high = dset.attrs['end_iter']
            low = self.burnin
        elif section == 'burnin':
            high = dset.attrs['end_iter']
            low = 0
            if high > self.burnin:
                high = self.burnin
        elif section == 'entire':
            low = 0
            high = dset.attrs['end_iter']
        elif section != 'slice':
            raise ValueError('`section` ({}) not recognized'.format(section))
        
        if section != 'slice':
            if slices is not None:
                slices = [slice(low,high,None)] + list(slices)
            else:
                slices = slice(low,high,None)

        ret = dset[slices]
        self.close()
        return ret

    def get_iter(self, name: str) -> int:
        '''Return the end index that has been saved so far for variable with
        name `name`

        Parameters
        ----------
        name : str
            Name of the variable
        
        Returns
        -------
        int
        '''
        self.open()
        dset = self.f[name]
        i = dset.attrs['end_iter']
        self.close()
        return i

    def continue_inference(self, gibb_step_start: int):
        '''Restart the inference at the gibbs step provided

        Parameters
        ----------
        gibb_step_start : int
            Gibb step to start at
        '''
        self.open()
        for name in self.f:
            dset = self.f[name]
            dset.attrs['end_iter'] = gibb_step_start
        self.close()


def r_hat(chains: Iterator[BaseMCMC], vname: str, start: int, end: int, idx: Union[int, slice]=None, 
    returnBW: bool=False) -> Union[float, Dict[str, float]]:
    '''Calculate the measure `R^` for the variable called `vname` at the index `idx`.
    If `idx` is None then we assume that the variable is scalar.

    Definition
    ----------
    `R^` is defined in [1] as:
        Let m be the number of distinct chains
        Let n be the length of each chain

        \mu_j = mean for chain j
        \mu = mean over all the chains
        X_{ij} = Value in posterior for sample i in chain j

        % Variance between sequences
        B = [ n/(m-1) ] * \sum^m_{j=1} ( \mu_j - \mu )^2

        % Variance within sequences
        W = [ \sum^m_{j=1} [ \sum^n_{i=1} ( X_{ij} - \mu_j )^2 ] ] / m

        \hat{R} = \sqrt{ [ (n-1)*W/n + B/n ] / W }

    It is assumed that these chains were run with different initial conditions
    and that the data is the same. An error will be thrown if the total number of
    burn-in or total samples are different.

    The values range from [1, inf], where 1 is completely mixed.

    Parameters
    ----------
    chains : list(pylab.inference.BaseMCMC),list(str)
        An iterable object of pl.inference.BaseMCMC objects. If it is a string then
        it is the saved location of the chain object.
    vanme : str, int
        This is the index of the variable we want to calculate `R^` from. This can
        either be the name (str) or the graph ID (int) to identify it.
    idx : int, tuple(int)
        This is the index of the item you're looking at. This is only necessary if
        this variable is a vector. If it is a scalar then this is ignored.
    returnBW : bool
        If True, returns B (between sequence variance) and W (within sequence variance)
        as well as the :math:`\hat{R}` metric. Returns it as a dictionary

    From the R documentation (https://www.rdocumentation.org/packages/asbio/versions/1.6-5/topics/R.hat):
    Gelman et al. (2003, pg. 296) provides insufficient details to reproduce this function.
    To get the real function see Gelman and Rubin (1992). The authors list one other change
    in their Statlab version of this function at http://lib.stat.cmu.edu/S/itsim. They
    recommend multiplying sqrt(postvar/W) by sqrt((df + 3)/t(df + 1)). The original code
    and this function can produce estimates below 1.

    References
    ----------
    - [1] Gelman, A. and D. B. Rubin (1992) Inference from iterative simulation using multiple
        sequences (with discussion). Statistical Science, 7:457-511.
    - [2] Brooks and Gelman (1998). Journal of Computational and Graphical Statistics, 7(4)434-455.
    - [3] A. Gelman, H. S. Stern, J. B. Carlin, D. B. Dunson, A. Vehtari, and D. B. Rubin, Bayesian Data
        Analysis Third Edition. Chapman and Hall/CRC, 2013.
    - [4] https://stats.stackexchange.com/questions/348984/stan-hatr-versus-gelman-rubin-hatr-definition
    '''
    # Check that all of the chains are consistent with each other
    if not util.isarray(chains):
        raise TypeError('`chains` ({}) must be array_like'.format(type(chains)))
    if len(chains) <= 1:
        raise ValueError('There must be at least 2 items in chains. There are {}'.format(
            len(chains)))
    if not util.isint(start):
        raise TypeError('`start` ({}) must be an int'.format(start))
    if not util.isint(end):
        raise TypeError('`end` ({}) must be an int'.format(end))
    for i in range(len(chains)):
        chain = chains[i]
        if util.isstr(chain):
            chains[i] = BaseMCMC.load(chains[i])
            chain = chains[i]
        if not isMCMC(chain):
            raise TypeError('`chain` ({}) must be a pylab.inference.BaseMCMC object'.format(
                type(chain)))

    # Get the variables from each of the chains and check that they are the proper shapes
    if not (util.isstr(vname) or util.isint(vname)):
        raise TypeError('`vname` ({}) must be an int or a str'.format(type(vname)))
    traces = [chain.graph[vname].get_trace_from_disk(section='entire') for chain in chains]
    traces = [trace[start:end] for trace in traces]
    if idx is not None:
        traces = [trace[:, idx] for trace in traces]

    traces = np.nan_to_num(traces, nan=0.0)

    # Calculate r_hat
    m = traces.shape[0]  # no. of sequences
    n = traces.shape[1]  # length of each chain after burn-in
    try:
        num_var = traces.shape[2]  # Number of indexed variables being sampled
    except:
        # This fails when the variable is a scalar, reshape it so that it 
        # has a final dimension of 1
        traces = traces.reshape(n,m,1)
        num_var = 1
    s_hat_i = np.var(traces, axis=1, ddof=1)
    x_bar_i = np.mean(traces, axis=1)
    x_bar = np.mean(x_bar_i, axis=0)

    # W: Mean of sample variances
    W = np.mean(s_hat_i, axis=0)

    # B: Sample variance of means (times n)
    B = n * np.var(x_bar_i, axis=0, ddof=1)

    # Sigma-hat and V-hat:
    sigma_hat = ((n - 1) / n) * W + (1 / n) * B
    v_hat = sigma_hat + B * (1 / (n * m))

    var_v_hat = (
            (((n - 1) / n) ** 2) * (1 / m) * np.var(s_hat_i, axis=0)
            + (((m + 1) / (m * n)) ** 2) * (2 / (m - 1)) * np.power(B, 2)
            + 2 * (m + 1) * (n - 1) * (1 / m ** 2) * (1 / n) * np.array([
                np.cov(s_hat_i[:, k], np.power(x_bar_i[:, k], 2))[0, 1]
                - 2 * x_bar[k] * np.cov(s_hat_i[:, k], x_bar_i[:, k])[0, 1]
                for k in range(num_var)
            ])
    )

    # degrees of freedom
    dof = 2 * np.power(v_hat, 2) / var_v_hat

    # R-hat (Empirical conditional variance, divided by W)
    rhat = np.sqrt(
        v_hat * (dof+3) * np.reciprocal(W * (dof + 1))
    )

    if returnBW:
        return {'B': B, 'W': W, 'rhat': rhat}
    else:
        return rhat