'''pylab.multiprocessing

This module has classes for specialized types of multiprocessing that can 
dramatically decrease the run time if there large I/O costs for certain
types of problems. These classes are not better than the standard Python
`multiprocessing` package if the problem requires very little I/O overhead.

These classes are built off of the standard Python `multiprocessing` library
so it is portable to any system that can run Python.

NOTE: These classes are not used during inference. Need to pass the random state
for each computation orelse the results are not reproducable. To do this, either
pass the random state or do no sampling in the multiprocessed object.
'''
import sys
import multiprocessing
import copy
import logging
import os
import signal
import time

# Typing
from typing import TypeVar, Generic, Any, Union, Dict, Iterator, Tuple, \
    Callable

from .errors import NeedToImplementError
from .graph import isgraph, Graph
from . import util

def ispersistentworker(x: Any) -> bool:
    '''Checks whether the input is a subclass of PersistentWorker

    Parameters
    ----------
    x : any
        Input instance to check the type of PersistentWorker

    Returns
    -------
    bool
        True if `x` is of type PersistentWorker, else False
    '''
    return x is not None and issubclass(x.__class__, PersistentWorker)

def ispersistentpool(x: Any) -> bool:
    '''Checks whether the input is a subclass of PersistentPool

    Parameters
    ----------
    x : any
        Input instance to check the type of PersistentPool

    Returns
    -------
    bool
        True if `x` is of type PersistentPool, else False
    '''
    return x is not None and issubclass(x.__class__, PersistentPool)

def isDASW(x: Any) -> bool:
    '''Checks whether the PersistentPool is Different Argument
    Single Worker

    Parameters
    ----------
    x : any
        Input instance to check the type of
        If it is not a persistent pool type then it will return False

    Returns
    -------
    bool
    '''
    if not ispersistentpool(x):
        return False
    return x.ptype == 'dasw'

def isSADW(x: Any) -> bool:
    '''Checks whether the PersistentPool is Same Argument
    Different Worker

    Parameters
    ----------
    x : any
        Input instance to check the type of
        If it is not a persistent pool type then it will return False

    Returns
    -------
    bool
    '''
    if not ispersistentpool(x):
        return False
    return x.ptype == 'sadw'


class PersistentWorker:
    '''Base class of the persistent worker. If you want to add functionality to
    each worker that gets passed in, then do it here.
    '''
    pass


class PersistentPool:
    '''Manages a set of PersistentWorker objects for a pooling.

    By persistent we mean that the worker processes are always active, i.e. they
    are not created after the `map` function is called nor are they destroyed when
    the pool ends - They wait until arguments are sent to them. 
    This also means they have to be explicitly killed at the end of the program with
    the `kill` function and have to be explicitly created before `map` is called.

    An instance that you would want to use pylab.multiprocessing.PersistentPool
    class instead of the regular multiprocessing.Pool class is when you have very
    large arguments (large matrix, dictionaries, etc.) that do not change over time
    or you only need to send them once to the workers. This way you are getting rid of
    unnecessary I/O. It also gets rid of the overhead of creating and destroying
    Processes every time you call `map`.

    To add a worker to the pool, it must be an object that inherits from the class
    `PersistentWorker`. When the `map` function is called by the pool 
    (`PersistentPool`), it calls the function defined by the `func` parameter of `map`.

    NOTE: There is *NO* guarenteed order of the return of the arguments because in either
    case we do not assume an order of the workers or the arguments. Make sure you
    can somehow determine which arguments belong where.

    There are two different types of parallelization that can be accomplished using this
    class, which are specified using the `ptype` (parallelization type); 'sadw' and 'dasw', 
    which are explained below:

    SADW (Same Arguments Different Worker)
    --------------------------------------
    This type of parallelization has different instantiations at each of the workers
    and all of the workers get the same arguments when `map` is called. An instance 
    you would use pooling like this is if you have 5 distinct large matrices each with 
    different data and you want to do the same operations on them. Here you pass in 
    a single dictionary of arguments.

    DASW (Different Arguments Same Worker)
    --------------------------------------
    This type of parallelization has the same worker (operations) at each of the workers and we 
    want to run many different arguments. This is analogous to the classical type
    mapping. An instance you want to use this is when you have a single operation that
    takes a long time and you want to run that will multiple arguments. Here you pass
    in a list of dictionaries that have the arguments in them.
    
    Staged map
    -------------
    This is an option ONLY for DASW type of mapping. This lets you send out arguments in 
    stages instead of all at once. This is useful if it takes a while to generate the
    arguments. This allows you to send out the arguments as you get them instead of 
    waiting until you have all of the arguments. There are three parts to this:
        `staggered_map_start`: Defines `func` that you are mapping over
        `staggered_map_put`: Send argument/s to the available processes
        `staggered_map_get`: Get all of the arguments when they are ready

    Passing in a graph
    ------------------
    There is an option to pass in a graph. This is useful if you are running
    your code in inference. All this does is sends a pointer to the 
    graph so that if something fails and there is an exception, the graph
    can call `kill` on all the variables that have persistent workers. THIS IS
    NOT A NECESSARY ARGUMENT TO RUN THE MULTIPROCESSING.

    What type of parallelizations
    -----------------------------
    The default ptype is set during initialization of the pool. If you want to do the 
    other type of parallelization then you need to specify it with the `ptype` argument
    in `PersistentPool.map`. You can also change the default parallelization type
    with the function `PersistentPool.change_ptype`.
    
    Pickling
    --------
    During checkpointing, we are unable to serialize multiprocessed objects, so we
    do not include the attributes `tasks`, `results`, or `workers` when we pickle 
    object - thus we cannot load this object

    Parameters
    ----------
    ptype : str
        This defines the default type of parallelization to do.
        Options: 
            'sadw': Same argument, different worker
            'dasw': Different argument, same worker
    G : pylab.graph.Graph, None
        Optional graph to pass in
    '''
    def __init__(self, ptype: str, G: Graph=None):
        if not util.isstr(ptype):
            raise TypeError('`ptype` ({}) must be a str'.format(type(ptype)))
        if ptype not in ['sadw', 'dasw']:
            raise ValueError('`ptype` ({}) not recognized'.format(ptype))

        self.ptype = ptype # This is the type of multiprocessing to do
        self.tasks = multiprocessing.JoinableQueue() # This is a queue for the jobs to do
        self.results = multiprocessing.Queue() # This is a queue of the results
        self.num_workers = 0
        self.busy = False
        self.workers = [] # A list of the persistent objects
        self._worker_pids = set([]) # pids of the workers
        self._staged_running = False

        if G is not None:
            if not isgraph(G):
                raise TypeError('`G` ({}) must be a graph'.format(type(G)))
            G._persistent_pntr.append(self)

    def __len__(self) -> int:
        return len(self.workers)

    def reset(self):
        '''Kill everything if necessary - start as if new
        '''
        if type(self.tasks) == multiprocessing.JoinableQueue:
            self.kill()
        self.tasks = multiprocessing.JoinableQueue()
        self.results = multiprocessing.Queue()
        self.num_workers = 0
        self.busy = False
        self.workers = []

    def add_worker(self, obj: Any) -> int:
        '''Add a persistent worker

        Parameter
        ---------
        args : dict
            These are the persistent arguments for the process

        Returns
        -------
        int
        '''
        try:
            if not ispersistentworker(obj):
                raise TypeError('`obj` ({}) must be a subtype of PersistentWorker'.format(
                    type(obj)))
            self.workers.append(_PersistentWorker(
                task_queue=self.tasks,
                result_queue=self.results,
                obj=copy.deepcopy(obj)))
            self.num_workers += 1
            self.workers[-1].start()
        except:
            if type(self.tasks) == DestroyedFromPickling:
                raise SerializationError('You cannot load this object from a pickle' \
                    ' because `tasks`, `results`, and `workers` are nonserializable.' \
                    ' You must reinitialize it')
            else:
                raise
        pid = self.workers[-1].pid
        self._worker_pids.add(pid)
        return pid

    def change_ptype(self, ptype: str):
        '''Change the default ptype

        Parameters
        ----------
        ptype : str
            This is the parallelization type you are setting
        '''
        if not util.isstr(ptype):
            raise TypeError('`ptype` ({}) must be a str'.format(type(ptype)))
        if ptype not in ['sadw', 'dasw']:
            raise ValueError('`ptype` ({}) not recognized'.format(ptype))
        self.ptype = ptype

    def map(self, func: Callable, args: Union[Dict[int, Dict[str, Any]], Iterator[Dict[str, Any]]], 
        ptype: str=None) -> Iterator[Any]:
        '''Maps the function `func` over the list of arguments `lst`

        Parameters
        ----------
        func : str
            This is the name of the function that is defined in the object
        args : list(dict), dict
            If SADW, it should be a dict
            If DASW, it should be:
                dict(int -> dict)
                    If you care where the arguments get sent, specify the worker's pid for 
                    each set of arguments by mapping the name to the respective worker. 
                    The pid is returned when you add the worker to the pool with `PersistentPool.add_worker`
                list(dict)
                    If you do not care where the arguments get sent to
        ptype : str, optional
            Override the default ptype with this

        Returns
        -------
        list(any)
            Returns a list of results from the function `func`

        See also
        --------
        `pylab.multiprocessing.PersistentPool.change_ptype`
        '''
        if not util.isstr(func):
            self.kill()
            raise TypeError('`func` ({}) must be a str'.format(type(func)))

        if ptype is None:
            ptype = self.ptype
        else:
            if not util.isstr(ptype):
                raise TypeError('`ptype` ({}) must be a str'.format(type(ptype)))
            if ptype not in ['sadw', 'dasw']:
                raise ValueError('`ptype` ({}) not recognized'.format(ptype))

        if ptype == 'sadw':
            if not util.isdict(args):
                self.kill()
                raise TypeError('`args` ({}) must be a dict'.format(type(args)))
            return self._sadw_map(func=func, args=args)
        else:
            if self._staged_running:
                self.kill()
                raise PoolTypeError('staged_map is already running. You must call ' \
                    '`staged_map_get` before you call `map`.')
            if util.isarray(args):
                for arg in args:
                    if not util.isdict(arg):
                        self.kill()
                        raise TypeError('Each arg in `args` ({}) must be a dict'.format(type(arg)))
            elif util.isdict(args):
                for key, val in args.items():
                    if key not in self._worker_pids:
                        raise IndexError('`pid` ({}) not recognized. Valid pids: {}'.format( 
                            key, list(self._worker_pids)))
                    if not util.isdict(val):
                        self.kill()
                        raise TypeError('Each value in `args` ({}) must be a dict'.format(type(val)))
            else:
                raise TypeError('`args` ({}) type not recognized'.format(type(args)))
                    
            return self._dasw_map(func=func, args=args)

    def _sadw_map(self, func, args):
        '''Same argument different workers map
        '''
        try:
            self.busy = True
            ret = []
            
            try:
                for _ in range(self.num_workers):
                    self.tasks.put((func, args))
                self.tasks.join()
                for _ in range(self.num_workers):
                    result = self.results.get()
                    ret.append(result)
                self.busy = False
                return ret
            except:
                self.kill()
                logging.critical('A child threw an error')
                logging.critical('Error: {}'.format(sys.exc_info()[0]))
                raise
        except:
            if type(self.tasks) == DestroyedFromPickling:
                raise SerializationError('You cannot load this object from a pickle' \
                    ' because `tasks`, `results`, and `workers` are nonserializable.' \
                    ' You must reinitialize it')
            else:
                raise

    def _dasw_map(self, func, args):
        '''Different arguemnts same worker map
        '''
        try:
            if type(args) == list:
                # Send the args to each one, dont care
                self.busy = True
                ret = []
                
                try:
                    for i in range(len(args)):
                        self.tasks.put((func, args[i]))
                    self.tasks.join()
                    for _ in range(len(args)):
                        result = self.results.get()
                        ret.append(result)
                    self.busy = False
                    return ret
                except:
                    self.kill()
                    logging.critical('A child threw an error')
                    logging.critical('Error: {}'.format(sys.exc_info()[0]))
                    raise
            else:
                # Send each args to specific processes
                raise NotImplementedError('Not implemented')
        except:
            if type(self.tasks) == DestroyedFromPickling:
                raise SerializationError('You cannot load this object from a pickle' \
                    ' because `tasks`, `results`, and `workers` are nonserializable.' \
                    ' You must reinitialize it')
            else:
                raise

    def staged_map_start(self, func: Callable):
        '''Staged mapping for DASW mappings

        Parameters
        ----------
        func : str
            This is the function name we are calling over the arguments
        '''
        if self.ptype != 'dasw':
            self.kill()
            raise PoolTypeError('Pool type is `{}`, must be DASW'.format(self.ptype))
        if self._staged_running:
            self.kill()
            raise PoolTypeError('staggered_map is already running. You must call ' \
                '`staggered_map_get` before you call it again.')
        self.func = func
        self.n = 0
        self._staged_running = True

    def staged_map_put(self, args: Union[Dict[str, Any], Iterator[Dict[str, Any]]]):
        '''Add arguemnts to the queue

        Parameters
        ----------
        args : list(dict), dict
            Argument/s to send
        '''
        if not self._staged_running:
            self.kill()
            raise PoolTypeError('You must call `staggered_map_start` before this function')
        if util.isdict(args):
            args = [args]
        if not util.isarray(args):
            self.kill()
            raise TypeError('`args` ({}) must be a list or dict'.format(type(args)))
        for arg in args:
            if not util.isdict(arg):
                self.kill()
                raise TypeError('Each arg in `args` ({}) must be a dict'.format(type(arg)))
        try:
            for arg in args:
                self.tasks.put((self.func, arg))
            self.n += len(args)
        except:
            self.kill()
            if type(self.tasks) == DestroyedFromPickling:
                raise SerializationError('You cannot load this object from a pickle' \
                    ' because `tasks`, `results`, and `workers` are nonserializable.' \
                    ' You must reinitialize it')
            else:
                logging.critical('A child threw an error')
                logging.critical('Error: {}'.format(sys.exc_info()[0]))
                raise

    def staged_map_get(self, timeout: float=None) -> Iterator[Any]:
        '''Get the results of a staged mapping

        Returns
        -------
        list
        '''
        if not self._staged_running:
            self.kill()
            raise PoolTypeError('You must call `staggered_map_start` before this function')
        
        try:
            self.tasks.join()
        except KeyboardInterrupt:
            self.kill()
            raise
        ret = []
        for i in range(self.n):
            ret.append(self.results.get())
        
        self.n = None
        self.func = None
        self._staged_running = False
        return ret

    def kill(self):
        '''Kill all of the workers
        '''
        try:
            for worker in self.workers:
                os.kill(worker.pid, signal.SIGTERM)
        except:
            if type(self.tasks) == DestroyedFromPickling:
                raise SerializationError('You cannot load this object from a pickle' \
                    ' because `tasks`, `results`, and `workers` are nonserializable.' \
                    ' You must reinitialize it')
            else:
                raise

    def __getstate__(self) -> Dict[str, Any]:
        '''Do not include the multiprocessed objects during serialization
        '''
        state = self.__dict__.copy()
        state.pop('tasks')
        state.pop('results')
        state.pop('workers')
        return state

    def __setstate__(self, state: Dict[str, Any]):
        '''Set the multiprocessed objects as destroyed from serialization
        '''
        self.__dict__.update(state)
        self.workers = DestroyedFromPickling
        self.results = DestroyedFromPickling
        self.tasks = DestroyedFromPickling
        self.reset()


class _PersistentWorker(multiprocessing.Process):
    '''Custom worker class for running a process with persistent args
    and process location.

    Parameters
    ----------
    task_queue : multiprocessing.JoinableQueue((int, callable, dict))
        These are the tasks to get with some extra information:
        int 
            This is the index to put the result in the result queue
        callable
            This is the function to call
        dict
            These are the kwargs for the function
    result_queue : multiprocessing.Queue
        These are where the results go (int, any)
    obj : PersistentWorker
        Object to keep here and call `persistent_run`
    '''
    def __init__(self, task_queue: multiprocessing.JoinableQueue, 
        result_queue: multiprocessing.Queue, obj: PersistentWorker):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.obj = obj

    def run(self):
        '''Wait until there is a task for it to do. When there is,
        get it and put the result in the result_queue.
        '''
        while True:
            new_args = self.task_queue.get()
            if new_args is None:
                # Poison pill means shutdown
                # logging.critical('{}: Exiting'.format(self.name))
                self.task_queue.task_done()
                break
            func, kwargs = new_args

            # Push the exception if one is thrown
            try:
                answer = getattr(self.obj, func)(**kwargs)
            except:
                self.task_queue.task_done()
                self.result_queue.put(None)
                logging.critical('Child {} ({}) failed on function `{}`'.format(
                    self.name, os.getpid(), func))
                raise

            self.task_queue.task_done()
            self.result_queue.put(answer)
        return


class DestroyedFromPickling:
    def __init__(self):
        self.value = 'Must reinitialize'


class SerializationError(Exception):
    pass


class PoolTypeError(Exception):
    pass