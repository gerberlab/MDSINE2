import os
from typing import Union, Any

import numpy as np
import pickle
from pathlib import Path

from .. import util as plutil
from .graph import Node


class Saveable:
    '''Implements baseline saving classes with pickle for classes
    '''
    def save(self, filename: Union[str, Path]=None):
        '''Pickle the object

        Paramters
        ---------
        filename : str
            This is the location to store the file. Overrides the location if
            it is set using `pylab.base.Saveable.set_save_location`. If None
            it means that we are using the file location set in
            set_location.
        '''
        if filename is None:
            if not hasattr(self, '_save_loc'):
                raise TypeError('`filename` must be specified if you have not set the save location')
            filename = self._save_loc

        if isinstance(filename, Path):
            filename = Path(filename)

        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename: str):
        '''Unpickle the object

        Paramters
        ---------
        cls : type
            Type
        filename : str
            This is the location of the file to unpickle
        '''
        with open(str(filename), 'rb') as handle:
            b = pickle.load(handle)

        # redo the filename to the new path if it has a save location
        if not hasattr(b, '_save_loc'):
            filename = os.path.abspath(filename)
            b._save_loc = filename

        return b

    def set_save_location(self, filename: str):
        '''Set the save location for the object.

        Internally converts this to the absolute path

        Parameters
        ----------
        filename : str
            This is the path to set it to
        '''
        if not plutil.isstr(filename):
            raise TypeError('`filename` ({}) must be a str'.format(type(filename)))
        filename = os.path.abspath(filename)
        self._save_loc = filename

    def get_save_location(self) -> str:
        try:
            return self._save_loc
        except:
            raise AttributeError('Save location is not set.')


class TraceableNode(Node):
    '''
    Defines the functionality for a Node to interact with the Graph tracer object
    '''
    def __init__(self, dtype, **kwargs):
        super().__init__(**kwargs)
        self.dtype = dtype

    def set_trace(self):
        '''Initialize the trace arrays for the variable in the Tracer object.

        It will initialize a buffer the size of the checkpoint size in Tracer
        '''
        raise NotImplementedError('User needs to define this function')

    def add_trace(self):
        '''Adds the current value to the trace. If the buffer is full
        it will end it to disk
        '''
        raise NotImplementedError('User needs to define this function')

    def get_trace_from_disk(self, section: str = 'posterior', slices: slice = None) -> np.ndarray:
        '''Returns the entire trace (after burnin) writen on the disk. NOTE: This may/may not
        include the samples in the local buffer trace and could be very large

        Parameters
        ----------
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
        return self.G.tracer.get_trace(name=self.name, section=section, slices=slices)

    def overwrite_entire_trace_on_disk(self, data: np.ndarray, **kwargs):
        '''Overwrites the entire trace of the variable with the given data.

        Parameters
        ----------
        data : np.ndarray
            Data you are overwriting the trace with.
        '''
        self.G.tracer.overwrite_entire_trace_on_disk(
            name=self.name, data=data, dtype=self.dtype, **kwargs
        )

    def get_iter(self) -> int:
        '''Get the number of iterations saved to the hdf5 file of the variable

        Returns
        -------
        int
        '''
        return self.G.tracer.get_iter(name=self.name)


def issavable(x: Any) -> bool:
    '''Checks whether the input is a subclass of Savable

    Parameters
    ----------
    x : any
        Input instance to check the type of Savable

    Returns
    -------
    bool
        True if `x` is of type Savable, else False
    '''
    return x is not None and issubclass(x.__class__, Saveable)


def istraceable(x: Any) -> bool:
    '''Checks whether the input is a subclass of Traceable

    Parameters
    ----------
    x : any
        Input instance to check the type of Traceable

    Returns
    -------
    bool
        True if `x` is of type Traceable, else False
    '''
    return x is not None and issubclass(x.__class__, TraceableNode)
