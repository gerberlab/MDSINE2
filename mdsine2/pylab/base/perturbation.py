from typing import Dict, Union
from .taxa import Subject
from .. import util as plutil


class BasePerturbation:
    '''Base perturbation class.

    Does not have to be applied to all subjects, and each subject can have a different start and
    end time to each other.

    Parameters
    ----------
    name : str, None
        - This is the name of the perturabtion. If nothing is given then the name will be
          set to the perturbation index
    starts, ends : dict, None
        - This is a map to the start and end times for the subject that have this perturbation
    '''
    def __init__(self, name: str, starts: Dict[str, float]=None, ends: Dict[str, float]=None):
        if not plutil.isstr(name):
            raise TypeError('`name` ({}) must be a str'.format(type(name)))
        if (starts is not None and ends is None) or (starts is None and ends is not None):
            raise ValueError('If `starts` or `ends` is specified, the other must be specified.')
        if starts is not None:
            if not plutil.isdict(starts):
                raise TypeError('`starts` ({}) must be a dict'.format(starts))
            if not plutil.isdict(ends):
                raise TypeError('`ends` ({}) must be a dict'.format(ends))

        self.starts = starts
        self.ends = ends
        self.name = name

    def __str__(self) -> str:
        s = 'Perturbation {}:\n'.format(self.name)
        if self.starts is not None:
            for subj in self.starts:
                s += '\tSubject {}: ({}, {})\n'.format(subj, self.starts[subj],
                    self.ends[subj])
        return s

    def __contains__(self, a: Union[str]) -> bool:
        '''Checks if subject name `a` is in this perturbation
        '''
        if isinstance(a, Subject):
            a = a.name
        return a in self.starts

    def isactive(self, time: Union[float, int], subj: str) -> bool:
        '''Returns a `bool` if the perturbation is on at time `time`.

        Parameters
        ----------
        time : float, int
            Time to check
        subj : str
            Subject to check

        Returns
        -------
        bool

        Raises
        ------
        ValueError
            If there are no start and end times set
        '''
        if self.starts is None:
            raise ValueError('`start` is not set in {}'.format(self.name))
        try:
            start = self.starts[subj]
            end = self.ends[subj]
        except:
            raise KeyError('`subj` {} not specified for {}'.format(subj, self.name))

        return time > start and time <= end


class Perturbations:
    '''Aggregator for individual perturbation obejcts
    '''
    def __init__(self):
        self._d = {}
        self._rev_idx = []

    def __len__(self) -> int:
        return len(self._d)

    def __getitem__(self, a: Union[BasePerturbation, int, str]) -> BasePerturbation:
        '''Get the perturbation either by index, name, or object
        '''
        if isinstance(a, BasePerturbation):
            if a.name in self:
                return a
            else:
                raise KeyError('`a` ({}) not contained in this Set'.format(a))
        if plutil.isstr(a):
            return self._d[a]
        elif plutil.isint(a):
            return self._d[self._rev_idx[a]]
        else:
            raise KeyError('`a` {} ({}) not recognized'.format(a, type(a)))

    def __contains__(self, a: Union[BasePerturbation, str, int]) -> bool:
        try:
            _ = self[a]
            return True
        except:
            False

    def __iter__(self):
        for a in self._d:
            yield self._d[a]

    def append(self, a: BasePerturbation):
        '''Add a perturbation

        a : mdsine2.BasePertubration
            Perturbation to add
        '''
        self._d[a.name] = a
        self._rev_idx.append(a.name)

    def remove(self, a: Union[BasePerturbation, str, int]):
        '''Remove the perturbation `a`. Can be either the name, index, or
        the object itself.

        Parameters
        ----------
        a : str, int, mdsine2.BasePerturbation
            Perturbation to remove

        Returns
        -------
        mdsine2.BasePerturbation
        '''
        a = self[a]
        self._d.pop(a.name, None)
        self._rev_idx = []
        for mer in self._d:
            self._rev_idx.append(mer.name)
        return a
