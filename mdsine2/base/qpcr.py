from typing import Union
import numpy as np


class qPCRdata:
    '''Single entry of qpcr data at a timepoint with maybe multiple technical replicates.
    Assumes that the dilution factor is constant between the replicate runs

    The normalized data is assumed to be:
        (cfus * dilution_factor / mass) * scaling_factor

    scaling_factor is a scale that we impose on the data so that the numbers don't get
    super large in the numerical calculations and we get errors, it does nothing to affect
    the empirical variance of the data.

    Parameters
    ----------
    cfus : np.ndarray
        These are the raw CFUs - it can be a single CFU measurement or a list of all
        the measurements
    mass : float
        This is the mass of the sample in grams
    dilution_factor : float
        This is the dilution factor of the samples
        Example:
            If the sample was diluted to 1/100 of its original concentration,
            the dilution factor is 100, NOT 1/100.

    '''
    def __init__(self, cfus: np.ndarray, mass: float=1., dilution_factor: float=1.):
        self._raw_data = np.asarray(cfus) # array of raw CFU values
        self.mass = mass
        self.dilution_factor = dilution_factor
        self.scaling_factor = 1 # Initialize with no scaling factor
        self.recalculate_parameters()

    def recalculate_parameters(self):
        '''Generate the normalized abundances and recalculate the statistics
        '''
        if len(self._raw_data) == 0:
            return

        self.data = (self._raw_data*self.dilution_factor/self.mass)*self.scaling_factor # array of normalized values
        self.log_data = np.log(self.data)

        self.loc = np.mean(self.log_data)

        if len(self._raw_data) == 1:
            self.scale = 0
            self.scale2 = 0
        else:
            self.scale = np.std(self.log_data - self.loc)
            self.scale2 = self.scale ** 2


        self._mean_dist = np.exp(self.loc + (self.scale2/2) )
        self._var_dist = (np.exp(self.scale2) - 1) * np.exp(2*self.loc + self.scale2)
        self._std_dist = np.sqrt(self._var_dist)
        self._gmean = (np.prod(self.data))**(1/len(self.data))

    def __str__(self) -> str:
        s = 'cfus: {}\nmass: {}\ndilution_factor: {}\n scaling_factor: {}\n' \
            'data: {}\nlog_data: {}\nloc: {}\n scale: {}'.format(
                self._raw_data, self.mass, self.dilution_factor, self.scaling_factor,
                self.data, self.log_data, self.loc, self.scale)
        return s

    def add(self, raw_data: Union[np.ndarray, float, int]):
        '''Add a single qPCR measurement to add to the set of observations

        Parameters
        ----------
        raw_data : float, array_like
            This is the measurement to add
        '''
        self._raw_data = np.append(self._raw_data,raw_data)
        self.recalculate_parameters()

    def set_to_nan(self):
        '''Set all attributes to `np.nan`
        '''
        self._raw_data *= np.nan
        self.data *= np.nan
        self.mass = np.nan
        self.dilution_factor = np.nan
        self._mean_dist = np.nan
        self._std_dist = np.nan
        self._var_dist = np.nan
        self._gmean = np.nan
        self.loc = np.nan
        self.scale = np.nan
        self.scale2 = np.nan
        self.scaling_factor = np.nan

    def set_scaling_factor(self, scaling_factor: float):
        '''Resets the scaling factor

        Parameters
        ----------
        scaling_factor : float, int
            This is the scaling factor to set everything to
        '''
        if scaling_factor <= 0:
            raise ValueError('The scaling factor must strictly be positive')
        self.scaling_factor = scaling_factor
        self.recalculate_parameters()

    def mean(self, qpcr_unnormalize: bool = False) -> float:
        '''Return the geometric mean
        '''
        ans = self.gmean()
        if qpcr_unnormalize:
            ans = (1 / self.scaling_factor) * ans
        return ans

    def var(self) -> float:
        return self._var_dist

    def std(self) -> float:
        return self._std_dist

    def gmean(self) -> float:
        return self._gmean
