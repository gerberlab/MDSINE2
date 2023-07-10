from typing import Union, Dict, Tuple, List
import numpy as np
import pandas as pd

from mdsine2.pylab import util as plutil
from mdsine2.pylab import Saveable

from .perturbation import Perturbations
from .taxa import OTU, Taxon, TaxaSet
from .constants import *
from .qpcr import qPCRdata
from .util import taxaname_formatter

from mdsine2.logger import logger


class Subject(Saveable):
    """Data for a single subject
    The TaxaSet order is done with respect to the ordering in the `reads_table`

    Parameters
    ----------
    parent : Study
        This is the parent class (we have a reverse pointer)
    name : str
        This is the name of the subject
    """
    def __init__(self, parent: 'Study', name: str, use_spikein=False):
        self.name = name # str
        self.id = id(self)
        self.parent = parent
        self.qpcr = {} # dict: time (float) -> qpcr object (qPCRData)
        self.reads = {} # dict: time (float) -> reads (np.ndarray)
        self.times = np.asarray([]) # times in order
        self._reads_individ = {} # for taking out aggregated taxa
        self.use_spikein = use_spikein
        
    def add_time(self, timepoint: Union[float, int]):
        """Add the timepoint `timepoint`. Set the reads and qpcr at that timepoint
        to None

        Parameters
        ----------
        timepoint : float, int
            Time point to add
        """
        if timepoint in self.times:
            return
        self.times = np.sort(np.append(self.times, timepoint))
        # self.reads[timepoint] = None
        # self.qpcr[timepoint] = None

    def add_reads(self, timepoints: Union[np.ndarray, int, float], reads: np.ndarray):
        """Add the reads for timepoint `timepoint`

        Parameters
        ----------
        timepoint : numeric, array
            This is the time that the measurement occurs. If it is an array, then
            we are adding for multiple timepoints
        reads : np.ndarray(N_TAXA, N_TIMEPOINTS)
            These are the reads for the taxa in order. Assumed to be in the
            same order as the TaxaSet. If it is a dataframe then we use the rows
            to index the taxon names. If timepoints is an array, then we are adding
            for multiple timepoints. In this case we assume that the rows index  the
            taxon and the columns index the timepoint.
        """
        if not plutil.isarray(timepoints):
            timepoints = [timepoints]
        for timepoint in timepoints:
            if not plutil.isnumeric(timepoint):
                raise TypeError('`timepoint` ({}) must be a numeric'.format(type(timepoint)))
        if not plutil.isarray(reads):
            raise TypeError('`reads` ({}) must be an array'.format(type(reads)))

        if reads.ndim == 1:
            reads = reads.reshape(-1,1)
        if reads.ndim != 2:
            raise ValueError('`reads` {} must be a matrix'.format(reads.shape))
        if reads.shape[0] != len(self.taxa) or reads.shape[1] != len(timepoints):
            raise ValueError('`reads` shape {} does not align with the number of taxa ({}) ' \
                'or timepoints ({})'.format(reads.shape, len(self.taxa), len(timepoints)))

        for tidx, timepoint in enumerate(timepoints):
            if timepoint in self.reads:
                if self.reads[timepoint] is not None:
                    logger.debug('There are already reads specified at time `{}` for subject `{}`, overwriting'.format(
                        timepoint, self.name))

            self.reads[timepoint] = reads[:,tidx]
            if timepoint not in self.times:
                self.times = np.sort(np.append(self.times, timepoint))
        return self

    def add_qpcr(self,
                 timepoints: Union[np.ndarray, int, float],
                 qpcr: np.ndarray,
                 masses: Union[np.ndarray, int, float]=None,
                 dilution_factors: Union[np.ndarray, int, float]=None):
        """Add qpcr measurements for timepoints `timepoints`

        Parameters
        ----------
        timepoint : numeric, array
            This is the time that the measurement occurs. If it is an array, then
            we are adding for multiple timepoints
        qpcr : np.ndarray(N_TIMEPOINTS, N_REPLICATES)
            These are the qPCR measurements in order of timepoints. Assumed to be in the
            same order as timepoints.If timepoints is an array, then we are adding
            for multiple timepoints. In this case we assume that the rows index the
            timepoint and the columns index the replicates of the qpcr measurement.
        masses : numeric, np.ndarray
            These are the masses for each on of the qPCR measurements. If this is not
            specified, then this assumes that the numbers in `qpcr` are already normalized
            by their sample weight.
        dilution_factors : numeric, np.ndarray
            These are the dilution factors for each of the qPCR measurements. If this is
            not specified, then this assumes that each one of the numbers in `qpcr` are
            already normalized by the dilution factor
        """
        if not plutil.isarray(timepoints):
            timepoints = [timepoints]
        for timepoint in timepoints:
            if not plutil.isnumeric(timepoint):
                raise TypeError('`timepoint` ({}) must be a numeric'.format(type(timepoint)))
        if masses is not None:
            if plutil.isnumeric(masses):
                masses = [masses]
            for mass in masses:
                if not plutil.isnumeric(mass):
                    raise TypeError('Each mass in `masses` ({}) must be a numeric'.format(type(mass)))
                if mass <= 0:
                    raise ValueError('Each mass in `masses` ({}) must be > 0'.format(mass))
            if len(masses) != len(timepoints):
                raise ValueError('Number of timepoints ({}) and number of masses ({}) ' \
                    'must be equal'.format(len(timepoints), len(masses)))
        if dilution_factors is not None:
            if plutil.isnumeric(dilution_factors):
                dilution_factors = [dilution_factors]
            for dilution_factor in dilution_factors:
                if not plutil.isnumeric(dilution_factor):
                    raise TypeError('Each dilution_factor in `dilution_factors` ({}) ' \
                        'must be a numeric'.format(type(dilution_factor)))
                if dilution_factor <= 0:
                    raise ValueError('Each dilution_factor in `dilution_factors` ({}) ' \
                        'must be > 0'.format(dilution_factor))
            if len(dilution_factors) != len(timepoints):
                raise ValueError('Number of timepoints ({}) and number of dilution_factors ({}) ' \
                    'must be equal'.format(len(timepoints), len(dilution_factors)))

        if not plutil.isarray(qpcr):
            raise TypeError('`qpcr` ({}) must be an array'.format(type(qpcr)))
        if qpcr.ndim == 1:
            qpcr = qpcr.reshape(1,-1)
        if qpcr.ndim != 2:
            raise ValueError('`qpcr` {} must be a matrix'.format(qpcr.shape))
        if qpcr.shape[0] != len(timepoints):
            raise ValueError('`qpcr` shape {} does not align with the number of timepoints ({}) ' \
                ''.format(qpcr.shape, len(timepoints)))

        for tidx, timepoint in enumerate(timepoints):
            if timepoint in self.qpcr:
                if self.qpcr[timepoint] is not None:
                    logger.debug('There are already qpcr measurements specified at time `{}` for subject `{}`, overwriting'.format(
                        timepoint, self.name))
            if masses is not None:
                mass = masses[tidx]
            else:
                mass = 1
            if dilution_factors is not None:
                dil = dilution_factors[tidx]
            else:
                dil = 1

            self.qpcr[timepoint] = qPCRdata(cfus=qpcr[tidx,:], mass=mass,
                dilution_factor=dil)

            if timepoint not in self.times:
                self.times = np.sort(np.append(self.times, timepoint))
        return self

    @property
    def perturbations(self) -> Perturbations:
        return self.parent.perturbations

    @property
    def taxa(self) -> TaxaSet:
        return self.parent.taxa

    @property
    def index(self) -> int:
        """Return the index of this subject in the Study file
        """
        for iii, subj in enumerate(self.parent):
            if subj.name == self.name:
                return iii
        raise ValueError('Should not get here')

    def matrix(self) -> Dict[str, np.ndarray]:
        """Make a numpy matrix out of our data - returns the raw reads,
        the relative abundance, and the absolute abundance.

        If there is no qPCR (or spikein) data, then the absolute abundance is set to None.
        """
        shape = (len(self.taxa), len(self.times))
        raw = np.zeros(shape=shape, dtype=int)
        rel = np.zeros(shape=shape, dtype=float)
        abs = np.zeros(shape=shape, dtype=float)

        # Raw reads of the actual (non-spikein) taxa
        for i,t in enumerate(self.times):
            raw[:,i] = self.reads[t]
            rel[:, i] = raw[:, i] / np.sum(raw[:, i])
            # rel[:, i] = raw[:, i] / (np.sum(raw[:, i]) + self.spikein_reads[t].sum())

        if self.use_spikein:
            if len(self.spikein_reads) > 0:
                for i,t in enumerate(self.times):
                    rel[:, i] = raw[:, i] / (np.sum(raw[:, i]) + self.spikein_reads[t].sum())
                    abs[:,i] = self.spikein_abundance_observed[t] / self.spikein_reads[t].sum() * raw[:,i]
                    # abs[:,i] = rel[:,i] * self.qpcr[t].mean()
        else:
            if len(self.qpcr) > 0:
                for i,t in enumerate(self.times):
                    abs[:,i] = rel[:,i] * self.qpcr[t].mean()

        return {'raw':raw, 'rel': rel, 'abs':abs}

    def df(self) -> Dict[str, pd.DataFrame]:
        """Returns a dataframe of the data - same as matrix
        """
        d = self.matrix()
        index = self.taxa.names.order
        times = self.times
        for key in d:
            d[key] = pd.DataFrame(data=d[key], index=index, columns=times)
        return d

    def read_depth(self, t: Union[int, float]=None) -> Union[np.ndarray, int]:
        """Get the read depth at time `t`. If nothing is given then return all
        of them

        Parameters
        ----------
        t : int, float, Optional
            Get the read depth at this time. If nothing is provided, all of the read depths for this
            subject are returned
        """
        if t is None:
            return np.sum(self.matrix()['raw'], axis=0)
        if t not in self.reads:
            raise ValueError('`t` ({}) not recognized. Valid times: {}'.format(
                t, self.times))
        return np.sum(self.reads[t])

    def cluster_by_taxlevel(self, dtype: str, taxlevel: str, index_formatter: str=None,
        smart_unspec: bool=True) -> Tuple[pd.DataFrame, Dict[str,str]]:
        """Clusters the taxa into the taxonomic level indicated in `taxlevel`.

        Smart Unspecified
        -----------------
        If True, returns the higher taxonomic classification while saying the desired taxonomic level
        is unspecified. Example: 'Order ABC, Family NA'. Note that this overrides the `index_formatter`.

        Parameters
        ----------
        dtype : str
            This is the type of data to cluster. Options are:
                'raw': These are the counts
                'rel': This is the relative abundances
                'abs': This is the absolute abundance (qPCR * rel)
        taxlevel : str, None
            This is the taxonomic level to aggregate the data at. If it is
            None then we do not do any collapsing (this is the same as 'asv')
        index_formatter : str
            How to make the index using `taxaname_formatter`. Note that you cannot
            specify anything at a lower taxonomic level than what youre clustering at. For
            example, you cannot cluster at the 'class' level and then specify '%(genus)s'
            in the index formatter.
            If nothing is specified then only return the specified taxonomic level
        smart_unspec : bool
            If True, if the taxonomic level is not not specified for that OTU/Taxon, then use the
            lowest taxonomic level instead.

        Returns
        -------
        pandas.DataFrame
            Dataframe of the data
        dict (str->str)
            Maps taxon name to the row it got allocated to
        """
        # Type checking
        if not plutil.isstr(dtype):
            raise TypeError('`dtype` ({}) must be a str'.format(type(dtype)))
        if dtype not in ['raw', 'rel', 'abs']:
            raise ValueError('`dtype` ({}) not recognized'.format(dtype))
        if not plutil.isstr(taxlevel):
            raise TypeError('`taxlevel` ({}) must be a str'.format(type(taxlevel)))
        if taxlevel not in ['kingdom', 'phylum', 'class',  'order', 'family',
            'genus', 'species', 'asv']:
            raise ValueError('`taxlevel` ({}) not recognized'.format(taxlevel))
        if index_formatter is None:
            index_formatter = taxlevel
        if index_formatter is not None:
            if not plutil.isstr(index_formatter):
                raise TypeError('`index_formatter` ({}) must be a str'.format(type(index_formatter)))

            for tx in TAX_IDXS:
                if tx in index_formatter and TAX_IDXS[tx] > TAX_IDXS[taxlevel]:
                    raise ValueError('You are clustering at the {} level but are specifying' \
                        ' {} in the `index_formatter`. This does not make sense. Either cluster' \
                        'at a lower tax level or specify the `index_formatter` to a higher tax ' \
                        'level'.format(taxlevel, tx))

        index_formatter = index_formatter.replace('%(asv)s', '%(name)s')

        # Everything is valid, get the data dataframe and the return dataframe
        taxaname_map = {}
        df = self.df()[dtype]
        cols = list(df.columns)
        cols.append(taxlevel)
        dfnew = pd.DataFrame(columns = cols).set_index(taxlevel)

        # Get the level in the taxonomy, create a new entry if it is not there already
        taxa = {} # lineage -> label
        for i, taxon in enumerate(self.taxa):
            row = df.index[i]
            tax = taxon.get_lineage(level=taxlevel)
            tax = tuple(tax)
            tax = str(tax).replace("'", '')
            if tax in taxa:
                dfnew.loc[taxa[tax]] += df.loc[row]
            else:
                if not taxon.tax_is_defined(taxlevel) and smart_unspec:
                    # Get the least common ancestor above the taxlevel
                    taxlevelidx = TAX_IDXS[taxlevel]
                    ttt = None
                    while taxlevelidx > -1:
                        if taxon.tax_is_defined(TAX_LEVELS[taxlevelidx]):
                            ttt = TAX_LEVELS[taxlevelidx]
                            break
                        taxlevelidx -= 1
                    if ttt is None:
                        raise ValueError('Could not find a single taxlevel: {}'.format(str(taxon)))
                    taxa[tax] = '{} {}, {} NA'.format(ttt.capitalize(),
                        taxon.taxonomy[ttt], taxlevel.capitalize())
                else:
                    taxa[tax] = taxaname_formatter(format=index_formatter, taxon=taxon, taxa=self.taxa)
                toadd = pd.DataFrame(np.array(list(df.loc[row])).reshape(1,-1),
                    index=[taxa[tax]], columns=dfnew.columns)
                dfnew = dfnew.append(toadd)

            if taxa[tax] not in taxaname_map:
                taxaname_map[taxa[tax]] = []
            taxaname_map[taxa[tax]].append(taxon.name)

        return dfnew, taxaname_map

    def _split_on_perturbations(self):
        """If there are perturbations, then we take out the data on perturbations
        and we set the data in the different segments to different subjects

        Internal funciton, should not be used by the user
        """
        if len(self.parent.perturbations) == 0:
            logger.info('No perturbations to split on, do nothing')
            return

        # Get the time intervals for each of the times that we are not on perturbations
        start_tidx = 0
        not_perts = []
        in_pert = False
        for i in range(len(self.times)):
            # check if the time is in a perturbation
            a = False
            for pert in self.parent.perturbations:
                if self.name not in pert:
                    continue
                start = pert.starts[self.name]
                end = pert.ends[self.name]
                # check if in the perturbation
                if self.times[i] > start and self.times[i] <= end:
                    a = True
                    break
            if a:
                # If the current time point is in a perturbation and we previously
                # have no been in a perturbation, this means we can add the previous
                # interval into the intervals that we want to keep
                if not in_pert:
                    not_perts.append((start_tidx, i))
                in_pert = True
            else:
                # If we are not currently in a perturbation but we previously were
                # then we restart to `start_tidx`
                if in_pert:
                    start_tidx = i
                    in_pert = False
        # If we have finished and we are out of a perturbation at the end, then
        # we can add the rest of the times at the end to a valid not in perturbation time
        if not in_pert:
            not_perts.append((start_tidx, len(self.times)))

        # For each of the time slices recorded, make a new subject
        if len(in_pert) == 0:
            raise ValueError('THere are perturbations ({}), this must not be zero.' \
                ' Something went wrong'.format(len(self.parent.perturbations)))
        ii = 0
        for start,end in not_perts:
            mid = self.name+'_{}'.format(ii)
            self.parent.add_subject(name=mid)
            for i in range(start,end):
                t = self.times[i]
                self.parent[mid].qpcr[t] = self.qpcr[t]
                self.parent[mid].reads[t] = self.reads[t]
            self.parent[mid].times = self.times[start:end]

    def _deaggregate_item(self, agg: OTU, other: str):
        """Deaggregate the sequence `other` from OTU `agg`.
        `other` is then appended to the end. This is called from
        `mdsine2.Study.deaggregate_item`.

        Parameters
        ----------
        agg : OTU
            This is an OTU with multiple sequences contained. Must
            have the name `other` in there
        other : str
            This is the name of the taxon that should be taken out of `agg`
        """
        # Append the reads of the deaggregated at the bottom and subtract them
        # from the aggregated index
        if other not in self._reads_individ:
            raise ValueError('`other` ({}) reads not found in archive. This probably ' \
                'happened because you called `aggregate_items` from the TaxaSet object' \
                ' instead from this object. Study object not consistent. Failing.'.format(other))

        aggidx = agg.idx
        for t in self.times:
            try:
                new_reads = self._reads_individ[other][t]
            except:
                raise ValueError('Timepoint `{}` added into subject `{}` after ' \
                    'Taxon `{}` was removed. Study object is not consistent. You ' \
                    'cannot add in other timepoints after you aggregate taxa. Failing.'.format(
                        t, self.name, other))
            self.reads[t][aggidx] = self.reads[t][aggidx] - new_reads
            self.reads[t] = np.append(self.reads[t], new_reads)
        self._reads_individ.pop(other)
        return

    def aggregate_items(self, study: 'Study', taxon_components: List[List[Taxon]]) -> 'Subject':
        """
        Aggregate the taxon `other` into `anchor`. This is called from
        `mdsine2.Study.aggregate_items`.

        Parameters
        ----------
        anchor, other : OTU, Taxon
            These are the s to combine
        """
        agg_subj = Subject(parent=study, name=self.name)
        agg_subj.times = self.times
        agg_subj.qpcr = self.qpcr
        agg_subj._reads_individ = {}

        for t in self.times:
            agg_subj.reads[t] = np.zeros(len(taxon_components), dtype=int)
            for aidx, components in enumerate(taxon_components):
                subset_idxs = [taxon.idx for taxon in components]
                agg_subj.reads[t][aidx] = np.sum(self.reads[t][subset_idxs])

        for otu_list in taxon_components:
            for taxon in otu_list:
                agg_subj._reads_individ[taxon.name] = {
                    t: self.reads[t][taxon.idx]
                    for t in self.times
                }

        return agg_subj
