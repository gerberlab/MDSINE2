import copy
from typing import Union, Tuple, Dict, Iterable, List
import numpy as np
import pandas as pd

from mdsine2.pylab import Saveable
from .taxa import TaxaSet, OTU, Taxon, OTUTaxaSet
from .subject import Subject
from .perturbation import Perturbations, BasePerturbation
from mdsine2.pylab import util as plutil
from mdsine2.logger import logger

class Study(Saveable):
    """Holds data for all the subjects

    Paramters
    ---------
    taxa : TaxaSet, Optional
        Contains all of the s
    """
    def __init__(self, taxa: TaxaSet, name: str='unnamed-study'):
        self.name = name
        self.id = id(self)
        self._subjects = {}
        self.perturbations = None
        self.qpcr_normalization_factor = None
        self.taxa = taxa

        self._samples = {}

    def __getitem__(self, key: Union[str, int, Subject]) -> Subject:
        if plutil.isint(key):
            name = self.names()[key]
            return self._subjects[name]
        elif plutil.isstr(key):
            return self._subjects[key]
        elif isinstance(key, Subject):
            if key.name not in self:
                raise ValueError('Subject not found in study ({})'.format(key.name))
        else:
            raise KeyError('Key ({}) not recognized'.format(type(key)))

    def __len__(self) -> int:
        return len(self._subjects)

    def __iter__(self) -> Subject:
        for v in self._subjects.values():
            yield v

    def __contains__(self, key: Union[str, int, Subject]) -> bool:
        if plutil.isint(key):
            return key < len(self)
        elif plutil.isstr(key):
            return key in self._subjects
        elif isinstance(key, Subject):
            return key.name in self._subjects
        else:
            raise KeyError('Key ({}) not recognized'.format(type(key)))

    def parse(self, metadata: pd.DataFrame, reads: pd.DataFrame=None, qpcr: pd.DataFrame=None,
        perturbations: pd.DataFrame=None):
        """Parse tables of samples and cast in Subject sets. Automatically creates
        the subject classes with the respective names.

        Parameters
        ----------
        metadata : pandas.DataFrame
            Contains the meta data for each one of the samples
            Columns:
                'sampleID' -> str : This is the name of the sample
                'subject' -> str : This is the name of the subject
                'time' -> float : This is the time the sample takes place
                'perturbation:`name`' -> int : This is a perturbation meta data where the
                    name of the perturbation is `name`
        reads : pandas.DataFrame, None
            Contains the reads for each one of the samples and taxa
                index (str) : indexes the taxon name
                columns (str) : indexes the sample ID
            If nothing is passed in, the reads are set to None
        qpcr : pandas.DataFrame, None
            Contains the qpcr measurements for each sample
                index (str) : indexes the sample ID
                columns (str) : Name is ignored. the values are set to the measurements
        perturbations : pandas.DataFrame, None
            Contains the times and subjects for each perturbation
            columns:
                'name' -> str : Name of the perturbation
                'start' -> float : This is the start time for the perturbation
                'end' -> float : This is the end time for the perturbation
                'subject' -> str : This is the subject name the perturbation is applied to
        """
        if not plutil.isdataframe(metadata):
            raise TypeError('`metadata` ({}) must be a pandas.DataFrame'.format(type(metadata)))

        # Add the samples
        # ---------------
        if 'sampleID' in metadata.columns:
            metadata = metadata.set_index('sampleID')
        for sampleid in metadata.index:

            sid = str(metadata['subject'][sampleid])
            t = float(metadata['time'][sampleid])

            if sid not in self:
                self.add_subject(name=sid)
            if t not in self[sid].times:
                self[sid].add_time(timepoint=t)

            self._samples[str(sampleid)] = (sid,t)

        # Add the perturbations if there are any
        # --------------------------------------
        if perturbations is not None:
            logger.debug('Reseting perturbations')
            self.perturbations = Perturbations()
            if not plutil.isdataframe(perturbations):
                raise TypeError('`metadata` ({}) must be a pandas.DataFrame'.format(type(metadata)))
            try:
                for pidx in perturbations.index:
                    pname = perturbations['name'][pidx]
                    subj = str(perturbations['subject'][pidx])

                    if pname not in self.perturbations:
                        # Create a new one
                        pert = BasePerturbation(
                            name=pname,
                            starts={subj: perturbations['start'][pidx]},
                            ends={subj: perturbations['end'][pidx]})
                        self.perturbations.append(pert)
                    else:
                        # Add this subject name to the pertubration
                        self.perturbations[pname].starts[subj] = perturbations['start'][pidx]
                        self.perturbations[pname].ends[subj] = perturbations['end'][pidx]
            except KeyError as e:
                logger.critical(e)
                raise KeyError('Make sure that `subject`, `start`, and `end` are columns')

        # Add the reads if necessary
        # --------------------------
        if reads is not None:
            if not plutil.isdataframe(reads):
                raise TypeError('`reads` ({}) must be a pandas.DataFrame'.format(type(reads)))

            if 'name' in reads.columns:
                reads = reads.set_index('name')

            for sampleid in reads.columns:
                if sampleid not in self._samples:
                    raise ValueError('sample {} not contained in metadata. abort'.format(sampleid))
                sid, t = self._samples[sampleid]
                self[sid].add_reads(timepoints=t, reads=reads[sampleid].to_numpy())

        # Add the qPCR measurements if necessary
        # --------------------------------------
        if qpcr is not None:
            if not plutil.isdataframe(qpcr):
                raise TypeError('`qpcr` ({}) must be a pandas.DataFrame'.format(type(qpcr)))
            if 'sampleID' in qpcr.columns:
                qpcr = qpcr.set_index('sampleID')

            for sampleid in qpcr.index:
                try:
                    sid, t = self._samples[sampleid]
                except:
                    raise ValueError('Sample ID `{}` not found in metadata ({}). Make sure ' \
                        'you set the sample ID as the index in the `qpcr` dataframe'.format(
                            sampleid, list(self._samples.keys())))
                cfuspergram = qpcr.loc[sampleid].to_numpy()
                self[sid].add_qpcr(timepoints=t, qpcr=cfuspergram)
        return self

    def write_metadata_to_csv(self, path: str=None, sep:str='\t') -> pd.DataFrame:
        """Write the internal metadata to a table. If a path is provided
        then write it to that path.

        Parameters
        ----------
        path : str, None
            This is the location to save the metadata file
            If this is not provided then just return the dataframe
        sep : str
            This is the separator of the table
        """
        columns = ['sampleID', 'subject', 'time']
        data = []
        for sampleid in self._samples:
            sid, t = self._samples[sampleid]
            if t not in self[sid].times:
                continue
            data.append([sampleid, sid, t])
        df = pd.DataFrame(data, columns=columns)
        if path is not None:
            df.to_csv(path, sep=sep, index=False, header=True)
        return df

    def write_reads_to_csv(self, path: str=None, sep:str='\t') -> pd.DataFrame:
        """Write the reads to a table. If a path is provided then
        we write to that path

        Parameters
        ----------
        path : str
            This is the location to save the reads file
            If this is not provided then just return the dataframe
        sep : str
            This is the separator of the table
        """
        data = [[taxon.name for taxon in self.taxa]]
        index = ['name']
        for sampleid in self._samples:
            sid, t = self._samples[sampleid]
            if t not in self[sid].times:
                continue

            index.append(sampleid)
            reads = self[sid].reads[t]
            data.append(reads)

        df = pd.DataFrame(data, index=index).T
        if path is not None:
            df.to_csv(path, sep=sep, index=False, header=True)
        return df

    def write_qpcr_to_csv(self, path: str=None, sep:str='\t') -> pd.DataFrame:
        """Write the qPCR measurements to a table. If a path is provided then
        we write to that path

        Parameters
        ----------
        path : str
            This is the location to save the qPCR file
            If this is not provided then we do not save
        sep : str
            This is the separator of the table
        """
        max_n_measurements = -1
        data = []
        for sampleid in self._samples:
            sid, t = self._samples[sampleid]
            if t not in self[sid].times:
                continue
            subj = self[sid]
            ms = subj.qpcr[t].data
            if len(ms) > max_n_measurements:
                max_n_measurements = len(ms)
            ms = [sampleid] + ms.tolist()
            data.append(ms)

        for i, ms in enumerate(data):
            if len(ms)-1 < max_n_measurements:
                data[i] = np.append(
                    ms,
                    np.nan * np.ones(max_n_measurements - len(ms)))

        columns = ['sampleID'] + ['measurement{}'.format(i+1) for i in range(max_n_measurements)]

        df = pd.DataFrame(data, columns=columns)
        if path is not None:
            df.to_csv(path, sep=sep, index=False, header=True)
        return df

    def write_perturbations_to_csv(self, path: str=None, sep:str='\t') -> pd.DataFrame:
        """Write the perturbations to a table. If a path is provided then
        we write to that path

        Parameters
        ----------
        path : str
            This is the location to save the perturbations file
            If this is not provided then we do not save
        sep : str
            This is the separator of the table
        """
        columns = ['name', 'start', 'end', 'subject']
        data = []
        for perturbation in self.perturbations:
            for subjname in perturbation.starts:
                data.append([
                    perturbation.name,
                    perturbation.starts[subjname],
                    perturbation.ends[subjname],
                    subjname])

        df = pd.DataFrame(data, columns=columns)
        if path is not None:
            df.to_csv(path, sep=sep, index=False, header=True)
        return df

    def names(self) -> Iterable[str]:
        """List the names of the contained subjects

        Returns
        -------
        list(str)
            List of names of the subjects in order
        """
        return [subj.name for subj in self]

    def iloc(self, idx: int) -> Subject:
        """Get the subject as an index

        Parameters
        ----------
        idx : int
            Index of the subject

        Returns
        -------
        pl.base.Subject
        """
        for i,sid in enumerate(self._subjects):
            if i == idx:
                return self._subjects[sid]
        raise IndexError('Index ({}) not found'.format(idx))

    def add_subject(self, name: str):
        """Create a subject with the name `name`

        Parameters
        ----------
        name : str
            This is the name of the new subject
        """
        self.add_subject_obj(Subject(name=name, parent=self))
        return self

    def add_subject_obj(self, subj: Subject):
        if subj.name not in self._subjects:
            self._subjects[subj.name] = subj
        else:
            logger.warning(f"Subject `{subj.name}` already found. Skipping")

    def pop_subject(self, sid: Union[int, str, Iterable[str]],
        name: str='unnamed-study') -> 'Study':
        """Remove the indicated subject id

        Parameters
        ----------
        sid : list(str), str, int
            This is the subject name/s or the index/es to pop out.
            Return a new Study with the specified subjects removed.
        name : str
            Name of the new study to return
        """
        if not plutil.isarray(sid):
            sids = [sid]
        else:
            sids = sid

        for i in range(len(sids)):
            if plutil.isint(sids[i]):
                sids[i] = list(self._subjects.keys())[sids[i]]
            elif not plutil.isstr(sids[i]):
                raise ValueError('`sid` ({}) must be a str'.format(type(sids[i])))
        ret = Study(taxa=self.taxa, name=name)
        ret.qpcr_normalization_factor = self.qpcr_normalization_factor

        for s in sids:
            if s in self._subjects:
                ret._subjects[s] = self._subjects.pop(s, None)
                ret._subjects[s].parent = ret
            else:
                raise ValueError('`sid` ({}) not found'.format(sid))

        ret.perturbations = copy.deepcopy(self.perturbations)

        # Remove the names of the subjects in the perturbations
        for study in [ret, self]:
            for perturbation in study.perturbations:
                names = list(perturbation.starts.keys())
                for subjname in names:
                    if subjname not in study:
                        perturbation.starts.pop(subjname, None)
                names = list(perturbation.ends.keys())
                for subjname in names:
                    if subjname not in study:
                        perturbation.ends.pop(subjname, None)

        return ret

    def pop_taxa_like(self, study: 'Study'):
        """Remove s in the TaxaSet so that it matches the TaxaSet in `study`

        Parameters
        ----------
        study : mdsine2.study
            This is the study object we are mirroring in terms of taxa
        """
        to_delete = []
        for taxon in self.taxa:
            if taxon.name not in study.taxa:
                to_delete.append(taxon.name)
        self.pop_taxa(to_delete)

    def pop_taxa(self, oids: Union[str, int, Iterable[str], Iterable[int]]):
        """Delete the taxa indicated in oidxs. Updates the reads table and
        the internal TaxaSet

        Parameters
        ----------
        oids : str, int, list(str/int)
            These are the identifiers for each of the taxon/taxa to delete
        """
        # get indices
        oidxs = []
        for oid in oids:
            oidxs.append(self.taxa[oid].idx)

        # Delete the s from taxaset
        for oid in oids:
            self.taxa.del_taxon(oid)

        # Delete the reads
        for subj in self:
            for t in subj.reads:
                subj.reads[t] = np.delete(subj.reads[t], oidxs)
        return self

    def deaggregate_item(self, agg: OTU, other: str) -> Taxon:
        """Deaggregate the sequence `other` from OTU `agg`.
        `other` is then appended to the end

        Parameters
        ----------
        agg : OTU, str
            This is an OTU with multiple sequences contained. Must
            have the name `other` in there
        other : str
            This is the name of the taxon that should be taken out of `agg`

        Returns
        -------
        mdsine2.Taxon
            This is the deaggregated taxon
        """
        agg = self.taxa[agg]
        if not plutil.isstr(other):
            raise TypeError('`other` ({}) must be a str'.format(type(other)))
        if other not in agg.aggregated_taxa:
            raise ValueError('`other` ({}) is not contained in `agg` ({}) ({})'.format(
                other, agg.name, agg.aggregated_taxa))

        for subj in self:
            subj._deaggregate_item(agg=agg, other=other)
        return self.taxa.deaggregate_item(agg, other)

    def aggregate_items_like(self, study: 'Study', prefix: str=None) -> 'Study':
        """Aggregate s like they are in study `study`

        Parameters
        ----------
        study : mdsine2.Study
            Data object we are mirroring
        prefix : str
            If provided, this is how you rename the Taxas after aggregation
        """
        if isinstance(study.taxa, OTUTaxaSet):
            raise RuntimeError("aggregate_items_like() requires an OTUTaxaSet to model from.")

        components = []
        for otu in study.taxa:
            components.append(otu.components)

        other = self.aggregate_items(components)
        if prefix is not None:
            other.taxa.rename(prefix=prefix)
        return other

    def aggregate_items(self, components: List[List[Taxon]]) -> 'Study':
        """Aggregates the abundances of `taxon1` and `taxon2`. Updates the reads table and
        internal TaxaSet

        Parameters
        ----------
        taxon1, taxon2 : str, int, mdsine2.Taxon, mdsine2.OTU
            These are the taxa you are agglomerating together

        Returns
        -------
        mdsine2.OTU
            This is the new aggregated taxon containing anchor and other
        """
        other = Study(
            taxa=self.taxa.aggregate_items(components),
            name=self.name
        )

        for subj in self:
            agg_subj = subj.aggregate_items(other, components)
            other.add_subject_obj(agg_subj)

        other.perturbations = self.perturbations
        other.qpcr_normalization_factor = self.qpcr_normalization_factor
        other._samples = self._samples
        return other

    def pop_times(self, times: Union[int, float, np.ndarray], sids: Union[str, int, Iterable[int]]='all'):
        """
        Discard the times in `times` for the subjects listed in `sids`.
        If a timepoint is not found in a subject, no error is thrown.

        Parameters
        ----------
        times : numeric, list(numeric)
            Time/s to delete
        sids : str, int, list(int)
            The Subject ID or a list of subject IDs that you want to delete the timepoints
            from. If it is a str:
                'all' - delete from all subjects
        """
        if plutil.isstr(sids):
            if sids == 'all':
                sids = list(self._subjects.keys())
            else:
                raise ValueError('`sids` ({}) not recognized'.format(sids))
        elif plutil.isint(sids):
            if sids not in self._subjects:
                raise IndexError('`sid` ({}) not found in subjects'.format(
                    list(self._subjects.keys())))
            sids = [sids]
        elif plutil.isarray(sids):
            for sid in sids:
                if not plutil.isint(sid):
                    raise TypeError('Each sid ({}) must be an int'.format(type(sid)))
                if sid not in self._subjects:
                    raise IndexError('Subject {} not found in subjects ({})'.format(
                        sid, list(self._subjects.keys())))
        else:
            raise TypeError('`sids` ({}) type not recognized'.format(type(sids)))
        if plutil.isnumeric(times):
            times = [times]
        elif plutil.isarray(times):
            for t in times:
                if not plutil.isnumeric(t):
                    raise TypeError('Each time ({}) must be a numeric'.format(type(t)))
        else:
            raise TypeError('`times` ({}) type not recognized'.format(type(times)))

        for t in times:
            for sid in sids:
                subj = self._subjects[sid]
                if t in subj.times:
                    subj.qpcr.pop(t, None)
                    subj.reads.pop(t,None)
                    subj.times = np.sort(list(subj.reads.keys()))

    def normalize_qpcr(self, max_value: float):
        """Normalize the qPCR values such that the largest value is the max value
        over all the subjects

        Parameters
        ----------
        max_value : float, int
            This is the maximum qPCR value to

        Returns
        -------
        self
        """
        if type(max_value) not in [int, float]:
            raise ValueError('max_value ({}) must either be an int or a float'.format(
                type(max_value)))

        if self.qpcr_normalization_factor is not None:
            logger.warning('qPCR is already rescaled. unscaling and rescaling')
            self.denormalize_qpcr()

        temp_max = -1
        for subj in self:
            for key in subj.qpcr:
                temp_max = np.max([temp_max, subj.qpcr[key].mean()])

        self.qpcr_normalization_factor = max_value/temp_max
        logger.info('max_value found: {}, scaling_factor: {}'.format(
            temp_max, self.qpcr_normalization_factor))

        for subj in self:
            for key in subj.qpcr:
                subj.qpcr[key].set_scaling_factor(scaling_factor=
                    self.qpcr_normalization_factor)
        return self

    def denormalize_qpcr(self):
        """Denormalizes the qpcr values if necessary

        Returns
        -------
        self
        """
        if self.qpcr_normalization_factor is None:
            logger.warning('qPCR is not normalized. Doing nothing')
            return
        for subj in self:
            for key in subj.qpcr:
                subj.qpcr[key].set_scaling_factor(scaling_factor=1)
        self.qpcr_normalization_factor = None
        return self

    def add_perturbation(self, a: Union[Dict[str, float], BasePerturbation], ends: Dict[str, float]=None,
        name: str=None):
        """Add a perturbation.

        We can either do this by passing a perturbation object
        (if we do this then we do not need to specify `ends`) or we can
        specify the start and stop times (if we do this them we need to
        specify `ends`).

        `starts` and `ends`
        -------------------
        If `a` is a dict, this corresponds to the start times for each subject in the
        perturbation. Each dict maps the name of the subject to the timepoint that it
        either starts or ends, respectively.

        Parameters
        ----------
        a : dict, BasePerturbation
            If this is a dict, then this corresponds to the starts
            times of the perturbation for each subject. If this is a Pertubration object
            then we just add this.
        ends : dict
            Only necessary if `a` is a dict
        name : str, None
            Only necessary if `a` is a dict. Name of the perturbation

        Returns
        -------
        self
        """
        if self.perturbations is None:
            self.perturbations = Perturbations()
        if plutil.isdict(a):
            if not plutil.isdict(ends):
                raise ValueError('If `a` is a dict, then `ends` ({}) ' \
                    'needs to be a dict'.format(type(ends)))
            if not plutil.isstr(name):
                raise ValueError('`name` ({}) must be defined as a str'.format(type(name)))
            self.perturbations.append(BasePerturbation(starts=a, ends=ends, name=name))
        elif isinstance(a, BasePerturbation):
            self.perturbations.append(a)
        else:
            raise ValueError('`a` ({}) must be a subclass of ' \
                'pl.base.BasePerturbation or a dict'.format(type(a)))
        return self

    def split_on_perturbations(self):
        """Make new subjects for the time points that are divided by perturbations.
        Throw out all of the data  where the perturbations are active.

        Returns
        -------
        self
        """
        for subj in self:
            subj._split_on_perturbations()
        return self

    def times(self, agg: str) -> np.ndarray:
        """Aggregate the times of all the contained subjects

        These are the types of time aggregations:
            'union': Take  theunion of the times of the subjects
            'intersection': Take the intersection of the times of the subjects
        You can manually specify the times to include with a list of times. If times are not
        included in any of the subjects then we set them to NAN.

        Parameters
        ----------
        agg : str
            Type of aggregation to do of the times. Options: 'union', 'intersection'
        """
        if agg not in ['union', 'intersection']:
            raise ValueError('`agg` ({}) not recognized'.format(agg))

        all_times = []
        for subj in self:
            all_times = np.append(all_times, subj.times)
        all_times = np.sort(np.unique(all_times))
        if agg == 'union':
            times = all_times
        elif agg == 'intersection':
            times = []
            for t in all_times:
                addin = True
                for subj in self:
                    if t not in subj.times:
                        addin = False
                        break
                if addin:
                    times = np.append(times, t)
        else:
            raise ValueError('`agg` ({}) not recognized'.format(agg))
        return times

    def _matrix(self, dtype: str, agg: str, times: Union[str, np.ndarray], qpcr_unnormalize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        if dtype not in ['raw', 'rel', 'abs']:
            raise ValueError('`dtype` ({}) not recognized'.format(dtype))

        if agg == 'mean':
            aggfunc = np.nanmean
        elif agg == 'median':
            aggfunc = np.nanmedian
        elif agg == 'sum':
            aggfunc = np.nansum
        elif agg == 'max':
            aggfunc = np.nanmax
        elif agg == 'min':
            aggfunc = np.nanmin
        else:
            raise ValueError('`agg` ({}) not recognized'.format(agg))

        if plutil.isstr(times):
            times = self.times(agg=times)
        elif plutil.isarray(times):
            times = np.array(times)
        else:
            raise TypeError('`times` type ({}) not recognized'.format(type(times)))

        shape = (len(self.taxa), len(times))
        M = np.zeros(shape, dtype=float)
        for tidx, t in enumerate(times):
            temp = None
            for subj in self:
                if t not in subj.times:
                    continue
                if dtype == 'raw':
                    a = subj.reads[t]
                elif dtype == 'rel':
                    a = subj.reads[t]/np.sum(subj.reads[t])
                else:
                    rel = subj.reads[t]/np.sum(subj.reads[t])
                    a = rel * subj.qpcr[t].mean(qpcr_unnormalize=qpcr_unnormalize)
                if temp is None:
                    temp = (a.reshape(-1,1), )
                else:
                    temp = temp + (a.reshape(-1,1), )
            if temp is None:
                temp = np.zeros(len(self.taxa)) * np.nan
            else:
                temp = np.hstack(temp)
                temp = aggfunc(temp, axis=1)
            M[:, tidx] = temp

        return M, times

    def matrix(self, dtype: str, agg: str, times: Union[str, np.ndarray], qpcr_unnormalize: bool = False) -> np.ndarray:
        """Make a matrix of the aggregation of all the subjects in the subjectset

        Aggregation of subjects
        -----------------------
        What are the values for the taxa? Set the aggregation type using the parameter `agg`.
        These are the types of aggregations:
            'mean': Mean abundance of the taxon at a timepoint over all the subjects
            'median': Median abundance of the taxon at a timepoint over all the subjects
            'sum': Sum of all the abundances of the taxon at a timepoint over all the subjects
            'max': Maximum abundance of the taxon at a timepoint over all the subjects
            'min': Minimum abundance of the taxon at a timepoint over all the subjects

        Aggregation of times
        --------------------
        Which times to include? Set the times to include with the parameter `times`.
        These are the types of time aggregations:
            'union': Take  theunion of the times of the subjects
            'intersection': Take the intersection of the times of the subjects
        You can manually specify the times to include with a list of times. If times are not
        included in any of the subjects then we set them to NAN.

        Parameters
        ----------
        dtype : str
            What kind of data to return. Options:
                'raw': Count data
                'rel': Relative abundance
                'abs': Abundance data
        agg : str
            Type of aggregation of the values. Options specified above.
        times : str, array
            The times to include

        Returns
        -------
        np.ndarray(n_taxa, n_times)
        """
        M, _ =  self._matrix(dtype=dtype, agg=agg, times=times, qpcr_unnormalize=qpcr_unnormalize)
        return M

    def df(self, dtype: str, agg: str, times: Union[str, np.ndarray]) -> pd.DataFrame:
        """Returns a dataframe of the data in matrix. Rows are taxa, columns are times.

        Parameters
        ----------
        dtype : str
            What kind of data to return. Options:
                'raw': Count data
                'rel': Relative abundance
                'abs': Abundance data
        agg : str
            Type of aggregation of the values. Options specified above.
        times : str, array
            The times to include

        Returns
        -------
        pandas.DataFrame

        See Also
        --------
        mdsine2.Study.matrix
        """
        M, times = self._matrix(dtype=dtype, agg=agg, times=times)
        index = [taxon.name for taxon in self.taxa]
        return pd.DataFrame(data=M, index=index, columns=times)
