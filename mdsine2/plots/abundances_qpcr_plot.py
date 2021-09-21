import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TAXLEVEL_PLURALS = {'genus': 'Genera', 'Genus': 'Genera', 'family': 'Families',
                    'Family': 'Families', 'order': 'Orders', 'Order': 'Orders',
                    'class': 'Classes', 'Class': 'Classes', 'phylum': 'Phyla',
                    'Phylum': 'Phyla', 'kingdom': 'Kingdoms', 'Kingdom': 'Kingdoms'}

TAXLEVEL_INTS = ["species", "genus", "family", "order", "class", "phylum", "kingdom"]
TAXLEVEL_REV_IDX = {"species": 0, "genus": 1, "family": 2, "order": 3, "class": 4, "phylum": 5, "kingdom": 6}
CUTOFF_FRAC_ABUNDANCE = 0.01


def get_df(subjset, taxlevel="family"):
    """
       @parameters
       subjset : (pl.Subject)
    """
    taxidx = TAXLEVEL_REV_IDX[taxlevel]
    upper_tax = TAXLEVEL_INTS[taxidx + 1]
    lower_tax = TAXLEVEL_INTS[taxidx]

    df = None
    times = []
    for subj in subjset:
        times = np.append(times, subj.times)

    times = np.sort(np.unique(times))  # the times at which samples were collected
    t2idx = {}
    for i, t in enumerate(times):
        t2idx[t] = i
    times_cnts = np.zeros(len(times))  # the times at which samples were taken

    # update the data frame for each subject
    for subj in subjset:
        dfnew, taxaname_map = subj.cluster_by_taxlevel(dtype='abs',
                                                       taxlevel=taxlevel,
                                                       index_formatter='%({})s %({})s'.format(upper_tax,
                                                                                              lower_tax),
                                                       smart_unspec=False)

        df, times_cnts = _add_unequal_col_dataframes(df=df, dfother=dfnew,
                                                     times=times, times_cnts=times_cnts, t2idx=t2idx)

    df = df / df.sum(axis=0)

    # Only plot the OTUs that have a total percent abundance over a threshold
    if CUTOFF_FRAC_ABUNDANCE is not None:
        df = _get_top(df, cutoff_frac_abundance=CUTOFF_FRAC_ABUNDANCE,
                      taxlevel=taxlevel)

    return df


def _cnt_times(df, times, times_cnts, t2idx):
    """counts the number of times data at a given point were collected"""

    for col in df.columns:
        if col in times:
            times_cnts[t2idx[col]] += 1

    return times_cnts


def _add_unequal_col_dataframes(df, dfother, times, times_cnts, t2idx):
    """
    Add the contents of both the dataframes. This controls for the
    columns in the dataframes `df` and `dfother` being different.
    """

    times_cnts = _cnt_times(dfother, times, times_cnts, t2idx)
    if df is None:
        return dfother, times_cnts

    cols_toadd_df = []
    cols_toadd_dfother = []
    for col in dfother.columns:
        if col not in df.columns:
            cols_toadd_df.append(col)
    for col in df.columns:
        if col not in dfother.columns:
            cols_toadd_dfother.append(col)

    df = pd.concat([df,
                    pd.DataFrame(np.zeros(shape=(len(df.index), len(cols_toadd_df))),
                                 index=df.index, columns=cols_toadd_df)], axis=1)
    dfother = pd.concat([dfother,
                         pd.DataFrame(np.zeros(shape=(len(dfother.index), len(cols_toadd_dfother))),
                                      index=dfother.index, columns=cols_toadd_dfother)], axis=1)

    return dfother.reindex(df.index) + df, times_cnts


def _get_top(df, cutoff_frac_abundance, taxlevel, taxaname_map=None):
    """
       selects the data associated with taxon (at taxlevel) whose abundace is
       greater than the cutoff_frac_abundance
    """
    matrix = df.values
    abunds = np.sum(matrix, axis=1)
    namemap = {}

    a = abunds / abunds.sum()
    a = np.sort(a)[::-1]

    cutoff_num = None
    for i in range(len(a)):
        if a[i] < cutoff_frac_abundance:
            cutoff_num = i
            break
    if cutoff_num is None:
        raise ValueError('Error')

    idxs = np.argsort(abunds)[-cutoff_num:][::-1]
    dfnew = df.iloc[idxs, :]

    if taxaname_map is not None:
        indexes = df.index
        for idx in idxs:
            namemap[indexes[idx]] = taxaname_map[indexes[idx]]

    # Add everything else as 'Other'
    vals = None
    for idx in range(len(df.index)):
        if idx not in idxs:
            if vals is None:
                vals = df.values[idx, :]
            else:
                vals += df.values[idx, :]

    dfother = pd.DataFrame([vals], columns=df.columns, index=['{} with <{}% total abund'.format(
        TAXLEVEL_PLURALS[taxlevel], cutoff_frac_abundance * 100)])
    df = dfnew.append(dfother)

    return df


def plot_rel(df,
             ax,
             color_set):
    """
    plots the relative abundance and perturbation
    """
    matrix = df.values
    matrix = np.flipud(matrix)
    times = np.asarray(list(df.columns))

    # ====================== Plot relative abundance, Create a stacked bar chart ========================
    offset = np.zeros(matrix.shape[1])
    for row in range(matrix.shape[0]):
        color = color_set[row]
        ax.bar(np.arange(len(times)),
               matrix[row, :],
               bottom=offset,
               color=color,
               label=df.iloc[row].name,
               width=1, linewidth=1)
        offset = offset + matrix[row, :]

    # set the xlabels
    locs = np.arange(0, len(times), step=10)
    ticklabels = times[locs]
    ax.set_xticks(locs)
    ax.set_xticklabels(ticklabels)
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(24)
    ax.legend()
    ax.set_xlabel('Time (d)', size=24, fontweight='bold')
    ax.set_ylabel('Relative Abundance (Stacked)', size=24, fontweight='bold')
    ax.set_ylim(bottom=0, top=1)


def plot_qpcr(subjset, ax):
    qpcr_meas = {}
    for subj in subjset:
        for t in subj.times:
            if t not in qpcr_meas:
                qpcr_meas[t] = []
            qpcr_meas[t].append(subj.qpcr[t].mean())
    # geometric mean of qpcr
    for key in qpcr_meas:
        vals = qpcr_meas[key]
        a = 1
        for val in vals:
            a *= val
        a = a ** (1 / len(vals))
        qpcr_meas[key] = a
    times_qpcr = np.sort(list(qpcr_meas.keys()))
    vals = np.zeros(len(times_qpcr))
    for iii, t in enumerate(times_qpcr):
        vals[iii] = qpcr_meas[t]
    max_qpcr_value = np.max(vals)
    ax.plot(np.arange(len(times_qpcr)), vals, marker='o', linestyle='-', color='black')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_minor_locator(plt.NullLocator())

    ax.set_ylabel('CFUs/g', size=24, fontweight='bold')
    ax.set_title("QPcr", fontsize=30, fontweight='bold')

    ax.set_ylim(bottom=1e9, top=max_qpcr_value * 1.25)
    ax.set_yscale('log')
