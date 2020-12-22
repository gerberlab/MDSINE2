'''Visualization funtions for pylab

Heatmap Rendering
-----------------
- These functions render heatmaps of matrix data:
    - render_bayes_factors
    - render_cocluster_probabilities
    - render_interaction_strength
    - render_growth_vector

Abundance functions
-------------------
- These functions either plot the abundance of the data or a metric of them:
    - alpha_diversity_over_time
    - qpcr_over_time
    - abundance_over_time
    - taxonomic_distribution_over_time

Tracing functions
-----------------
- These functions plot how the value of a variable changes over inference:
    - render_acceptance_rate_trace
    - render_trace

Linewidths are automatically shutoff if the number of taxa is greater than 75
'''
import numpy as np
import logging
import math
import warnings
import re
import copy
import pandas
import sys
import numba

from typing import Union, Dict, Iterator, Tuple, List, Any, IO, Callable

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.pyplot import arrow
import matplotlib.ticker as plticker
import matplotlib

from . import pylab as pl
from .pylab.base import DEFAULT_TAXLEVEL_NAME, Perturbations, Subject, TaxaSet, Study, OTU
from .pylab import Clustering, variables

warnings.filterwarnings('ignore')
_plt_labels = ['title', 'xlabel', 'ylabel']

# Constants
DEFAULT_TAX_LEVEL = None
PERTURBATION_COLOR = 'orange'
XTICK_FREQUENCY = 5 # in days
DEFAULT_SNS_CMAP = 'deep'

DEFAULT_MAX_BAYES_FACTOR = 15
DEFAULT_LINEWIDTHS = 0.8
DEFAULT_INCLUDE_COLORBAR = True
DEFAULT_INCLUDE_TICK_MARKS = False
DEFAULT_ACCEPTANCE_RATE_PREV = 50
DEFAULT_PLT_TYPE = 'both'
DEFAULT_TRACE_COLOR = 'steelblue'
PLT_TITLE_LABEL = 'title'
PLT_XLABEL_LABEL = 'xlabel'
PLT_YLABEL_LABEL = 'ylabel'

# ----------------
# Global Functions
# ----------------
def set_default_tax_level(level: str):
    '''This sets the default taxonomic level to plot at.
    
    Parameters
    ----------
    level : str
        This is the level to set it at. It must be either:
            'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species', 'asv'
    '''
    global DEFAULT_TAX_LEVEL
    if not pl.isstr(level):
        raise ValueError('`level` ({}) must be a str'.format(type(level)))
    if level not in ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species', 'asv']:
        raise ValueError('`level` ({}) not valid'.format(level))
    DEFAULT_TAX_LEVEL = level

def set_perturbation_color(color: Any):
    '''Set the color for the perturbation shading. Must be matplotlib compatible.

    Parameters
    ----------
    color : any
        This is the color to set it to.

    See Also
    --------
    https://matplotlib.org/2.0.2/api/colors_api.html
    '''
    global PERTURBATION_COLOR
    PERTURBATION_COLOR = color

def set_xtick_frequency(x: Union[float, int]):
    '''Sets the xtick frequency (how often a label on the x axis should occur)

    Parameters
    ----------
    x : numeric
        How often it should occur (in days)
    '''
    global XTICK_FREQUENCY
    if not pl.isnumeric(x):
        raise ValueError('x ({}) must be a numeric'.format(type(x)))
    if x <= 0:
        raise ValueError('x ({}) must be >= 0'.format(x))
    XTICK_FREQUENCY = x

def set_default_trace_color(color: Any):
    '''Sets defalt color of the trace. Must be matplotlib compatible.

    Parameters
    ----------
    color : any
        This is the color to set it to.

    See Also
    --------
    https://matplotlib.org/2.0.2/api/colors_api.html
    '''
    global DEFAULT_TRACE_COLOR
    DEFAULT_TRACE_COLOR = color

def shade_in_perturbations(ax: matplotlib.pyplot.Axes, perturbations: Perturbations, 
    subj: Subject, textcolor: str='black', textsize: Union[float, int]=None, 
    alpha: float=0.25, label: bool=True) -> matplotlib.pyplot.Axes:
    '''Shade in the axis where there are perturbations and adds the label of
    the perturbation above it.

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        Axis we are plotting on
    perturbations : mdsine2.Perturbations
        List of perturbations we are coloring in
    subj : mdsine2.Subject, str

    Returns
    -------
    matplotlib.pyplot.Axes
    '''
    from . import pylab as pl

    if pl.issubject(subj):
        subj = subj.name
    if not pl.isstr(subj):
        raise ValueError('`Cannot recognize {}'.format(subj))
    if perturbations is None or len(perturbations) == 0:
        return ax

    pert_locs = []
    pert_names = []
    for pidx, perturbation in enumerate(perturbations):

        if subj not in perturbation.starts or subj not in perturbation.ends:
            continue

        ax.axvspan(
            xmin=perturbation.starts[subj],
            xmax=perturbation.ends[subj], 
            facecolor=PERTURBATION_COLOR, 
            alpha=alpha, zorder=-10000)
        pert_locs.append((perturbation.ends[subj] + perturbation.starts[subj]) / 2)
        name = perturbation.name
        if name is None:
            name = 'pert{}'.format(pidx)
        pert_names.append(name)

    if label:
        # Set the names on the top x-axis
        ax2 = ax.twiny()

        # # Set the visibility of the twin axis to see through
        # ax2.spines['top'].set_visible(False)
        # ax2.spines['bottom'].set_visible(False)
        # ax2.spines['left'].set_visible(False)
        # ax2.spines['right'].set_visible(False)
        # ax2.xaxis.set_major_locator(plt.NullLocator())
        # ax2.xaxis.set_minor_locator(plt.NullLocator())
        # ax2.yaxis.set_major_locator(plt.NullLocator())
        # ax2.yaxis.set_minor_locator(plt.NullLocator())

        left,right = ax.get_xlim()
        ax2.set_xlim(ax.get_xlim())
        pl = []
        pn = []
        for idx, loc in enumerate(pert_locs):
            if loc > left and loc < right:
                pl.append(loc)
                pn.append(pert_names[idx])
        ax2.set_xticks(pl)
        ax2.set_xticklabels(pn)
        ax2.tick_params('x', which='both', length=0, colors=textcolor, 
            labelsize=textsize)

    return ax

# --------
# Heatmaps
# --------
def render_bayes_factors(bayes_factors: np.ndarray, taxa: TaxaSet=None, ax:matplotlib.pyplot.Axes=None,
    n_colors: int=100, max_value: Union[float, int]=None, xticklabels: str='%(index)s', 
    yticklabels: str='%(name)s %(index)s', include_tick_marks: bool=False, linewidths: float=0.8, 
    linecolor: str='black', cmap: str='Blues',
    include_colorbar: bool=True, title: str='Microbe Interaction Bayes Factors', figure_size: Tuple[float, float]=None,
    order: Union[Iterator[int], np.ndarray]=None) -> matplotlib.pyplot.Axes:
    '''Renders the bayes factors for each of the interactions. Self interactions
    are automatically set to np.nan.

    Parameters
    ----------
    bayes_factors : 2-dim np.ndarray
        - Square matrix indicating the bayes factors of the interaction
    taxa : pl.base.TaxaSet, None
        - This is the object that contains all of the Taxa metadata
        - If this is None, then we do no checking if the size of the `interaction_matrix`
          corresponds to the size of taxa
    clustering : pylab.cluster.ClusteringBase, Optional
        - Clustering object if you want the Taxa in the same cluster to be grouped
          together
    ax : matplotlib.pyplot.Axes, Optional
        - The axes to plot on. If nothing is provided a new figure will be created
    n_colors : int, Optional
        - The number of colors to generate for the colormap.
    max_value : float, int, Optional
        - Clips all of the values above this limit
        - If None, then there is no clipping
    xticklabels, yticklabels : str, list, None, Optional
        - These are the labels for the x and y axis, respectively.
        - If it is a list then it must have `taxa.n_taxa` elements.
        - If it is a string, it is the formatter for each of the rows/columns.
        - If it is None, then do not make any ticklabels
    include_tick_marks : bool, Optional
        - If True, include tick marks. If False get rid of them
    linewidths : float, Optional
        - The width of the lines separating the squares
    linecolor : str, Optional
        - The color of the lines separating the squares
    cmap : colormap object, Optional
        - Overrides the default colormap. If specified, we ignore `n_colors`
    include_colorbar : bool, Optional
        - If True, it will render the colorbar, if not then it wont.
    title : str, None, Optional
        - Title of the figure
        - If None then do not put any title
    figure_size : 2-tuple, Optional
        - This is the size of the figure (in inches)
        - If nothing is specified it will default to adding 10 inches in each
          dimension for every 50 Taxa
    
    Returns
    -------
    matplotlib.pyplot.Axes
        Axes object that has the image rendering on it
    '''
    if max_value is None:
        max_value = DEFAULT_MAX_BAYES_FACTOR
    # Set default parameters
    d = _set_heatmap_default_args(linewidths=linewidths, linecolor=linecolor, 
        n_colors=n_colors, xticklabels=xticklabels, yticklabels=yticklabels, 
        include_colorbar=include_colorbar, include_tick_marks=include_tick_marks)
    linewidths = d['linewidths']
    linecolor = d['linecolor']
    n_colors = d['n_colors']
    xticklabels = d['xticklabels']
    yticklabels = d['yticklabels']
    include_colorbar = d['include_colorbar']
    include_tick_marks = d['include_tick_marks']

    # Type checking and initialization
    d = _init_parameters_heatmap(matrix=bayes_factors, clustering=None,
        taxa=taxa, xticklabels=xticklabels,
        yticklabels=yticklabels, ax=ax, figure_size=figure_size,
        linewidths=linewidths, order=order)
    ax = d['ax']
    bayes_factors = d['matrix']
    xticklabels = d['xticklabels']
    yticklabels = d['yticklabels']
    cbar_kws = d['cbar_kws']
    linewidths = d['linewidths']

    # if np.any(bayes_factors < 0):
    #     raise ValueError('There should be no negative values in `bayes_factors`')
    for i in range(bayes_factors.shape[0]):
        bayes_factors[i,i] = np.nan
    if cmap is None:
        cmap = sns.color_palette('Blues', n_colors=n_colors)
    if max_value is not None:
        bayes_factors[bayes_factors > max_value] = max_value

    ax = sns.heatmap(
        data=bayes_factors,
        square=True,
        linewidths=linewidths,
        linecolor=linecolor,
        cmap=cmap,
        cbar=include_colorbar,
        robust=True,
        cbar_kws=cbar_kws,
        xticklabels=xticklabels,
        yticklabels=yticklabels)

    if not include_tick_marks:
        ax.tick_params(bottom=False, left=False)
    if title is not None:
        ax.set_title(title)
    plt.yticks(rotation=0)
    return ax

def render_cocluster_probabilities(coclusters: np.ndarray, taxa: TaxaSet, ax: matplotlib.pyplot.Axes=None,
    n_colors: int=100, max_value: Union[float, int]=None, xticklabels: str='%(index)s', 
    yticklabels: str='%(name)s %(index)s', include_tick_marks: bool=False, linewidths: float=0.8, 
    linecolor: str='black', cmap: str='Blues',
    include_colorbar: bool=True, title: str='Microbe Co-cluster Probabilities', figure_size: Tuple[float, float]=None,
    order: Union[Iterator[int], np.ndarray]=None) -> matplotlib.pyplot.Axes:
    '''Render the cocluster proportions. Values in coclusters should be [0,1].

    Parameters
    ----------
    coclusters : 2-dim np.ndarray
        - Square matrix indicating the cocluster proportions
    taxa : pylab.base.TaxaSet
        - This is the object that contains all of the Taxa metadata
    ax : matplotlib.pyplot.Axes, Optional
        - The axes to plot on. If nothing is provided a new figure will be created
    n_colors : int, Optional
        - The number of colors to generate for the colormap.
    xticklabels, yticklabels : str, list, None, Optional
        - These are the labels for the x and y axis, respectively.
        - If it is a list then it must have `taxa.n_taxa` elements.
        - If it is a string, it is the formatter for each of the rows/columns.
        - If it is None, then do not make any ticklabels
    include_tick_marks : bool, Optional
        - If True, include tick marks. If False get rid of them
    linewidths : float, Optional
        - The width of the lines separating the squares
    linecolor : str, Optional
        - The color of the lines separating the squares
    cmap : colormap object, Optional
        - Overrides the default colormap. If specified, we ignore `n_colors`
    include_colorbar : bool, Optional
        - If True, it will render the colorbar, if not then it wont.
    title : str, None, Optional
        - Title of the figure
        - If None then do not put any title
    figure_size : 2-tuple, Optional
        - This is the size of the figure (in inches)
        - If nothing is specified it will default to adding 10 inches in each
          dimension for every 50 Taxa

    Returns
    -------
    matplotlib.pyplot.Axes
        Axes object that has the image rendering on it
    '''
    # Set default parameters
    d = _set_heatmap_default_args(linewidths=linewidths, linecolor=linecolor, 
        n_colors=n_colors, xticklabels=xticklabels, yticklabels=yticklabels, 
        include_colorbar=include_colorbar, include_tick_marks=include_tick_marks)
    linewidths = d['linewidths']
    linecolor = d['linecolor']
    n_colors = d['n_colors']
    xticklabels = d['xticklabels']
    yticklabels = d['yticklabels']
    include_colorbar = d['include_colorbar']
    include_tick_marks = d['include_tick_marks']

    # Type checking and initialization
    d = _init_parameters_heatmap(matrix=coclusters,
        taxa=taxa, xticklabels=xticklabels, clustering=None,
        yticklabels=yticklabels, ax=ax, figure_size=figure_size,
        linewidths=linewidths, order=order)
    ax = d['ax']
    coclusters = d['matrix']
    xticklabels = d['xticklabels']
    yticklabels = d['yticklabels']
    cbar_kws = d['cbar_kws']
    linewidths = d['linewidths']

    if cmap is None:
        cmap = sns.color_palette('Blues', n_colors=n_colors)
    if np.any(coclusters < 0) or np.any(coclusters > 1):
        raise ValueError('All values of coclusters should be in [0,1]')

    ax = sns.heatmap(
        data=coclusters,
        square=True,
        linewidths=linewidths,
        linecolor=linecolor,
        cmap=cmap,
        cbar_kws=cbar_kws,
        cbar=include_colorbar,
        xticklabels=xticklabels,
        robust=True,
        yticklabels=yticklabels)

    if not include_tick_marks:
        ax.tick_params(bottom=False, left=False)
    if title is not None:
        ax.set_title(title)
    plt.yticks(rotation=0)
    return ax

def render_interaction_strength(interaction_matrix: np.ndarray, log_scale: bool, taxa: TaxaSet, 
    clustering: Clustering=None, ax: matplotlib.pyplot.Axes=None, center_colors: bool=False, 
    n_colors: int=100, xticklabels: str='%(index)s', vmax: Union[float, int]=None, vmin: Union[float, int]=None, 
    yticklabels: str='%(name)s %(index)s', include_tick_marks: bool=False, linewidths: float=0.8, linecolor: str='black',
    cmap: Any=None, include_colorbar: bool=True, title: str='Microbe Interaction Strength',
    figure_size: Tuple[float, float]=None, order: Union[Iterator[int], np.ndarray]=None) -> matplotlib.pyplot.Axes:
    '''Render the interaction strength matrix. If you want the values in log scale,
    it will annotate the box with the sign of the interaction and plot the absolute
    value of the interaction. If you want the Taxa in the same clusters to be grouped
    together, specify the clustering object in `cluster`.

    Parameters
    ----------
    interaction_matrix : 2-dim np.ndarray
        - Square matrix indicating the interaction strengths
    log_scale : bool
        - If True, plots with log scale. If False it plots with regular
          scale
    taxa : pylab.base.TaxaSet, None
        - This is the object that contains all of the Taxa metadata
        - If this is None, then we do no checking if the size of the `interaction_matrix`
          corresponds to the size of taxa
    clustering : pylab.cluster.ClusteringBase, Optional
        - Clustering object if you want the Taxa in the same cluster to be grouped
          together
    ax : matplotlib.pyplot.Axes, Optional
        - The axes to plot on. If nothing is provided a new figure will be created
    center_colors : bool, Optional
        - If True, it will center the colors for the colormap
        - This is overriden if `log_scale` is True
    n_colors : int, Optional
        - The number of colors to generate for the colormap.
    vmax, vmin : float
        - Lower and upper values to plot. If nothing is provided then it is
          infered from the data
    xticklabels, yticklabels : str, list, None, Optional
        - These are the labels for the x and y axis, respectively.
        - For details on the format, look at `pylab.taxaname_formatter`
        - If it is a list then it must have `taxa.n_taxa` elements.
        - If it is a string, it is the formatter for each of the rows/columns.
        - If it is None, then do not make any ticklabels
    include_tick_marks : bool, Optional
        - If True, include tick marks. If False get rid of them
    linewidths : float, Optional
        - The width of the lines separating the squares
    linecolor : str, Optional
        - The color of the lines separating the squares
    cmap : colormap object, Optional
        - Overrides the default colormap. If specified, we ignore `n_colors`
    include_colorbar : bool, Optional
        - If True, it will render the colorbar, if not then it wont.
    title : str, None, Optional
        - Title of the figure
        - If None then do not put any title
    figure_size : 2-tuple, Optional
        - This is the size of the figure (in inches)
        - If nothing is specified it will default to adding 10 inches in each
          dimension for every 50 Taxa
    
    Returns
    -------
    matplotlib.pyplot.Axes
        Axes object that has the image rendering on it
    '''
    # Set default parameters
    d = _set_heatmap_default_args(linewidths=linewidths, linecolor=linecolor, 
        n_colors=n_colors, xticklabels=xticklabels, yticklabels=yticklabels, 
        include_colorbar=include_colorbar, include_tick_marks=include_tick_marks)
    linewidths = d['linewidths']
    linecolor = d['linecolor']
    n_colors = d['n_colors']
    xticklabels = d['xticklabels']
    yticklabels = d['yticklabels']
    include_colorbar = d['include_colorbar']
    include_tick_marks = d['include_tick_marks']

    # Type checking and initialization
    d = _init_parameters_heatmap(matrix=interaction_matrix,
        taxa=taxa, clustering=clustering, xticklabels=xticklabels,
        yticklabels=yticklabels, ax=ax, figure_size=figure_size,
        linewidths=linewidths, order=order)
    ax = d['ax']
    interaction_matrix = d['matrix']
    xticklabels = d['xticklabels']
    yticklabels = d['yticklabels']
    cbar_kws = d['cbar_kws']
    linewidths = d['linewidths']

    if cmap is None:
        cmap = sns.cubehelix_palette(n_colors)
    if center_colors and not log_scale:
        cmap = sns.color_palette("RdBu_r", n_colors)
        center_colors=0 # center at 0
        if vmin is not None and vmax is not None:
            absmax = np.max(np.absolute([vmin,vmax]))
            vmin = -absmax
            vmax = absmax

    log_scale = bool(log_scale)
    center_colors = bool(center_colors)
    include_tick_marks = bool(include_tick_marks)
    linewidths = float(linewidths)
    linecolor = str(linecolor)
    include_colorbar = bool(include_colorbar)
    n_colors = int(n_colors)

    # Add annotation and override parameters if we are doing log scale
    if log_scale:
        ann = []
        for i in range(interaction_matrix.shape[0]):
            ann.append([])
            for j in range(interaction_matrix.shape[1]):
                b = '+'
                if interaction_matrix[i,j] < 0:
                    b = '\u2212' # uicode minus sign
                ann[-1].append(b)
        annot = np.asarray(ann)
        interaction_matrix = np.log10(np.absolute(interaction_matrix))
        interaction_matrix[~np.isfinite(interaction_matrix)] = np.nan
        interaction_matrix[interaction_matrix == 0] = np.nan

        center_colors = None
        # if vmax is not None:
        #     vmax=np.absolute(vmax)
        # if vmin is not None:
        #     vmin=np.absolute(vmin)
    else:
        annot = None

    try:
        ax = sns.heatmap(
            data=interaction_matrix,
            annot=annot,
            fmt='',
            center=center_colors,
            square=True,
            # vmin=vmin,
            # vmax=vmax,
            robust=True,
            linewidths=linewidths,
            linecolor=linecolor,
            cmap=cmap,
            cbar=include_colorbar,
            cbar_kws=cbar_kws,
            xticklabels=xticklabels,
            yticklabels=yticklabels)
        if not include_tick_marks:
            ax.tick_params(bottom=False, left=False)
        if title is not None:
            ax.set_title(title)
        if log_scale:
            ax.collections[0].colorbar.ax.set_title("$\\log_{10}$")
        plt.yticks(rotation=0)
    except Exception as e:
        logging.critical('Could not plot heatmap because of error message: "{}".' \
            ' This is likely because `interaction_matrix` has either only NaNs ' \
            'or 0s ({}). We are clearing the current axis and are going to skip' \
            'plotting this axis.'.format(str(e), _is_just_zero_or_nan(interaction_matrix)))
    return ax

# ------
# Traces
# ------
def render_acceptance_rate_trace(var: variables.Variable, idx: Union[int, Tuple[int, int]]=None, 
    prev: Union[int, str]='default', ax: matplotlib.pyplot.Axes=None, include_burnin: bool=True, 
    scatter: bool=True, n_burnin: int=None, section: str='posterior', 
    **kwargs) -> matplotlib.pyplot.Axes:
    '''Visualize the acceptance rate over time for a
    metropolis._BaseKernel object.

    Parameters
    ----------
    var : array, pl.variables.Variable
        - Array or variable to see the trace on
    idx : int, tuple, Optional
        - If the variable is not a scalar, this indexes the index that you want to trace on
    prev : int, str, None, Optional
        - For each iteration, calculate the acceptance rate based on the previous `prev`
          iterations. If None, does it over the entire trace. If it is a str, the options
          are:
            'default': sets it to the default 
            'all': Same as None
    ax : matplotlib.pyplot.Axes, Optional
        - Axes to plot on
        - If nothing is provided, a new figure with a single subplot
          will be created and returned
    inlcude_burnin : bool, Optional
        - If True, it will plot the burnin trace as negative numbers
          in the trace as well
    scatter : bool
        - Only applies to plt_type=='trace'
        - If True, it will plot the points as a scatter plot
        - Else it will plot the points as a line
    n_burnin : int
        - Only required if `var` is an array, not a pylab.variables.Variable.
        - Tells how big the burnin array is.
    section : str
        This is only used if var is a pylab variable and we have to retrieve the section
        of the chain.
    kwargs : dict, Optional
        - Optional arguments:
            - 'color' : str
                - Default color is 'blue'
            - 'alpha' : float
                - Default alpha is 0.5 for the trace
            - 'title' : str
                - Nothing
            - 'xlabel' : str
                - Default is 'Iteration'
            - 'ylabel' : str
                - Default is 'Acceptance Rate

    Returns
    -------
    matplotlib.pyplot.Axes
        - Axis object that contains the rendering of the acceptance rate
    '''
    if pl.isstr(prev):
        if prev == 'all':
            prev = float('inf')
        elif prev == 'default':
            prev = DEFAULT_ACCEPTANCE_RATE_PREV
        else:
            raise ValueError('str `default` ({}) not recognized'.format(prev))
    elif prev is None:
        prev = 50
    elif not pl.isint(prev):
        raise TypeError('`prev` ({}) must either be None or an int'.format(type(prev)))
    if prev < 0:
        raise ValueError('`prev` ({}) must be postiive'.format(prev))

    if not pl.isbool(include_burnin):
        raise TypeError('`include_burnin` ({}) must be a bool'.format(type(include_burnin)))

    if idx is not None:
        # Then it is multidimensional
        if type(idx) == int:
            idxs = (..., idx)
        elif type(idx) == tuple:
            idxs = (..., ) + idx
        else:
            raise ValueError('`idx` ({}) must either be an int or a tuple'.format(
                type(idx)))
    else:
        idxs = ()

    if pl.isarray(var):
        trace = np.asarray(var)
        if pl.isint(n_burnin):
            if n_burnin < 0:
                raise ValueError('`n_burnin` ({}) must be >= 0'.format(n_burnin))
            points = np.append(
                np.arange(-n_burnin, 0, 1),
                np.arange(len(trace) - n_burnin))
        else:
            points = np.arange(trace.shape[0])
        trace = trace[idxs]
        include_burnin = False

    elif pl.isVariable(var):
        if section == 'entire':
            section = 'posterior'
            include_burnin = True

        trace = var.get_trace_from_disk(section=section)
        trace = trace[idxs]
        
        if section == 'burnin':
            points = np.arange(-len(trace),0,1) 
        elif include_burnin:
            burnin_trace = var.get_trace_from_disk(section='burnin')
            burnin_trace = burnin_trace[idxs]

            points = np.append(
                np.arange(-len(burnin_trace),0,1),
                np.arange(len(trace)))
            trace = np.append(burnin_trace, trace)
        else:
            points = np.arange(len(trace))
    else:
        raise ValueError('`var` ({}) must either an array or a metropolis Kernel'.format(
            type(var)))

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    # extract labels
    if 'xlabel' not in kwargs:
        kwargs['xlabel'] = 'Iteration'
    if 'ylabel' not in kwargs:
        kwargs['ylabel'] = 'Acceptance Rate'
    if 'label' not in kwargs:
        kwargs['label'] = 'Acceptance Rate'
    ax, kwargs = _set_plt_labels(kwargs, ax)

    # Calculate the acceptance rate overtime
    value = np.zeros(len(trace), dtype=float)
    value = _calc_acceptance_rate(ret=value, trace=trace, prev=prev)

    if 'alpha' not in kwargs:
        kwargs['alpha'] = 0.5
    try:
        if scatter:
            ax.scatter(points, np.squeeze(value), s=1.5, **kwargs)
        else:
            ax.plot(points, np.squeeze(value), **kwargs)
    except:
        logging.info('`ax.plot` failed. No points to plot')
    return ax

def render_trace(var: variables.Variable, idx: Union[int, Tuple[int, int]]=None, 
    ax: matplotlib.pyplot.Axes=None, plt_type: str=None, include_burnin: bool=True, 
    scatter: bool=True, log_scale: bool=False, n_burnin: int=None, section:str='posterior', 
    **kwargs) -> matplotlib.pyplot.Axes:
    '''
    Visualizes the Trace of a random variable.
    Produces a historgram of the values and a plot of the sample
    values over the iterations.

    Parameters
    ----------
    var : array, pylab.variables.Variable
        - Array to do the trace on.
        - If it is not a pylab variable object, then we need to pass in how many 
          burnin iterations there are with the parameter `n_burnin` orelse we 
          automatically set `include_burnin` to False
    idx : int, tuple, Optional
        - If the variable is not a scalar, this indexes the index that you want to trace on
    ax : matplotlib.pyplot.Axes, Optional
        - Axes to plot on
        - If nothing is provided, a new figure with a single subplot
          will be created and returned
    plt_type : str, Optional
        - The type of plot that will make
        - Options:
            - 'hist'
                - Histogram of the values. This is the posterior
            - 'trace' (Default)
                - The value of the random variable for each iteration
            - 'both' (Default)
                - Makes a new figure and plots 'hist' on the left
                  and 'trace' on the right.
                - If specified, `ax` is ignored.
    inlcude_burnin : bool, Optional
        - If True, it will plot the burnin trace as negative numbers
          in the trace as well
        - If `plt_type` == 'hist', `include_burnin` is automatically set to False
    scatter : bool
        - Only applies to plt_type=='trace'
        - If True, it will plot the points as a scatter plot
        - Else it will plot the points as a line
    log_scale : bool
        If True, plots the points in log-scale. If any of the points are negative, then we 
        take the absolute value of the values that we are plotting
    n_burnin : int
        Only required if `var` is an array, not a pylab.variables.Variable.
        Tells how big the burnin array is.
    kwargs : Optional values for the figure
        - Optional arguments:
            - 'color' : str
                - Default color is 'blue'
            - 'alpha' : float
                - Default alpha is 0.5 for the trace
            - 'title' : str
                - Nothing
            - 'xlabel' : str
                - Label of the horizontal axis
                - Default is 'Iteration' if the `plt_type` is 'trace'.
                - Default is 'Value' if the `plt_type` is 'hist'
            - 'ylabel' : str
                - Label of the vertical axis
                - Default is 'Value' if the `plt_type` is 'trace'.
                - Default is 'Count' if the `plt_type` is 'hist'
        
    Returns
    -------
    matplotlib.pyplot.Axes, 2-tuple
        - This is the Axis that contains the rendered figure
        - If `plt_type` == 'both' then it will return both Axes
    '''
    _valid_kwargs = ['color', 'alpha', 'xlabel', 'ylabel', 'label', 'title', 'rasterized']

    # Set defaults
    if plt_type is None:
        plt_type = DEFAULT_PLT_TYPE
    if 'color' not in kwargs:
        kwargs['color'] = DEFAULT_TRACE_COLOR
    if not pl.isbool(scatter):
        raise TypeError('`scatter` ({}) must be a bool'.format(type(scatter)))
    if not pl.isbool(log_scale):
        raise TypeError('`log_scale` ({}) must be a bool'.format(type(log_scale)))
    
    # check the kwargs
    for k in kwargs:
        if k not in _valid_kwargs:
            raise TypeError('`render_trace` got unexpected keyword argument "{}"'.format(k))

    if plt_type == 'both':
        fig = plt.figure()
        ax1 = render_trace(
            var=var, ax=fig.add_subplot(1,2,1), plt_type='hist',
            idx=idx, log_scale=log_scale, 
            n_burnin=n_burnin, section=section,  **kwargs)
        ax2 = render_trace(
            var=var, ax=fig.add_subplot(1,2,2), plt_type='trace',
            scatter=scatter, log_scale=log_scale, idx=idx, 
            n_burnin=n_burnin, section=section, **kwargs)
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        return ax1, ax2
    elif plt_type == 'hist':
        include_burnin = False

    if idx is not None:
        # Then it is multidimensional
        if type(idx) == int:
            idxs = (..., idx)
        elif type(idx) == tuple:
            idxs = (..., ) + idx
        else:
            raise ValueError('`idx` ({}) must either be an int or a tuple'.format(
                type(idx)))
    else:
        idxs = ()

    if not pl.isVariable(var):
        trace = np.asarray(var)
        if pl.isint(n_burnin):
            if n_burnin < 0:
                raise ValueError('`n_burnin` ({}) must be >= 0'.format(n_burnin))
            points = np.append(
                np.arange(-n_burnin, 0, 1),
                np.arange(len(trace) - n_burnin))
        else:
            points = np.arange(trace.shape[0])
        trace = trace[idxs]
        include_burnin = False
        
    else:
        if section == 'entire':
            section = 'posterior'
        if var.G.inference.tracer_filename is not None:
            trace = var.get_trace_from_disk(section=section)[idxs]
        else:
            trace = var.trace[idxs]

        if section == 'burnin':
            start = -var.G.inference.burnin
            end = len(trace) - var.G.inference.burnin
            points = np.arange(start,end,1)
        elif include_burnin:
            if var.G.inference.tracer_filename is not None:
                burnin_trace = var.get_trace_from_disk(section='burnin')
            else:
                burnin_trace = var.trace[:var.G.inference.burnin, ...]
            burnin_trace = burnin_trace[idxs]
            points = np.append(
                np.arange(-len(burnin_trace),0,1),
                np.arange(len(trace)))
            trace = np.append(burnin_trace, trace)
        else:
            points = np.arange(len(trace))

    try:
        if trace.ndim > 1:
            raise ValueError('`render_trace` only supports vectors ({})'.format(trace.shape))
    except:
        print(trace.shape)
        print(idxs)
        raise

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    # extract labels
    if 'xlabel' not in kwargs:
        if plt_type == 'hist':
            kwargs['xlabel'] = 'Value'
        else:
            kwargs['xlabel'] = 'Iteration'
    if 'ylabel' not in kwargs:
        if plt_type == 'hist':
            kwargs['ylabel'] = 'Probability'
        else:
            kwargs['ylabel'] = 'Value'
    if 'title' not in kwargs:
        if plt_type == 'hist':
            kwargs['title'] = 'Posterior'
        else:
            kwargs['title'] = 'Trace'
    ax, kwargs = _set_plt_labels(kwargs, ax)

    if plt_type == 'trace':
        if 'alpha' not in kwargs:
            kwargs['alpha'] = 0.5
        if 'label' not in kwargs:
            kwargs['label'] = 'Trace'
        try:
            if log_scale:
                if np.any(trace < 0):
                    logging.warning('Some values in trace are negative, take absolute value of vector')
                    trace = np.absolute(trace)
                ax.set_yscale('log')
            if scatter:
                ax.scatter(points, np.squeeze(trace), s=1.5, **kwargs)
            else:
                ax.plot(points, np.squeeze(trace), **kwargs)
        except:
            logging.info('`ax.plot` failed. No points to plot')
        if log_scale:
            ax.set_yscale('log')
    elif plt_type == 'hist':
        if 'alpha' not in kwargs:
            kwargs['alpha'] = 0.25
        if 'label' not in kwargs:
            kwargs['label'] = 'Posterior'
        try:
            if log_scale:
                if np.any(trace < 0):
                    logging.warning('Some values in trace are negative, take absolute value of vector')
                    trace = np.absolute(trace)
                hist, bins = np.histogram(trace, bins=30)
                logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
                ax.hist(x=trace, bins=logbins, density=True, **kwargs)
                ax.set_xscale('log')
            else:
                ax.hist(x=trace, density=True, bins=20, **kwargs)
            # ax.axvline(x=np.mean(trace), color = 'red')
            # ax.axvline(x=np.median(trace), color = 'blue')

        except Exception as e:
            logging.info('`ax.hist` failed: {}'.format(e))
            return None
    else:
        raise ValueError('plt_type ({}) not recognized'.format(plt_type))
    return ax

# ----------
# Abundances
# ----------
def alpha_diversity_over_time(subjs: Union[Subject, List[Subject]], metric: Callable, taxlevel: str=None,
    highlight: List[float]=None, marker: str='o', markersize: Union[float, int]=4, shade_perturbations: bool=True, 
    legend: bool=True, cmap: Any=None, alpha: float=1.0, linestyle: str='-', ax: matplotlib.pyplot.Axes=None, 
    grid: bool=False, colors: Any=None, **kwargs) -> matplotlib.pyplot.Axes:
    '''Plots the alpha diversity over time for the subject

    Parameters
    ----------
    subjs : pylab.base.Subject, list(pylab.base.Subject)
        This is the subject that we are getting the data from, or a list of subjects
        we are doing together
    metric : callable
        This is the function we want to calculate the alpha diversity with.
        This function usually comes from `diversity.alpha`
    taxlevel : str, None
        This is the taxonomic level to aggregate the data at. If 'default' is specified
        then it defaults the DEFAULT_TAX_LEVEL. If None then there will be no aggregation.
    highlight : list(float), None
        These are a list of tuples (subjectname, timepoint) we want to highlight (circle). 
        Each element must be a time in `subj.times`. If nothing is specified then we do not
        circle anything
    marker : str
        Type of marker to have on the plot
    markersize : numeric
        How big to make the marker
    shade_perturbations : bool
        If True, shade in the perturbations
    legend : bool
        If True, add a legend
    cmap : str
        This is the colormap to use. It uses `seaborn.color_palette` to generate the 
        colormap and `camp` is which colormap to use. Default (None) is `DEFAULT_SNS_CMAP`.
    alpha : float
        How dark to make the line. Default is 1.0
    linestyle : str
        Linestyle for matplotlib. The type is not checked with this because it is checked 
        within matplotlib
    kwargs : dict
        xlabel, ylabel, title : str
            Default 'title' is the type of alpha diversity and the level
        ax : matplotlib.pyplot.Axes, None
            This is the axis to plot on. If nothing is specified then we will create a 
            new figure with this as the only axis.
        figsize : 2-tuple
            Size of the figure
        legend : bool
            If True, add a legend
        Others for matplotlib.pylot.plot
    
    Returns
    -------
    matplotlib.pyplot.Axes
    '''
    # Type checking
    if not pl.isbool(grid):
        raise TypeError('`grid` ({}) must be a bool'.format(type(grid)))
    if pl.issubject(subjs):
        subjs = [subjs]
    if not pl.isarray(subjs):
        raise ValueError('`subjs` ({}) must be a pylab.base.Subject or an array'.format(type(subjs)))
    if not callable(metric):
        raise ValueError('`metric` ({}) must be callable'.format(type(metric)))
    if not pl.isbool(shade_perturbations):
        raise ValueError('`shade_perturbations` ({}) must be a bool'.format(
            type(shade_perturbations)))
    if not pl.isbool(legend):
        raise ValueError('If `legend` ({}) is specified, it must be a bool'.format(
            type(legend)))
    if cmap is None:
        cmap = DEFAULT_SNS_CMAP
    elif not pl.isstr(cmap):
        raise TypeError('`cmap` ({}) must either be None or a str'.format(type(cmap)))

    if 'title' not in kwargs:
        kwargs['title'] = '{} over time, {} level'.format( 
            metric.__name__.capitalize(), taxlevel)
    if 'xlabel' not in kwargs:
        kwargs['xlabel'] = 'Day'
    ax, kwargs = _set_plt_labels(d=kwargs, ax=ax)
    taxlevel = _set_taxlevel(taxlevel)

    if colors is None and 'color' not in kwargs:
        colors = sns.color_palette(cmap, n_colors=len(subjs))
    for sidx, subj in enumerate(subjs):
        df = subj.cluster_by_taxlevel(dtype='raw', taxlevel=taxlevel)[0]

        # Calculate the alpha diversity over time and plot
        vals = np.zeros(len(df.columns))
        for i in range(len(vals)):
            vals[i] = metric(df.values[:,i])
        if colors is not None:
            kwargs['color'] = colors[sidx]
        ax.plot(subj.times, vals, label=subj.name, marker=marker, markersize=markersize,
            alpha=alpha, linestyle=linestyle, **kwargs)

    # Check if there are any points to highlight
    if highlight is not None:
        for a in highlight:
            if type(a) != tuple:
                raise ValueError('Each element in highlight must be a tuple ({})'.format(
                    type(a)))
            if len(a) != 2:
                raise ValueError('The tuple must be length 2 ({})'.format(len(a)))
            
            subjname, timepoint = a
            subj = None
            for b in subjs:
                if b.name == subjname:
                    subj = b
            if subj is None:
                raise ValueError('`subjname` ({}) not a valid subject name'.format(subjname))
            if not pl.isnumeric(timepoint):
                raise ValueError('invalid timepoint ({}) in `highlight`. ' \
                    'It must be a numeric'.format(type(timepoint)))
            if timepoint not in subj.times:
                raise ValueError('timepoint ({}) not found in times ({})'.format(
                    timepoint, subj.times))
            tidx = np.searchsorted(subj.times, timepoint)
            ax.scatter( 
                [timepoint],
                [vals[tidx]],
                s=140,
                facecolor='None',
                edgecolors='black',
                zorder=100)

    ax = _set_xticks(ax)
    if shade_perturbations:
        ax = shade_in_perturbations(ax, subjs[0].parent.perturbations, subj=subjs[0])
    if legend:
        ax.legend(bbox_to_anchor=(1,1))
    if grid:
        ax.grid()
    return ax

def abundance_over_time(subj: Union[Subject, Study, List[Subject]], dtype: str, taxlevel: str=None, 
    yscale_log: bool=None, plot_abundant: int=None, plot_specific: Iterator[str]=None, 
    plot_clusters: Iterator[int]=None, highlight: List[float]=None, marker: str='o',
    ylim: Tuple[float, float]=None, markersize: Union[float, int]=4, shade_perturbations: bool=True, 
    legend: bool=True, set_0_to_nan: bool=False, color_code_clusters: bool=False, clustering: Clustering=None, 
    cmap: Any=None, alpha: float=1.0, linestyle: str='-', ax: matplotlib.pyplot.Axes=None,
    include_errorbars: bool=False, grid: bool=False, label_formatter: str=None, 
    label_func: Callable=None, **kwargs) -> matplotlib.pyplot.Axes:
    '''Plots the abundance over time for the Taxa in `subj`.

    What you're plotting
    --------------------
    - There are several different types of abundances you can plot, which is specified using
      the `dtype` (str) parameter:
        - 'raw'
            - This plots the counts of the Taxa. `subj` must be a single pl.base.Subject object.
        - 'rel'
            - This plots the relative abundance of the Taxa. `subj` must be a single 
        pl.base.Subject object.
        - 'abs'
            - This plots the absolute abundance of the Taxa. `subj` must be a single 
              pl.base.Subject object.
        - 'qpcr'
            - This plots the qPCR measurements at each time point. `subj` can also be a 
        list of pl.base.Subject objects, or a `pl.base.Study` object.
        - 'read-depth'
            - These are the the read depths at each timepoint. `subj` can also be a 
              list of pl.base.Subject objects, or a `pl.base.Study` object.
    
    Aggregating by taxanomic level
    ------------------------------
    If the taxonomy of the Taxa are specified, you can aggregate Taxa into specific 
    taxonomic levels and plot them as a trajectory by using the parameter `taxlevel`.
    Example: if `taxlevel='phylum'` then we add all of the abundances/reads of the Taxa
    that are in the same Phylum. If `taxlevel=None` then we do no aggregation. If you 
    set `taxlevel='default'` then it aggregates at the default taxonomic level, which 
    can be set using the function `plotting.set_default_tax_level(level)`. NOTE: these 
    are only necessary if dtype is either 'raw', 're', or 'abs'.

    The `label_formatter` (str) tells the function how to set the index of the dataframe
    it returns using `pylab.taxaname_formatter`. If nothing is specified then it 
    will return the entire taxonomy as a label for the taxon. NOTE, you cannot specifiy
    a taxonomy *below* that youre clustering at. For example, you cannot cluster at the 
    'class' level and then specify `'%(genus)s'` in `label_formatter`.

    What to plot?
    -------------
    - You can plot a subset of the Taxa by using the `plot_` arguments. If None of those
      parameters are specified, then it will plot everything. NOTE: You can only specify 
      one of the `plot_` at a time. NOTE: these are only necessary if dtype is either 
      'raw', 're', or 'abs'.

    - plot_abundant : int
        - If you want to only plot the x most abundant Taxa, specify that number 
          with `plot_abundant` (`int`) as a positive number. Example: `plot_abundant = 15` will
          only plot the 15 most abundant. If `plot_abundant` is a negative number, it will
          plot the least abundant. Example: `plot_abundant = -15` will only plot the 15 least 
          abundant. 
    
    - plot_specific : list
        - If you want to only plot specific Taxa, you can specify them by any identification
          (index, name, ID, etc.) as a list of Taxa to plot.
          NOTE: If you specify `plot_specific` and you are clustering along a taxonomic level, then
          you specify the names at the taxonomic level you clustered at.
        - Example::
            - If taxlevel = 'phylum'
            - VALID: plot_specific = [('Bacteria', 'Bacteroidetes'), ('Bacteria', 'Firmicutes')]
            - INVALID: plot_specific = ['Bacteroidetes', 'Firmicutes'] # Need full taxonomy
            - INVALID: plot_specific = ['Bacteroidia', 'Clostridia'] # This is at the class level
            - INVALID: plot_specific = ['Taxa_32'] # These names are no longer valid

    - plot_clusters : list(int), int
        - If you want to plot specific clusters (or a single cluster), you can specify the cluster
          ID/s to plot. Note that you must also specify the `clustering` parameter as well and you
          cannot aggregate the data into taxonomic classes.

    Log-scale and NaN's
    -------------------
    We will automatically plot the yscale as log for the 'raw' and 'abs' 
    datatypes and regular for 'rel'. If you want to do something different
    then set the `yscale_log` parameter.

    If you want to NaN out all of the points that have a zero abundance so that there are not
    vertical lines everywhere, use parameter `set_0_to_nan` to True

    Parameters
    ----------
    subj : pylab.base.Subject, pylab.base.Study, list(pylab.base.Subject)
        Subject/s we are getting the data from. Must be a single object if 
        `dtype` is 'raw', 'rel', or 'abs'. Must be multiple or single object.
        If `dtype` is 'qpcr' or 'read-depth', then this can also be a list of 
        Subject objects or a Study object,
    dtype : str
        Datatype to plot.
            'raw': Count data
            'rel': Relative abundance
            'abs': Absolute abudance 
            'qpcr': qPCR measurements
            'read-depth': read depths
    taxlevel : str, None
        This is the taxonomic level to aggregate the data at. If 'default' is specified
        then it defaults the DEFAULT_TAX_LEVEL. If None then there will be no aggregation.
    yscale : bool, None
        If True, it will plot the y-axis in log scale. If nothing is specified then it 
        will pick it automatically.
    plot_abundant : int, None
        If specified, it will plot only this number most abundant Taxa
    plot_specific : array_like, None
        If specified, it will only plot the Taxa specified indicated in here. Else
        it will plot everything
    plot_clusters : array_like, None
        If specified, only plots the clusters specified. Note that the `clustering` parameter
        must also be specified
    highlight : list(float), None
        These are a list of tuples (taxon, timepoint) we want to highlight (circle). 
        Each element must be a time in `subj.times`. If nothing is specified then we do not
        circle anything
    ylim : 2-tuple, Optional
        Sets lower and upper bounds for the y-axis. 
    marker : str
        Type of marker to have on the plot
    markersize : numeric
        How big to make the marker
    legend : bool
        If True, add a legend
    shade_perturbations : bool
        If True, shade in the perturbations
    color_code_clusters : bool
        If True, set all of the Taxa in a cluster to the same color. Note that the `clustering` 
        parameter must also be specified. NOTE: if you are aggregating the data at a taxonomic 
        level (taxlevel is not None or 'taxon'), then this is automatically overridden to False.
        If specified, it overrides the legend to False.
    clustering : pylab.cluster.Clustering
        Clustering object. Only necessary if `plot_clusters` or `color_code_clusters` are specified
    cmap : str
        This is the colormap to use. It uses `seaborn.color_palette` to generate the 
        colormap and `camp` is which colormap to use. Default (None) is `DEFAULT_SNS_CMAP`.
    alpha : float
        How dark to make the line. Default is 1.0
    linestyle : str
        Linestyle for matplotlib. The type is not checked with this because it is checked 
        within matplotlib
    include_errorbars : bool
        If True, we include the errorbars (standard deviation) for each point. This is only
        used if `dtype` == 'qpcr'
    highlight : list(2-tuple), None
        If this is specified, this highlights the points specified in the 2-tuple:
        (subject ID, timepoint). This is only used if `dtype` == 'qpcr' or 'read-depth'
    grid : bool
        If True, plots with a grid
    kwargs : dict
        xlabel, ylabel, title : str
            Default 'title' is the type of alpha diversity and the level
        ax : matplotlib.pyplot.Axes, None
            This is the axis to plot on. If nothing is specified then we will create a 
            new figure with this as the only axis.
        figsize : 2-tuple
            Size of the figure

    Returns
    -------
    matplotlib.pyplot.Axes
    '''
    # Type checking
    taxlevel = _set_taxlevel(taxlevel)
    if not pl.isstr(dtype):
        raise TypeError('`dtype` ({}) must be a str'.format(type(dtype)))
    if dtype not in ['raw', 'rel', 'abs', 'qpcr', 'read-depth']:
        raise TypeError('`dtype` ({}) not recognized'.format(dtype))
    if dtype == 'qpcr':
        if not pl.isbool(include_errorbars):
            raise ValueError('`include_errorbars ({}) must be a bool'.format(type(include_errorbars)))
    if not pl.isbool(grid):
        raise TypeError('`grid` ({}) must be a bool'.format(type(grid)))
    if not pl.issubject(subj):
        if dtype not in ['qpcr', 'read-depth']:
            raise TypeError('`subj` ({}) must be a pylab.base.Subject'.format(type(subj)))
        else:
            if pl.isarray(subj):
                for s in subj:
                    if not pl.issubject(s):
                        raise TypeError('Every element in `subj` ({}) must be a ' \
                            'pl.base.Subject'.format(type(s)))
            if not pl.isstudy(subj):
                raise TypeError('`subj` ({}) type not recognized'.format(type(subj)))
    if yscale_log is None:
        if dtype == 'rel':
            yscale_log = False
        else:
            yscale_log = True
    if not pl.isbool(yscale_log):
        raise TypeError('`yscale_log` ({}) must be a bool'.format(
            type(yscale_log)))
    if not pl.isbool(shade_perturbations):
        raise TypeError('`shade_perturbations` ({}) must be a bool'.format(
            type(shade_perturbations)))
    if not pl.isbool(legend):
        raise TypeError('If `legend` ({}) is specified, it must be a bool'.format(
            type(legend)))
    if not pl.isbool(set_0_to_nan):
        raise TypeError('`set_0_to_nan` ({}) must be a bool'.format(type(set_0_to_nan)))
    if clustering is not None:
        if not pl.isclustering(clustering):
            raise TypeError('`clustering` ({}) must be a pylab.cluster.Clustering' \
                ' object if specified'.format(type(clustering)))
    if color_code_clusters and clustering is None:
        raise ValueError('If `color_code_clusters` is True, `clustering` must also' \
            ' be specified')
    if cmap is None:
        cmap = DEFAULT_SNS_CMAP
    elif not pl.isstr(cmap):
        raise TypeError('`cmap` ({}) must either be None or a str'.format(type(cmap)))
    if dtype in ['raw', 'rel', 'abs']:
        if plot_abundant is not None:
            if not pl.isint(plot_abundant):
                raise TypeError('`plot_abundant` ({}) must be an int'.format(
                    type(plot_abundant)))
            if plot_abundant == 0:
                raise ValueError('`plot_abundant` cannot be zero')
        if plot_specific is not None:
            if not pl.isarray(plot_specific):
                plot_specific = [plot_specific]
        if pl.isint(plot_clusters):
            plot_clusters = [plot_clusters]
        if plot_clusters is not None:
            if not pl.isarray(plot_clusters):
                raise TypeError('If `plot_clusters` ({}) is specified, it must either be ' \
                    'an int or an array'.format(type(plot_clusters)))
            else:
                for ele in plot_clusters:
                    if not pl.isint(ele):
                        raise TypeError('Every element in `plot_clusters` ({}) ({}) must ' \
                            'be an int'.format(plot_clusters,
                                pl.itercheck(plot_clusters, pl.isint)))
            if taxlevel != 'taxon':
                raise ValueError('Cannot plot clusters (`plot_clusters` ({})) and aggregate by a' \
                    ' taxonomic level (`taxlevel` ({}))'.format(plot_clusters, taxlevel))            
            if not pl.isclustering(clustering):
                raise ValueError('If `plot_clusters` is specified, then clustering ({}) must be ' \
                    'specified'.format(type(clustering)))
        if not pl.isbool(color_code_clusters):
            raise TypeError('`color_code_clusters` ({}) must be a bool'.format(
                type(color_code_clusters)))
        if color_code_clusters:
            if taxlevel != 'taxon':
                logging.warning('Overriding `color_code_clusters` to False because `taxlevel`' \
                    ' ({}) is not None nor "taxon"'.format(taxlevel))
                color_code_clusters = False
            if legend:
                logging.warning('Overriding `legend` to False because `color_code_clusters` is True')
            legend = False

        _cumm = 0
        if plot_abundant is not None:
            _cumm += 1
        if plot_specific is not None:
            _cumm += 1
        if plot_clusters is not None:
            _cumm += 1
        if _cumm > 1:
            raise ValueError('Only one `plot_` parameter can be specified. You specified {}'.format(_cumm))
        
        if taxlevel == 'asv':
            df = subj.df()[dtype]
        else:
            df = subj.cluster_by_taxlevel(dtype=dtype, taxlevel=taxlevel,
                index_formatter=label_formatter)[0]
        times = subj.times

        if 'title' not in kwargs:
            kwargs['title'] = 'Abundance, Subject {}, {} level'.format(subj.name, taxlevel)
        if 'xlabel' not in kwargs:
            kwargs['xlabel'] = 'Days'
        if 'ylabel' not in kwargs:
            if dtype == 'rel':
                kwargs['ylabel'] = 'Relative abundance'
            elif dtype == 'raw':
                kwargs['ylabel'] = 'Counts'
            else:
                kwargs['ylabel'] = 'CFUs/g'

        ax, kwargs = _set_plt_labels(d=kwargs, ax=ax)
        idxs = np.arange(len(df.index))
        if plot_specific is not None:
            idxs = []
            for a in plot_specific:
                temp_taxa = subj.taxa[a]
                idxs.append(temp_taxa.idx)
        elif plot_abundant is not None:
            matrix = df.values
            abnds = np.sum(matrix, axis=1)
            if plot_abundant < 0:
                plot_abundant *= -1
                idxs = np.argsort(abnds)[:plot_abundant]
            else:
                idxs = np.argsort(abnds)[-plot_abundant:]

        elif plot_clusters is not None:
            idxs = []
            for cid in plot_clusters:
                idxs = np.append(idxs, list(clustering[cid].members))
            idxs = np.asarray(idxs, dtype=int)

        if color_code_clusters:
            colors = {}
            ccs = sns.color_palette(cmap, n_colors=len(clustering.clusters))
            for i, cluster in enumerate(clustering):
                for idx in cluster.members:
                    colors[idx] = ccs[i]
        else:
            ccs = sns.color_palette(cmap, n_colors=len(idxs))
            colors = {}
            for i,idx in enumerate(idxs):
                colors[idx] = ccs[i]

                
        for idx in idxs:
            label = df.index[idx]
            # Only get the last name of the entire taxonomy
            label = re.findall(r"([^,)(' ]+)",label)[-1]
            datapoints = np.asarray(list(df.iloc[idx]))

            if label_func is not None:
                label = label_func(subj.taxa[label], taxa=subj.taxa)
            # print(label) 

            if set_0_to_nan:
                for i,val in enumerate(datapoints):
                    if val == 0:
                        datapoints[i] = np.nan
            
            ax.plot(times, datapoints, label=label, marker=marker, color=colors[idx], 
                markersize=markersize, alpha=alpha, linestyle=linestyle, **kwargs)
        
        if highlight is not None:
            logging.warning('`highlight` is not implemented for dtype ({}). Skipping'.format(
                dtype))
    else:
        if dtype == 'qpcr':
            if 'title' not in kwargs:
                kwargs['title'] = 'qPCR over time'
            if 'xlabel' not in kwargs:
                kwargs['xlabel'] = 'Day'
            if 'ylabel' not in kwargs:
                kwargs['ylabel'] = 'CFUs/g'
        else:
            if 'title' not in kwargs:
                kwargs['title'] = 'Read depth over time'
            if 'xlabel' not in kwargs:
                kwargs['xlabel'] = 'Day'
            if 'ylabel' not in kwargs:
                kwargs['ylabel'] = 'Counts'
        ax, kwargs = _set_plt_labels(d=kwargs, ax=ax)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if pl.issubject(subj):
            subj = [subj]
        colors = sns.color_palette(cmap, n_colors=len(subj))
        
        for sidx, ss in enumerate(subj):
            subj_kwargs = {'label': ss.name, 'marker': marker,
                'color': colors[sidx], 'markersize': markersize, 
                'alpha': alpha, 'linestyle': linestyle}

            if dtype == 'qpcr':
                mean = np.zeros(len(ss.times))
                std = np.zeros(len(ss.times))
                for i, t in enumerate(ss.times):
                    mean[i] = ss.qpcr[t].mean()
                    std[i] = ss.qpcr[t].std()
                if not include_errorbars:
                    std = None
                ax.errorbar(ss.times, mean, yerr=std, **subj_kwargs)
            else:
                rds = []
                for t in ss.times:
                    rds.append(np.sum(ss.reads[t]))
                ax.plot(ss.times, rds, **subj_kwargs)
        
        # Check if there are any points to highlight
        if highlight is not None:
            for a in highlight:
                if type(a) != tuple:
                    raise ValueError('Each element in highlight must be a tuple ({})'.format(
                        type(a)))
                if len(a) != 2:
                    raise ValueError('The tuple must be length 2 ({})'.format(len(a)))
                
                subjname, timepoint = a
                ss = None
                for b in subj:
                    if b.name == subjname:
                        ss = b
                if ss is None:
                    raise ValueError('`subjname` ({}) not a valid subject name'.format(subjname))
                if not pl.isnumeric(timepoint):
                    raise ValueError('invalid timepoint ({}) in `highlight`. ' \
                        'It must be a numeric'.format(type(timepoint)))
                if timepoint not in ss.times:
                    raise ValueError('timepoint ({}) not found in times ({})'.format(
                        timepoint, ss.times))
                if dtype == 'qpcr':
                    yy = ss.qpcr[timepoint].mean()
                else:
                    yy = np.sum(ss.reads[timepoint])
                ax.scatter( 
                    [timepoint],
                    [yy],
                    s=140,
                    facecolor='None',
                    edgecolors='black',
                    zorder=100)

    # Set the other parameters for the plot
    if yscale_log:
        ax.set_yscale('log')
    ax = _set_xticks(ax)
    if shade_perturbations:
        if pl.issubject(subj):
            perturbations = subj.parent.perturbations
        elif pl.isstudy(subj):
            perturbations = subj.perturbations
        else:
            # Else it is an array of subjsets
            perturbations = subj[0].parent.perturbations
        ax = shade_in_perturbations(ax, perturbations, subj=subj)

    if legend:
        ax.legend(bbox_to_anchor=(1,1))
    if ylim is not None:
        ax.set_ylim(*ylim)
    if grid:
        ax.grid()
    return ax

def taxonomic_distribution_over_time(subj: Union[Subject, Study], taxlevel: str=None,
    legend: bool=True, ax: matplotlib.pyplot.Axes=None, plot_abundant: int=None, 
    label_formatter: str=None, dtype: str='rel', shade_perturbations: bool=True, 
    **kwargs) -> matplotlib.pyplot.Axes:
    '''Produces a taxonomic bar graph for each datapoint

    Aggregating by taxanomic level
    ------------------------------
    If the taxonomy of the Taxa are specified, you can aggregate Taxa into specific 
    taxonomic levels and plot them using the parameter `taxlevel`.
    Example: if `taxlevel='phylum'` then we add all of the reads of the Taxa
    that are in the same Phylum. If `taxlevel=None` then we do no aggregation. If you 
    set `taxlevel='default'` then it aggregates at the default taxonomic level, which 
    can be set using the function `plotting.set_default_tax_level(level)`.

    The `label_formatter` (str) tells the function how to set the index of the dataframe
    it returns using `pylab.taxaname_formatter`. If nothing is specified then it 
    will return the entire taxonomy as a label for the taxon. NOTE, you cannot specifiy
    a taxonomy *below* that youre clustering at. For example, you cannot cluster at the 
    'class' level and then specify `'%(genus)s'` in `label_formatter`.

    plot_abundant : int
    If you want to only plot the x most abundant Taxa, specify that number 
    with `plot_abundant` (`int`) as a positive number. Example: `plot_abundant = 15` will
    only plot the 15 most abundant. If `plot_abundant` is a negative number, it will
    plot the least abundant. Example: `plot_abundant = -15` will only plot the 15 least 
    abundant. The abundances are calculated using the reads.

    Parameters
    ----------
    subj : pylab.base.Subject, pl.base.Study
        Subject we are getting the data from. If it is a subjectset then we average over
        all of the subjects
    taxlevel : str, None
        This is the taxonomic level to aggregate the data at. If 'default' is specified
        then it defaults the DEFAULT_TAX_LEVEL. If None then there will be no aggregation.
    plot_abundant: int, None
        If specified, only plots the top or bottom `plot_abundant` elements.
        If None then nothing happens
    label_formatter : str, None
        If specified, it will tell how to make the legend using the taxaname_formatter.
        This can only be used if the `taxlevel` is not specified.
    kwargs : dict
        xlabel, ylabel, title : str
            Default 'title' is the type of alpha diversity and the level
        ax : matplotlib.pyplot.Axes, None
            This is the axis to plot on. If nothing is specified then we will create a 
            new figure with this as the only axis.
        figsize : 2-tuple
            Size of the figure
        legend : bool
            If True, add a legend
    
    Returns
    -------
    matplotlib.pyplot.Axes
    ''' 
    # Type checking
    if not (pl.issubject(subj) or pl.isstudy(subj)):
        raise ValueError('`subj` ({}) must be a pylab.base.Subject or pl.base.Study'.format(type(subj)))
    if not pl.isbool(legend):
        raise ValueError('`legend` ({}) must be a bool'.format(type(legend)))
    taxlevel = _set_taxlevel(taxlevel)

    if plot_abundant is not None:
        if not pl.isint(plot_abundant):
            raise TypeError('`plot_abundant` ({}) must be an int'.format(
                type(plot_abundant)))
        if plot_abundant == 0:
            raise ValueError('`plot_abundant` cannot be zero')

    if 'title' not in kwargs:
        kwargs['title'] = 'Taxonomic Distribution, {} level'.format(taxlevel.capitalize())
    if 'xlabel' not in kwargs:
        kwargs['xlabel'] = 'Days'
    if 'ylabel' not in kwargs:
        kwargs['ylabel'] = 'Relative Abundance'

    ax, kwargs = _set_plt_labels(d=kwargs, ax=ax)
    if len(kwargs) != 0:
        raise ValueError('Arguemnts {} not recognized'.format(list(kwargs.keys())))

    if pl.isstudy(subj):
        raise NotImplementedError('Not implemented yet')
        # # average over all of the subjects
        # dfs = []

        # # Only have the times that are consistent across all subjects        
        # for s in subj:

        #     dfs.append(s.cluster_by_taxlevel(dtype=dtype, taxlevel=taxlevel, 
        #         index_formatter=label_formatter))
        # df = dfs[0]
        # for i in range(1, len(dfs)):
        #     df = df + dfs[i]
        # df = df/len(dfs)

    else:
        df = subj.cluster_by_taxlevel(dtype=dtype, taxlevel=taxlevel, 
            index_formatter=label_formatter)[0]

    if plot_abundant is not None:
        matrix = df.values
        abnds = np.sum(matrix, axis=1)
        if plot_abundant < 0:
            plot_abundant *= -1
            idxs = np.argsort(abnds)[:plot_abundant]
        else:
            idxs = np.argsort(abnds)[-plot_abundant:]
        df = df.iloc[idxs, :]

        # Add everything else as 'Everything else'
        vals = []
        for col in df.columns:
            vals.append(1 - df[col].sum())
        df2 = pandas.DataFrame([vals], columns=df.columns, index=['Other'])
        df = df.append(df2)
        
    kwargs = {}
    if dtype == 'abs':
        kwargs['logy'] = True

    ax = df.T.plot(ax=ax, kind='bar', stacked=True, **kwargs)
    if shade_perturbations:
        if pl.isstudy(subj):
            for sss in subj:
                ax = shade_in_perturbations(ax, subj.perturbations, subj=sss)
        else:
            ax = shade_in_perturbations(ax, subj.parent.perturbations, subj=subj)

    if legend:
        # handles, labels = ax.get_legend_handles_labels()
        # Reverse the inherit order of the legend so it matches the graph
        # ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1,1))
        ax.legend(bbox_to_anchor=(1,1), )

    # ax = _set_xticks(ax)
    return ax

def aggregate_taxa_abundances(subj: Subject, agg: Union[str, OTU, int], dtype: str='rel', 
    yscale_log: bool=True, ax: matplotlib.pyplot.Axes=None, title: str='Subject %(subjectname)s', 
    xlabel: str='auto', ylabel: str='auto', vmin: Union[float, int]=None, vmax: Union[float, int]=None,
    alpha_agg: float=0.5, alpha_asv: float=1., legend: bool=True, fontstyle: str=None, 
    shade_perturbations: bool=True) -> matplotlib.pyplot.Axes:
    '''Plot the abundances of the aggregated Taxa within the OTU `agg` for the subject `subj`

    Each subject within the study has its own axis within the figure. If you want to
    plot only specific subjects, pass them in with the parameter `subjs`

    Parameters
    ----------
    subj : mdsine2.Subject
        This is the subject oobject that contains all of the data for the subjsets as well as the Taxa
    agg : str, mdsine2.OTU, int
        This is the identifier for the aggregate Taxa you want to plot
    dtype : str
        This is the type of plot you want. Options:
            'raw' : Counts
            'rel' : Relative abundance
            'abs' : Total abundance data
    yscale_log : bool
        If True, plot the yscale in log scale.
    ax : matplotlib.pyplot.Axes
        If passed in, this is the Axes to plot on. Else we create a new figure where
        it only has one axis
    title : str, None
        This is the format to set the titles for each one of the subplots. Options:
            '%(subjectname)s' : Replaces this with the name of the subject
            '%(subjectid)s' : Replaces this with the id of the subject
            '%(subjectindex)s' : Replaces this with the index of the subject
        If None then do not set
    ylabel : str, None
        This is the y-axis label of the figure. If None, do not plot. If `ylabel='auto'`:
            `ylabel = 'Counts'` if `dtype='raw'`
            `ylabel = 'Relative Abundance'` if `dtype='rel'`
            `ylabel = 'CFU/g'` if `dtype='abs'`
    xlabel : str, None
        This is the x-axis label. If None, do not set. If `xlabel='auto'`, set to 'Time (days)'.
    vmin, vmax : float
        These are the minimum and maximum values in the yaxis, respectively.
    alpha_agg : float
        This is the alpha of the aggregate Taxa plot
    alpha_asv : float
        This is the alpha of the individual Taxa plots
    legend : bool
        If True, render the legend on the right hand side
    fontstyle : str
        If None, set to default fontsize of matplotlib. Options:
            'paper' : set to paper style fontsize
            'poster' : set to poster style fontsize
    shade_perturbations : bool
        If True, shade in and lable the perturbations

    Returns
    -------
    matplotlib.pyplot.Axes
    '''
    def _tax_is_defined(tax, level):
        return (type(tax) != float) and (tax != DEFAULT_TAXLEVEL_NAME) and (tax != '')

    def _agg_taxaname_for_paper(agg, taxaname):
        '''Makes the name in the format needed for the paper for an OTU
        '''
        if _tax_is_defined(agg.aggregated_taxonomies[taxaname], 'species'):
            species = agg.aggregated_taxonomies[taxaname]['species']
            species = species.split('/')
            if len(species) >= 3:
                species = species[:2]
            species = '/'.join(species)
            label = '{genus} {spec} {name}'.format(
                    genus=agg.aggregated_taxonomies[taxaname]['genus'],
                    spec=species,
                    name=taxaname)
        elif _tax_is_defined(agg.aggregated_taxonomies[taxaname], 'genus'):
            label = '* {genus} {name}'.format(
                    genus=agg.aggregated_taxonomies[taxaname]['genus'],
                    name=taxaname)
        elif _tax_is_defined(agg.aggregated_taxonomies[taxaname], 'family'):
            label = '** {family} {name}'.format(
                    family=agg.aggregated_taxonomies[taxaname]['family'],
                    name=taxaname)
        elif _tax_is_defined(agg.aggregated_taxonomies[taxaname], 'order'):
            label = '*** {order} {name}'.format(
                    order=agg.aggregated_taxonomies[taxaname]['order'],
                    name=taxaname)
        elif _tax_is_defined(agg.aggregated_taxonomies[taxaname], 'class'):
            label = '**** {clas} {name}'.format(
                    clas=agg.aggregated_taxonomies[taxaname]['class'],
                    name=taxaname)
        elif _tax_is_defined(agg.aggregated_taxonomies[taxaname], 'phylum'):
            label = '***** {phylum} {name}'.format(
                    phylum=agg.aggregated_taxonomies[taxaname]['phylum'],
                    name=taxaname)
        elif _tax_is_defined(agg.aggregated_taxonomies[taxaname], 'kingdom'):
            label = '****** {kingdom} {name}'.format(
                    kingdom=agg.aggregated_taxonomies[taxaname]['kingdom'],
                    name=taxaname)
        return label

    if not pl.issubject(subj):
        raise TypeError('`subj` ({}) must be a mdsine2.Subject object'.format(type(subj)))
    if agg not in subj.taxa:
        raise ValueError('`agg` ({}) not found in study'.format(agg))
    if dtype not in ['rel', 'abs', 'raw']:
        raise ValueError('`dtype` ({}) not recognized'.format(dtype))
    if fontstyle is None:
        titlefontsize = None
        labelfontsize = None
        tickfontsize = None
        legendfontsize = None
    elif fontstyle == 'paper':
        titlefontsize = None
        labelfontsize = None
        tickfontsize = None
        legendfontsize = None
    elif fontstyle == 'poster':
        titlefontsize = None
        labelfontsize = None
        tickfontsize = None
        legendfontsize = None
    else:
        raise ValueError('`fontstyle` ({}) not recognized'.format(fontstyle))

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
    agg = subj.taxa[agg]
    M = subj.matrix()[dtype]

    # Plot the aggregate
    labelotu = pl.taxaname_for_paper(taxon=agg, taxa=subj.taxa)
    ax.plot(subj.times, M[agg.idx, :], label=labelotu, alpha=alpha_agg, linewidth=7, 
        marker='x')

    individ_trajs = {}
    for taxaname in agg.aggregated_taxa:
        if taxaname not in subj._reads_individ:
            raise ValueError('This should not happend. Failing.')
        temp = []
        for t in subj.times:
            abund = subj._reads_individ[taxaname][t]

            if dtype == 'rel':
                abund = abund / np.sum(subj.reads[t])
            if dtype == 'abs':
                abund = abund * subj.qpcr[t].mean()
            temp.append(abund)

        label = _agg_taxaname_for_paper(agg=agg, taxaname=taxaname)
        individ_trajs[label] = temp
    
    for label in individ_trajs:
        ax.plot(subj.times, individ_trajs[label], label=label, alpha=alpha_asv, 
            linewidth=2, marker='x')

    if vmin is not None:
        ax.set_ylim(bottom=vmin)
    if vmax is not None:
        ax.set_ylim(top=vmax)

    if yscale_log:
        ax.set_yscale('log')
    if title is not None:
        title = title.replace('%(subjectname)s', subj.name)
        title = title.replace('%(subjectid)s', str(subj.id))
        title = title.replace('%(subjectindex)s', str(subj.index))
        ax.set_title(title, fontsize=titlefontsize)

    if xlabel is not None:
        if xlabel == 'auto':
            xlabel = 'Time (days)'
        ax.set_xlabel(xlabel, fontsize=labelfontsize)
    if ylabel is not None:
        if ylabel == 'auto':
            if dtype == 'raw':
                ylabel = 'Counts'
            elif dtype == 'rel':
                ylabel = 'Relative Abundance'
            else:
                ylabel = 'CFU/g'
        ax.set_ylabel(ylabel, fontsize=labelfontsize)

    if legend:
        ax.legend(fontsize=legendfontsize, bbox_to_anchor=(1.05, 1))
    
    ax = _set_xticks(ax)
    ax = _set_tick_fontsize(ax, fontsize=tickfontsize)
    if shade_perturbations:
        ax = shade_in_perturbations(ax, perturbations=subj.perturbations, subj=subj,
            textsize=legendfontsize)

    return ax

# ---------------
# INNER FUNCTIONS
# ---------------
def _is_just_zero_or_nan(matrix: np.ndarray) -> bool:
    '''Returns True if the input matrix has only zero or only NaNs

    Parameters
    ----------
    matrix : np.ndarray(n,n)
        - Square matrix

    Returns
    -------
    bool
        True if `matrix` has only 0s or `np.nan`s
    '''
    matrix = np.asarray(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not (matrix[i,j] == 0 or np.isnan(matrix[i,j])):
                return False
    return True

def _format_ticklabel(format: str, order: Union[List[int], np.ndarray], taxa: TaxaSet) -> List[str]:
    '''Format the xtick labels witha  slightly different format than that in 
    `pylab.taxaname_formatter`: Overrides the %(index)s to be where it appears 
    in the local order, not the global order

    Parameters
    ----------
    format : str
        - This is the format to make the ticklabel for each Taxa. The format can be 
          seen in `pylab.taxaname_formatter`.
    order : array_like
        - This is the list of the Taxa indices in the order that they should appear.
    taxa : pylab.base.TaxaSet
        - This is where all the information is stored for the Taxa

    Returns
    -------
    list(str)
        - These are the list of strings that are the labels for the ticks

    See also
    --------
    pylab.taxaname_formatter
    '''
    ticklabels =[]

    for num, idx in enumerate(order):
        fmt = format.replace('%(index)s', str(num))
        label = pl.taxaname_formatter(format=fmt, taxon=idx, taxa=taxa)
        ticklabels.append(label)
    return ticklabels

def _set_heatmap_default_args(linewidths: float=None, linecolor: str=None, n_colors: int=None, 
    xticklabels: str=None, yticklabels: str=None, include_colorbar: bool=None, 
    include_tick_marks: str=None) -> Dict[str, Any]:
    '''Sets the defaults of the above parameters if they are None.

    Parameters
    ----------
    linewidths : float
        How wide the lines should be on a heatmap
    linecolor : str
        What color the lines should be on the heatmap
    n_colors : int
        How many colors to make the palette with
    xticklabels : str
        Label format on the x-axis of the heatmap
    yticklabels : str
        Label format on the y-axis of the heatmap
    include_colorbar : bool
        If True, includes the colorbar on the Axes render
    include_tick_marks : bool
        If True includes the tickmarks along the perimeter of the axis
    
    Returns
    -------
    dict
        Dictionary of all the values
    '''
    if linewidths is None:
        linewidths = DEFAULT_LINEWIDTHS
    if linecolor is None:
        linecolor = 'blue'
    if n_colors is None:
        n_colors = 100
    if xticklabels is None:
        xticklabels = '%(index)s'
    if yticklabels is None:
        yticklabels = '%(name)s %(index)s'
    if include_colorbar is None:
        include_colorbar = True
    if include_tick_marks is None:
        include_tick_marks = False

    return {'linewidths': linewidths, 'linecolor': linecolor, 'n_colors': n_colors, 
        'xticklabels': xticklabels, 'yticklabels': yticklabels, 
        'include_colorbar': include_colorbar, 'include_tick_marks': include_tick_marks}

def _init_parameters_heatmap(matrix: np.ndarray, taxa: TaxaSet, clustering: Clustering, 
    xticklabels: str, yticklabels: str, ax: matplotlib.pyplot.Axes, figure_size: Tuple[float, float],
    linewidths: Union[float, int], order: Iterator[int]) -> Dict[str, Any]:
    '''Checks if the parameters are initialized correctly for the standard arguments
    for `render_interaction_strength`, `render_cocluster_probabilities`, and
    `render_bayes_factors`.

    Parameters
    ----------
    matrix : array_like (2-dimensional)
        - 2D matrix that can be casted to a numpy ndarray
    taxa : pylab.Base.TaxaSet
        - This is the TaxaSet that contains all of the information about the Taxa
    clustering : pylab.cluster.Clustering
        - This is the clustering object that tells the assignments for each of the Taxa
    xticklables, yticklables : str, array_like
        - These are either the string formats to make each of the labels or they are 
          the actual labels for each of the entries for the horizontal or vertical axis
    ax : matplotlib.pyplot.Axes, None
        - This is the Axis to plot on. If nothing is provided then a Figure and an Axis
          object will be created
    figure_size : 2-tuple(numeric, numeric), None
        - This is the size to create the figure. If nothing is provided it will be
          automatically set based on the number of Taxa
    linewidths : numeric
        - If the number of Taxa exceeds a certain number (75), it will automatically set 
          the linewidths to 0
    order : array_like, None
        - This is the order to set for the Taxa. The order has the Taxa identifier (name, index, 
          id, etc.) for each Taxa. If None then it does it by the order of the Taxa or by the 
          clustering object.
    Returns
    -------
    dict
        'matrix'
            - This is the input matrix, reorganized if necessary
        'xticklabels'
            - These are the labels for the horizontal axis based on the input format, if 
              necessary
        'yticklabels'
            - These are the labels for the vertical axis based on the input format, if 
              necessary
        'ax'
            - This is the matplotlib.pyplot.Axes object that has the appropriate scaled 
              size
        'fig'
            - This is the matplotlib.pyplot.Figure object that is associated with `ax`
        'cbar_kws'
            - These are the colobar arguments, if any
        'linewidths'
            - These are the linewidths for the rendering
    '''
    # Type checking
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError('`matrix` ({}) must be 2 dimensions'.format(
            matrix.ndim))
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError('`matrix` ({}) must be square'.format(
            matrix.shape))
    if taxa is not None:
        if not pl.istaxaset(taxa):
            raise ValueError('`taxa` ({}) must be a subclass of pylab.data.TaxaSet'.format(
                taxa.__class__))
        if matrix.shape[0] != taxa.n_taxa:
            raise ValueError('The size of the interaction matrix ({}) must ' \
                'equal the number of Taxa in `taxa` ({})'.format(
                    matrix.shape[0], taxa.n_taxa))
    if clustering is not None:
        if not pl.isclustering(clustering):
            raise ValueError('`clustering` ({}) must be a subclass of ' \
                'pylab.cluster.ClusteringBase'.format(clustering.__class__))
    if not (xticklabels is None or type(xticklabels) == str or pl.isarray(xticklabels)):
        raise ValueError('xticklabels ({}) must either be None, str, or a list/np.ndarray' \
            ''.format(type(xticklabels)))
    if type(xticklabels) == list or type(xticklabels) == np.ndarray:
        if len(xticklabels) != matrix.shape[0]:
            raise ValueError('If xticklabels is a list, the length ({}) must ' \
                'be the same as the matrix shape ({})'.format(len(xticklabels), matrix.shape[0]))
    if not (yticklabels is None or type(yticklabels) == str or type(yticklabels) == list \
        or type(yticklabels) == np.ndarray):
        raise ValueError('yticklabels ({}) must either be None, str, or a list/np.ndarray' \
            ''.format(type(yticklabels)))
    if type(yticklabels) == list or type(yticklabels) == np.ndarray:
        if len(yticklabels) != matrix.shape[0]:
            raise ValueError('If yticklabels is a list, the length ({}) must ' \
                'be the same as the matrix shape'.format(len(yticklabels), matrix.shape[0]))
    if order is not None:
        if not pl.isarray(order):
            raise TypeError('`order` ({}) if specified must be an array'.format(type(order)))
        if len(order) != len(taxa):
            raise TypeError('`order` ({}) must be {} elements'.format(len(order), len(taxa)))
        for i in order:
            if i not in taxa:
                raise IndexError(' identifier ({}) not found in s'.format(i))

    if taxa is not None:
        N = len(taxa)
    else:
        N = matrix.shape[0]

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        # Set figure size
        if figure_size is None:
            # For each 50 Taxa add 10 inches to each dimension
            x = math.ceil(N/50)
            l = int(10*x)
            figure_size = (l,l)
        fig.set_size_inches(*figure_size)
    else:
        fig = plt.gcf()

    # Make taxa order based on clusters if necessary
    if order is not None and matrix.shape[0] == len(taxa):
        temp = []
        for oidx in order:
            temp.append(taxa[oidx].idx)
        order = temp
    elif clustering is not None:
        # get the index order
        reordered_idxs = []
        for cid in clustering.clusters:
            for oidx in clustering.clusters[cid].members:
                reordered_idxs.append(oidx)
        order = reordered_idxs
    else:
        order = np.arange(N)

    # reorganize
    temp = np.zeros(shape=matrix.shape, dtype=float)
    for i in range(temp.shape[0]):
        for j in range(temp.shape[1]):
            temp[i,j] = matrix[
                order[i],
                order[j]]
    matrix = temp
        
    if xticklabels is None:
        xticklabels = False
    if yticklabels is None:
        yticklabels = False
    if type(xticklabels) == str:
        if taxa is None:
            logging.warning('Automatically setting xlabels as index because there are no taxa')
            xticklabels = ['{}'.format(i+1) for i in range(matrix.shape[0])]
        else:
            xticklabels = _format_ticklabel(format=xticklabels, order=order, taxa=taxa)
    
    if type(yticklabels) == str:
        if taxa is None:
            logging.warning('Automatically setting xlabels as index because there are no taxa')
            yticklabels = ['{}'.format(i+1) for i in range(matrix.shape[0])]
        else:
            yticklabels = _format_ticklabel(format=yticklabels, order=order, taxa=taxa)
    if N > 75:
        linewidths = 0

    return {'matrix':matrix, 'xticklabels':xticklabels, 'yticklabels':yticklabels,
        'ax':ax, 'fig':fig, 'cbar_kws': None, 'linewidths':linewidths}

def _set_plt_labels(d: Dict[str, Any], ax: matplotlib.pyplot.Axes=None) -> Tuple[matplotlib.pyplot.Axes, Dict]:
    '''Removes labels form the dictionay and puts it in
    a new dictionary. This is useful so that we can pass
    **kwargs into matplotlib functions without throwing
    errors.

    Parameters
    ----------
    d : dict
        - This is a dictionary of arguements that contains arguments both
          for matplotlib.pyplot.Axes objects and that are not
    ax : matplotlib.pyplot.Axes
        - Object that we are plotting on.

    Returns
    -------
    dict
        This is an augmented dictionary form the input that removes the labels that 
        were used for Axes object
    '''
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    labels = {}
    for ele in _plt_labels:
        if ele in d:
            labels[ele] = d[ele]
            d.pop(ele, None)

    if PLT_TITLE_LABEL in labels:
        ax.set_title(labels[PLT_TITLE_LABEL])
    if PLT_XLABEL_LABEL in labels:
        ax.set_xlabel(labels[PLT_XLABEL_LABEL])
    if PLT_YLABEL_LABEL in labels:
        ax.set_ylabel(labels[PLT_YLABEL_LABEL])
    return ax, d

def _set_xticks(ax: matplotlib.pyplot.Axes) -> matplotlib.pyplot.Axes:

    loc = plticker.MultipleLocator(base=XTICK_FREQUENCY)
    ax.xaxis.set_major_locator(loc)
    return ax

def _set_tick_fontsize(ax: matplotlib.pyplot.Axes, fontsize: Union[int, float]) -> matplotlib.pyplot.Axes:

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
    return ax

def _set_taxlevel(taxlevel: str) -> str:
    if taxlevel == 'default':
        taxlevel = DEFAULT_TAX_LEVEL
    if taxlevel is None:
        taxlevel = 'asv'
    elif not pl.isstr(taxlevel):
        raise ValueError('`taxlevel` ({}) must be a str'.format(type(taxlevel)))
    elif taxlevel not in ['kingdom', 'phylum', 'class', 
        'order', 'family', 'genus', 'species', 'asv']:
        raise ValueError('taxlevel ({}) not recognized'.format(taxlevel))
    return taxlevel

def _set_default_matplotlib_params(ax: matplotlib.pyplot.Axes=None, title: str=None, 
    xlabel: str=None, ylabel: str=None, figsize: Tuple[float, float]=None) -> matplotlib.pyplot.Axes:
    '''Sets standard stuff with plotting and matplotlib

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes, None
        Axes we are plotting on. If None we make a new one
    title : str
        Title to set or the Axes. If None we do not make a title
    xlabel, ylabel : str
        Label for the x and y axis. If None we do not make any
    figsize : 2-tuple
        Only necessary is `ax` is `None`. This is the size of the figure

    Returns
    -------
    matplotlib.pyplot.Axes
    '''
    if ax is None:
        if figsize is None:
            fig = plt.figure()
        else:
            fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    return ax

@numba.jit(nopython=True, parallel=True)
def _calc_acceptance_rate(ret: np.ndarray, trace: np.ndarray, prev: int):
    '''Calculate the acceptance rate over a scalar trace

    Parameters
    ----------
    ret : np.ndarray
        This is the return array
    trace : np.ndarray
        This is the trace
    prev : int
        This is how many samples to look over
    '''
    for i in range(1,len(ret)):
        # Get start
        if i <= prev:
            start = 0
        else:
            start = i - prev

        # Get number the same
        cumm = 0
        for j in range(start+1, i):
            cumm += trace[j] != trace[j-1]
        
        ret[i] = cumm / (i-start)

    return ret
