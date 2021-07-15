import numpy as np
from pathlib import Path
import mdsine2 as md2

from typing import List, Tuple, Union
from mdsine2.logger import logger

import matplotlib.pyplot as plt


def run_forward_sim(growth: np.ndarray,
                    interactions: np.ndarray,
                    initial_conditions: np.ndarray,
                    perturbations: Union[List[np.ndarray], None],
                    perturbations_start: List[float],
                    perturbations_end: List[float],
                    dt: float,
                    sim_max: float,
                    n_days: float):
    """
    Forward simulate with the given dynamics, with the option to apply perturbations during specified timeframes.

    Parameters
    ----------
    growth : np.ndarray(n_gibbs, n_taxa)
        Growth parameters
    interactions : np.ndarray(n_gibbs, n_taxa, n_taxa)
        Interaction parameters
    initial_conditions : np.ndarray(n_taxa)
        Initial conditions of the taxa
    perturbations : List of np.ndarray(n_gibbs, n_taxa)
        Perturbation effects
    perturbations_start : List of float
        Time to start the perturbation (in days)
    perturbations_end : List of float
        Time at which perturbation ends (in days)
    dt : float
        Step size to forward simulate with
    sim_max : float, None
        Maximum clip for forward sim
    n_days : float
        Total number of days
    """
    dyn = md2.model.gLVDynamicsSingleClustering(
        growth=growth,
        interactions=interactions,
        perturbations=perturbations,
        perturbation_starts=perturbations_start,
        perturbation_ends=perturbations_end,
        start_day=0,
        sim_max=sim_max
    )

    x = md2.integrate(
        dynamics=dyn,
        initial_conditions=initial_conditions,
        dt=dt,
        n_days=n_days,
        subsample=False
    )
    fwsim_values = x['X']
    return fwsim_values


def plot_fwsim_comparison(
        taxa: md2.Taxon,
        taxa_trajectory: np.ndarray,
        trajectory_times: np.ndarray,
        subject: md2.Subject,
        out_path: Path,
        mcmc_display_method: str = "quantiles",
        figsize: Tuple = (10, 8),
        ylim: Tuple = (1e5, 1e12)
):
    times = subject.times
    subject_truth = subject.matrix()['abs'][taxa.idx, :]

    valid_indices = ~np.isnan(taxa_trajectory).any(axis=1)
    taxa_trajectory = taxa_trajectory[valid_indices]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if taxa_trajectory.shape[0] == 0:
        logger.info("Taxa `{}` had no simulations without NaNs. Check for numerical integrity.".format(taxa.name))
    else:
        if mcmc_display_method == "quantiles":
            low = np.percentile(taxa_trajectory, q=25, axis=0)
            high = np.percentile(taxa_trajectory, q=75, axis=0)
            median = np.percentile(taxa_trajectory, q=50, axis=0)

            ax.fill_between(trajectory_times, y1=low, y2=high, alpha=0.2)
            ax.plot(trajectory_times, median, label='Forward Sim')
        elif mcmc_display_method == "all":
            cmap = plt.get_cmap("blues")
            for mcmc_idx in range(taxa_trajectory.shape[0]):
                ax.plot(times, taxa_trajectory[mcmc_idx, :], c=cmap(mcmc_idx / taxa_trajectory.shape[0]), linewidth=0.8)
        else:
            raise ValueError("Unrecognized mcmc_display_argument value `{}`".format(mcmc_display_method))

    # Ground truth data.
    ax.plot(times, subject_truth, label='Data', marker='x', color='black', linestyle=':')

    md2.visualization.shade_in_perturbations(ax, perturbations=subject.perturbations, subj=subject)

    ax.set_yscale('log')
    ax.set_ylim(bottom=ylim[0], top=ylim[1])
    ax.legend()
    ax.set_title("{taxa}, Subject: {subj}".format(
        taxa=taxa.name,
        subj=subject.name
    ))

    plt.savefig(out_path)
