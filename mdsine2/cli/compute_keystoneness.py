"""
Run keystoneness analysis using the specified MDSINE2 output.
"""

import argparse
import csv
from typing import List, Union, Set

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
import seaborn as sns

import mdsine2 as md2
from mdsine2 import TaxaSet
from mdsine2.logger import logger

from .base import CLIModule
from .helpers.fwsim_helper import run_forward_sim


# Make font editable in AI
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


class KeystonenessCLI(CLIModule):
    def __init__(self, subcommand="evaluate-keystoneness"):
        super().__init__(
            subcommand=subcommand,
            docstring=__doc__
        )

    def create_parser(self, parser: argparse.ArgumentParser):
        # Inputs
        parser.add_argument('--extraction-dir', '-e', dest='extraction_dir', type=str, required=True,
                            help="The directory containing the extracted .npy interaction/growth/coclustering files.")

        parser.add_argument('--study', '-s', dest='study', type=str, required=True,
                            help="The path to the relevant Study object containing the input data (subjects, taxa).")

        # Optional:
        parser.add_argument('--initial-conditions-file', '-if', type=str, dest='initial_condition_path',
                            required=False,
                            help='The path to a file specifying the initial conditions. File will be interpreted as a '
                                 'two-column TSV file (Taxa name, Absolute abundance). '
                                 'If not specified, then the user must specify initial_condition_study_tidx argument.')
        parser.add_argument('--initial-conditions-study-time', '-it', type=int, dest='initial_condition_study_tidx',
                            required=False,
                            help='The time index of the study to pull out initial conditions from.')

        # Outputs
        parser.add_argument('--output-dir', '-o', type=str, dest='out_dir',
                            required=True,
                            help='This is where you are saving the posterior renderings')

        # Simulation params
        parser.add_argument('--n-days', type=int, dest='n_days', required=False,
                            help='Total number of days to simulate for', default=180)
        parser.add_argument('--simulation-dt', '-dt', type=float, dest='dt', required=False,
                            help='Timesteps we go in during forward simulation', default=0.01)
        parser.add_argument('--sim-max', dest='sim_max', type=float, required=False,
                            help='Maximum value of abundance.', default=1e20)
        parser.add_argument('--limit-of-detection', dest='limit_of_detection', required=False,
                            help='If any of the taxa have a 0 abundance at the start, then we ' \
                                 'set it to this value.', default=1e5, type=float)
        parser.add_argument('--simulate-every-n', dest='simulate_every_n', type=int, required=False, default=1,
                            help='Specify to skip a certain number of gibbs steps and thin out the samples '
                                 '(for faster calculations)')

        parser.add_argument('--width', default=10., type=float, required=False)
        parser.add_argument('--height', default=10., type=float, required=False)

    def main(self, args: argparse.Namespace):
        study = md2.Study.load(args.study)
        logger.info(f"Loading initial conditions from {args.initial_condition_path}")

        # Load initial conditions
        if args.initial_condition_path is not None:
            initial_conditions_master = load_initial_conditions(study, args.initial_condition_path)
        else:
            if args.initial_condition_study_tidx is None:
                raise RuntimeError("If initial-conditions-file is not specified, user must provide initial-conditions-study-time.")
            initial_conditions_master = initial_conditions_from_study(study, args.initial_condition_study_tidx)
        lb = args.limit_of_detection
        logger.info(f"Using limit of detection = {lb}")
        initial_conditions_master[initial_conditions_master < lb] = lb

        # Load MCMC posterior
        extraction_dir = Path(args.extraction_dir)
        interactions = np.load(str(extraction_dir / 'interactions.npy'))
        growths = np.load(str(extraction_dir / 'growth.npy'))
        agglom = np.load(str(extraction_dir / 'agglomeration.npy'))
        modules = []
        for cidx in range(np.max(agglom) + 1):  # Make sure to do the (+1) to count the last module.
            module = {oidx for oidx in np.where(agglom == cidx)[0]}
            modules.append(module)

        # Run analysis
        out_dir = Path(args.out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)

        df_path = out_dir / f"{study.name}_steady_states.tsv"
        fwsim_df = retrieve_ky_simulations(
            study,
            df_path,
            interactions,
            growths,
            modules,
            initial_conditions_master,
            args.n_days, args.dt, args.sim_max,
            args.simulate_every_n
        )
        fwsim_df['ExcludedCluster'] = fwsim_df['ExcludedCluster'].astype("string")

        # Render figure
        fig = plt.figure(figsize=(args.width, args.height))
        ky = Keystoneness(
            study,
            fwsim_df,
            modules
        )
        ky.plot(fig)
        ky.save_ky(out_dir / f"{study.name}_keystoneness.tsv")
        plt.savefig(out_dir / f"{study.name}_keystoneness.pdf", format="pdf")


def retrieve_ky_simulations(
        study: md2.Study,
        df_path: Path,
        interactions: np.ndarray,
        growths: np.ndarray,
        modules: List[Set[int]],
        initial_conditions_master: np.ndarray,
        n_days: float, dt: float, sim_max: float, simulate_every: int
):
    if df_path.exists():
        logger.info(f"Loading previously computed results ({df_path})")
        return pd.read_csv(df_path, sep='\t', index_col=False)

    logger.info(f"Computing new steady states (target: {df_path})")
    df_entries = []

    # Baseline
    compute_forward_sim(
        interactions, growths, study.taxa,
        None,
        initial_conditions_master,
        n_days,
        dt,
        sim_max,
        df_entries,
        simulate_every
    )

    # Cluster exclusion
    for cluster_idx, cluster in enumerate(modules):
        initial_conditions = exclude_cluster_from(initial_conditions_master, cluster)
        logger.info(f"Now excluding cluster idx={cluster_idx}")
        logger.info("Using initial conditions: {}".format(initial_conditions.flatten()))

        compute_forward_sim(
            interactions, growths, study.taxa,
            cluster_idx,
            initial_conditions,
            n_days,
            dt,
            sim_max,
            df_entries,
            simulate_every
        )

    df = pd.DataFrame(df_entries)
    df.to_csv(df_path, sep='\t', index=False)
    return df


def exclude_cluster_from(initial_conditions_master: np.ndarray, cluster: Set[int]):
    initial_conditions = np.copy(initial_conditions_master)
    for oidx in cluster:
        initial_conditions[oidx] = 0.0
    return initial_conditions


def compute_forward_sim(
        interactions: np.ndarray,
        growths: np.ndarray,
        taxa: TaxaSet,
        cluster_idx: Union[int, None],
        initial_conditions: np.ndarray,
        n_days: float,
        dt: float,
        sim_max: float,
        df_entries: List,
        simulate_every: int
):
    n_samples = interactions.shape[0]

    # forward simulate and add results to dataframe.
    if cluster_idx is None:
        tqdm_disp = "Baseline"
    else:
        tqdm_disp = f"Cluster {cluster_idx}"

    for gibbs_idx, fwsim in tqdm(
            do_fwsims(interactions, growths, initial_conditions, n_days, dt, sim_max, simulate_every),
            total=n_samples // simulate_every,
            desc=tqdm_disp
    ):
        for entry in fwsim_entries(taxa, fwsim, dt=dt):
            entry['SampleIdx'] = gibbs_idx
            entry['ExcludedCluster'] = str(cluster_idx)
            df_entries.append(entry)


def fwsim_entries(taxa: TaxaSet, fwsim, dt):
    n = int(0.5 / dt)  # number of timepoints to average over.
    stable_states = fwsim[:, -n:].mean(axis=1)  # 100 indices = 1 day, if dt = 0.01
    for otu in taxa:
        yield {
            "OTU": otu.name,
            "StableState": stable_states[otu.idx]
        }


def do_fwsims(interactions: np.ndarray,
              growths: np.ndarray,
              initial_conditions: np.ndarray,
              n_days: float,
              dt: float,
              sim_max: float,
              simulate_every: int
              ):
    n_samples = interactions.shape[0]

    # Do the forward sim.
    for gibb in range(0, n_samples, simulate_every):
        gibbs_step_sim, _ = run_forward_sim(
            growth=growths[gibb],
            interactions=interactions[gibb],
            initial_conditions=initial_conditions.reshape(-1, 1),
            perturbations=None,
            perturbations_start=[],
            perturbations_end=[],
            dt=dt,
            start_time=0.,
            sim_max=sim_max,
            n_days=n_days
        )
        yield gibb, gibbs_step_sim


def initial_conditions_from_study(study: md2.Study, tidx: int) -> np.ndarray:
    M = study.matrix(dtype='abs', agg='mean', times='intersection', qpcr_unnormalize=True)
    return M[:, tidx]  # Day 20


def load_initial_conditions(study: md2.Study, initial_condition_path: str) -> np.ndarray:
    taxa_to_abundance = {}
    with open(initial_condition_path, "r") as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for row in rd:
            if len(row) != 2:
                raise ValueError(
                    "Input file for initial_condition_path must be a two-column format. "
                    "Found {} columns instead.".format(
                        len(row)
                    )
                )
            taxa_to_abundance[row[0]] = float(row[1])

    abundances = np.zeros(len(study.taxa), dtype=np.float)
    for tidx, taxa in enumerate(study.taxa):
        try:
            abundances[tidx] = taxa_to_abundance[taxa.name]
        except KeyError:
            raise KeyError("Could not find initial condition value for taxa `{}`.".format(taxa.name))
    return abundances.reshape(-1, 1)


def cluster_nonmembership_df(taxaset: TaxaSet, clustering: List[Set[int]]):
    entries = []
    for cluster_idx, cluster in enumerate(clustering):
        for otu in taxaset:
            if otu.idx not in cluster:
                entries.append({
                    "ClusterID": f"{cluster_idx}",
                    "OTU": otu.name
                })
    return pd.DataFrame(entries)


def cluster_membership_df(taxaset: TaxaSet, clustering: List[Set[int]]):
    entries = []
    for cluster_idx, cluster in enumerate(clustering):
        for oidx in cluster:
            otu = taxaset[oidx]
            entries.append({
                "ClusterOfOTU": f"{cluster_idx}",
                "OTU": otu.name
            })
    return pd.DataFrame(entries)


def create_cmap(tag, nan_value="red"):
    cmap = cm.get_cmap(tag)
    cmap.set_bad(color=nan_value)
    return cmap


class Keystoneness(object):
    def __init__(self, study: md2.Study, fwsim_df: pd.DataFrame, clustering: List[Set[int]]):
        self.fwsim_df = fwsim_df
        self.clustering = clustering
        self.study = study

        logger.info("Compiling dataframe.")
        self.ky_df: pd.DataFrame = self.generate_keystoneness_df()

        logger.info("Compiling abundance data.")
        self.abundance_array = self.get_abundance_array()

        logger.info("Extracting keystoneness values.")
        agg_ky_df = self.ky_df.groupby(level=0).mean()
        self.ky_array = np.array(
            [
                agg_ky_df.loc[f'{cluster_idx}', "Ky"]
                for cluster_idx in range(len(clustering))
            ]
        )

        logger.info("Extracting baseline abundances.")
        self.day20_array = self.get_day20_abundances()

    def generate_keystoneness_df(self) -> pd.DataFrame:
        nonmembers_df = cluster_nonmembership_df(self.study.taxa, self.clustering)

        baseline = self.fwsim_df.loc[self.fwsim_df["ExcludedCluster"] == "None"]

        altered = self.fwsim_df.loc[self.fwsim_df["ExcludedCluster"] != "None"]
        altered = altered.merge(
            right=nonmembers_df,
            how="inner",
            left_on=["ExcludedCluster", "OTU"],
            right_on=["ClusterID", "OTU"]
        )

        merged = altered.merge(
            baseline[["OTU", "SampleIdx", "StableState"]],
            how="left",
            left_on=["OTU", "SampleIdx"],
            right_on=["OTU", "SampleIdx"],
            suffixes=["", "Base"]
        )

        merged["DiffStableState"] = np.log10(merged["StableStateBase"] + 1e5) - np.log10(merged["StableState"] + 1e5)

        return merged[
            ["ExcludedCluster", "SampleIdx", "DiffStableState"]
        ].groupby(
            ["ExcludedCluster", "SampleIdx"]
        ).mean().rename(columns={"DiffStableState": "Ky"})

    def get_abundance_array(self):
        membership_df = cluster_membership_df(self.study.taxa, self.clustering)
        merged_df = self.fwsim_df.merge(
            membership_df,
            how="left",
            left_on="OTU",
            right_on="OTU"
        )

        abund_array = np.zeros(shape=(len(self.clustering) + 1, len(self.clustering)))

        # Baseline abundances (no cluster removed) -- sum across OTUs (per sample), median across samples.
        subset_df = merged_df.loc[merged_df["ExcludedCluster"] == "None"]
        subset_df = subset_df[
            ["ClusterOfOTU", "SampleIdx", "StableState"]
        ].groupby(
            ["ClusterOfOTU", "SampleIdx"]
        ).sum(
            # Aggregate over OTUs  (e.g. Baseline abundance of a cluster is the sum of its constituents.)
        ).groupby(
            level=0
        ).median(
            # Aggregate over samples
        )
        for cluster_idx in range(len(self.clustering)):
            abund_array[0, cluster_idx] = subset_df.loc[f'{cluster_idx}']

        # Altered abundances (remove 1 cluster at a time)
        for removed_cluster_idx in range(len(self.clustering)):
            subset_df = merged_df.loc[merged_df["ExcludedCluster"] == f'{removed_cluster_idx}']

            # Compute the total abundance (over OTUs) for each cluster, for each sample.
            # Then aggregate (median) across samples.
            subset_df = subset_df[
                ["ClusterOfOTU", "SampleIdx", "StableState"]
            ].groupby(
                ["ClusterOfOTU", "SampleIdx"]
            ).sum(
                # Aggregate over OTUs
            ).groupby(
                level=0
            ).median(
                # Aggregate over samples
            )

            for cluster_idx in range(len(self.clustering)):
                if cluster_idx == removed_cluster_idx:
                    continue
                abund_array[removed_cluster_idx + 1, cluster_idx] = subset_df.loc[f'{cluster_idx}']
        return abund_array

    def get_agg_ky(self) -> pd.DataFrame:
        """
        Evalautes the keystoneness as an aggregate across samples.

        :return:
        """
        return self.ky_df.groupby(level=0).mean()  # Aggregate across per-sample Ky values

    def get_day20_abundances(self):
        M = self.study.matrix(dtype='abs', agg='mean', times='intersection', qpcr_unnormalize=True)
        day20_state = M[:, 19]
        cluster_day20_abundances = np.zeros(len(self.clustering))

        for cidx, cluster in enumerate(self.clustering):
            cluster_day20_abundances[cidx] = np.sum(
                [day20_state[oidx] for oidx in cluster]
            )
        return cluster_day20_abundances

    def save_ky(self, tsv_path: Path):
        self.ky_df.to_csv(tsv_path, sep='\t')

    def plot(self, fig):
        # Main abundance grid shows the _difference_ from baseline, instead of the abundances itself.
        n_clusters = len(self.ky_array)

        # =========== Pre-sorting. ===========
        ky_order = np.argsort(self.ky_array)
        ky_order = ky_order[::-1]
        ky_array = self.ky_array[ky_order]

        day20_array = self.day20_array[ky_order].reshape(1, len(self.day20_array))

        baseline_array = self.abundance_array[[0], :]
        baseline_array = baseline_array[:, ky_order]

        cluster_exclusion_array = self.abundance_array[1 + ky_order, :]  # Reorder the rows first (exclude the baseline row),
        cluster_exclusion_array = cluster_exclusion_array[:, ky_order]  # Then reorder the columns.

        baseline_diff_array = np.log10(baseline_array + 1e5) - np.log10(cluster_exclusion_array + 1e5)
        for i in range(baseline_diff_array.shape[0]):
            baseline_diff_array[i, i] = np.nan

        # =========== Heatmap settings. ========
        # Colors and normalization (abund)
        abund_min = np.max([
            np.min(self.abundance_array[self.abundance_array > 0]),
            1e5
        ])
        abund_max = np.min([
            np.max(self.abundance_array[self.abundance_array > 0]),
            1e13
        ])

        abund_cmap = create_cmap("Greens", nan_value="white")
        abund_norm = matplotlib.colors.LogNorm(vmin=abund_min, vmax=abund_max)

        # Colors and normalization (Ky)
        gray = np.array([0.95, 0.95, 0.95, 1.0])
        red = np.array([1.0, 0.0, 0.0, 1.0])
        blue = np.array([0.0, 0.0, 1.0, 1.0])
        n_interp = 128

        top = blue
        bottom = red
        top_middle = 0.05 * top + 0.95 * gray
        bottom_middle = 0.05 * bottom + 0.95 * gray

        ky_cmap = matplotlib.colors.ListedColormap(
            np.vstack(
                [(1 - t) * bottom + t * bottom_middle for t in np.linspace(0, 1, n_interp)]
                +
                [(1 - t) * top_middle + t * top for t in np.linspace(0, 1, n_interp)]
            ),
            name='Keystoneness'
        )
        ky_cmap.set_bad(color="white")

        diff_min = np.min(baseline_diff_array[~np.isnan(baseline_diff_array) & (cluster_exclusion_array > 0)])
        diff_max = np.max(baseline_diff_array[~np.isnan(baseline_diff_array) & (cluster_exclusion_array > 0)])
        ky_min = 0.90 * np.min(ky_array) + 0.10 * diff_min
        ky_max = 0.90 * np.max(ky_array) + 0.10 * diff_max

        def _forward(x):
            y = x.copy()
            positive_part = x[x > 0]
            y[x > 0] = np.sqrt(positive_part / ky_max)

            negative_part = x[x < 0]
            y[x < 0] = -np.sqrt(np.abs(negative_part / ky_min))
            return y

        def _reverse(x):
            y = x.copy()
            positive_part = x[x > 0]
            y[x > 0] = ky_max * np.power(positive_part, 2)

            negative_part = x[x < 0]
            y[x < 0] = -np.abs(ky_min) * np.power(negative_part, 2)
            return y

        ky_norm = mcolors.FuncNorm((_forward, _reverse), vmin=ky_min, vmax=ky_max)

        # Seaborn Heatmap Kwargs
        abund_heatmapkws = dict(square=False,
                                cbar=False,
                                cmap=abund_cmap,
                                linewidths=0.5,
                                norm=abund_norm)
        ky_heatmapkws = dict(square=False, cbar=False, cmap=ky_cmap, linewidths=0.5, norm=ky_norm)

        # ========== Plot layout ===========
        #     [left, bottom, width, height]
        main_x = 0.67
        main_y = 0.5
        box_unit = 0.03
        main_width = box_unit * n_clusters
        main_height = main_width
        main_left = main_x - 0.5 * main_width
        main_bottom = main_y - 0.5 * main_width
        # print("Left: {}, bottom: {}, width: {}, height: {}".format(main_left, main_bottom, main_width, main_height))
        # print("Right: {}, Top: {}".format(main_left + main_width, main_bottom + main_height))

        ky_ax = fig.add_axes([main_left + main_width + 0.5 * box_unit, main_bottom, box_unit, main_height])
        abundances_ax = fig.add_axes([main_left, main_bottom, main_width, main_height])
        obs_ax = fig.add_axes([main_left, main_bottom + main_height + 1.5 * box_unit, box_unit * n_clusters, box_unit])
        baseline_ax = fig.add_axes(
            [main_left, main_bottom + main_height + 0.5 * box_unit, box_unit * n_clusters, box_unit])

        # ========= Rendering. ==========
        # ====== Bottom left: Keystoneness
        hmap_ky = sns.heatmap(
            ky_array.reshape(len(ky_array), 1),
            ax=ky_ax,
            xticklabels=False,
            yticklabels=False,
            **ky_heatmapkws
        )
        hmap_ky.xaxis.set_tick_params(width=0)
        fig.text(main_left + main_width + 2 * box_unit, main_y, "Keystoneness", ha='center', va='center', rotation=-90)

        for _, spine in hmap_ky.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(1.0)

        # ====== Top right 1: Observed levels (day 20)
        hmap_day20_abund = sns.heatmap(day20_array,
                                       ax=obs_ax,
                                       xticklabels=False,
                                       yticklabels=["Observation"],
                                       **abund_heatmapkws)
        hmap_day20_abund.set_yticklabels(hmap_day20_abund.get_yticklabels(), rotation=0)
        for _, spine in hmap_day20_abund.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
        fig.text(main_x, main_bottom + main_height + 3 * box_unit, "Steady State Abundance", ha='center', va='center')

        # ====== Top right 2: Baseline abundances
        hmap_base_abund = sns.heatmap(baseline_array,
                                      ax=baseline_ax,
                                      xticklabels=False,
                                      yticklabels=["Simulation"],
                                      **abund_heatmapkws)
        hmap_base_abund.set_yticklabels(hmap_base_abund.get_yticklabels(), rotation=0)
        for _, spine in hmap_base_abund.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(1.0)

        # ====== Bottom right: Abundances with clusters removed.
        ticklabels = [
            f"M{c_idx+1}"
            for c_idx in ky_order
        ]
        hmap_removed_cluster_abund = sns.heatmap(
            baseline_diff_array,
            ax=abundances_ax,
            xticklabels=ticklabels,
            yticklabels=ticklabels,
            **ky_heatmapkws
        )
        # Draw a marker ("X") on top of NaNs.
        abundances_ax.scatter(*np.argwhere(np.isnan(baseline_diff_array.T)).T + 0.5, marker="x", color="black", s=100)
        abundances_ax.set_ylabel("Module Removed")
        abundances_ax.set_xlabel("Per-Module Change")
        for _, spine in hmap_removed_cluster_abund.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
        hmap_removed_cluster_abund.xaxis.set_ticks_position('bottom')
        hmap_removed_cluster_abund.set_xticklabels(
            hmap_removed_cluster_abund.get_xticklabels(), rotation=90, horizontalalignment='center'
        )
        hmap_removed_cluster_abund.set_yticklabels(
            hmap_removed_cluster_abund.get_yticklabels(), rotation=0, horizontalalignment='right'
        )
        abundances_ax.tick_params(direction='out', length=0, width=0)

        # ======= Draw the colormaps ========
        cbar_from_main = 0.25
        cbar_width = 0.01
        cbar_height = 0.35

        # Cbar on the right (steady state diff, green)
        cax = fig.add_axes([main_left - cbar_from_main, main_y - 0.5 * cbar_height, cbar_width, cbar_height])
        sm = matplotlib.cm.ScalarMappable(cmap=abund_cmap, norm=abund_norm)
        sm.set_array(np.array([]))
        cbar = fig.colorbar(sm, cax=cax)

        yticks = cbar.get_ticks()
        yticklabels = [str(np.log10(y)) for y in yticks]
        yticklabels[0] = "<{}".format(yticklabels[0])
        cax.set_yticklabels(yticklabels)
        cax.set_ylabel("Log-Abundance")

        # Cbar on the left (Keyst., RdBu)
        cax = fig.add_axes(
            [main_left - cbar_from_main - 2 * cbar_width, main_y - 0.5 * cbar_height, cbar_width, cbar_height])
        sm = matplotlib.cm.ScalarMappable(cmap=ky_cmap, norm=ky_norm)
        sm.set_array(np.array([]))
        cbar = fig.colorbar(sm, cax=cax)
        cax.yaxis.set_ticks_position('left')
        cax.set_ylabel("Log-Difference from Base")
        cax.yaxis.set_label_position("left")

        yticks = cbar.get_ticks()
        yticklabels = ["{:.1f}".format(y) for y in yticks]
        cax.set_yticklabels(yticklabels)

        # Legend label text
        fig.text(
            main_left - cbar_from_main - cbar_width,
            main_y + 0.5 * cbar_height + 0.05,
            "Legend",
            ha='center', va='center',
            fontweight='bold'
        )
