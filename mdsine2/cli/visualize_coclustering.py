"""
Plots the posterior co-clustering probabilities.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import mdsine2 as md2
import os

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial import distance as dist

from .helpers import figure_helper as helper
from .base import CLIModule


class CoclusteringVisualizationCLI(CLIModule):
    def __init__(self, subcommand="visualize-coclustering"):
        super().__init__(
            subcommand=subcommand,
            docstring=__doc__
        )

    def create_parser(self, parser: argparse.ArgumentParser):
        parser.add_argument("-file1", "--healthy_prob", required=True,
                            help=".tsv file containing co-clustering probailities for healthy cohort")
        parser.add_argument("-file2", "--uc_prob", required=True,
                            help=".tsv file containing co-clustering probailities for UC cohort")
        parser.add_argument("-file3", "--healthy_cluster", required=True,
                            help=".tsv file containing consensus cluster assignment for healthy cohort")
        parser.add_argument("-file4", "--uc_cluster", required=True,
                            help=".tsv file containing consensus cluster assignment for UC cohort")
        parser.add_argument("-file5", "--healthy_pkl", required=True,
                            help="pickled md2.base.Study file for healthy subjects")
        parser.add_argument("-file6", "--uc_pkl", required=True,
                            help="pickled md2.base.Study file for UC subjects")
        parser.add_argument("-opt", "--enable_opt", required=True,
                            help=".tsv file containing consensus cluster assignment for UC cohort")

    def main(self, args: argparse.Namespace):
        enable_opt = True
        if args.enable_opt == "False":
            enable_opt = False
        if enable_opt:
            uc_savename = "uc_opt"
            healthy_savename = "healthy_opt"
        else:
            uc_savename = "uc_no_opt"
            healthy_savename = "healthy_no_opt"

        subjset_healthy = md2.Study.load(args.healthy_pkl)
        subjset_uc = md2.Study.load(args.uc_pkl)

        cluster_healthy = helper.parse_cluster(args.healthy_cluster)
        healthy_cocluster_prob = pd.read_csv(args.healthy_prob, sep="\t",
                                             index_col=0)
        healthy_order_li = list(healthy_cocluster_prob.columns)
        healthy_order_d = {healthy_order_li[i]: i for i in range(len(healthy_order_li))}
        healthy_opt_order = get_leaves_order(healthy_cocluster_prob.to_numpy(),
                                             healthy_order_li, cluster_healthy, enable_opt)
        x_healthy, y_healthy = helper.get_axes_names(healthy_opt_order,
                                                     cluster_healthy, subjset_healthy)
        print("Making Figure 6")
        co_clustering_probability(healthy_cocluster_prob.to_numpy(), x_healthy,
                                  y_healthy, healthy_opt_order, healthy_order_d, healthy_savename)

        cluster_uc = helper.parse_cluster(args.uc_cluster)
        uc_cocluster_prob = pd.read_csv(args.uc_prob, sep="\t",
                                        index_col=0)

        uc_order_li = list(uc_cocluster_prob.columns)
        uc_order_d = {uc_order_li[i]: i for i in range(len(uc_order_li))}
        uc_opt_order = get_leaves_order(uc_cocluster_prob.to_numpy(), uc_order_li,
                                        cluster_uc, enable_opt)
        x_uc, y_uc = helper.get_axes_names(uc_opt_order, cluster_uc, subjset_uc)
        print("Making Figure 7")
        co_clustering_probability(uc_cocluster_prob.to_numpy(), x_uc, y_uc,
                                  uc_opt_order, uc_order_d, uc_savename)


def co_clustering_probability(prob_mat, x_names, y_names, otus_order, index_d,
                              name):
    """
       ready the variables for plotting the co-clustering prbability

       @parameters
       ------------------------------------------------------------------------
       prob mat : (np.array) probability matrix
       x_names, y_names : ([str]) names of x and y ticks
       otus_order : [str] order of OTUs in prob_mat
       index_d : (dict (str) otu -> (int) index of the OTU)
       name : (str) name of the figure

    """
    print("Making Co-clustering probability Heatmap " + name)

    uc_mat_reordered = helper.reorder(prob_mat, otus_order, index_d)
    plot_coclustering_heatmap(uc_mat_reordered, x_names, y_names,
                              24, 27, 17, 14, name + "_coclustering_matrix")


def plot_coclustering_heatmap(data, x_names, y_names, figsize_x, figsize_y,
                              fontsize_x, fontsize_y, figname):
    """
       plots the co-clustering probability heatmap and saves it as figname.pdf

       @parameters
       ------------------------------------------------------------------------
       data : (np.array) the co-clusteing probability
       x_names : ([str]) xticklabels
       y_names ([str]) yticklabels
       figsize_x, figsize_y : (float) size of the plt figure
       fontsize_x, fontsize_y : (float) font of the x and y ticklabels
       figname : (str) name of the figure

    """

    df = pd.DataFrame(data, index=y_names, columns=x_names)
    fig = plt.figure(figsize=(figsize_x, figsize_y))
    axes = fig.add_subplot(1, 1, 1)
    heatmap = sns.heatmap(df, xticklabels=2, yticklabels=True,
                          ax=axes, cbar_kws={"shrink": 0.75}, linewidth=0.1,
                          cmap="magma_r")

    axes.set_xticklabels(axes.get_xticklabels(), fontsize=fontsize_x,
                         rotation=90)
    axes.set_yticklabels(axes.get_yticklabels(), fontsize=fontsize_y,
                         rotation=0)
    axes.tick_params(length=7.5, width=1, left=True, bottom=True,
                     color="black")

    cbar = heatmap.collections[0].colorbar  # colorbar for p-values
    # cbar.ax.set_yticklabels(["<1", "2", "3", "4", "5", "6", "7", "8", "9", ">10"])
    cbar.ax.tick_params(labelsize=20, length=15, width=1)
    # cbar.ax.tick_params(length = 10, width = 1, which = "minor")
    cbar.ax.set_title("  Co-clustering \n Probability \n", fontweight="bold",
                      fontsize=22.5)

    for _, spine in axes.spines.items():
        spine.set_visible(True)

    legend = "Taxonomy Key \n* : Genus, ** : Family, *** : Order, **** : Class," \
             "***** : Phylum, ****** : Kingdom"
    pos = axes.get_position()
    fig.text(pos.x0, pos.y0 - 0.05, legend, fontsize=25, fontweight="bold")

    loc = "output_figures/"
    if not os.path.exists(loc):
        os.makedirs(loc, exist_ok=True)

    plt.savefig(loc + figname + ".pdf", dpi=100, bbox_inches="tight")
    print("done")


def get_leaves_order(data, otu_li, cluster_d, enable_opt=True):
    """
       get the optimal order for the heatmap; if enable_opt is False, then
       return the OTU order based on consensus cluster dict

       @parameters
        data : (numpy) a square matrix
        otu_li : a list containing the list of OTU ids in serial order

        @returns
        [str] : a list consisting of OTU ids
    """
    dist_mat = dist.squareform(1 - data)

    if enable_opt:
        linkage_ = linkage(dist_mat, "average", optimal_ordering=True)
        leaves = dendrogram(linkage_)["leaves"]
        order = [otu_li[i] for i in leaves]
        return order
    else:
        return [otu for id in cluster_d for otu in cluster_d[id]]
