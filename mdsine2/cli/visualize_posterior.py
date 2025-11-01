"""
Visualize the posteriors of the MCMC sampled parameters.
Iterates through all of the parameters that were learned and visualizes the posterior

Implemented for:
    Growth parameters
    Self-interactions parameters
    Interaction parameters (value, bayes factors)
    Perturabtion parameters (value, bayes factors)
    Clustering (cocluster and number of clusters)
    Filtering
"""
import os
import argparse
import pandas as pd
import mdsine2 as md2
from mdsine2.logger import logger
from mdsine2.names import STRNAMES

from .base import CLIModule


class PosteriorVisualizationCLI(CLIModule):
    """ Visualize the posterior distribution of a MDSINE2 Markov Chain. """
    def __init__(self, subcommand="visualize-posterior"):
        super().__init__(
            subcommand=subcommand,
            docstring=self.__doc__
        )

    def create_parser(self, parser: argparse.ArgumentParser):
        parser.add_argument('--chain', '-c', type=str, dest='chain',
                            required=True,
                            help='This is the path of the chain for inference.')
        parser.add_argument('--output-basepath', '-o', type=str, dest='basepath',
                            required=True,
                            help='This is where you are saving the posterior renderings')
        parser.add_argument('--section', '-s', type=str, dest='section',
                            required=False, default="posterior",
                            choices=['posterior', 'burnin', 'entire'],
                            help='Section to plot the variables of.')
        parser.add_argument('--is-fixed-clustering', dest='fixed_clustering',
                            action="store_true",
                            help='If flag is set, plot the posterior with fixed clustering options.')

    def main(self, args: argparse.Namespace):
        mcmc = md2.BaseMCMC.load(args.chain)
        basepath = args.basepath
        section = args.section
        os.makedirs(basepath, exist_ok=True)

        # Plot Process variance
        # ---------------------
        logger.info('Process variance')
        mcmc.graph[STRNAMES.PROCESSVAR].visualize(
            path=os.path.join(basepath, 'processvar.pdf'), section=section)

        # Plot growth
        # -----------
        logger.info('Plot growth')
        growthpath = os.path.join(basepath, 'growth')
        os.makedirs(growthpath, exist_ok=True)
        dfvalues = mcmc.graph[STRNAMES.GROWTH_VALUE].visualize(basepath=growthpath,
                                                               taxa_formatter='%(paperformat)s', section=section)
        dfmean = mcmc.graph[STRNAMES.PRIOR_MEAN_GROWTH].visualize(
            path=os.path.join(growthpath, 'mean.pdf'), section=section)
        dfvar = mcmc.graph[STRNAMES.PRIOR_VAR_GROWTH].visualize(
            path=os.path.join(growthpath, 'var.pdf'), section=section)
        df = pd.concat([dfvalues, dfmean, dfvar], ignore_index=True)
        df.to_csv(os.path.join(growthpath, 'values.tsv'), sep='\t', index=True, header=True)

        # Plot self-interactions
        # ----------------------
        logger.info('Plot self-interactions')
        sipath = os.path.join(basepath, 'self_interactions')
        os.makedirs(sipath, exist_ok=True)
        dfvalues = mcmc.graph[STRNAMES.SELF_INTERACTION_VALUE].visualize(basepath=sipath,
                                                                         taxa_formatter='%(paperformat)s',
                                                                         section=section)
        dfmean = mcmc.graph[STRNAMES.PRIOR_MEAN_SELF_INTERACTIONS].visualize(
            path=os.path.join(sipath, 'mean.pdf'), section=section)
        dfvar = mcmc.graph[STRNAMES.PRIOR_VAR_SELF_INTERACTIONS].visualize(
            path=os.path.join(sipath, 'var.pdf'), section=section)
        df = pd.concat([dfvalues, dfmean, dfvar], ignore_index=True)
        df.to_csv(os.path.join(sipath, 'values.tsv'), sep='\t', index=True, header=True)

        # Plot clustering
        # ---------------
        logger.info('Plot clustering')
        if args.fixed_clustering:
            f = open(os.path.join(basepath, 'clustering.txt'), 'w')
            f.write('Cluster assignments\n')
            f.write('-------------------\n')
            clustering = mcmc.graph[STRNAMES.CLUSTERING_OBJ]
            taxa = mcmc.graph.data.taxa
            for cidx, cluster in enumerate(clustering):
                f.write('Cluster {}\n'.format(cidx + 1))
                for oidx in cluster.members:
                    f.write('\t{}\n'.format(
                        md2.taxaname_for_paper(taxon=taxa[oidx], taxa=taxa)))
        else:
            clusterpath = os.path.join(basepath, 'clustering')
            os.makedirs(clusterpath, exist_ok=True)
            f = open(os.path.join(clusterpath, 'overview.txt'), 'w')

            mcmc.graph[STRNAMES.CONCENTRATION].visualize(
                path=os.path.join(clusterpath, 'concentration.pdf'), f=f,
                section=section)
            mcmc.graph[STRNAMES.CLUSTERING].visualize(basepath=clusterpath, f=f,
                                                      section=section)

        # Plot interactions
        # -----------------
        logger.info('Plot interactions')
        interactionpath = os.path.join(basepath, 'interactions')
        os.makedirs(interactionpath, exist_ok=True)
        f = open(os.path.join(interactionpath, 'overview.txt'), 'w')
        mcmc.graph[STRNAMES.PRIOR_MEAN_INTERACTIONS].visualize(
            path=os.path.join(interactionpath, 'mean.pdf'), f=f, section=section)
        mcmc.graph[STRNAMES.PRIOR_VAR_INTERACTIONS].visualize(
            path=os.path.join(interactionpath, 'variance.pdf'), f=f, section=section)
        mcmc.graph[STRNAMES.CLUSTER_INTERACTION_INDICATOR_PROB].visualize(
            path=os.path.join(interactionpath, 'probability.pdf'), f=f, section=section)
        mcmc.graph[STRNAMES.CLUSTER_INTERACTION_INDICATOR].visualize(basepath=interactionpath,
                                                                     section=section, vmax=10,
                                                                     fixed_clustering=args.fixed_clustering)
        mcmc.graph[STRNAMES.CLUSTER_INTERACTION_VALUE].visualize(basepath=interactionpath,
                                                                 section=section,
                                                                 fixed_clustering=args.fixed_clustering)

        # Plot Perturbations
        # ------------------
        if mcmc.graph.data.subjects.perturbations is not None:
            logger.info('Plot perturbations')
            for pidx, perturbation in enumerate(mcmc.graph.data.subjects.perturbations):
                logger.info('Plot {}'.format(perturbation.name))
                perturbationpath = os.path.join(basepath, perturbation.name)
                os.makedirs(perturbationpath, exist_ok=True)

                f = open(os.path.join(perturbationpath, 'overview.txt'), 'w')
                mcmc.graph[STRNAMES.PRIOR_MEAN_PERT].visualize(
                    path=os.path.join(perturbationpath, 'mean.pdf'),
                    f=f, section=section, pidx=pidx)
                mcmc.graph[STRNAMES.PRIOR_VAR_PERT].visualize(
                    path=os.path.join(perturbationpath, 'var.pdf'),
                    f=f, section=section, pidx=pidx)
                mcmc.graph[STRNAMES.PERT_INDICATOR_PROB].visualize(
                    path=os.path.join(perturbationpath, 'probability.pdf'),
                    f=f, section=section, pidx=pidx)
                f.close()
                mcmc.graph[STRNAMES.PERT_INDICATOR].visualize(
                    path=os.path.join(perturbationpath, 'bayes_factors.tsv'),
                    section=section, pidx=pidx, fixed_clustering=args.fixed_clustering)
                mcmc.graph[STRNAMES.PERT_VALUE].visualize(
                    basepath=perturbationpath, section=section, pidx=pidx,
                    taxa_formatter='%(paperformat)s', fixed_clustering=args.fixed_clustering)

        # Plot Filtering
        # --------------
        logger.info('Plot filtering')
        filteringpath = os.path.join(basepath, 'filtering')
        os.makedirs(filteringpath, exist_ok=True)
        mcmc.graph[STRNAMES.FILTERING].visualize(basepath=filteringpath,
                                                 taxa_formatter='%(paperformat)s')
