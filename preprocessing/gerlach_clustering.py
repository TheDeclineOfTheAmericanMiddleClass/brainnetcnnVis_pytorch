import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.util_funcs import NEOFFIdomain_latent_transform, Bunch


# src_dir = os.path.abspath(os.path.join(os.curdir, 'personality-types'))
# sys.path.insert(1, src_dir)

def main(args):
    bunch = Bunch(args)

    assert bunch.dataset_to_cluster in ['HCP',
                                        'IMAGEN'], f'{bunch.dataset_to_cluster} not available for latent dimension transform'
    assert bunch.Q == 5, f'Latent feature extraction is only implemented for calculation over BigFive traits, Q==5'

    src_dir = os.path.abspath(os.path.join(os.curdir, 'personality-types'))
    sys.path.insert(1, src_dir)

    from src.analysis import preprocessing_01_filter_data, preprocessing_02_questions_vs_domains, \
        preprocessing_03_factor_analysis, analysis_clustering_02_meaningful_clusters_kernel_density

    # reading Gerlach subject IPIP-NEO data and running factor analysis
    if not os.path.isfile('personality-types/data_filter/df_filter_IPIP300.csv'):  # unpack data if never unpacked
        preprocessing_01_filter_data.main()
    if not os.path.isfile(
            'personality-types/data_filter/IPIP300_questions_domains.txt'):  # calc IPIP scores if never done
        preprocessing_02_questions_vs_domains.main()

    # running factor analysis
    preprocessing_03_factor_analysis.main(args)

    # loading in correct dataset
    if bunch.dataset_to_cluster == 'IMAGEN':
        # # reaading in IMAGEN personality data
        IMAGEN_HCP = pd.read_csv('/raid/projects/BIGFIVE/New_Tim_Hahn_GraphVar/BEHAV_IMAGEN_and_HCP_clean.csv')
        IMAGEN_only = IMAGEN_HCP[IMAGEN_HCP.ID.str.startswith('IMAGEN')]
        # NEO_keys = [x.startswith('NEO') for x in IMAGEN_only.keys()]
        # IMAGEN_NEO = IMAGEN_only[IMAGEN_only.keys()[NEO_keys]]
        IMAGEN_subs = [int(x[7:]) for x in IMAGEN_only.ID]
        data = IMAGEN_only

    elif bunch.dataset_to_cluster == 'HCP':
        # _, behavioral = read_dem_data() # TODO: clustering run on all available subject with non-NaN values
        # data = behavioral
        data = bunch.cdata

    # transforming data to FA latent dimensions
    #  saving as 'personality-types/data_filter/{dataset}_ipip{Q}_domain_latent_transform.npy'
    NEOFFIdomain_latent_transform(data, dataset=bunch.dataset_to_cluster, Q=bunch.Q)

    # clustering, and saving results as .pkl dictionary
    analysis_clustering_02_meaningful_clusters_kernel_density.main(args)

    # End product: gmm cluster dictionary, with cluster locations in space of latent dimensions
    gcd = pickle.load(
        open(f'personality-types/data_filter/{bunch.dataset_to_cluster}_gmm_cluster13_IPIP{bunch.Q}.pkl', "rb"))

    # # functions for plotting cluster distribution beween subjects
    def plot_most_likely_clusters(gmm_cluster_dict=gcd):
        '''Plots a histogram of likelihoods across all subjects, from highest to least likely cluster membership.'''
        ml_cluster = np.argsort(
            gmm_cluster_dict['labels'])  # plotting histogram to see distribution of most likely cluster
        ns_cluster_args = (gmm_cluster_dict['enrichment'] > 1.25) & (
                gmm_cluster_dict['pval'] < .01)  # non-spurious clusters

        fig, axs = plt.subplots(3, 5, sharey=True, sharex=True)
        fig.subplots_adjust(top=.85, left=.15, right=.9, wspace=.14, hspace=.28)
        axs = axs.ravel()
        for i, x in enumerate(ml_cluster.T[::-1] + 1):  # for each cluster of highest likelihood for all subjects...

            d = np.diff(np.unique(x)).min()
            left_of_first_bin = x.min() - float(d) / 2
            right_of_last_bin = x.max() + float(d) / 2

            tito = axs[i].hist(x, np.arange(left_of_first_bin, right_of_last_bin + d, d))

            axs[i].set_title(i + 1)
            axs[i].vlines(np.argwhere(ns_cluster_args).squeeze() + 1, ymin=0, ymax=tito[0].max(), colors='r',
                          label='non-spurious clusters')  # plotting lines at non-spurious clusters
            axs[i].set_xlabel('cluster')
            axs[i].set_ylabel('subject count')

            if i == 0:
                axs[i].legend()
        fig.suptitle(
            f'Distribution of {ml_cluster.shape[0]} HCP Subjects\' personality cluster membership \n (arranged from most-to-least likely, 13 cluster-bins)')

    def plot_per_cluster_probability(gmm_cluster_dict=gcd):
        '''Plots subjects' likelihood of cluster membership, for each cluster.'''
        ns_cluster_args = (gmm_cluster_dict['enrichment'] > 1.25) & (
                gmm_cluster_dict['pval'] < .01)  # non-spurious clusters

        # # plotting histogram of probability distribution of non-spurious vs. spurious clusters
        fig, axs = plt.subplots(3, 5)
        fig.subplots_adjust(top=.85, left=.15, right=.9, wspace=.14, hspace=.28)
        axs = axs.ravel()
        for i, x in enumerate(gmm_cluster_dict['labels'].T):  # for all clusters...
            axs[i].hist(x, bins=50, range=(0, 1), log=True)

            if ns_cluster_args[i]:
                axs[i].set_title(f'Cluster {i + 1}, non-spurious')
            else:
                axs[i].set_title(f'Cluster {i + 1}, spurious')
        fig.suptitle(
            f'Distribution of {gmm_cluster_dict["labels"].shape[0]} HCP Subjects\' personality cluster log-likelihood\n(per cluster, 50 bins)')


if __name__ == '__main__':
    main()
