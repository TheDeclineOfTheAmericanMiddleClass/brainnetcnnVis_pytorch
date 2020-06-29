## Start by running the below notebooks:
# personality-types/notebooks/preprocessing_01-filter-data.ipynb
# personality-types/notebooks/preprocessing_02-questions-vs-domains.ipynb
# personality-types/notebooks/preprocessing_03-factor-analysis.ipynb
# personality-types/notebooks/analysis_clustering-01_number-of-clusters-BIC.ipynb

import pickle

import matplotlib.pyplot as plt
import numpy as np

from preprocessing.read_data import cdata


def HCP_latent_transform(cdata, Q=5):
    '''Calculating factor-transformed, varimax-rotated latent dimensions of personality from HCP dataset.
    Saving as a .npy file.'''

    # HCP personality info
    HCP_domain = cdata[['NEOFAC_O', 'NEOFAC_C', 'NEOFAC_E', 'NEOFAC_A', 'NEOFAC_N']].to_array().values.T

    # dictionary of data necessary to transform HCP data
    mvtr = pickle.load(open(f'personality-types/data_filter/ipip{Q}-mvtr-1.pkl', "rb"))

    # z-score acc. to the Gerlach mean/std
    z_muvar = np.load(f'personality-types/data_filter/ipip{Q}-pre_cluster_zscore_mu_var-1.npy')
    z_mu, z_var = z_muvar[0], z_muvar[1]

    # transforming HCP data
    HCP_latent = (HCP_domain - mvtr['mu']) @ mvtr['trans_mat']  # applying scaling & factor analysis fit-transform
    HCP_latent = (mvtr['rot_mat'] @ HCP_latent.T).T  # varimax rotation
    HCP_latent = (HCP_latent - z_mu) / z_var  # z-scoring

    # saving as file, to be run through soft-cluster anaylsis
    np.save(f'personality-types/data_filter/ipip{Q}_HCP_latent_transform.npy', HCP_latent)


HCP_latent_transform(cdata)

# # Run personality-types/notebooks/analysis_clustering-02_meaningful-clusters-kernel-density.ipynb..

# End product: gmm cluster dictionary, with cluster locations in space of latent dimensions
#  (calculated from Gerlach subjects' BigFive traits)
HCP_gcd = pickle.load(open('personality-types/data_filter/gmm_cluster13_IPIP5.pkl', "rb"))


def plot_most_likely_clusters(gmm_cluster_dict=HCP_gcd):
    '''Plots a histogram of likelihoods across all subjects, from highest to least likely cluster membership.'''
    ml_cluster = np.argsort(gmm_cluster_dict['labels'])  # plotting histogram to see distribution of most likely cluster
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


def plot_per_cluster_probability(gmm_cluster_dict=HCP_gcd):
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
