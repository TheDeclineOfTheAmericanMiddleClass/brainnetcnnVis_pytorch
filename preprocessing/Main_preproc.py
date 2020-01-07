import torch
from preprocessing.Read_raw_data import read_raw_data
from preprocessing.Preproc_funcs import *
from preprocessing.Analyze_raw_data import plot_raw_data, test_raw_data
from preprocessing.Tangent_transform import tangent_transform

# from preprocessing.Partitioning import partition

# # Everything to be put on a GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()

# setting data directory
dataDir = 'data/3T_HCP1200_MSMAll_d300_ts2_RIDGE'  # TODO: figure out whether ICA300 dataset was created by Lea or HCP
# dataDir = 'data/POWER_264_FIXEXTENDED_RIDGEP'
# dataDir = 'data/edge_betweenness'

# read in the connectivity data
cdata, restricted, behavioral, subnums = read_raw_data(dataDir)

# # testing if arbitrary data matrices are positive definite
# test_raw_data(cdata, nMat=2)

# # plotting arbitrary matrix/matrices to ensure data looks okay
# plot_raw_data(cdata, dataDir, nMat=2)

# # # if non-PD matrices, convert to closes PD matrix
# # pddata = PD_transform(cdata)
# pddata = np.load('data/transformed_data/pd_300.npy')


# # # projecting matrices into tangent space
# # tdata = tangent_transform(pddata, ref='euclidean')
tdata = np.load('data/transformed_data/pd_tan300.npy')

# TODO: implement shrinking of tangent space data, ?implement optimal shrinkage parameter
# from sklearn.covariance import LedoitWolf
# cov = LedoitWolf().fit()  # must be fit with samples x features
# shrink = .7 # arbitrary
# regcov = (1-shrink) * cov.covariance_ + shrink * np.trace(cov)/len(cov) * np.identity(len(cov) # regularized covariance
# stdata =

# defining what do use...connectivity data, rescaled connectivity, or tangent space data?
# data = cdata
# data = cdata/np.max(np.max(cdata, axis=1), axis=1)[:, np.newaxis, np.newaxis]  # rescaling btwn 0 and 1
data = tdata

# # partition data by twins
# partition(restricted)

