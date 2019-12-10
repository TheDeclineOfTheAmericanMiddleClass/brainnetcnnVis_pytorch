from preprocessing.Read_raw_data import read_raw_data
from preprocessing.Preproc_funcs import PD_transform
from preprocessing.Analyze_raw_data import plot_raw_data, test_raw_data
# from preprocessing.Partitioning import partition
from preprocessing.Tangent_transform import tangent_transform
import torch

# # Everything to be put on a GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()

# TODO: figure out whether ICA300 dataset created by Lea or HCP
# setting data directory
dataDir = 'data/3T_HCP1200_MSMAll_d300_ts2_RIDGE'
# dataDir = 'data/POWER_264_FIXEXTENDED_RIDGEP'

# read in the data
data, restricted, behavioral, subnums = read_raw_data(dataDir)

# load data used to train model
from analysis.Load_model_data import *

# # testing if arbitrary data matrix is positive definite
# test_raw_data(data, nMat=2)

# # plotting arbitrary matrix/matrices to ensure data looks okay
# plot_raw_data(data, dataDir, nMat=2)

# # if non-PD matrices, convert to closes PD matrix
# pddata = PD_transform(data)

# # TODO: get to source of pyriemman error
# # ValueError: Covariance matrices must be positive definite. Add regularization to avoid this error.
# # convert it tangent space
# tsdata = tangent_transform(data)

# # partition it by twins
# partition(restricted)

