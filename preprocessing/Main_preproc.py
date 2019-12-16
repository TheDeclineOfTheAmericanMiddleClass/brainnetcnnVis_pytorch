from preprocessing.Read_raw_data import read_raw_data
from preprocessing.Preproc_funcs import *
from preprocessing.Analyze_raw_data import plot_raw_data, test_raw_data
# from preprocessing.Partitioning import partition
from preprocessing.Tangent_transform import tangent_transform
import torch

# # Everything to be put on a GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()

# setting data directory
# dataDir = 'data/3T_HCP1200_MSMAll_d300_ts2_RIDGE' # TODO: figure out whether ICA300 dataset created by Lea or HCP
dataDir = 'data/POWER_264_FIXEXTENDED_RIDGEP'
# dataDir = 'data/edge_betweenness'

# read in the connectivity data
cdata, restricted, behavioral, subnums = read_raw_data(dataDir)

# # # testing if arbitrary data matrices are positive definite
# test_raw_data(data, nMat=2)
#
# # # plotting arbitrary matrix/matrices to ensure data looks okay
# plot_raw_data(data, dataDir, nMat=2)
#
# # if non-PD matrices, convert to closes PD matrix
pddata = PD_transform(cdata)

# projecting matrices into tangent space
tdata = tangent_transform(pddata[0:10], ref='euclidean')

# defining what do use...connectivity data, or tangent space data?
# data = cdata
data = tdata

# # partition data by twins
# partition(restricted)

