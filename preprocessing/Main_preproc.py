import os
import torch

from preprocessing.Read_raw_data import read_raw_data
from preprocessing.Read_task_data import *
from preprocessing.Analyze_raw_data import plot_raw_data, test_raw_data, PD_transform
from preprocessing.Read_demographic_data import read_dem_data
from preprocessing.Preproc_funcs import *
from preprocessing.Tangent_transform import tangent_transform

# # Everything to be put on a GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()

# # Setting data directory
# dataDir = 'data/3T_HCP1200_MSMAll_d300_ts2_RIDGE'  # equivalent to HCP partial correaltion matrices @ rho = .01
# dataDir = 'data/POWER_264_FIXEXTENDED_RIDGEP' # power 264 resting PCORR matrices
# dataDir = 'data/edge_betweenness  # edge-betweeness created by Lea
# dataDir = 'data/cfHCP900_FSL_GM'  # emotion task 268x268
# dataDir = 'data/HCP_created_ICA300_mats/pcorr'  # HCP-created ICA300 rsfc
dataDir = 'data/self_created_HCP_mats/ICA300_rho50_pcorr'  # self-created ICA 300 resting PCORR matrices with rho of 0.5
# dataDir = 'data/self_created_HCP_mats/ICA300_corr'

if dataDir == 'data/cfHCP900_FSL_GM':  # read in task-based connectivity data
    toi = 'tfMRI_EMOTION'  # task of interest
    cdata = reshape_task_matrix(taskCorr, toi, taskdata)
    subnums = taskIDs

else:  # read in the resting-state connectivity data
    toi = 'rsfc'
    cdata, subnums = read_raw_data(dataDir)

# Read in demographic data, based on subjects given task
restricted, behavioral = read_dem_data(subnums)

# # Plotting arbitrary matrix/matrices to ensure data looks okay
# plot_raw_data(cdata, dataDir, nMat=2)

# TODO: Deconfound data before transforming to PD or tangent space
# deconfound the dataset


# Test all matrices for positive definiteness
num_notPD, which = areNotPD(cdata)
print(f'There are {num_notPD} non-PD matrices in {dataDir}...\n')

# If data set has non-PD matrices, convert to closest PD matrix
if num_notPD != 0:
    saved_pd = f'data/transformed_data/positive_definite/pd_{toi}_ICA300_rho50_pcorr.npy'  # TODO: change this on running new datasets
    if os.path.isfile(saved_pd):
        print('Loading saved positive definite matrices ...\n')
        pddata = np.load(saved_pd)
    else:
        print('Transforming non-PD matrices to nearest PD neighbor ...\n')
        pdmats = PD_transform(cdata[which])
        pddata = cdata.copy()
        pddata[which] = pdmats
        np.save(saved_pd, pddata)
else:
    pddata = cdata

# If data set non-existent, projecting matrices into tangent space
saved_tan = f'data/transformed_data/tangent/tan_{toi}_ICA300_rho50_pcorr.npy'  # TODO: change this on running new datasets
if os.path.isfile(saved_tan):
    print('Loading saved tangent space matrices ...\n')
    tdata = np.load(saved_tan)
else:
    print('Transforming all matrices into tangent space ...\n')
    tdata = tangent_transform(pddata, ref='euclidean')
    np.save(saved_tan, tdata)

# TODO: implement shrinking of tangent space data, ?implement optimal shrinkage parameter
# from sklearn.covariance import LedoitWolf
# cov = LedoitWolf().fit()  # must be fit with saamples x features
# shrink = .7 # arbitrary
# regcov = (1-shrink) * cov.covariance_ + shrink * np.trace(cov)/len(cov) * np.identity(len(cov) # regularized covariance
# stdata =

# Defining what do use for training...connectivity data, rescaled connectivity, or tangent space data?
data = tdata
# data = cdata
# data = cdata/np.max(np.max(cdata, axis=1), axis=1)[:, np.newaxis, np.newaxis]  # rescaling btwn 0 and 1

# # partition data by twins
# partition(restricted)
