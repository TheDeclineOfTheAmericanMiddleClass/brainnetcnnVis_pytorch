import torch

from preprocessing.Preproc_funcs import *
from preprocessing.Read_demographic_data import read_dem_data
from preprocessing.Read_raw_data import read_raw_data
from preprocessing.Read_task_data import *

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
    cdata, subnums = read_raw_data(dataDir, actually_read=True)

# pruning subjects without FFI data, for personality classification, 1003 subject dataset
# no_ffisubs = np.array([47,  80,  88, 225, 841, 922])
# no_weightsubs = 510
nan_subs = [47, 80, 88, 225, 510, 841, 922]
subnums = np.delete(subnums, nan_subs, axis=0)
if not cdata == []:
    cdata = np.delete(cdata, nan_subs, axis=0)

# Read in demographic data, based on subjects given task
restricted, behavioral = read_dem_data(subnums)

# # Plotting arbitrary matrix/matrices to ensure data looks okay
# plot_raw_data(cdata, dataDir, nMat=2)

