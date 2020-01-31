import torch

from preprocessing.Analyze_raw_data import plot_raw_data
from preprocessing.Preproc_funcs import *
from preprocessing.Read_demographic_data import read_dem_data
from preprocessing.Read_raw_data import read_raw_data
from preprocessing.Read_task_data import *

# # Everything to be put on a GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()

# # Setting data directory
dataDirs = {'HCP_rsfc_pCorr01_300': 'data/3T_HCP1200_MSMAll_d300_ts2_RIDGE',  # HCP partial correaltion @ rho = .01
            'HCP_rsfc_pCorr01_264': 'data/POWER_264_FIXEXTENDED_RIDGEP',  # power 264 resting PCORR matrices
            'HCP_rsfc_Corr_300': 'data/HCP_created_ICA300_mats/corr',  # HCP-created ICA300 rsfc
            'HCP_face_268': 'data/cfHCP900_FSL_GM',  # emotion task 268x268 # TODO: partial or full correlation?
            'Lea_EB_rsfc_300': 'data/edge_betweenness',  # edge-betweeness created by Lea
            'Adu_rsfc_pCorr50_300': 'data/self_created_HCP_mats/ICA300_rho50_pcorr',
            # Adu's ICA 300 resting PCORR with rho of 0.5
            'Adu_rsfc_Corr_300': 'data/self_created_HCP_mats/ICA300_corr'  # should be equivalen to 'HCP_rsfc_Corr_300'
            }

dataDir = dataDirs['HCP_face_268']

if dataDir == dataDirs['HCP_face_268']:  # read in task-based connectivity data
    toi = 'tfMRI_EMOTION'  # task of interest
    cdata = reshape_task_matrix(taskCorr, toi, taskdata)
    subnums = taskIDs

else:  # read in the resting-state connectivity data
    toi = 'rsfc'
    cdata, subnums = read_raw_data(dataDir, actually_read=True)

############################
# DEGREES OF FREEDOM IN MODEL
############################
# Not pictured here: specific confounds, various architectures
multi_outcome = True  # if changed, must also change final layer output size in Define_model.py and outcome in Load_model_data
deconfound = True
deconfound_flavor = 'X1Y1'  # or 'X1Y0'
scaled = True  # whether confound are scaled [0,1]
tan_mean = 'euclidean'
data_to_use = 'tdata'  # 'pddata', 'cdata'

if deconfound and len(cdata) == 1003:
    # pruning subjects without necessary FFI data, for personality classification, 1003 subject dataset
    # no_FFISubs = np.array([47,  80,  88, 225, 841, 922]) # subjects without FFI scores
    # no_WHSubs = 510 # subject without weight/height
    nan_subs = [47, 80, 88, 225, 510, 841, 922]
    subnums = np.delete(subnums, nan_subs, axis=0)
    if not cdata == []:
        cdata = np.delete(cdata, nan_subs, axis=0)

# Read in demographic data, based on subjects given task
restricted, behavioral = read_dem_data(subnums)

# Plotting arbitrary matrix/matrices to ensure data looks okay
plot_raw_data(cdata, dataDir, nMat=2)
