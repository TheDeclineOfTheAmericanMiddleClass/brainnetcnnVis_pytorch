import torch

from preprocessing.Preproc_funcs import *
from preprocessing.Read_demographic_data import read_dem_data
from preprocessing.Read_raw_data import read_raw_data
from preprocessing.Read_task_data import *

# # Everything to be put on a GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()

# Posssibledata directories
dataDirs = {'HCP_rsfc_pCorr01_300': 'data/3T_HCP1200_MSMAll_d300_ts2_RIDGE',  # HCP partial correaltion @ rho = .01
            'HCP_rsfc_pCorr01_264': 'data/POWER_264_FIXEXTENDED_RIDGEP',  # power 264 resting PCORR matrices
            'HCP_rsfc_Corr_300': 'data/HCP_created_ICA300_mats/corr',  # HCP-created ICA300 rsfc
            'HCP_face_268': 'data/cfHCP900_FSL_GM',  # emotion task 268x268 # TODO: partial or full correlation?
            'Lea_EB_rsfc_300': 'data/edge_betweenness',  # edge-betweeness created by Lea
            'Adu_rsfc_pCorr50_300': 'data/self_created_HCP_mats/ICA300_rho50_pcorr',
            # Adu's ICA 300 resting PCORR with rho of 0.5
            'Adu_rsfc_Corr_300': 'data/self_created_HCP_mats/ICA300_corr'  # should be equivalen to 'HCP_rsfc_Corr_300'
            }

############################
# DEGREES OF FREEDOM IN MODEL
############################
# Not pictured here: specific confounds, various architectures
dataDir = dataDirs['Adu_rsfc_pCorr50_300']  # Choosing data directory for training
predicted_outcome = 'neuro'  # 'neuro', 'age', 'sex', 'allFFI'
multi_outcome = False  # TODO: if number of outcomes is changed, (1) final layer output size in Define_model.py and (2) outcome in Load_model_data must also be changed
deconfound_flavor = 'X0Y0'  # or 'X1Y0', 'X1Y1', 'X0Y0'
scaled = False  # whether confound are scaled [0,1]
tan_mean = 'euclidean'
data_to_use = 'tdata'  # 'pddata', 'cdata', 'tdata'


if dataDir == dataDirs['HCP_face_268']:  # read in task-based connectivity data
    toi = 'tfMRI_EMOTION'  # task of interest
    cdata = reshape_task_matrix(taskCorr, toi, taskdata)
    subnums = taskIDs

else:  # read in the resting-state connectivity data
    toi = 'rsfc'
    cdata, subnums = read_raw_data(dataDir, actually_read=True)

# If testing for any FFI outcome and 1003 subjects in dataset, delete subjects with no FFI scores
if np.any(np.isin(['allFFI', 'neuro'], predicted_outcome)) and len(cdata) == 1003:
    # pruning subjects without necessary FFI data, for personality classification, 1003 subject dataset
    nan_subs = []
    no_FFISubs = [47, 80, 88, 225, 841, 922]  # subjects without FFI scores
    nan_subs.extend(no_FFISubs)
    if deconfound_flavor == 'X1Y1' or deconfound_flavor == 'X1Y0':
        no_WHSubs = [510]  # subject without weight/height
        nan_subs.extend(no_WHSubs)
    subnums = np.delete(subnums, nan_subs, axis=0)
    if not cdata == []:
        cdata = np.delete(cdata, nan_subs, axis=0)

# Read in demographic data, based on subjects given task
restricted, behavioral = read_dem_data(subnums)

# # Plotting arbitrary matrix/matrices to ensure data looks okay
# plot_raw_data(cdata, dataDir, nMat=2)
