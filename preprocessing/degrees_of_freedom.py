############################
# DEGREES OF FREEDOM IN MODEL
############################

# Directories of posssible datasets to use
directories = {'HCP_rsfc_pCorr01_300': 'data/3T_HCP1200_MSMAll_d300_ts2_RIDGE',  # HCP partial correlation @ rho = .01
               'HCP_rsfc_pCorr01_264': 'data/POWER_264_FIXEXTENDED_RIDGEP',  # power 264 resting PCORR matrices
               'HCP_rsfc_Corr_300': 'data/HCP_created_ICA300_mats/corr',  # HCP-created ICA300 rsfc
               'HCP_alltasks_268': 'data/cfHCP900_FSL_GM',  # all HCP tasks, 268x268, z-scored
               'Lea_EB_rsfc_264': 'data/edge_betweenness',  # edge-betweenness created by Lea
               'Adu_rsfc_pCorr50_300': 'data/self_created_HCP_mats/ICA300_rho50_pcorr',
               # Adu's ICA 300 resting PCORR with rho of 0.5
               'Adu_rsfc_Corr_300': 'data/self_created_HCP_mats/ICA300_corr',
               # should be equivalent to 'HCP_rsfc_Corr_300'
               'Johann_mega_graph': 'data/Send_to_Tim/HCP_IMAGEN_ID_mega_file.txt'
               }

# Tasks in cfHCP900_FSL_GM dataset
tasks = {'rest1': 'rfMRI_REST1',
         'working_memory': 'tfMRI_WM',
         'gambling': 'tfMRI_GAMBLING',
         'motor': 'tfMRI_MOTOR',
         'rest2': 'rfMRI_REST2',
         'language': 'tfMRI_LANGUAGE',
         'social': 'tfMRI_SOCIAL',
         'relational': 'tfMRI_RELATIONAL',
         'faces': 'tfMRI_EMOTION',
         'NA': ''}

# Degrees of freedom in the model input/output
chosen_dir = ['HCP_alltasks_268']  # list of data keys for training
chosen_tasks = list(tasks.keys())[
               :-1]  # ['NA'] # list of 'HCP_alltasks_268' tasks for training; set to ['NA'] if directory unused
predicted_outcome = ['NEOFAC_O', 'NEOFAC_C', 'NEOFAC_E', 'NEOFAC_A',
                     'NEOFAC_N']  # 'NEOFAC_O', 'NEOFAC_C', 'NEOFAC_E', 'NEOFAC_A', 'NEOFAC_N', 'Gender', 'Age_in_Yrs'
one_hot = True  # only relevant for classification-based outcomes (i.e. sex)
data_to_use = 'untransformed'  # 'positive definite', 'untransformed', 'tangent' # TODO: implement transformations for multi-input/xarray data
tan_mean = 'euclidean'  # euclidean, harmonic
deconfound_flavor = 'X0Y0'  # or 'X1Y0', 'X1Y1', 'X0Y0' # TODO: implement for multi-input/xarray data
confound_names = None  # ['Weight','Height','Handedness', 'Age_in_Yrs', 'PSQI_Score'] # (sleep quality)
scale_confounds = False  # whether confound are scaled by confound's max value in training set
architecture = 'usama'  # 'yeo_sex', 'kawahara', 'usama', 'parvathy_v2', 'FC90Net'
optimizer = 'sgd'  # 'sgd', 'adam'

# Hyper parameters for training
momentum = 0.9  # momentum
lr = .0001  # learning rate
wd = .0005  # weight decay
max_norm = 1.5  # maximum value of normed gradients, to prevent explosion/vanishing

# Epochs over which the network is trained
nbepochs = 300  # number of epochs to run
early = True  # early stopping or nah
ep_int = 5  # early stopping interval
min_ep = 50  # minimum epochs after which to check for stagnation in learning

# various measures of interest from the HCP dataset
r_vars = ['Family_ID', 'Subject', 'Weight', 'Height', 'Handedness', 'Age_in_Yrs']
b_vars = ['NEOFAC_O', 'NEOFAC_C', 'NEOFAC_E', 'NEOFAC_A', 'NEOFAC_N', 'PSQI_Score', 'Gender']

################################################
## Automated setting of conditional variables ##
################################################

# setting number of classes per outcome
if predicted_outcome == 'Gender':
    num_classes = 2
else:
    num_classes = 1

# setting number of outcomes to predict
num_outcome = len(predicted_outcome)  # number of outcomes predicted

# necessary booleans for model output & architecture
multi_outcome = (num_outcome > 1)

if num_classes > 1:
    multiclass = True
else:
    multiclass = False

# setting necessary string for saving model, plotting
if scale_confounds:
    scl = '_scaled'
else:
    scl = ''

# logic for accurate calculation of multi_input
if chosen_tasks == 'NA':
    assert sum([directory == 'HCP_alltasks_268' for directory in
                chosen_dir]) == 0, 'Please choose at least one task for HCP_AllTasks'
if sum([directory == 'HCP_alltasks_268' for directory in chosen_dir]) == 0:
    assert chosen_tasks == ['NA'], 'Please set variable chosen_tasks to [\'NA\'] to continue.'

# boolean logic for multiple input matrices
num_input = len(chosen_dir) - 1 + len(chosen_tasks)

if num_input == 1:
    multi_input = False
else:
    multi_input = True

# logic for
if chosen_dir == ['Johann_mega_graph']:
    data_are_matrices = False
else:
    data_are_matrices = True
