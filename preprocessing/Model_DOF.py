############################
# DEGREES OF FREEDOM IN MODEL
############################

# Posssible data directories
dataDirs = {'HCP_rsfc_pCorr01_300': 'data/3T_HCP1200_MSMAll_d300_ts2_RIDGE',  # HCP partial correlation @ rho = .01
            'HCP_rsfc_pCorr01_264': 'data/POWER_264_FIXEXTENDED_RIDGEP',  # power 264 resting PCORR matrices
            'HCP_rsfc_Corr_300': 'data/HCP_created_ICA300_mats/corr',  # HCP-created ICA300 rsfc
            'HCP_face_268': 'data/cfHCP900_FSL_GM',  # emotion task 268x268 # TODO: partial or full correlation?
            'Lea_EB_rsfc_264': 'data/edge_betweenness',  # edge-betweenness created by Lea
            'Adu_rsfc_pCorr50_300': 'data/self_created_HCP_mats/ICA300_rho50_pcorr',
            # Adu's ICA 300 resting PCORR with rho of 0.5
            'Adu_rsfc_Corr_300': 'data/self_created_HCP_mats/ICA300_corr'  # should be equivalent to 'HCP_rsfc_Corr_300'
            }

# Degrees of freedom in the model input/output  (not pictured here: specific confounds, various architectures)
dataDir = dataDirs['Adu_rsfc_pCorr50_300']  # Choosing data directory for training
predicted_outcome = 'sex'  # 'neuro', 'age', 'sex', 'allFFI', 'open'
num_classes = 2  # number of classes per outcome
num_outcome = 1  # number of outcomes predicted
multi_outcome = (num_outcome > 1)  # necessary boolean for model output & architecture
# multi_input = False  # TODO: implement multiple input matrices
data_to_use = 'untransformed'  # 'positive definite', 'untransformed', 'tangent'
tan_mean = 'euclidean'  # euclidean, harmonic, log euclidean, riemannian, kullback
deconfound_flavor = 'X0Y0'  # or 'X1Y0', 'X1Y1', 'X0Y0'
scaled = False  # whether confound are scaled by confound's max value in trainin set
architecture = 'yeo'  # 'yeo', 'kawahara', 'usama'# Setting criterion

# Setting hyper parameters for training
momentum = 0.9  # momentum
lr = 1e-4  # learning rate, changed from 0.00001 on 2.24.20
wd = .0005
ep_int = 2  # setting early stopping interval, how frequently changes in mae are checked


# Setting necessary variables for labeling plots and saved files
if multi_outcome and predicted_outcome == 'allFFI':  # TODO: logic must be changed if predicting various combinations of features
    outcome_names = ['O', 'C', 'E', 'A', 'N']
else:
    outcome_names = predicted_outcome

if type(outcome_names) != str:
    assert num_outcome == len(outcome_names), 'Number of outcomes must be same as outcome names !'

if scaled:
    scl = '_scaled'
else:
    scl = ''

if num_classes > 1:
    multiclass = True
else:
    multiclass = False
