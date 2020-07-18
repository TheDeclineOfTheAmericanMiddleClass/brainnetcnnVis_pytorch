from utils.var_names import multiclass_outcomes

# TODO: create utils folder
# Degrees of freedom in the model input/output # TODO: read in changing DoF from dof_parser
chosen_dir = ['HCP_alltasks_268']  # list of data keys in var(data_directories), pointing to data to train on
chosen_tasks = ['rest1']  # list of 'HCP_alltasks_268' tasks to train on; set to ['NA'] if directory unused
predicted_outcome = ['Gender']  # TODO: identify non-spurious clusters to predict

transformations = 'untransformed'  # 'positive definite', 'untransformed', 'tangent'
tan_mean = 'harmonic'  # euclidean, harmonic
deconfound_flavor = 'X0Y0'  # or 'X1Y0', 'X1Y1', 'X0Y0' # TODO: implement X1Y1 multi-input/xarray data
confound_names = None  # ['Weight','Height','Handedness', 'Age_in_Yrs', 'PSQI_Score'], None
scale_confounds = True  # whether confound are scaled by train set confounds's max value
architecture = 'usama'  # 'yeo_sex', 'kawahara', 'usama', 'parvathy_v2', 'FC90Net', 'yeo_58'
optimizer = 'sgd'  # 'sgd', 'adam'
edge_betweenness = False  # if reading in data_directories['Johann_mega_graph'], whether to include edgebetweeness vector

# params for Gerlach personality clsutering
dataset_to_cluster = 'HCP'  # 'HCP', 'IMAGEN'
Q = 5  # number of FFIitems/domains {5, 300} from which to calculate latent dimensions

# Hyper parameters for training
momentum = 0.9  # momentum
lr = .0001  # learning rate
wd = .0005  # weight decay
max_norm = 1.5  # maximum value of normed gradients, to prevent explosion/vanishing

# Epochs over which the network is trained
nbepochs = 300  # number of epochs to run
early = False  # early stopping or nah
ep_int = 5  # early stopping interval
min_ep = 50  # minimum epochs after which to check for stagnation in learning
cv_folds = 5  # cross validation folds, for shallow networks

################################################
## Automated setting of conditional variables ##
###############################################

# TODO: fix for auto read-in of Gerlach data, perhaps write a class to store conditional vars and update them after read_data

# setting necessary string for saving model, plotting
if scale_confounds:
    scl = '_scaled'
else:
    scl = ''

# boolean logic for multiple input matrices
num_input = len(chosen_dir) - 1 + len(chosen_tasks)
if num_input == 1:
    multi_input = False
else:
    multi_input = True

multiclass = bool(predicted_outcome[0] in multiclass_outcomes)  # necessary bool for single-outcome, multiclass problems

# logic for transformations, etc.
if chosen_dir == ['Johann_mega_graph']:
    data_are_matrices = False
else:
    data_are_matrices = True

# setting names for xarray data variables
chosen_Xdatavars = chosen_dir.copy()
if 'HCP_alltasks_268' in chosen_Xdatavars:
    chosen_Xdatavars.remove('HCP_alltasks_268')
    for task in chosen_tasks:
        datavar = f'HCP_alltasks_268_{task}'
        chosen_Xdatavars.append(datavar)

# exit logic for mutually exclusive dof
if 'HCP_alltasks_268' in chosen_dir:
    assert 'NA' not in chosen_tasks, 'Please choose at least non-NA task(s) for HCP_alltasks_268'

if 'HCP_alltasks_268' not in chosen_dir:
    assert 'NA' not in chosen_tasks, 'Please set variable chosen_tasks to [\'NA\'] if chosen_dir != HCP_alltasks_268.'
