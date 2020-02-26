from preprocessing.Analyze_raw_data import *
from preprocessing.Model_DOF import *
from preprocessing.Read_demographic_data import read_dem_data
from preprocessing.Read_raw_data import read_raw_data
from preprocessing.Read_task_data import *

# # Everything to be put on a GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()

if dataDir == dataDirs['HCP_face_268']:  # read in task-based connectivity data
    taskIDs, taskCorr, taskNames, taskdata = read_task_data()
    toi = 'tfMRI_EMOTION'  # task of interest
    cdata = reshape_task_matrix(taskCorr, toi, taskdata, r_transform=False)
    subnums = taskIDs

else:  # read in the resting-state connectivity data
    if dataDir == dataDirs['Lea_EB_rsfc_264']:
        toi = 'eb'  # edge betweenness
    else:
        toi = 'rsfc'
    cdata, subnums = read_raw_data(dataDir, actually_read=True)

# TODO: include clause to delete subs if not a .npy file
# If predicting any FFI outcome and 1003 subjects in dataset, delete subjects with no FFI scores
if np.any(np.isin(['allFFI', 'neuro', 'open'], predicted_outcome)) and len(cdata) == 1003:
    nan_subs = []
    no_FFISubs = [47, 80, 88, 225, 841, 922]  # subjects without FFI scores
    nan_subs.extend(no_FFISubs)

    # assuming deconfounding with weight, delete subject with no weight measurement
    if deconfound_flavor == 'X1Y1' or deconfound_flavor == 'X1Y0':
        no_WHSubs = [510]  # subject without weight/height
        nan_subs.extend(no_WHSubs)

    # removing undesirable subjects
    subnums = np.delete(subnums, nan_subs, axis=0)
    if not cdata == []:
        cdata = np.delete(cdata, nan_subs, axis=0)

# Read in demographic data, based on subjects given task
restricted, behavioral = read_dem_data(subnums)

# # Plotting arbitrary matrix/matrices to ensure data looks okay
# plot_raw_data(cdata, dataDir, nMat=2)
