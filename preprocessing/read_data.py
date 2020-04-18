import os

import torch
import xarray as xr

from preprocessing.degrees_of_freedom import *
from preprocessing.preproc_funcs import *

# # Everything to be put on a GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()

if multi_input:  # create xarray to hold all matrices
    cdata = []

    for i, cd in enumerate(chosen_dir):  # for all directories

        if chosen_dir != ['HCP_alltasks_268']:  # setting chosen_tasks if multiple are available in the directory
            chosen_tasks_in_dir = ['NA']

        elif chosen_dir == ['HCP_alltasks_268']:
            chosen_tasks_in_dir = chosen_tasks

            # TODO: implement intelligent reading in of HC900_FSL_GM data and AFTER other directories read in, their concatenation
            if os.path.isfile('/raid/projects/Adu/brainnetcnnVis_pytorch/data/cfHCP900_FSL_GM/cfHCP900_FSL_GM.nc'):
                cdata = xr.load_dataset(
                    '/raid/projects/Adu/brainnetcnnVis_pytorch/data/cfHCP900_FSL_GM/cfHCP900_FSL_GM.nc')
                tasknames = ['_'.join((cd, taskname)) for taskname in
                             chosen_tasks_in_dir]  # creating keys for chosen tasks
                cdata = cdata[tasknames]
                break

        for j, taskname in enumerate(chosen_tasks_in_dir):  # for each task in the directory, read in the matrix
            try:
                partial, subnums = read_mat_data(directories[x], toi=tasks[taskname])
            except KeyError:
                raise KeyError(f'\'{taskname}\' is an invalid task name for dataset {x}...')

            nodes = [f'node {x}' for x in np.arange(partial.shape[-1])]
            partial = xr.DataArray(partial.squeeze(), coords=[subnums, nodes, nodes], dims=['subject', 'dim1', 'dim2'])
            partial.name = '_'.join((cd, taskname))

            cdata.append(partial)

    if os.path.isfile('/raid/projects/Adu/brainnetcnnVis_pytorch/data/cfHCP900_FSL_GM/cfHCP900_FSL_GM.nc'):
        subnums = cdata.subject.values

    elif not os.path.isfile('/raid/projects/Adu/brainnetcnnVis_pytorch/data/cfHCP900_FSL_GM/cfHCP900_FSL_GM.nc'):
        cdata = xr.align(*cdata, join='inner', exclude=['dim1',
                                                        'dim2'])  # 'inner' takes intersection, removing subjects without matrices in both datasets
        cdata = xr.merge(cdata, compat='override', join='exact')  # 'exact' merges on all dimensions exactly

        cdata = cdata.sortby(['subject', 'dim1', 'dim2'], ascending=True)
        subnums = cdata.subject.values

elif not multi_input:

    if chosen_dir != ['HCP_alltasks_268']:  # setting chosen_tasks if multiple are available in the directory
        chosen_tasks_in_dir = ['NA']
    elif chosen_dir == ['HCP_alltasks_268']:
        chosen_tasks_in_dir = chosen_tasks

    cdata, subnums = read_mat_data(directories[chosen_dir[0]], toi=tasks[chosen_tasks_in_dir[0]])

# TODO: implement better handling of subject with NaN-valued data
# Rudimentary handling of subjects with NaN values in FFI and confounds
# nan_subs = []
# if len(cdata) == 1003:
#     # assuming deconfounding with weight, delete subject with no weight measurement
#     if deconfound_flavor == 'X1Y1' or deconfound_flavor == 'X1Y0':
#         no_WHSubs = [510]  # subject without weight/height
#         nan_subs.extend(no_WHSubs)
#
#     # If predicting any FFI outcome and 1003 subjects in dataset, delete subjects with no FFI scores
#     if np.any(np.isin(['allFFI', 'neuro', 'open'], predicted_outcome)) and len(cdata) == 1003:
#         no_FFISubs = [47, 80, 88, 225, 841, 922]  # subjects without FFI scores
#         nan_subs.extend(no_FFISubs)
#
# # removing undesirable subjects
# subnums = np.delete(subnums, nan_subs, axis=0)
# if not cdata == []:
#     cdata = np.delete(cdata, nan_subs, axis=0)

# Read in demographic data, based on subjects given task
restricted, behavioral = read_dem_data(subnums)

# # Plotting arbitrary matrix/matrices to ensure data looks okay
# plot_mat_data(cdata, dataDir, nMat=2)
