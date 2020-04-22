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
                partial, subnums = read_mat_data(directories[cd], toi=tasks[taskname])
            except KeyError:
                raise KeyError(f'\'{taskname}\' is an invalid task name for dataset {cd}...')

            nodes = [f'node {x}' for x in np.arange(partial.shape[-1])]

            # creating dictionary of dims for xarray
            partial = xr.DataArray(partial.squeeze(), coords=[subnums, nodes, nodes],
                                   dims=['subject', 'dim1', 'dim2'], name='_'.join((cd, taskname)))
            cdata.append(partial)

    if chosen_dir == ['HCP_alltasks_268']:
        if os.path.isfile('/raid/projects/Adu/brainnetcnnVis_pytorch/data/cfHCP900_FSL_GM/cfHCP900_FSL_GM.nc'):
            subnums = cdata.subject.values

    else:
        cdata = xr.align(*cdata, join='inner')  # 'inner' takes intersection of cdata objects
        cdata = xr.merge(cdata, compat='override', join='exact')  # 'exact' merges on all dimensions exactly
        cdata = cdata.sortby(['subject', 'dim1', 'dim2'], ascending=True)

elif not multi_input:

    if chosen_dir != ['HCP_alltasks_268']:  # setting chosen_tasks if multiple are available in the directory
        chosen_tasks_in_dir = ['NA']

    elif chosen_dir == ['HCP_alltasks_268']:
        chosen_tasks_in_dir = chosen_tasks

    cdata, subnums = read_mat_data(directories[chosen_dir[0]], toi=tasks[chosen_tasks_in_dir[0]])
    nodes = [f'node {x}' for x in np.arange(cdata.shape[-1])]

    try:
        cdata = xr.DataArray(cdata, coords=[subnums, nodes, nodes], dims=['subject', 'dim1', 'dim2'],
                             name=chosen_dir[0]).to_dataset()
    except ValueError:
        cdata = xr.DataArray(cdata, coords=[subnums, nodes], dims=['subject', 'dim1'], name=chosen_dir[0]).to_dataset()

print('Finished reading in matrix data...adding restricted and behavioral data to dataset.\n')
# adding restricted and behavioral data to dataset
r_vars = ['Family_ID', 'Subject', 'Weight', 'Height', 'Handedness', 'Age_in_Yrs']
b_vars = ['NEOFAC_O', 'NEOFAC_C', 'NEOFAC_E', 'NEOFAC_A', 'NEOFAC_N', 'PSQI_Score', 'Gender']
restricted, behavioral = read_dem_data(cdata.subject.values)
brx = xr.merge([restricted[r_vars].to_xarray(), behavioral[b_vars].to_xarray()], join='inner').swap_dims(
    {'index': 'Subject'})
brx = brx.rename_dims({'Subject': 'subject'})
cdata = xr.merge([cdata, brx], join='inner')  # finding intersection
cdata = cdata.dropna(dim='subject')  # dropping nan values

# cdata.to_netcdf('data/cfHCP900_FSL_GM/cfHCP900_FSL_GM.nc') # Saving when necessary

# setting keys for xarray data variables
chosen_datavars = chosen_dir.copy()
if np.isin(chosen_datavars, 'HCP_alltasks_268')[0]:
    chosen_datavars.remove('HCP_alltasks_268')
    for task in chosen_tasks:
        datavar = f'HCP_alltasks_268_{task}'
        chosen_datavars.append(datavar)
