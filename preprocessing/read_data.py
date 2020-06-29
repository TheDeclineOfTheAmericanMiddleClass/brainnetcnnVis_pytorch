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
                # cdata = cdata[tasknames]
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
                             name=chosen_Xdatavars[0]).to_dataset()
    except ValueError:
        cdata = xr.DataArray(cdata, coords=[subnums, nodes], dims=['subject', 'dim1'], name=chosen_dir[0]).to_dataset()

print('Finished reading in matrix data...adding restricted and behavioral data to dataset.\n')

# adding restricted and behavioral data to dataset
r_vars = ['Family_ID', 'Subject', 'Weight', 'Height', 'Handedness', 'Age_in_Yrs']
b_vars = ['NEOFAC_O', 'NEOFAC_C', 'NEOFAC_E', 'NEOFAC_A', 'NEOFAC_N', 'PSQI_Score', 'Gender', 'PMAT24_A_CR']

# if cdata doesn't have behavioral and restricted data, add it
if np.all([r_var not in list(cdata.data_vars) for r_var in r_vars]) \
        and np.all([b_var not in list(cdata.data_vars) for b_var in b_vars]):
    restricted, behavioral = read_dem_data(cdata.subject.values)  # reading data for only available subjects
    brx = xr.merge([restricted[r_vars].to_xarray(), behavioral[b_vars].to_xarray()], join='inner').swap_dims(
        {'index': 'Subject'})
    brx = brx.rename_dims({'Subject': 'subject'})
    cdata = xr.merge([cdata, brx], join='inner')  # finding intersection
    cdata = cdata.dropna(dim='subject')  # dropping nan values

# add Gerlach soft-clustering scores to cdata
if chosen_dir == ['HCP_alltasks_268']:
    import pickle

    HCP_gmm_cluster_IPIP5 = pickle.load(
        open('personality-types/data_filter/gmm_cluster13_IPIP5.pkl', "rb"))  # read in file

    cluster_labels = HCP_gmm_cluster_IPIP5['labels'].T
    cluster_labels = np.log10(
        cluster_labels)  # log-likelihoods of cluster membership TODO: comment out if no improvement
    ns_cluster_args = (HCP_gmm_cluster_IPIP5['enrichment'] > 1.25) & (
            HCP_gmm_cluster_IPIP5['pval'] < .01)  # non-spurious clusters

    # HCP_ns_softcluster = HCP_gmm_cluster_IPIP5['labels'][:, ns_cluster_args]  # soft-cluster likelihoods of non-spurious
    # cdata['IPIP_ns_softcluster'] = xr.DataArray(HCP_ns_softcluter, dims=['subject','ns_cluster'])
    # HCP_ns_hardcluster = HCP_ns_softcluter.argmax(axis=1) # hard-cluster likelihoods of non-spurious
    # cdata['IPIP_ns_hardcluster'] = xr.DataArray(HCP_ns_hardcluster, dims='subject')

    cdata['hardcluster'] = xr.DataArray(cluster_labels.argmax(axis=0), dims='subject')
    for i, x in enumerate(cluster_labels):
        cdata[f'softcluster_{i + 1}'] = xr.DataArray(x, dims='subject').assign_attrs(
            dict(enrichment=HCP_gmm_cluster_IPIP5['enrichment'][i], pval=HCP_gmm_cluster_IPIP5['pval'][i],
                 non_spurious=ns_cluster_args[i]))

subnums = cdata.subject.values

# removing edge betweeness from mega file data
if 'Johann_mega_graph' in chosen_Xdatavars and not edge_betweenness:
    cdata = cdata.drop_sel(dim1=[f'node {x}' for x in range(1586, len(cdata.dim1.values))])

# cdata.to_netcdf('data/cfHCP900_FSL_GM/cfHCP900_FSL_GM.nc') # Saving when necessary
