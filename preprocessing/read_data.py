import os

import numpy as np
import torch
import xarray as xr

from utils.util_funcs import Bunch, read_dem_data, read_mat_data
from utils.var_names import HCP268_tasks, data_directories


def main(args):
    bunch = Bunch(args)

    # # Everything to be put on a GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_cuda = torch.cuda.is_available()

    if bunch.multi_input:  # create xarray to hold all matrices
        cdata = []

        for i, cd in enumerate(bunch.chosen_dir):  # for all data_directories

            if bunch.chosen_dir != [
                'HCP_alltasks_268']:  # setting chosen_tasks if multiple are available in the directory
                chosen_tasks_in_dir = ['NA']

            elif bunch.chosen_dir == ['HCP_alltasks_268']:
                chosen_tasks_in_dir = bunch.chosen_tasks

                # TODO: implement intelligent reading in of HC900_FSL_GM data and AFTER other data_directories read in, their concatenation
                if os.path.isfile('/raid/projects/Adu/brainnetcnnVis_pytorch/data/cfHCP900_FSL_GM/cfHCP900_FSL_GM.nc'):
                    cdata = xr.load_dataset(
                        '/raid/projects/Adu/brainnetcnnVis_pytorch/data/cfHCP900_FSL_GM/cfHCP900_FSL_GM.nc')
                    tasknames = ['_'.join((cd, taskname)) for taskname in
                                 chosen_tasks_in_dir]  # creating keys for chosen tasks
                    # cdata = cdata[tasknames]
                    break

            for j, taskname in enumerate(chosen_tasks_in_dir):  # for each task in the directory, read in the matrix
                try:
                    partial, subnums = read_mat_data(data_directories[cd], toi=HCP268_tasks[taskname])
                except KeyError:
                    raise KeyError(f'\'{taskname}\' is an invalid task name for data {cd}...')

                nodes = [f'node {x}' for x in np.arange(partial.shape[-1])]

                # creating dictionary of dims for xarray
                partial = xr.DataArray(partial.squeeze(), coords=[subnums, nodes, nodes],
                                       dims=['subject', 'dim1', 'dim2'], name='_'.join((cd, taskname)))
                cdata.append(partial)

        if bunch.chosen_dir == ['HCP_alltasks_268']:
            if os.path.isfile('/raid/projects/Adu/brainnetcnnVis_pytorch/data/cfHCP900_FSL_GM/cfHCP900_FSL_GM.nc'):
                subnums = cdata.subject.values

        else:
            cdata = xr.align(*cdata, join='inner')  # 'inner' takes intersection of cdata objects
            cdata = xr.merge(cdata, compat='override', join='exact')  # 'exact' merges on all dimensions exactly
            cdata = cdata.sortby(['subject', 'dim1', 'dim2'], ascending=True)

    elif not bunch.multi_input:

        if bunch.chosen_dir != ['HCP_alltasks_268']:  # setting chosen_tasks if multiple are available in the directory
            chosen_tasks_in_dir = ['NA']

        elif bunch.chosen_dir == ['HCP_alltasks_268']:
            chosen_tasks_in_dir = bunch.chosen_tasks

        cdata, subnums = read_mat_data(data_directories[bunch.chosen_dir[0]], toi=HCP268_tasks[chosen_tasks_in_dir[0]])
        nodes = [f'node {x}' for x in np.arange(cdata.shape[-1])]

        try:
            cdata = xr.DataArray(cdata, coords=[subnums, nodes, nodes], dims=['subject', 'dim1', 'dim2'],
                                 name=bunch.chosen_Xdatavars[0]).to_dataset()
        except ValueError:
            cdata = xr.DataArray(cdata, coords=[subnums, nodes], dims=['subject', 'dim1'],
                                 name=bunch.chosen_dir[0]).to_dataset()

    print('Finished reading in matrix data...reading HCP restricted and behavioral data...')

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
    if bunch.chosen_dir == ['HCP_alltasks_268']:  # TODO: change to be specified form subject numbers
        import pickle

        gmm_cluster = pickle.load(
            open(f'personality-types/data_filter/gmm_cluster13_IPIP5.pkl',
                 "rb"))  # read in file # TODO: update for IMAGEN, change hard coding
        # open(f'personality-types/data_filter/{bunch.dataset_to_cluster}_gmm_cluster13_IPIP{bunch.Q}.pkl', "rb"))

        cluster_labels = gmm_cluster['labels'].T
        cluster_labels = np.log10(cluster_labels)  # loglik of cluster membership
        ns_cluster_args = (gmm_cluster['enrichment'] > 1.25) & (
                gmm_cluster['pval'] < .01)  # non-spurious clusters

        cdata['hardcluster'] = xr.DataArray(cluster_labels.argmax(axis=0), dims='subject',
                                            coords=dict(subject=cdata.subject.values))
        for i, x in enumerate(cluster_labels):
            cdata[f'softcluster_{i + 1}'] = xr.DataArray(x, dims='subject').assign_attrs(
                dict(enrichment=gmm_cluster['enrichment'][i], pval=gmm_cluster['pval'][i],
                     non_spurious=ns_cluster_args[i]))

    subnums = cdata.subject.values

    # removing edge betweeness from mega file data
    if 'Johann_mega_graph' in bunch.chosen_Xdatavars and not bunch.edge_betweenness:
        cdata = cdata.drop_sel(dim1=[f'node {x}' for x in range(1586, len(cdata.dim1.values))])

    # cdata.to_netcdf('data/cfHCP900_FSL_GM/cfHCP900_FSL_GM.nc') # Saving when necessary

    return dict(subnums=subnums, cdata=cdata, use_cuda=use_cuda, device=device)


if __name__ == '__main__':
    main()
