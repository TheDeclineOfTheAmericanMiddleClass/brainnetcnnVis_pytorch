import os
import pickle

import numpy as np
import torch
import xarray as xr

from utils.util_funcs import Bunch, read_dem_data, read_mat_data
from utils.var_names import HCP268_tasks, data_directories, r_vars, b_vars


def main(args):
    bunch = Bunch(args)

    # # Everything to be put on a GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_cuda = torch.cuda.is_available()

    read_in_anew = False  # bool determines whether data will be read in anew or loaded

    cdata = []

    for i, cd in enumerate(bunch.chosen_dir):  # for all data_directories

        if bunch.chosen_dir != ['HCP_alltasks_268']:  # set chosen_tasks if multiple are available in the directory
            chosen_tasks_in_dir = ['NA']

        # TODO: implement concatenation of multiple compatible directories
        elif bunch.chosen_dir == ['HCP_alltasks_268']:
            chosen_tasks_in_dir = bunch.chosen_tasks

            # load in, if all tasks are already saved. if not reads all in anew
            if os.path.isfile('/raid/projects/Adu/brainnetcnnVis_pytorch/data/cfHCP900_FSL_GM/cfHCP900_FSL_GM.nc'):
                cdata = xr.load_dataset(
                    '/raid/projects/Adu/brainnetcnnVis_pytorch/data/cfHCP900_FSL_GM/cfHCP900_FSL_GM.nc')
                tasknames = bunch.chosen_Xdatavars.copy()  # keys for chosen tasks

                # only task-relevant dec, pd, and tan matrices read in
                if bunch.deconfound_flavor in ['X1Y0', 'X1Y1']:
                    print('\nchecking for saved deconfounded matrices...')
                    dec_vars = ['_'.join(['dec', '_'.join(bunch.confound_names), x]) for x in tasknames]
                    avail_dec_vars = [var for var in dec_vars if var in list(cdata.data_vars)]  # those available
                    tasknames.extend(avail_dec_vars)

                if bunch.transformations in ['positive definite', 'tangent']:
                    print('checking for saved positive matrices...')
                    pd_vars = [f'pd_{datavar}' for datavar in dec_vars]  # name for PD matrices
                    avail_pd_vars = [var for var in pd_vars if var in list(cdata.data_vars)]  # those available
                    tasknames.extend(avail_pd_vars)

                if bunch.transformations in ['tangent']:
                    print('checking for saved tangent matrices...\n')
                    tan_vars = [f'tan_{datavar}' for datavar in dec_vars]  # name for tangent matrices
                    avail_tan_vars = [var for var in tan_vars if var in list(cdata.data_vars)]  # those available
                    tasknames.extend(avail_tan_vars)

                try:
                    cdata = cdata[tasknames]  # checking if all tasks there, and loading in only necessaries
                    break
                except KeyError:
                    cdata = []

        for j, taskname in enumerate(chosen_tasks_in_dir):  # for each task in the directory, read in the matrix

            read_in_anew = True

            try:
                print(f'reading in task ({taskname}) from directory {bunch.chosen_dir}..')
                partial, subnums = read_mat_data(data_directories[cd], toi=HCP268_tasks[taskname])
            except KeyError:
                raise KeyError(f'\'{taskname}\' is an invalid task name for data {cd}...')

            nodes = [f'node {x}' for x in np.arange(partial.shape[-1])]

            try:
                partial = xr.DataArray(partial.squeeze(), coords=[subnums, nodes, nodes],
                                       dims=['subject', 'dim1', 'dim2'], name='_'.join((cd, taskname)))
            except ValueError:
                partial = xr.DataArray(partial.squeeze(), coords=[subnums, nodes], dims=['subject', 'dim1'],
                                       name=bunch.chosen_dir[0])
            cdata.append(partial)

    if read_in_anew:
        cdata = xr.align(*cdata, join='inner')  # 'inner' takes intersection of cdata objects
        cdata = xr.merge(cdata, compat='override', join='exact')  # 'exact' merges on all dimensions exactly

    # TODO: note 'node' sorting here isn't 1 - 300, but rather 0, 1, 10, etc.
    try:
        cdata = cdata.sortby(['subject', 'dim1', 'dim2'], ascending=True)
    except KeyError:
        cdata = cdata.sortby(['subject', 'dim1'], ascending=True)

    print('Finished reading in matrix data...reading HCP restricted and behavioral data...')

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
    if bunch.chosen_dir == ['HCP_alltasks_268']:
        gmm_cluster = pickle.load(open(f'personality-types/data_filter/gmm_cluster13_IPIP5.pkl', "rb"))  # read in file
        # TODO: change hard coding, update if reading in IMAGEN data
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
                     non_spurious=int(ns_cluster_args[i])))

    subnums = cdata.subject.values

    # removing edge betweeness from mega file data
    if 'Johann_mega_graph' in bunch.chosen_Xdatavars and not bunch.edge_betweenness:
        cdata = cdata.drop_sel(dim1=[f'node {x}' for x in range(1586, len(cdata.dim1.values))])

    if read_in_anew:  # saving when necessary
        cdata.to_netcdf('data/cfHCP900_FSL_GM/cfHCP900_FSL_GM.nc')

    return dict(subnums=subnums, cdata=cdata, use_cuda=use_cuda, device=device)


if __name__ == '__main__':
    main()
