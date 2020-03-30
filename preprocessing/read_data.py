import torch
import xarray as xr

from preprocessing.degrees_of_freedom import *
from preprocessing.preproc_funcs import *

# # Everything to be put on a GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()

if multi_input:  # create xarray to hold all matrices
    for i, x in enumerate(chosen_dir):
        partial, subnums = read_mat_data(directories[x])
        nodes = [f'node {x}' for x in np.arange(partial.shape[-1])]
        partial = xr.DataArray(partial.squeeze(), coords=[subnums, nodes, nodes], dims=['subject', 'dim1', 'dim2'])
        partial.name = x

        if i == 0:
            cdata = partial
        if i > 0:
            cdata = xr.align(cdata, partial, join='inner', exclude=['dim1',
                                                                    'dim2'])  # takes intersection, removing subjects without matrices in both datasets

    multi_cdata = xr.merge(cdata, compat='override',
                           join='exact')  # (join='inner') only holds for matrices of same parcellation
    multi_cdata = multi_cdata.sortby(['subject', 'dim1', 'dim2'], ascending=True)
    subnums = multi_cdata.subject.values

    cdata = multi_cdata  # TODO: remove if redundant

elif not multi_input:
    cdata, subnums = read_mat_data(directories[chosen_dir[0]])

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
