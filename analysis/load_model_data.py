import xarray as xr

from preprocessing.degrees_of_freedom import predicted_outcome, tasks, chosen_dir, tan_mean, data_are_matrices, \
    chosen_tasks, deconfound_flavor, scale_confounds, confound_names, chosen_Xdatavars, transformations, \
    subnum_paths, multiclass_outcomes, multiclass
from preprocessing.preproc_funcs import *
from preprocessing.read_data import subnums, cdata

# # from cdata, calculating number of classes and number of outcomes
# setting number of classes per outcome
if predicted_outcome[
    0] in multiclass_outcomes:  # NOTE: can only handle multiclass problems with single outcome prediction
    num_classes = np.unique(cdata[predicted_outcome[0]]).size
else:
    num_classes = 1

# specifying outcome and its shape
if multiclass:  # TODO: later, implement logic for deconfounding with (multiclass + continuous) outcomes
    outcome = np.where(np.array([pd.get_dummies(cdata[x].values, dtype=bool).to_numpy()
                                 for x in predicted_outcome]).squeeze())[1].astype(float)
else:
    outcome = np.array([cdata[x].values for x in predicted_outcome],
                       dtype=float).squeeze()  # Setting variable for network to predict

if outcome.shape[0] != len(cdata.subject):  # assuming 2dim outcome array, ensure first dim is subject,
    outcome = outcome.T

# updating number of outcomes with outcome shape
if outcome.shape.__len__() > 1:
    num_outcome = outcome.shape[-1]
else:
    num_outcome = 1
multi_outcome = num_outcome > 1

###############################################################
# Defining train, test, validaton sets with Parvathy's partitions
###############################################################
# train_subs, train_ind, _ = np.intersect1d(subnums, np.loadtxt(train_subnum_path), return_indices=True)
# val_subs, val_ind, _ = np.intersect1d(subnums, np.loadtxt(val_subnum_path), return_indices=True)
# test_subs, test_ind, _ = np.intersect1d(subnums, np.loadtxt(test_subnum_path), return_indices=True)
# print(f'{train_ind.shape + test_ind.shape + val_ind.shape} subjects total included in test-train-validation sets '
#       f'({len(train_ind) + len(test_ind) + len(val_ind)} total)...\n')
partition_subs = dict()  # dict for subject numbers by set split
partition_inds = dict()  # dict for subject indices by set split

partition_subs["train"], partition_inds["train"], _ = np.intersect1d(subnums, np.loadtxt(subnum_paths["train"]),
                                                                     return_indices=True)
partition_subs["test"], partition_inds["test"], _ = np.intersect1d(subnums, np.loadtxt(subnum_paths["test"]),
                                                                   return_indices=True)
partition_subs["val"], partition_inds["val"], _ = np.intersect1d(subnums, np.loadtxt(subnum_paths["val"]),
                                                                 return_indices=True)

print(
    f'{partition_subs["test"].shape + partition_subs["train"].shape + partition_subs["val"].shape} subjects total included in test-train-validation sets '
    f'({len(partition_inds["train"]) + len(partition_inds["test"]) + len(partition_inds["val"])} total)...\n')

# # Dropping subjects not in train lists from all data variables in cdata
# difference = set(cdata[chosen_Xdatavars[0]].subject.values).difference(
#     set(np.hstack([partition_subs["train"], partition_subs["test"], partition_subs["val"]]).tolist()))
#
# for i, datavar in enumerate(list(cdata.data_vars)):
#     cdata[datavar] = cdata[datavar].drop_sel(dict(subject=list(difference)))

###############################################################
# # Deconfounding X and Y for data classes
###############################################################
if deconfound_flavor == 'X1Y1':
    raise NotImplementedError('X1Y1 not implementd yet. Please use another deconfounding method.')
    # TODO: implement X1Y1 deconfounding

if deconfound_flavor == 'X1Y1' or deconfound_flavor == 'X1Y0':  # If we have data to deconfound...
    for i, datavar in enumerate(chosen_Xdatavars):

        dec_Xvar = f'dec_{"_".join(confound_names)}_{datavar}'  # name for positive definite transformed mats
        # dec_Yvar = f'{"_".join(predicted_outcome)}_dec_{"_".join(confound_names)}_{datavar}'

        if dec_Xvar in list(cdata.data_vars):  # check if positive definite data already saved in xarray
            break

        cdata = cdata.assign({dec_Xvar: cdata[datavar]})

        if confound_names is not None:  # getting confounds
            confounds = [cdata[x].values for x in confound_names]

            if scale_confounds:  # scaling confounds per train set alone
                confounds = [x / np.max(np.abs(x[partition_inds["train"]])) for x in confounds]

        print(f'Deconfounding {dec_Xvar} data ...')
        X_corr, Y_corr, nan_ind = deconfound_dataset(data=cdata[datavar].values, confounds=confounds,
                                                     set_ind=partition_inds["train"], outcome=outcome)

        # if deconfound_flavor == 'X1Y1':  # load deconfounded Y data
        #     Y = Y_corr

        print('...deconfounding complete.')
        cdata[dec_Xvar] = xr.DataArray(X_corr,
                                       coords=dict(zip(list(cdata[dec_Xvar].dims),
                                                       [cdata[dec_Xvar][x].values for x in
                                                        list(cdata[dec_Xvar].dims)])),
                                       dims=list(cdata[dec_Xvar].dims))
        # cdata = cdata.assign({dec_Yvar: Y_corr})
        del X_corr

    # saving deconfounded matrices
    if chosen_dir == ['HCP_alltasks_268'] and chosen_tasks == list(tasks.keys())[:-1]:
        cdata.to_netcdf('data/cfHCP900_FSL_GM/cfHCP900_FSL_GM.nc')

    # updating chosen datavars
    chosen_Xdatavars = ['_'.join(['dec', '_'.join(confound_names), x]) for x in chosen_Xdatavars]

elif deconfound_flavor == 'X0Y0' or deconfound_flavor == 'X1Y0':
    Y = outcome


# Setting up multiclass classification with one-hot encoding
if multiclass:
    # if multiclass and one_hot:
    Y = multiclass_to_onehot(Y).astype(float)  # ensures Y is not of type object

if data_are_matrices:
    ###################################################################
    # # Projecting matrices into positive definite
    ###################################################################
    if transformations == 'positive definite' or transformations == 'tangent':
        for i, datavar in enumerate(chosen_Xdatavars):

            pd_var = f'pd_{datavar}'  # name for positive definite transformed mats

            if pd_var in list(cdata.data_vars):  # check if positive definite data already saved in xarray
                continue

            else:
                num_notPD, which = areNotPD(cdata[datavar].values)  # Test all matrices for positive definiteness
                print(f'There are {num_notPD} non-PD matrices in {datavar}...\n')

                print('Transforming non-PD matrices to nearest PD neighbor ...')
                cdata = cdata.assign({pd_var: cdata[datavar]})
                X_pd = PD_transform(cdata[pd_var].values[which])
                cdata[pd_var][which] = xr.DataArray(X_pd,
                                                    coords=dict(zip(list(cdata[pd_var].dims),
                                                                    [cdata[pd_var]['subject'].values[which],
                                                                     cdata[pd_var].dim1.values,
                                                                     cdata[pd_var].dim2.values])),
                                                    dims=list(cdata[pd_var].dims))

                del X_pd

        # saving positive definite matrices, for each data iteration
        if chosen_dir == ['HCP_alltasks_268'] and chosen_tasks == list(tasks.keys())[:-1]:
            cdata.to_netcdf('data/cfHCP900_FSL_GM/cfHCP900_FSL_GM.nc')

    ###################################################################
    # # Projecting matrices into tangent space
    ###################################################################
    if transformations == 'tangent':
        for i, datavar in enumerate(chosen_Xdatavars):

            tan_var = f'tan_{datavar}'  # name for tangent transformed mats

            # # If data set non-existent, projecting matrices into tangent space
            # saved_tan = f'data/transformed_data/tangent/{dir_str}{scl}_tangent.npy'
            # if os.path.isfile(saved_tan):
            #     print('Loading saved tangent space matrices ...\n')
            #     tdata = np.load(saved_tan)
            # else:
            #     np.save(saved_tan, tdata)

            if tan_var in list(cdata.data_vars):  # check if positive definite data already saved in xarray
                continue

            print('Transforming all matrices into tangent space ...')
            cdata = cdata.assign({tan_var: cdata[datavar]})
            X_tan = tangent_transform(refmats=cdata[datavar].loc[dict(subject=partition_subs["train"])],
                                      # tangent only from trainset
                                      projectmats=cdata[datavar].values,
                                      ref=tan_mean)

            cdata[tan_var] = xr.DataArray(X_tan,
                                          coords=dict(zip(list(cdata[tan_var].dims),
                                                          [cdata[tan_var]['subject'].values,
                                                           cdata[tan_var].dim1.values,
                                                           cdata[tan_var].dim2.values])),
                                          dims=list(cdata[tan_var].dims))

            del X_tan

        # saving tangent matrices, for each data iteration
        if chosen_dir == ['HCP_alltasks_268'] and chosen_tasks == list(tasks.keys())[:-1]:
            cdata.to_netcdf('data/cfHCP900_FSL_GM/cfHCP900_FSL_GM.nc')  # Saving when necessary

# updating list of chosen data for training
if transformations == 'tangent':
    chosen_Xdatavars = ['_'.join(['tan', x]) for x in chosen_Xdatavars]
elif transformations == 'positive definite':
    chosen_Xdatavars = ['_'.join(['pd', x]) for x in chosen_Xdatavars]

X = cdata
del cdata
