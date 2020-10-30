import itertools
import os
import pickle

import numpy as np
import pandas as pd
import xarray as xr

from utils.util_funcs import Bunch, deconfound_dataset, tangent_transform, areNotPD, PD_transform, multiclass_to_onehot
from utils.var_names import HCP268_tasks, subnum_paths, multiclass_outcomes


def main(args):
    bunch = Bunch(args)

    cdata = bunch.cdata
    chosen_Xdatavars = bunch.chosen_Xdatavars

    # # from cdata, calculating number of classes and number of outcomes
    # setting number of classes per outcome
    if bunch.predicted_outcome[0] in multiclass_outcomes:  # NOTE: can only handle multiclass, single outcome prediction
        num_classes = np.unique(cdata[bunch.predicted_outcome[0]]).size
    else:
        num_classes = 1
    multiclass = bool(num_classes > 1)

    # specifying outcome and its shape
    if 'Gender' in bunch.predicted_outcome:
        outcome = np.where(np.array([pd.get_dummies(cdata[x].values, dtype=bool).to_numpy()
                                     for x in bunch.predicted_outcome]).squeeze())[1].astype(float)
    else:
        outcome = np.array([cdata[x].values for x in bunch.predicted_outcome],
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
    partition_subs = dict()  # dict for subject numbers by set split
    partition_inds = dict()  # dict for subject indices by set split

    if bunch.cv_folds > 1:  # conditional allows fold-specific data loading in calc_new_metric.py
        fold_cdata_path = f'data/cfHCP900_FSL_GM/cfHCP900_FSL_GM_preprocessed_fold{bunch.fold}.nc'
        sbf_path = os.path.join('subject_splits', f'{bunch.cv_folds}_train_test_splits.pkl')  # subs by fold

        cdata = xr.load_dataset(fold_cdata_path)
        train_test_splits = pickle.load(open(sbf_path, "rb"))  # read in file
        train_folds = list(itertools.combinations(range(bunch.cv_folds), bunch.cv_folds - 1))
        test_folds = [list(set(range(bunch.cv_folds)) - set(train_folds))[0] for _, train_folds in
                      enumerate(train_folds)]
        train_subs = np.hstack([train_test_splits[i] for i in train_folds[bunch.fold]])
        test_subs = np.array(train_test_splits[test_folds[bunch.fold]]).squeeze()
        tf_str = [str(i) for i in train_folds[bunch.fold]]  # list of str denoting the folds used to transform data
        tf_str.sort()  # sorting for consistency
    else:
        train_subs = np.loadtxt(subnum_paths["train"])
        test_subs = np.loadtxt(subnum_paths["test"])

    partition_subs["train"], partition_inds["train"], _ = np.intersect1d(bunch.subnums, train_subs, return_indices=True)
    partition_subs["test"], partition_inds["test"], _ = np.intersect1d(bunch.subnums, test_subs, return_indices=True)
    partition_subs["val"], partition_inds["val"], _ = np.intersect1d(bunch.subnums, np.loadtxt(subnum_paths["val"]),
                                                                     return_indices=True)

    print(
        f'{partition_subs["test"].shape + partition_subs["train"].shape + partition_subs["val"].shape} subjects total included in test-train-validation sets '
        f'({len(partition_inds["train"]) + len(partition_inds["test"]) + len(partition_inds["val"])} total)...\n')

    ###############################################################
    # # Deconfounding X and Y for data classes
    ###############################################################
    if bunch.deconfound_flavor == 'X1Y1':
        raise NotImplementedError('X1Y1 not implementd yet. Please use another deconfounding method.')
        # TODO: implement X1Y1 deconfounding

    if bunch.deconfound_flavor == 'X1Y0':  # If we have data to deconfound...

        print(f'Checking if {len(bunch.chosen_Xdatavars)} data variable(s) were/was previously deconfounded...\n')

        for i, datavar in enumerate(bunch.chosen_Xdatavars):

            if bunch.cv_folds > 1:  # conditional allows fold-specific data loading in calc_new_metric.py
                dec_Xvar = f'dec{"".join(tf_str)}_{"_".join(bunch.confound_names)}_{datavar}'
            else:
                dec_Xvar = f'dec_{"_".join(bunch.confound_names)}_{datavar}'  # name for positive definite transformed mats
            # dec_Yvar = f'{"_".join(predicted_outcome)}_dec_{"_".join(confound_names)}_{datavar}'

            if dec_Xvar in list(cdata.data_vars):  # check if positive definite data already saved in xarray
                print(f"{dec_Xvar} is a saved data variable. Skipping over deconfounding of {datavar}.\n")
                continue

            cdata = cdata.assign({dec_Xvar: cdata[datavar]})

            if bunch.confound_names is not None:  # getting confounds
                confounds = [cdata[x].values for x in bunch.confound_names]

                for i, confound_name in enumerate(bunch.confound_names):  # one-hot encoding class confounds
                    if confound_name in multiclass_outcomes:
                        _, confounds[i] = np.unique(confounds[i], return_inverse=True)

                if bunch.scale_confounds:  # scaling confounds, per train set alone
                    confounds = [x / np.max(np.abs(x[partition_inds["train"]])) for x in confounds]

            print(f'Deconfounding {datavar} data using {bunch.confound_names} as confounds...')
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

            # saving deconfounded matrices in HCP_alltasks_268, ONLY if all were deconfounded
            if bunch.chosen_dir == ['HCP_alltasks_268'] and bunch.chosen_tasks == list(HCP268_tasks.keys())[:-1]:
                print(f'saving {dec_Xvar} matrices to xarray...\n')
                cdata.to_netcdf('data/cfHCP900_FSL_GM/cfHCP900_FSL_GM.nc')

        # updating chosen datavars
        if bunch.cv_folds > 1:
            chosen_Xdatavars = ['_'.join([f'dec{"".join(tf_str)}', '_'.join(bunch.confound_names),
                                          x]) for x in chosen_Xdatavars]
        else:
            chosen_Xdatavars = ['_'.join(['dec', '_'.join(bunch.confound_names), x]) for x in bunch.chosen_Xdatavars]

        # TODO: set logic here for X1Y1: Y = outcome

    if bunch.deconfound_flavor == 'X0Y0' or bunch.deconfound_flavor == 'X1Y0':
        Y = outcome

    # Setting up multiclass classification with one-hot encoding
    if multiclass:
        Y = multiclass_to_onehot(Y).astype(float)  # ensures Y is not of type object
        y_weights = Y.sum(axis=0) / len(Y)  # class weighting in dataset (inverse of class frequency)
        y_weights_dict = dict(zip(range(len(y_weights)), y_weights))

    if bunch.data_are_matrices:

        ###################################################################
        # # Projecting matrices into positive definite
        ###################################################################
        if bunch.transformations == 'positive definite' or bunch.transformations == 'tangent':
            for i, datavar in enumerate(chosen_Xdatavars):

                pd_var = f'pd_{datavar}'  # name for positive definite transformed mats

                if pd_var in list(cdata.data_vars):  # if positive definite data saved in xarray, do no transformation
                    print(f'Positive-definite {datavar} already saved. Skipping PD projection.\n')
                    continue

                else:
                    num_notPD, which = areNotPD(cdata[datavar].values)  # Test all matrices for positive definiteness
                    print(f'\nThere are {num_notPD} non-PD matrices in {datavar}...\n')

                    cdata = cdata.assign({pd_var: cdata[datavar]})

                    if int(num_notPD) == 0:
                        continue

                    print('Transforming non-PD matrices to nearest PD neighbor ...')
                    X_pd = PD_transform(cdata[pd_var].values[which])
                    cdata[pd_var][which] = xr.DataArray(X_pd,
                                                        coords=dict(zip(list(cdata[pd_var].dims),
                                                                        [cdata[pd_var]['subject'].values[which],
                                                                         cdata[pd_var].dim1.values,
                                                                         cdata[pd_var].dim2.values])),
                                                        dims=list(cdata[pd_var].dims))

                    del X_pd

                # saving positive definite matrices
                if bunch.chosen_dir == ['HCP_alltasks_268']:
                    print(f'Saving data variable {pd_var} to xarray...\n')
                    cdata.to_netcdf('data/cfHCP900_FSL_GM/cfHCP900_FSL_GM.nc')

        ###################################################################
        # # Projecting matrices into tangent space
        ###################################################################
        if bunch.transformations == 'tangent':
            for i, datavar in enumerate(chosen_Xdatavars):

                if bunch.cv_folds > 1:  # conditional allows fold-specific data loading in calc_new_metric.py
                    tan_var = f'tan{"".join(tf_str)}_{datavar}'
                else:
                    tan_var = f'tan_{datavar}'  # name for tangent transformed mats

                if tan_var in list(cdata.data_vars):  # if tangent data saved in xarray, do no projection
                    print(f'Tangent {datavar} already saved. Skipping tangent projection.\n')
                    continue

                print(f'Transforming all {datavar} matrices into tangent space ...')
                cdata = cdata.assign({tan_var: cdata[datavar]})
                # determine tangent transformation from trainset and project all sets' matrices
                X_tan = tangent_transform(refmats=cdata[datavar].loc[dict(subject=partition_subs["train"])],
                                          projectmats=cdata[datavar].values,
                                          ref=bunch.tan_mean)

                cdata[tan_var] = xr.DataArray(X_tan,
                                              coords=dict(zip(list(cdata[tan_var].dims),
                                                              [cdata[tan_var]['subject'].values,
                                                               cdata[tan_var].dim1.values,
                                                               cdata[tan_var].dim2.values])),
                                              dims=list(cdata[tan_var].dims))

                del X_tan

                # saving tangent matrices, for each transformed
                if bunch.chosen_dir == ['HCP_alltasks_268']:
                    print(f'Saving data variable {tan_var} to xarray...\n')
                    cdata.to_netcdf('data/cfHCP900_FSL_GM/cfHCP900_FSL_GM.nc')

    # updating list of chosen data for training
    if bunch.transformations == 'tangent':
        if bunch.cv_folds > 1:
            chosen_Xdatavars = ['_'.join([f'tan{"".join(tf_str)}', x]) for x in chosen_Xdatavars]
        else:
            chosen_Xdatavars = ['_'.join(['tan', x]) for x in chosen_Xdatavars]
    elif bunch.transformations == 'positive definite':
        chosen_Xdatavars = ['_'.join(['pd', x]) for x in chosen_Xdatavars]

    X = cdata
    del cdata

    out = dict(multi_outcome=multi_outcome, X=X, Y=Y, num_outcome=num_outcome, num_classes=num_classes,
               multiclass=multiclass, partition_subs=partition_subs, partition_inds=partition_inds,
               chosen_Xdatavars=chosen_Xdatavars)

    if multiclass:  # add class weights
        out['y_weights_dict'] = y_weights_dict
        out['y_weights'] = y_weights

    return out


if __name__ == '__main__':
    main()
