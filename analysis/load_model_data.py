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
    if 'Gender' in bunch.predicted_outcome:  # TODO: later, implement logic for (multiclass + continuous) outcomes
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

    partition_subs["train"], partition_inds["train"], _ = np.intersect1d(bunch.subnums,
                                                                         np.loadtxt(subnum_paths["train"]),
                                                                         return_indices=True)
    partition_subs["test"], partition_inds["test"], _ = np.intersect1d(bunch.subnums, np.loadtxt(subnum_paths["test"]),
                                                                       return_indices=True)
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

    if bunch.deconfound_flavor == 'X1Y1' or bunch.deconfound_flavor == 'X1Y0':  # If we have data to deconfound...
        for i, datavar in enumerate(bunch.chosen_Xdatavars):

            dec_Xvar = f'dec_{"_".join(bunch.confound_names)}_{datavar}'  # name for positive definite transformed mats
            # dec_Yvar = f'{"_".join(predicted_outcome)}_dec_{"_".join(confound_names)}_{datavar}'

            if dec_Xvar in list(cdata.data_vars):  # check if positive definite data already saved in xarray
                break

            cdata = cdata.assign({dec_Xvar: cdata[datavar]})

            if bunch.confound_names is not None:  # getting confounds
                confounds = [cdata[x].values for x in bunch.confound_names]

                if bunch.scale_confounds:  # scaling confounds per train set alone
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
        if bunch.chosen_dir == ['HCP_alltasks_268'] and bunch.chosen_tasks == list(HCP268_tasks.keys())[:-1]:
            cdata.to_netcdf('data/cfHCP900_FSL_GM/cfHCP900_FSL_GM.nc')

        # updating chosen datavars
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
            if bunch.chosen_dir == ['HCP_alltasks_268'] and bunch.chosen_tasks == list(HCP268_tasks.keys())[:-1]:
                cdata.to_netcdf('data/cfHCP900_FSL_GM/cfHCP900_FSL_GM.nc')

        ###################################################################
        # # Projecting matrices into tangent space
        ###################################################################
        if bunch.transformations == 'tangent':
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
                                          ref=bunch.tan_mean)

                cdata[tan_var] = xr.DataArray(X_tan,
                                              coords=dict(zip(list(cdata[tan_var].dims),
                                                              [cdata[tan_var]['subject'].values,
                                                               cdata[tan_var].dim1.values,
                                                               cdata[tan_var].dim2.values])),
                                              dims=list(cdata[tan_var].dims))

                del X_tan

            # saving tangent matrices, for each data iteration
            if bunch.chosen_dir == ['HCP_alltasks_268'] and bunch.chosen_tasks == list(HCP268_tasks.keys())[:-1]:
                cdata.to_netcdf('data/cfHCP900_FSL_GM/cfHCP900_FSL_GM.nc')  # Saving when necessary

    # updating list of chosen data for training
    if bunch.transformations == 'tangent':
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
