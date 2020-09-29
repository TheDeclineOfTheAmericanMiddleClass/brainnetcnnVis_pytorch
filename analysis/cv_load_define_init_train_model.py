import datetime
import gc
import itertools
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.data.dataset
import torch.utils.data.dataset
import xarray as xr
from scipy.stats import pearsonr
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import mean_absolute_error as mae
from torch.autograd import Variable

from utils.util_funcs import Bunch, create_cv_folds, deconfound_dataset, tangent_transform, areNotPD, PD_transform, \
    multiclass_to_onehot
from utils.var_names import subnum_paths, multiclass_outcomes


def main(args):
    bunch = Bunch(args)

    cdata = bunch.cdata
    chosen_Xdatavars = bunch.chosen_Xdatavars

    # # calculate number of classes and number of outcomes
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

    # creating filename to save training performance
    rundate = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M")
    model_preamble = f"BNCNN_{bunch.architecture}_{bunch.po_str}_{bunch.cXdv_str}" \
                     f"_{bunch.transformations}_{bunch.deconfound_flavor}{bunch.scl}_{bunch.early_str}_" + rundate

    nan_fill = np.repeat(np.nan, bunch.cv_folds)
    best_test_epochs = dict(zip([f'best_test_epoch_fold_{i}' for i in range(bunch.cv_folds)], nan_fill))
    estop_epochs = dict(zip([f'estop_epoch_fold_{i}' for i in range(bunch.cv_folds)], nan_fill))

    # setting up xarray to hold performance metrics
    sets = ['train', 'test', 'val']
    metrics = ['loss', 'accuracy', 'MAE', 'pearsonR', 'p_value']
    alloc_data = np.zeros((bunch.n_epochs, len(sets), len(metrics), num_outcome, bunch.cv_folds))
    alloc_data[:] = np.nan
    performance = xr.DataArray(alloc_data,
                               coords=[range(bunch.n_epochs), sets, metrics,
                                       bunch.predicted_outcome, range(bunch.cv_folds)],
                               dims=['epoch', 'set', 'metrics', 'outcome', 'cv_fold'])

    # calculate subject partitions for each fold
    train_subs = np.loadtxt(subnum_paths["train"])
    test_subs = np.loadtxt(subnum_paths["test"])

    # load dict of subs_by_fold (or create if necessary)
    sbf_path = os.path.join('subject_splits', f'{bunch.cv_folds}_train_test_splits.pkl')
    if os.path.isfile(sbf_path):
        train_test_splits = pickle.load(open(sbf_path, "rb"))  # read in file
    else:
        # creating folds without shuffling, to ensure same subjects in each split every run
        train_test_subs = np.hstack([np.intersect1d(bunch.subnums, train_subs),
                                     np.intersect1d(bunch.subnums, test_subs)])  # train & test subs
        subs_by_fold, _ = create_cv_folds(cdata.loc[dict(subject=train_test_subs)], n_folds=bunch.cv_folds,
                                          separate_families=False, shuffle=False)  # note: test fold size ~ 1 / n_folds
        train_test_splits = dict(zip(range(bunch.cv_folds), subs_by_fold))  # to dict
        pickle.dump(train_test_splits, open(sbf_path, "wb"))  # saving

    # for each of n_folds model trainings, pick an N choose k combination at random
    print('\npartitioning data into arrays to feed to model...')
    train_folds = list(itertools.combinations(range(bunch.cv_folds), bunch.cv_folds - 1))
    test_folds = [list(set(range(bunch.cv_folds)) - set(train_folds))[0] for _, train_folds in enumerate(train_folds)]

    # # start of cv_fold training loop
    for fold in range(bunch.cv_folds):

        gc.collect()  # cleaning up space before looping through fold

        # loading in cdata file for fold, if it exists
        fold_cdata_path = f'data/cfHCP900_FSL_GM/cfHCP900_FSL_GM_preprocessed_fold{fold}.nc'
        if os.path.isfile(fold_cdata_path):
            cdata = xr.load_dataset(fold_cdata_path)

        # defining train, test, validaton sets
        partition_subs = dict()  # dict for subject numbers by partition
        partition_inds = dict()  # dict for subject indices by partition

        if bunch.cv_folds > 1:  # set partition subs per the train_test_splits
            train_subs = np.hstack([train_test_splits[i] for i in train_folds[fold]])
            test_subs = np.array(train_test_splits[test_folds[fold]]).squeeze()

            tf_str = [str(i) for i in train_folds[fold]]  # list of str denoting the folds used to transform data
            tf_str.sort()  # sorting for consistency

        else:
            tf_str = []

        # if cv_folds == 1, using parvathy's origianl train, test subs; validation set always read in per this split
        partition_subs["train"], partition_inds["train"], _ = np.intersect1d(bunch.subnums, train_subs,
                                                                             return_indices=True)
        partition_subs["test"], partition_inds["test"], _ = np.intersect1d(bunch.subnums, test_subs,
                                                                           return_indices=True)
        partition_subs["val"], partition_inds["val"], _ = np.intersect1d(bunch.subnums, np.loadtxt(subnum_paths["val"]),
                                                                         return_indices=True)

        # print subject counts for each partition
        print(f'{partition_subs["test"].shape + partition_subs["train"].shape + partition_subs["val"].shape} '
              f'subjects total included in test-train-validation sets '
              f'({len(partition_inds["train"]) + len(partition_inds["test"]) + len(partition_inds["val"])} total)...\n')

        # # Deconfounding X and Y for data classes
        if bunch.deconfound_flavor == 'X1Y1':
            raise NotImplementedError('X1Y1 not implementd yet. Please use another deconfounding method.')
            # TODO: implement X1Y1 deconfounding

        if bunch.deconfound_flavor == 'X1Y0':  # If we have data to deconfound...

            print(f'Checking if {len(bunch.chosen_Xdatavars)} data variable(s) were/was previously deconfounded...\n')
            dec_count = 0

            # deconfounding all chosen data variables
            for i, datavar in enumerate(bunch.chosen_Xdatavars):

                gc.collect()  # cleaning up space before transformation

                # checking by the folds used to transform data
                dec_Xvar = f'dec{"".join(tf_str)}_{"_".join(bunch.confound_names)}_{datavar}'

                if dec_Xvar in list(cdata.data_vars):  # check if positive definite data already saved in xarray
                    print(f"{dec_Xvar} is a saved data variacble. Skipping over deconfounding of {datavar}...\n")
                    continue

                cdata = cdata.assign({dec_Xvar: cdata[datavar]})  # duplicate data variable

                if bunch.confound_names is not None:  # getting confounds
                    confounds = [cdata[x].values for x in bunch.confound_names]

                    for i, confound_name in enumerate(bunch.confound_names):  # one-hot encoding class confounds
                        if confound_name in multiclass_outcomes:
                            _, confounds[i] = np.unique(confounds[i], return_inverse=True)

                    if bunch.scale_confounds:  # scaling confounds, per train set alone
                        confounds = [x / np.max(np.abs(x[partition_inds["train"]])) for x in confounds]

                print(f'Deconfounding {datavar} data using {bunch.confound_names} as confounds...')
                dec_count += 1
                X_corr, Y_corr, nan_ind = deconfound_dataset(data=cdata[datavar].values, confounds=confounds,
                                                             set_ind=partition_inds["train"], outcome=outcome)

                print('...deconfounding complete.')
                cdata[dec_Xvar] = xr.DataArray(X_corr,
                                               coords=dict(zip(list(cdata[dec_Xvar].dims),
                                                               [cdata[dec_Xvar][x].values for x in
                                                                list(cdata[dec_Xvar].dims)])),
                                               dims=list(cdata[dec_Xvar].dims))
                del X_corr

            # saving any newly deconfounded matrices in HCP_alltasks_268, after all are transformed
            if bunch.chosen_dir == ['HCP_alltasks_268'] and bool(dec_count):
                # print(f'saving {dec_Xvar} matrices to xarray...\n')
                print(f'saving deconfounded matrix(es) to xarray...\n')
                cdata.to_netcdf(fold_cdata_path)

            # updating names of chosen datavars
            chosen_Xdatavars = ['_'.join([f'dec{"".join(tf_str)}', '_'.join(bunch.confound_names),
                                          x]) for x in bunch.chosen_Xdatavars]

        if bunch.deconfound_flavor == 'X0Y0' or bunch.deconfound_flavor == 'X1Y0':
            Y = outcome

        # Setting up multiclass classification with one-hot encoding
        if multiclass:
            Y = multiclass_to_onehot(Y).astype(float)  # ensures Y is not of type object
            y_weights = Y.sum(axis=0) / len(Y)  # class weighting in dataset (inverse of class frequency)
            y_weights_dict = dict(zip(range(len(y_weights)), y_weights))

        if bunch.data_are_matrices:
            if bunch.transformations == 'positive definite' or bunch.transformations == 'tangent':

                pd_count = 0  # count of pd-transformed matrices

                # project matrices into positive definite
                for i, datavar in enumerate(chosen_Xdatavars):

                    gc.collect()  # cleaning up space before transformation

                    pd_var = f'pd_{datavar}'  # name for positive definite transformed mats

                    if pd_var in list(
                            cdata.data_vars):  # if positive definite data saved in xarray, do no transformation
                        print(f'Positive-definite {datavar} already saved. Skipping PD projection.\n')
                        continue

                    else:
                        num_notPD, which = areNotPD(
                            cdata[datavar].values)  # Test all matrices for positive definiteness
                        print(f'\nThere are {num_notPD} non-PD matrices in {datavar}...\n')

                        cdata = cdata.assign({pd_var: cdata[datavar]})

                        if int(num_notPD) == 0:
                            continue

                        print('Transforming non-PD matrices to nearest PD neighbor ...')
                        pd_count += 1
                        X_pd = PD_transform(cdata[pd_var].values[which])
                        cdata[pd_var][which] = xr.DataArray(X_pd,
                                                            coords=dict(zip(list(cdata[pd_var].dims),
                                                                            [cdata[pd_var]['subject'].values[which],
                                                                             cdata[pd_var].dim1.values,
                                                                             cdata[pd_var].dim2.values])),
                                                            dims=list(cdata[pd_var].dims))

                        del X_pd

                # saving positive definite matrices, after all matrices transformed
                if bunch.chosen_dir == ['HCP_alltasks_268'] and bool(pd_count):
                    # print(f'Saving data variable {pd_var} to xarray...\n')
                    print(f'Saving positive definite data variable(s) to xarray...\n')
                    cdata.to_netcdf(fold_cdata_path)

            # project matrices into tangent space
            if bunch.transformations == 'tangent':

                tan_count = 0  # count of tangent transformed matrices

                for i, datavar in enumerate(chosen_Xdatavars):

                    gc.collect()  # cleaning up space before transformation

                    tan_var = f'tan{"".join(tf_str)}_{datavar}'  # name for tangent transformed mats

                    if tan_var in list(cdata.data_vars):  # if tangent data saved in xarray, do no projection
                        print(f'Tangent {datavar} already saved. Skipping tangent projection.\n')
                        continue

                    print(f'\nTransforming all {datavar} matrices into tangent space ...')
                    tan_count += 1
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

                # saving tangent matrices, after all are transformed
                if bunch.chosen_dir == ['HCP_alltasks_268'] and bool(tan_count):
                    # print(f'Saving data variable {tan_var} to xarray...\n')
                    print(f'Saving tangent data variable(s) to xarray...\n')
                    cdata.to_netcdf(fold_cdata_path)

        # updating list of chosen data for training
        if bunch.transformations == 'tangent':
            chosen_Xdatavars = ['_'.join([f'tan{"".join(tf_str)}', x]) for x in chosen_Xdatavars]
        elif bunch.transformations == 'positive definite':
            chosen_Xdatavars = ['_'.join(['pd', x]) for x in chosen_Xdatavars]

        try:  # if multi outcome
            Y = xr.DataArray(Y, coords=dict(subject=bunch.cdata.subject.values, outcome=bunch.predicted_outcome),
                             dims=['subject', 'outcome'])
        except ValueError:  # if single outcome
            Y = xr.DataArray(Y, coords=dict(subject=bunch.cdata.subject.values), dims='subject')

        X = cdata
        out = dict(multi_outcome=multi_outcome, X=X, Y=Y, num_outcome=num_outcome, num_classes=num_classes,
                   multiclass=multiclass, partition_subs=partition_subs, partition_inds=partition_inds,
                   chosen_Xdatavars=chosen_Xdatavars)

        if multiclass:  # add class weights
            out['y_weights_dict'] = y_weights_dict
            out['y_weights'] = y_weights

        # updating bunch with new arguments, cleaning up variable space
        args.update(out)
        bunch = Bunch(args)

        class HCPDataset(torch.utils.data.Dataset):

            def __init__(self, mode="train", transform=False):
                """
                Args:
                    directory (string): Path to the data.
                    mode (str): train = 75% Train, validation=15% Train, train+validation=100% train else test.
                    transform (callable, optional): Optional transform to be applied
                        on a sample.
                """
                self.mode = mode
                self.transform = transform

                # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

                if self.mode == "train":
                    x = xr.merge(
                        [bunch.X[var].sel(dict(subject=bunch.partition_subs['train'])) for var in
                         bunch.chosen_Xdatavars]).to_array().values
                    # y = bunch.Y[bunch.partition_inds["train"]]
                    y = bunch.Y.loc[dict(subject=bunch.partition_subs['train'])].values

                elif self.mode == "test":
                    x = xr.merge(
                        [bunch.X[var].sel(dict(subject=bunch.partition_subs['test'])) for var in
                         bunch.chosen_Xdatavars]).to_array().values
                    # y = bunch.Y[bunch.partition_inds["test"]]
                    y = bunch.Y.loc[dict(subject=bunch.partition_subs['test'])].values

                elif mode == "valid":
                    x = xr.merge(
                        [bunch.X[var].sel(dict(subject=bunch.partition_subs["val"])) for var in
                         bunch.chosen_Xdatavars]).to_array().values
                    # y = bunch.Y[bunch.partition_inds["val"]]
                    y = bunch.Y.loc[dict(subject=bunch.partition_subs["val"])].values

                x = x.reshape(-1, bunch.num_input, x.shape[-1], x.shape[-1]).squeeze()

                if bunch.multi_input:
                    self.X = torch.FloatTensor(x)
                if not bunch.multi_input:
                    self.X = torch.FloatTensor(np.expand_dims(x, 1))  # removed .astype(np.float64)
                self.Y = torch.FloatTensor(y)

                print(self.mode, self.X.shape, (self.Y.shape))

            def __len__(self):
                return self.X.shape[0]

            def __getitem__(self, idx):
                sample = [self.X[idx], self.Y[idx]]
                if self.transform:
                    sample[0] = self.transform(sample[0])
                return sample

        # Definite E2E
        class E2EBlock(torch.nn.Module):
            '''E2Eblock.'''

            def __init__(self, in_planes, planes, example, bias=False):
                super(E2EBlock, self).__init__()
                self.d = example.size(3)
                self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
                self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

            def forward(self, x):
                a = self.cnn1(x)
                b = self.cnn2(x)
                return torch.cat([a] * self.d, 3) + torch.cat([b] * self.d, 2)

        # BrainNetCNN edge to edge layer
        class Edge2Edge(nn.Module):
            def __init__(self, channel, dim, filters):
                super(Edge2Edge, self).__init__()
                self.channel = channel
                self.dim = dim
                self.filters = filters
                self.row_conv = nn.Conv2d(channel, filters, (1, dim))
                self.col_conv = nn.Conv2d(channel, filters, (dim, 1))

            # implemented by two conv2d with line filter
            def forward(self, x):
                size = x.size()
                row = self.row_conv(x)
                col = self.col_conv(x)
                row_ex = row.expand(size[0], self.filters, self.dim, self.dim)
                col_ex = col.expand(size[0], self.filters, self.dim, self.dim)
                return row_ex + col_ex

        # BrainNetCNN edge to node layer
        class Edge2Node(nn.Module):
            def __init__(self, channel, dim, filters):
                super(Edge2Node, self).__init__()
                self.channel = channel
                self.dim = dim
                self.filters = filters
                self.row_conv = nn.Conv2d(channel, filters, (1, dim))
                self.col_conv = nn.Conv2d(channel, filters, (dim, 1))

            def forward(self, x):
                row = self.row_conv(x)
                col = self.col_conv(x)
                return row + col.permute(0, 1, 3, 2)

        # BrainNetCNN node to graph layer
        class Node2Graph(nn.Module):
            def __init__(self, channel, dim, filters):
                super(Node2Graph, self).__init__()
                self.channel = channel
                self.dim = dim
                self.filters = filters
                self.conv = nn.Conv2d(channel, filters, (dim, 1))

            def forward(self, x):
                return self.conv(x)

        # # Parvathy's self-written script
        # class ParvathySex_BNCNN_original(nn.Module):
        #     def __init__(self, e2e, e2n, n2g, f_size, dropout):
        #         super(ParvathySex_BNCNN_original, self).__init__()
        #         print('\nInitializing BNCNN: Parvathy_Sex Original Architecture')
        #         self.n2g_filter = n2g
        #         self.e2e = Edge2Edge(1, f_size, e2e)
        #         self.e2n = Edge2Node(e2e, f_size, e2n)
        #         self.dropout = nn.Dropout(p=dropout)
        #         self.n2g = Node2Graph(e2n, f_size, n2g)
        #
        #         self.fc = nn.Linear(n2g, num_classes)
        #         self.BatchNorm = nn.BatchNorm1d(n2g)
        #
        #         for m in self.modules():
        #             if isinstance(m, nn.Conv2d):
        #                 init.xavier_uniform_(m.weight)
        #             elif isinstance(m, nn.Conv1d):
        #                 init.xavier_uniform_(m.weight)
        #             elif isinstance(m, nn.BatchNorm1d):
        #                 m.weight.data.fill_(1)
        #                 m.bias.data.zero_()
        #             elif isinstance(m, nn.Linear):
        #                 init.xavier_uniform_(m.weight)
        #
        #     def forward(self, x):
        #         x = self.e2e(x)
        #         x = self.dropout(x)
        #         x = self.e2n(x)
        #         x = self.dropout(x)
        #         x = self.n2g(x)
        #         x = self.dropout(x)
        #         x = x.view(-1, self.n2g_filter)
        #         x = self.fc(self.BatchNorm(x))
        #
        #         return x
        #

        # Adu's self-written script, using parameters of Parvathy's netowkr
        class ParvathySex_BNCNN_v2byAdu(torch.nn.Module):
            def __init__(self, example):  # removed num_classes=10
                super(ParvathySex_BNCNN_v2byAdu, self).__init__()
                print('\nInitializing BNCNN: Parvathy_Sex_v2 Architecture')
                self.in_planes = example.size(1)
                self.d = example.size(3)

                self.e2econv1 = E2EBlock(example.size(1), 16, example, bias=True)
                self.E2N = torch.nn.Conv2d(16, 128, (1, self.d))
                self.N2G = torch.nn.Conv2d(128, 26, (self.d, 1))
                self.dense1 = torch.nn.Linear(26, bunch.num_classes)

                for m in self.modules():  # initializing weights
                    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                        init.xavier_uniform_(m.weight)
                    elif isinstance(m, nn.BatchNorm1d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()

            def forward(self, x):
                out = F.dropout(self.e2econv1(x), p=0.6)
                out = F.dropout(self.E2N(out), p=0.6)
                out = F.dropout(self.N2G(out), p=0.6)
                out = out.view(out.size(0), -1)
                out = torch.sigmoid(self.dense1(out))  # adding sigmoid for binary sex

                return out

        # Yeo-version BNCNN for Multiclass Sex Classification (output = num_classes)
        class YeoSex_BNCNN(torch.nn.Module):
            def __init__(self, example):  # removed num_classes=10
                super(YeoSex_BNCNN, self).__init__()
                print('\nInitializing BNCNN: Yeo_Sex Architecture...')
                self.in_planes = example.size(1)
                self.d = example.size(3)

                self.e2econv1 = E2EBlock(example.size(1), 38, example, bias=True)
                self.E2N = torch.nn.Conv2d(38, 58, (1, self.d))
                self.N2G = torch.nn.Conv2d(58, 7, (self.d, 1))
                self.dense1 = torch.nn.Linear(7, bunch.num_classes)

                for m in self.modules():  # initializing weights
                    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                        init.xavier_uniform_(m.weight)
                    elif isinstance(m, nn.BatchNorm1d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()
                print('BNCNN instance initialized.\n')

            def forward(self, x):
                out = F.dropout(self.e2econv1(x), p=0.463)
                out = F.dropout(self.E2N(out), p=0.463)
                out = F.dropout(self.N2G(out), p=0.463)
                out = out.view(out.size(0), -1)
                out = torch.sigmoid(self.dense1(out))

                return out

        # Usama Pervaiz's BrainNetCNN Network
        class Usama_BNCNN(torch.nn.Module):
            def __init__(self, example):  # removed num_classes=10
                super(Usama_BNCNN, self).__init__()
                print('\nInitializing BNCNN: Usama Architecture')
                self.in_planes = example.size(1)
                self.d = example.size(3)

                self.e2econv1 = E2EBlock(example.size(1), 32, example, bias=True)
                self.e2econv2 = E2EBlock(32, 64, example, bias=True)
                self.E2N = torch.nn.Conv2d(64, 1, (1, self.d))
                self.N2G = torch.nn.Conv2d(1, 256, (self.d, 1))
                self.dense1 = torch.nn.Linear(256, 128)  # init
                self.dense2 = torch.nn.Linear(128, 30)
                if bunch.multiclass:
                    self.dense3 = torch.nn.Linear(30, bunch.num_classes)
                else:
                    self.dense3 = torch.nn.Linear(30, bunch.num_outcome)

                for m in self.modules():
                    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                        init.xavier_uniform_(m.weight)
                    elif isinstance(m, nn.BatchNorm1d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()

            def forward(self, x):
                out = F.dropout(F.leaky_relu(self.e2econv1(x), negative_slope=0.33), p=.5)
                out = F.dropout(F.leaky_relu(self.e2econv2(out), negative_slope=0.33), p=.5)
                out = F.leaky_relu(self.E2N(out), negative_slope=0.33)
                out = F.dropout(F.leaky_relu(self.N2G(out), negative_slope=0.33), p=0.5)
                out = out.view(out.size(0), -1)
                out = F.dropout(F.relu(self.dense1(out)), p=0.5)
                out = F.dropout(F.relu(self.dense2(out)), p=0.5)

                if bunch.multiclass:
                    out = torch.sigmoid(self.dense3(out))
                else:
                    out = F.relu(self.dense3(out))

                return out

        # Kawahara's BrainNetCNN Network
        class Kawahara_BNCNN(torch.nn.Module):
            def __init__(self, example):  # removed num_classes=10
                super(Kawahara_BNCNN, self).__init__()
                print('\nInitializing BNCNN: Kawahara Architecture')
                self.in_planes = example.size(1)
                self.d = example.size(3)

                self.e2econv1 = E2EBlock(1, 32, example, bias=True)
                self.e2econv2 = E2EBlock(32, 32, example, bias=True)
                self.E2N = torch.nn.Conv2d(32, 64, (1, self.d))
                self.N2G = torch.nn.Conv2d(64, 256, (self.d, 1))
                self.dense1 = torch.nn.Linear(256, 128)
                self.dense2 = torch.nn.Linear(128, 30)
                self.batchnorm = torch.nn.BatchNorm1d(30)
                self.dense3 = torch.nn.Linear(30, bunch.num_outcome)

                for m in self.modules():
                    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                        init.xavier_uniform_(m.weight)
                    elif isinstance(m, nn.BatchNorm1d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()

            # # forward from paper figure 1.
            # def forward(self, x):
            #     out = F.dropout(F.leaky_relu(self.e2econv1(x), negative_slope=0.33), p=.5)
            #     out = F.dropout(F.leaky_relu(self.e2econv2(out), negative_slope=0.33), p=.5)
            #     out = F.leaky_relu(self.E2N(out), negative_slope=0.33)
            #     out = F.dropout(F.leaky_relu(self.N2G(out), negative_slope=0.33), p=0.5)
            #     out = out.view(out.size(0), -1)
            #     out = F.dropout(F.relu(self.dense1(out)), p=0.5)
            #     out = F.dropout(F.relu(self.dense2(out)), p=0.5)
            #     out = F.relu(self.dense3(out))

            # forward from section 2.3 description
            def forward(self, x):
                out = F.leaky_relu(self.e2econv1(x), negative_slope=0.33)
                out = F.leaky_relu(self.e2econv2(out), negative_slope=0.33)
                out = F.leaky_relu(self.E2N(out), negative_slope=0.33)
                out = F.dropout(F.leaky_relu(self.N2G(out), negative_slope=0.33), p=0.5)
                out = out.view(out.size(0), -1)
                out = F.relu(self.dense1(out))
                out = F.dropout(F.relu(self.dense2(out)), p=0.5)
                out = F.relu(self.dense3(out))

                return out

        class FNN(nn.Module):

            def __init__(self, input_size, n_l1, n_l2, output_size, drop_out):
                super(FNN, self).__init__()

                self.fc1 = nn.Sequential(
                    nn.Dropout(drop_out),
                    nn.Linear(input_size, n_l1),
                    nn.Sigmoid(),
                    nn.BatchNorm1d(n_l1),

                )

                self.fc2 = nn.Sequential(
                    nn.Dropout(drop_out),
                    nn.Linear(n_l1, n_l2),

                    nn.Sigmoid(),
                    nn.BatchNorm1d(n_l2),

                )

                self.fc3 = nn.Sequential(
                    nn.Dropout(drop_out),
                    nn.Linear(n_l2, 1),

                )

                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        init.xavier_uniform_(m.weight)
                    elif isinstance(m, nn.BatchNorm1d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()

            def forward(self, x):

                x = self.fc1(x)
                x = self.fc2(x)
                x = self.fc3(x)
                return x

        class FC90Net_YeoSex(torch.nn.Module):
            def __init__(self, example):
                super(FC90Net_YeoSex, self).__init__()
                self.dense1 = torch.nn.Linear(bunch.num_input, 3, example)
                self.dense2 = torch.nn.Linear(3, 2)

            def forward(self, x):
                out = F.dropout(F.leaky_relu(self.dense1(x), negative_slope=.33), p=.00275)
                out = out.view(out.size(0), -1)
                out = F.leaky_relu(self.dense2(out), negative_slope=.33)

                return out

        # Yeo-version BNCNN for 58 HCP behavior prediction
        class Yeo58behaviors_BNCNN(torch.nn.Module):

            # https://www.sciencedirect.com/science/article/pii/S1053811919308675?via%3Dihub#appsec1

            def __init__(self, example):
                super(Yeo58behaviors_BNCNN, self).__init__()
                print('\nInitializing BNCNN: Yeo_58_behaviors Architecture...')
                self.in_planes = example.size(1)
                self.d = example.size(3)

                self.e2econv1 = E2EBlock(example.size(1), 18, example, bias=True)
                self.E2N = torch.nn.Conv2d(18, 19, (1, self.d))
                self.N2G = torch.nn.Conv2d(19, 84, (self.d, 1))

                if bunch.multiclass:
                    self.dense1 = torch.nn.Linear(84, bunch.num_classes)
                else:
                    self.dense1 = torch.nn.Linear(84, bunch.num_outcome)

                for m in self.modules():  # initializing weights
                    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                        init.xavier_uniform_(m.weight)
                    elif isinstance(m, nn.BatchNorm1d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()

                print('BNCNN instance initialized.\n')

            def forward(self, x):
                out = F.dropout(self.e2econv1(x), p=0.463)
                out = F.dropout(self.E2N(out), p=0.463)
                out = F.dropout(self.N2G(out), p=0.463)
                out = out.view(out.size(0), -1)
                out = self.dense1(out)

                return out

        # limiting CPU usage
        torch.set_num_threads(1)

        # Defining train, test, validation sets
        trainset = HCPDataset(mode="train")
        testset = HCPDataset(mode="test")
        valset = HCPDataset(mode="valid")

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, pin_memory=False, num_workers=1)
        testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, pin_memory=False, num_workers=1)
        valloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, pin_memory=False, num_workers=1)

        # Creating the model
        if bunch.predicted_outcome == ['Gender'] and bunch.architecture == 'yeo_sex':
            net = YeoSex_BNCNN(trainset.X)
        elif bunch.predicted_outcome == ['Gender'] and bunch.architecture == 'parvathy_v2':
            net = ParvathySex_BNCNN_v2byAdu(trainset.X)
        # elif predicted_outcome == 'Gender' and architecture == 'parvathy_orig':
        #     net = ParvathySex_BNCNN_original(e2e=16, e2n=128, n2g=26, f_size=trainset.X.shape[3], dropout=.6)
        elif bunch.architecture == 'kawahara':
            net = Kawahara_BNCNN(trainset.X)
        elif bunch.architecture == 'usama':
            net = Usama_BNCNN(trainset.X)
        elif bunch.architecture == 'FC90Net':
            net = FC90Net_YeoSex(trainset.X)
        elif bunch.architecture == 'yeo_58':
            net = Yeo58behaviors_BNCNN(trainset.X)
        else:
            print(
                f'"{bunch.architecture}" architecture not available for outcome(s) {", ".join(bunch.predicted_outcome)}. Using default \'usama\' architecture...\n')
            net = Usama_BNCNN(trainset.X)

        # Putting the model on the GPU
        if bunch.use_cuda:
            net = net.to(bunch.device)
            cudnn.benchmark = True

        # ensure model parameters are on GPU
        assert next(net.parameters()).is_cuda, 'Parameters are not on the GPU !'

        # Following function are only applied to linear layers
        def init_weights_he(m):
            """ Weights initialization for the dense layers using He Uniform initialization
             He et al., http://arxiv.org/abs/1502.01852
            https://keras.io/initializers/#he_uniform
        """
            print(m)
            if type(m) == torch.nn.Linear:
                fan_in = net.dense1.in_features
                print(f'In features for dense 1: {fan_in}')
                he_lim = np.sqrt(6 / fan_in)  # Note: fixed error in he limit calculation (Feb 10, 2020)
                print(f'he limit {he_lim}')
                m.weight.data.uniform_(-he_lim, he_lim)
                print(f'\nWeight initializations: {m.weight}')

        # net.apply(init_weights_he)

        if bunch.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=bunch.lr, momentum=bunch.momentum, nesterov=True,
                                        weight_decay=bunch.wd)
        elif bunch.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=bunch.lr, weight_decay=bunch.wd)
        else:
            raise KeyError(f'{bunch.optimizer} is not a valid optimizer. Please try again.')

        # # defining loss functions
        if bunch.multiclass:
            if bunch.num_classes == 2:
                criterion = nn.BCELoss(weight=torch.Tensor(bunch.y_weights).cuda())  # balanced Binary Cross Entropy
            elif bunch.num_classes > 2:
                criterion = nn.CrossEntropyLoss(weight=torch.Tensor(bunch.y_weights).cuda())
        else:
            criterion = torch.nn.MSELoss()

        def train():  # training in mini batches # TODO: pass cv_fold param
            net.train()
            running_loss = 0.0

            preds = []
            ytrue = []

            for batch_idx, (inputs, targets) in enumerate(trainloader):

                if bunch.use_cuda:
                    if not bunch.multi_outcome and not bunch.multiclass:
                        # print('unsqueezing target for vstack...')
                        inputs, targets = inputs.to(bunch.device), targets.to(bunch.device).unsqueeze(
                            1)  # unsqueezing for vstack
                    else:
                        # print('target left alone...')
                        inputs, targets = inputs.to(bunch.device), targets.to(bunch.device)

                optimizer.zero_grad()
                inputs, targets = Variable(inputs), Variable(targets)

                outputs = net(inputs)
                targets = targets.view(outputs.size())

                if bunch.multiclass and bunch.num_classes > 2:  # targets is encoded as one-hot by CrossEntropyLoss
                    loss = criterion(input=outputs, target=torch.argmax(targets.data, 1))
                else:
                    loss = criterion(input=outputs, target=targets)

                loss.backward()

                # print('\ngradient after backward: ')
                # for name, param in net.named_parameters(): # CONFIRMED! gradient issue
                #     print(name, param.grad.abs().sum())

                # TODO: see if max_norm size appropriate
                # prevents a vanishing / exploding gradient problem
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=bunch.max_norm)

                for p in net.parameters():
                    p.data.add_(-bunch.lr, p.grad.data)

                optimizer.step()

                running_loss += loss.data.mean(0)  # only predicting 1 feature

                # TODO: see if accumulated gradient speeds up training
                # # 16 accumulated gradient steps
                # scaled_loss = 0
                # for accumulated_step_i in range(16):
                #     outputs = net(inputs)
                #     targets = targets.view(outputs.size())
                #     loss = criterion(input=outputs, target=targets)
                #     loss.backward()
                #     scaled_loss += loss.data.mean(0)
                #
                # # update weights after 8 steps. effective batch = 8*16
                # optimizer.step()
                #
                # # loss is now scaled up by the number of accumulated batches
                # actual_loss = scaled_loss / 16
                # running_loss += actual_loss

                preds.append(outputs.data.cpu().numpy())
                ytrue.append(targets.data.cpu().numpy())

                if batch_idx % 10 == 9:  # print every 10 mini-batches
                    print('Training loss: %.6f' % (running_loss / 10))
                    running_loss = 0.0
                _, predicted = torch.max(outputs.data, 1, keepdim=True)

            # return running_loss / batch_idx

            if not bunch.multi_outcome and not bunch.multiclass:
                # print('y_true left well enough alone...')
                return np.vstack(preds), np.vstack(ytrue), running_loss / batch_idx
            else:
                # print('squeezing y_true...')
                return np.vstack(preds), np.vstack(ytrue).squeeze(), running_loss / batch_idx

        def test():
            global loss
            net.eval()
            test_loss = 0
            running_loss = 0.0

            preds = []
            ytrue = []

            for batch_idx, (inputs, targets) in enumerate(testloader):
                if bunch.use_cuda:
                    if not bunch.multi_outcome and not bunch.multiclass:
                        # print('unsqueezing target for vstack...')
                        inputs, targets = inputs.to(bunch.device), targets.to(bunch.device).unsqueeze(
                            1)  # unsqueezing for vstack
                    else:
                        # print('target left alone...')
                        inputs, targets = inputs.to(bunch.device), targets.to(bunch.device)

                with torch.no_grad():
                    inputs, targets = Variable(inputs), Variable(targets)

                    outputs = net(inputs)
                    targets = targets.view(outputs.size())

                    if bunch.multiclass and bunch.num_classes > 2:  # targets is encoded as one-hot by CrossEntropyLoss
                        loss = criterion(input=outputs, target=torch.argmax(targets.data, 1))
                    else:
                        loss = criterion(input=outputs, target=targets)

                    test_loss += loss.data.mean(0)  # only predicting 1 feature

                    preds.append(outputs.data.cpu().numpy())
                    ytrue.append(targets.data.cpu().numpy())

                running_loss += loss.data.mean(0)  # only predicting 1 feature

                # print statistics
                if batch_idx == len(testloader) - 1:  # print for final batch
                    print('\nTest loss: %.6f' % (running_loss / len(testloader) - 1))
                    running_loss = 0.0

            if not bunch.multi_outcome and not bunch.multiclass:
                # print('y_true left well enough alone...')
                return np.vstack(preds), np.vstack(ytrue), running_loss / batch_idx
            else:
                # print('squeezing y_true...')
                return np.vstack(preds), np.vstack(ytrue).squeeze(), running_loss / batch_idx

        def val():
            net.eval()
            test_loss = 0
            running_loss = 0.0

            preds = []
            ytrue = []

            for batch_idx, (inputs, targets) in enumerate(valloader):
                if bunch.use_cuda:
                    if not bunch.multi_outcome and not bunch.multiclass:
                        # print('unsqueezing target for vstack...')
                        inputs, targets = inputs.to(bunch.device), targets.to(bunch.device).unsqueeze(
                            1)  # unsqueezing for vstack
                    else:
                        # print('target left alone...')
                        inputs, targets = inputs.to(bunch.device), targets.to(bunch.device)

                with torch.no_grad():
                    inputs, targets = Variable(inputs), Variable(targets)

                    outputs = net(inputs)
                    targets = targets.view(outputs.size())

                    if bunch.multiclass and bunch.num_classes > 2:  # targets is encoded as one-hot by CrossEntropyLoss
                        loss = criterion(input=outputs, target=torch.argmax(targets.data, 1))
                    else:
                        loss = criterion(input=outputs, target=targets)

                    test_loss += loss.data.mean(0)  # only predicting 1 feature

                    preds.append(outputs.data.cpu().numpy())
                    ytrue.append(targets.data.cpu().numpy())

                running_loss += loss.data.mean(0)  # only predicting 1 feature

                # print statistics
                if batch_idx == len(valloader) - 1:  # print for final batch
                    print('Val loss: %.6f' % (running_loss / len(valloader) - 1))
                    running_loss = 0.0

            if not bunch.multi_outcome and not bunch.multiclass:
                # print('y_true left well enough alone...')
                return np.vstack(preds), np.vstack(ytrue), running_loss / batch_idx
            else:
                # print('squeezing y_true...')
                return np.vstack(preds), np.vstack(ytrue).squeeze(), running_loss / batch_idx

        def init_weights_XU(m):
            """Init weights per xavier uniform method"""
            print(m)
            if type(m) == torch.nn.Linear:
                fan_in = net.dense1.in_features
                fan_out = net.dense1.out_features
                print(f'In/out features for dense 1: {fan_in}/{fan_out}')
                he_lim = np.sqrt(6 / fan_in + fan_out)  # Note: fixed error in he limit calculation (Feb 10, 2020)
                print(f'he limit {he_lim}')
                m.weight.data.uniform_(-he_lim, he_lim)
                print(f'\nWeight initializations: {m.weight}')

        gc.collect()  # cleaning up space before training

        print(f'\nFold {fold}'
              f'\nUsing data: ', bunch.chosen_Xdatavars,
              f'\nPredicting: {", ".join(bunch.predicted_outcome)}')

        # initializing network and prediction from starting weights
        print("\nInit Network")

        testp, testy, loss_test = test()
        valp, valy, loss_val = val()

        # printing performance before any training
        if bunch.multi_outcome:  # calculate predictive performance of multiple variables

            # calculate performance metrics
            test_mae_all = np.array([mae(testy[:, i], testp[:, i]) for i in range(bunch.num_outcome)])
            test_pears_all = np.array([list(pearsonr(testy[:, i], testp[:, i])) for i in range(bunch.num_outcome)])

            val_mae_all = np.array([mae(valy[:, i], valp[:, i]) for i in range(bunch.num_outcome)])
            val_pears_all = np.array([list(pearsonr(valy[:, i], valp[:, i])) for i in range(bunch.num_outcome)])

            # print metrics
            for i in range(bunch.num_outcome):
                print(f"\n{bunch.predicted_outcome[i]}"
                      f"\nTest MAE : {100 * test_mae_all[i]:.2}, pearson R: {test_pears_all[i, 0]:.2} (p = {test_pears_all[i, 1]:.2})"
                      f"\nVal MAE : {100 * val_mae_all[i]:.2}, pearson R: {val_pears_all[i, 0]:.2} (p = {val_pears_all[i, 1]:.2})")

        elif bunch.multiclass:  # calculate classification performance

            # calculate performance metrics
            test_acc = balanced_accuracy_score(np.argmax(testy, 1), np.argmax(testp, 1))
            val_acc = balanced_accuracy_score(np.argmax(valy, 1), np.argmax(valp, 1))

            # print metrics
            print(f"Test accuracy : {test_acc:.3}")
            print(f"Val accuracy : {val_acc:.3}")

        elif not bunch.multiclass and not bunch.multi_outcome:  # calculate predictive performance of 1 variable
            test_mae = mae(testp[:, 0], testy[:, 0])
            test_pears = pearsonr(testp[:, 0], testy[:, 0])

            val_mae = mae(valp[:, 0], valy[:, 0])
            val_pears = pearsonr(valp[:, 0], valy[:, 0])

            print(f"Test Set : MAE : {test_mae:.2}, pearson R : {test_pears[0]:.2} (p = {test_pears[1]:.4})")
            print(f"Val Set : MAE : {val_mae:.2}, pearson R : {val_pears[0]:.2} (p = {val_pears[1]:.4})")

        # # train model
        for epoch in range(bunch.n_epochs):

            # prediction, true y, and loss for all sets
            trainp, trainy, loss_train = train()
            testp, testy, loss_test = test()
            valp, valy, loss_val = val()

            performance.loc[dict(epoch=epoch, set="test", metrics='loss', cv_fold=fold)] = [loss_test]
            performance.loc[dict(epoch=epoch, set="train", metrics='loss', cv_fold=fold)] = [loss_train]
            performance.loc[dict(epoch=epoch, set="val", metrics='loss', cv_fold=fold)] = [loss_val]

            print("\nEpoch %d" % epoch)

            # calculate performance from all sets, print for validation and test
            if bunch.multi_outcome:

                # calculate performance metrics
                test_mae_all, train_mae_all, val_mae_all = \
                    np.array([mae(testy[:, i], testp[:, i]) for i in range(bunch.num_outcome)]), \
                    np.array([mae(trainy[:, i], trainp[:, i]) for i in range(bunch.num_outcome)]), \
                    np.array([mae(valy[:, i], valp[:, i]) for i in range(bunch.num_outcome)])

                test_pears_all, train_pears_all, val_pears_all = \
                    np.array([list(pearsonr(testy[:, i], testp[:, i])) for i in range(bunch.num_outcome)]), \
                    np.array([list(pearsonr(trainy[:, i], trainp[:, i])) for i in range(bunch.num_outcome)]), \
                    np.array([list(pearsonr(valy[:, i], valp[:, i])) for i in range(bunch.num_outcome)])

                # print metrics
                for i in range(bunch.num_outcome):
                    print(f"{bunch.predicted_outcome[i]}"
                          f"\nTrain MAE : {train_mae_all[i]:.2}, pearson R: {train_pears_all[i, 0]:.2} (p = {train_pears_all[i, 1]:.2})"
                          f"\nTest MAE : {test_mae_all[i]:.2}, pearson R: {test_pears_all[i, 0]:.2} (p = {test_pears_all[i, 1]:.2})"
                          f"\nVal MAE : {val_mae_all[i]:.2}, pearson R: {val_pears_all[i, 0]:.2} (p = {val_pears_all[i, 1]:.2})\n")

                # save metrics to xarray
                performance.loc[dict(epoch=epoch, set="test", metrics=['MAE', 'pearsonR', 'p_value'], cv_fold=fold)] = \
                    [test_mae_all, test_pears_all[:, 0], test_pears_all[:, 1]]
                performance.loc[dict(epoch=epoch, set="train", metrics=['MAE', 'pearsonR', 'p_value'], cv_fold=fold)] = \
                    [train_mae_all, train_pears_all[:, 0], train_pears_all[:, 1]]
                performance.loc[dict(epoch=epoch, set="val", metrics=['MAE', 'pearsonR', 'p_value'], cv_fold=fold)] = \
                    [val_mae_all, val_pears_all[:, 0], val_pears_all[:, 1]]

            elif bunch.multiclass:

                # calculate performance metrics
                testp, testy, trainp, trainy, valp, valy = np.argmax(testp, 1), np.argmax(testy, 1), \
                                                           np.argmax(trainp, 1), np.argmax(trainy, 1), \
                                                           np.argmax(valp, 1), np.argmax(valy, 1)

                test_acc, train_acc, val_acc = balanced_accuracy_score(testy, testp), \
                                               balanced_accuracy_score(trainy, trainp), \
                                               balanced_accuracy_score(valy, valp)

                # print metrics
                print(f"{bunch.predicted_outcome},"
                      f"\nTrain accuracy : {train_acc:.3}"
                      f"\nTest accuracy: {test_acc:.3}"
                      f"\nVal accuracy: {val_acc:.3}\n")

                # save metrics to xarray
                performance.loc[dict(epoch=epoch, set="test", metrics=['accuracy'], cv_fold=fold)] = test_acc
                performance.loc[dict(epoch=epoch, set="train", metrics=['accuracy'], cv_fold=fold)] = train_acc
                performance.loc[dict(epoch=epoch, set="val", metrics=['accuracy'], cv_fold=fold)] = val_acc

            elif not bunch.multi_outcome and not bunch.multiclass:

                # calculate performance metrics 
                test_mae, train_mae, val_mae = mae(testp, testy), \
                                               mae(trainp, trainy), \
                                               mae(valp, valy)

                test_pears, train_pears, val_pears = pearsonr(testp[:, 0], testy[:, 0]), \
                                                     pearsonr(trainp[:, 0], trainy[:, 0]), \
                                                     pearsonr(valp[:, 0], valy[:, 0])

                # print metrics
                print(f"{bunch.predicted_outcome}"
                      f"\nTrain MAE : {train_mae:.3}, pearson R: {train_pears[0]:.3} (p = {train_pears[1]:.4})",
                      f"\nTest MAE : {test_mae:.3}, pearson R: {test_pears[0]:.3} (p = {test_pears[1]:.4})",
                      f"\nVal MAE : {val_mae:.3}, pearson R: {val_pears[0]:.3} (p = {val_pears[1]:.4})")

                # saving metrics to xarray
                performance.loc[dict(epoch=epoch, set="test", metrics=['MAE', 'pearsonR', 'p_value'], cv_fold=fold)] = \
                    np.array([test_mae, test_pears[0], test_pears[1]])[:, None]
                performance.loc[dict(epoch=epoch, set="train", metrics=['MAE', 'pearsonR', 'p_value'], cv_fold=fold)] = \
                    np.array([train_mae, train_pears[0], train_pears[1]])[:, None]
                performance.loc[dict(epoch=epoch, set="val", metrics=['MAE', 'pearsonR', 'p_value'], cv_fold=fold)] = \
                    np.array([val_mae, val_pears[0], val_pears[1]])[:, None]

            # save model parameters iteratively, for each best epoch during training
            if bunch.multiclass:
                best_epoch_yet = \
                    bool(epoch == performance.loc[dict(set='test', metrics='accuracy', cv_fold=fold)].argmax().values)
            elif bunch.multi_outcome:  # best epoch has lowest mean error
                best_epoch_yet = bool(epoch == performance.loc[dict(set='test', metrics='MAE', cv_fold=fold)].mean(
                    axis=-1).argmin().values)
            else:
                best_epoch_yet = bool(
                    epoch == performance.loc[dict(set='test', metrics='MAE', cv_fold=fold)].argmin().values)

            if best_epoch_yet:  # making a deep copy iteratively
                best_test_epoch = epoch
                best_net = net.state_dict()  # saving dict, not net object

            # Check every ep_int epochs. If there is no improvement on performance metrics, stop training early
            if bunch.early:
                if epoch > bunch.min_ep:
                    if bunch.multi_outcome:  # if model stops learning on at least half of predicted outcomes, break
                        majority = int(np.ceil(bunch.num_outcome / 2))

                        stagnant_mae = (np.nanmean(
                            performance[epoch - bunch.ep_int:-1].loc[dict(set='test', metrics='MAE', cv_fold=fold)],
                            axis=0) <=
                                        performance[epoch].loc[
                                            dict(set='test', metrics='MAE', cv_fold=fold)]).sum() >= majority

                        stagnant_r = (np.nanmean(
                            np.abs(performance[epoch - bunch.ep_int:-1].loc[
                                       dict(set='test', metrics='pearsonR', cv_fold=fold)]),
                            axis=0) <=
                                      np.abs(
                                          performance[epoch].loc[
                                              dict(set='test', metrics='pearsonR', cv_fold=fold)])).sum() >= majority

                        if stagnant_mae or stagnant_r:
                            estop_epochs[
                                f'estop_epoch_fold_{fold}'] = epoch - bunch.ep_int  # update early stopping epoch to dict
                            break

                    elif bunch.multiclass:
                        if np.nanmean(performance[epoch - bunch.ep_int:-1].loc[
                                          dict(set='test', metrics='accuracy', cv_fold=fold)]
                                      <= performance[epoch].loc[dict(set='test', metrics='accuracy', cv_fold=fold)]):
                            estop_epochs[
                                f'estop_epoch_fold_{fold}'] = epoch - bunch.ep_int  # update early stopping epoch to dict
                            break
                    elif not bunch.multiclass and not bunch.multi_outcome:
                        stagnant_mae = np.nanmean(
                            performance[epoch - bunch.ep_int:-1].loc[dict(set='test', metrics='MAE', cv_fold=fold)],
                            axis=0) <= performance[epoch].loc[dict(set='test', metrics='MAE', cv_fold=fold)]
                        stagnant_r = np.nanmean(
                            performance[epoch - bunch.ep_int:-1].loc[
                                dict(set='test', metrics='pearsonR', cv_fold=fold)],
                            axis=0) <= performance[epoch].loc[dict(set='test', metrics='pearsonR', cv_fold=fold)]
                        if stagnant_mae or stagnant_r:
                            estop_epochs[
                                f'estop_epoch_fold_{fold}'] = epoch - bunch.ep_int  # update early stopping epoch to dict
                            break

        best_test_epochs[f'best_test_epoch_fold_{fold}'] = best_test_epoch  # updating best_test_epoch to dict

        # saving model weights with best test-performance
        torch.save(best_net, os.path.join('models', model_preamble + f'_epoch-{best_test_epoch}_fold-{fold}_model.pt'))

        del best_net, net  # deleting from memory

    # Create attribute dicitonary, to hold training params
    attrs = dict(rundate=rundate, chosen_Xdatavars=bunch.cXdv_str,
                 predicted_outcome=bunch.po_str, transformations=bunch.transformations,
                 deconfound_flavor=bunch.deconfound_flavor, architecture=bunch.architecture,
                 multiclass=bunch.multiclass, multi_outcome=bunch.multi_outcome,
                 cv_folds=bunch.cv_folds)

    attrs.update(best_test_epochs)  # adding best test epochs as attributes

    if bunch.early:  # adding early stopping epochs as attributes, if early stopping
        attrs.update(estop_epochs)

    if bunch.confound_names:  # adding confounds, if any
        attrs.update(dict(confound_names='_'.join(bunch.confound_names)))

    performance.attrs = attrs  # saving attributes
    filename_performance = model_preamble + '_performance.nc'  # savepath
    performance.name = filename_performance  # updating xarray name
    performance.to_netcdf(f'performance/BNCNN/{filename_performance}')  # saving performance

    # calculating val set metrics, across folds, based on averaged best_test_epochs
    best_val_MAE = np.nanmean([performance.loc[
                                   dict(set='val', metrics='MAE', epoch=best_test_epochs[f'best_test_epoch_fold_{i}'],
                                        cv_fold=i)].values.tolist() for i in range(bunch.cv_folds)], axis=0)
    best_val_R = np.nanmean([performance.loc[dict(set='val', metrics='pearsonR',
                                                  epoch=best_test_epochs[f'best_test_epoch_fold_{i}'],
                                                  cv_fold=i)].values.tolist() for i in range(bunch.cv_folds)], axis=0)
    best_val_p = np.nanmean([performance.loc[
                                 dict(set='val', metrics='p_value', epoch=best_test_epochs[f'best_test_epoch_fold_{i}'],
                                      cv_fold=i)].values.tolist() for i in range(bunch.cv_folds)], axis=0)
    best_val_acc = np.nanmean([performance.loc[dict(set='val', metrics='accuracy',
                                                    epoch=best_test_epochs[f'best_test_epoch_fold_{i}'],
                                                    cv_fold=i)].values.tolist() for i in range(bunch.cv_folds)], axis=0)

    # print results
    print(f'\nBest val performance'
          f'\ndataset: {bunch.chosen_Xdatavars}'
          f'\noutcome: {bunch.predicted_outcome}'
          f'\nbest test epoch(s): {list(best_test_epochs.values())}'
          f"\nbest val MAE (mean across cv_folds): {best_val_MAE}"
          f"\nbest val pearson R (mean across cv_folds): {best_val_R}"
          f"\nbest val pearson p (mean across cv_folds): {best_val_p}"
          f"\nbest val accuracy (mean across cv_folds): {best_val_acc}")

    return dict(performance=performance)


if __name__ == '__main__':
    main()
