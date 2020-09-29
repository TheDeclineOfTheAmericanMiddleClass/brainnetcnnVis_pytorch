import datetime
import gc
import os

import numpy as np
import torch
import torch.utils.data.dataset
import xarray as xr
from scipy.stats import pearsonr
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import mean_absolute_error as mae

from utils.util_funcs import Bunch


def main(args):
    bunch = Bunch(args)

    net = bunch.net
    test = bunch.test
    train = bunch.train
    val = bunch.val

    # creating filename to save data
    rundate = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M")
    model_preamble = f"BNCNN_{bunch.architecture}_{bunch.po_str}_{bunch.cXdv_str}" \
                     f"_{bunch.transformations}_{bunch.deconfound_flavor}{bunch.scl}_{bunch.early_str}_" + rundate

    # setting up xarray to hold performance metrics
    sets = ['train', 'test', 'val']
    metrics = ['loss', 'accuracy', 'MAE', 'pearsonR', 'p_value']
    alloc_data = np.zeros((bunch.n_epochs, len(sets), len(metrics), bunch.num_outcome))
    alloc_data[:] = np.nan
    performance = xr.DataArray(alloc_data,
                               coords=[range(bunch.n_epochs), sets, metrics,
                                       bunch.predicted_outcome],
                               dims=['epoch', 'set', 'metrics', 'outcome'])

    gc.collect()  # cleaning up space

    print(f'\nUsing data: ', bunch.chosen_Xdatavars,
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

        performance.loc[dict(epoch=epoch, set="test", metrics='loss')] = [loss_test]
        performance.loc[dict(epoch=epoch, set="train", metrics='loss')] = [loss_train]
        performance.loc[dict(epoch=epoch, set="val", metrics='loss')] = [loss_val]

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
            performance.loc[dict(epoch=epoch, set="test", metrics=['MAE', 'pearsonR', 'p_value'])] = \
                [test_mae_all, test_pears_all[:, 0], test_pears_all[:, 1]]
            performance.loc[dict(epoch=epoch, set="train", metrics=['MAE', 'pearsonR', 'p_value'])] = \
                [train_mae_all, train_pears_all[:, 0], train_pears_all[:, 1]]
            performance.loc[dict(epoch=epoch, set="val", metrics=['MAE', 'pearsonR', 'p_value'])] = \
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
            performance.loc[dict(epoch=epoch, set="test", metrics=['accuracy'])] = test_acc
            performance.loc[dict(epoch=epoch, set="train", metrics=['accuracy'])] = train_acc
            performance.loc[dict(epoch=epoch, set="val", metrics=['accuracy'])] = val_acc

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
            performance.loc[dict(epoch=epoch, set="test", metrics=['MAE', 'pearsonR', 'p_value'])] = \
                np.array([test_mae, test_pears[0], test_pears[1]])[:, None]
            performance.loc[dict(epoch=epoch, set="train", metrics=['MAE', 'pearsonR', 'p_value'])] = \
                np.array([train_mae, train_pears[0], train_pears[1]])[:, None]
            performance.loc[dict(epoch=epoch, set="val", metrics=['MAE', 'pearsonR', 'p_value'])] = \
                np.array([val_mae, val_pears[0], val_pears[1]])[:, None]

        # save model parameters iteratively, for each best epoch during training
        if bunch.multiclass:
            best_epoch_yet = \
                bool(epoch == performance.loc[dict(set='test', metrics='accuracy')].argmax().values)
        elif bunch.multi_outcome:  # best epoch has lowest mean error
            best_epoch_yet = bool(epoch == performance.loc[dict(set='test', metrics='MAE')].mean(
                axis=-1).argmin().values)
        else:
            best_epoch_yet = bool(
                epoch == performance.loc[dict(set='test', metrics='MAE')].argmin().values)

        if best_epoch_yet:  # making a deep copy iteratively
            best_test_epoch = epoch
            best_net = net.state_dict()  # saving dict, not net object

        # Check every ep_int epochs. If there is no improvement on performance metrics, stop training early
        if bunch.early:
            if epoch > bunch.min_ep:
                if bunch.multi_outcome:  # if model stops learning on at least half of predicted outcomes, break
                    majority = int(np.ceil(bunch.num_outcome / 2))

                    stagnant_mae = (np.nanmean(
                        performance[epoch - bunch.ep_int:-1].loc[dict(set='test', metrics='MAE')],
                        axis=0) <=
                                    performance[epoch].loc[
                                        dict(set='test', metrics='MAE')]).sum() >= majority

                    stagnant_r = (np.nanmean(
                        np.abs(performance[epoch - bunch.ep_int:-1].loc[
                                   dict(set='test', metrics='pearsonR')]),
                        axis=0) <=
                                  np.abs(
                                      performance[epoch].loc[
                                          dict(set='test', metrics='pearsonR')])).sum() >= majority

                    if stagnant_mae or stagnant_r:
                        estop_epoch = epoch - bunch.ep_int
                        break

                elif bunch.multiclass:
                    if np.nanmean(performance[epoch - bunch.ep_int:-1].loc[
                                      dict(set='test', metrics='accuracy')]
                                  <= performance[epoch].loc[dict(set='test', metrics='accuracy')]):
                        estop_epoch = epoch - bunch.ep_int
                        break
                elif not bunch.multiclass and not bunch.multi_outcome:
                    stagnant_mae = np.nanmean(
                        performance[epoch - bunch.ep_int:-1].loc[dict(set='test', metrics='MAE')],
                        axis=0) <= performance[epoch].loc[dict(set='test', metrics='MAE')]
                    stagnant_r = np.nanmean(
                        performance[epoch - bunch.ep_int:-1].loc[
                            dict(set='test', metrics='pearsonR')],
                        axis=0) <= performance[epoch].loc[dict(set='test', metrics='pearsonR')]
                    if stagnant_mae or stagnant_r:
                        estop_epoch = epoch - bunch.ep_int
                        break

    # saving model weights with best test-performance
    torch.save(best_net, os.path.join('models', model_preamble + f'_epoch-{best_test_epoch}_model.pt'))

    # Create attribute dicitonary, to hold training params
    attrs = dict(rundate=rundate, chosen_Xdatavars=bunch.cXdv_str,
                 predicted_outcome=bunch.po_str, transformations=bunch.transformations,
                 deconfound_flavor=bunch.deconfound_flavor, architecture=bunch.architecture,
                 multiclass=bunch.multiclass, multi_outcome=bunch.multi_outcome,
                 cv_folds=bunch.cv_folds, best_test_epoch=best_test_epoch)

    if bunch.early:  # adding early stopping epochs as attributes, if early stopping
        attrs.update(dict(estop_epoch=estop_epoch))

    if bunch.confound_names:  # adding confounds, if any
        attrs.update(dict(confound_names='_'.join(bunch.confound_names)))

    performance.attrs = attrs  # saving attributes
    filename_performance = model_preamble + '_performance.nc'  # savepath
    performance.name = filename_performance  # updating xarray name
    performance.to_netcdf(f'performance/BNCNN/{filename_performance}')  # saving performance

    # calculating val set metrics, across folds, based on averaged best_test_epoch
    best_val_MAE = performance.loc[dict(set='val', metrics='MAE', epoch=best_test_epoch)].values.tolist()
    best_val_R = performance.loc[dict(set='val', metrics='pearsonR', epoch=best_test_epoch)].values.tolist()
    best_val_p = performance.loc[dict(set='val', metrics='p_value', epoch=best_test_epoch)].values.tolist()
    best_val_acc = performance.loc[dict(set='val', metrics='accuracy', epoch=best_test_epoch)].values.tolist()

    # print results
    print(f'\nBest val performance'
          f'\ndataset: {bunch.chosen_Xdatavars}'
          f'\noutcome: {bunch.predicted_outcome}'
          f'\nbest test epoch: {best_test_epoch}'
          f"\nbest val MAE: {best_val_MAE}"
          f"\nbest val pearson R: {best_val_R}"
          f"\nbest val pearson p: {best_val_p}"
          f"\nbest val accuracy: {best_val_acc}")

    return dict(performance=performance)


if __name__ == '__main__':
    main()
