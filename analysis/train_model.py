import datetime

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

    # setting up xarray to hold performance metrics
    sets = ['train', 'test']
    metrics = ['loss', 'accuracy', 'MAE', 'pearsonR', 'p_value']
    alloc_data = np.zeros((bunch.nbepochs, len(sets), len(metrics), bunch.num_outcome))
    alloc_data[:] = np.nan
    performance = xr.DataArray(alloc_data, coords=[np.arange(bunch.nbepochs), sets, metrics, bunch.predicted_outcome],
                               dims=['epoch', 'set', 'metrics', 'outcome'])

    print('Using data: ', bunch.chosen_Xdatavars, '\n Predicting:', ", ".join(bunch.predicted_outcome))

    # initial prediction from starting weights
    print("Init Network")
    preds, y_true, loss_test = test()

    if bunch.multi_outcome:  # calculate predictive performance of multiple variables
        mae_all = np.array([mae(y_true[:, i], preds[:, i]) for i in range(len(bunch.predicted_outcome))])
        pears_all = np.array([list(pearsonr(y_true[:, i], preds[:, i])) for i in range(len(bunch.predicted_outcome))])
        for i in range(len(bunch.predicted_outcome)):
            print(
                f"Test Set, {bunch.predicted_outcome[i]} : MAE : {100 * mae_all[i]:.02}, pearson R: {pears_all[i, 0]:.02}, p = {pears_all[i, 1]:.02}")

    elif bunch.multiclass:  # calculate classification performance
        preds, y_true = np.argmax(preds, 1), np.argmax(y_true, 1)
        # print(preds, '\n', y_true)
        acc_1 = balanced_accuracy_score(y_true, preds)
        print(f"Test Set : Accuracy for Engagement : {acc_1:.03}")

    elif not bunch.multiclass and not bunch.multi_outcome:  # calculate predictive performance of 1 variable
        mae_1 = mae(preds[:, 0], y_true[:, 0])
        pears_1 = pearsonr(preds[:, 0], y_true[:, 0])
        print(f"Test Set : MAE for Engagement : {mae_1:.02}")
        print("Test Set : pearson R for Engagement : %0.02f, p = %0.4f" % (pears_1[0], pears_1[1]))

    # # train model
    for epoch in range(bunch.nbepochs):

        trainp, trainy, loss_train = train()
        preds, y_true, loss_test = test()

        performance.loc[dict(epoch=epoch, set="test", metrics='loss')] = [loss_test]
        performance.loc[dict(epoch=epoch, set="train", metrics='loss')] = [loss_train]

        print("\nEpoch %d" % epoch)

        if bunch.multi_outcome:
            mae_all, trainmae_all = np.array(
                [mae(y_true[:, i], preds[:, i]) for i in range(len(bunch.predicted_outcome))]), \
                                    np.array(
                                        [mae(trainy[:, i], trainp[:, i]) for i in range(len(bunch.predicted_outcome))])
            pears_all, trainpears_all = np.array(
                [list(pearsonr(y_true[:, i], preds[:, i])) for i in range(len(bunch.predicted_outcome))]), \
                                        np.array([list(pearsonr(trainy[:, i], trainp[:, i])) for i in
                                                  range(len(bunch.predicted_outcome))])

            for i in range(len(bunch.predicted_outcome)):
                print(
                    f"{bunch.predicted_outcome[i]} : Test MAE : {mae_all[i]:.02}, pearson R: {pears_all[i, 0]:.02} (p = {pears_all[i, 1]:.02})")

            performance.loc[dict(epoch=epoch, set="test", metrics=['MAE', 'pearsonR', 'p_value'])] = [mae_all,
                                                                                                      pears_all[:, 0],
                                                                                                      pears_all[:, 1]]
            performance.loc[dict(epoch=epoch, set="train", metrics=['MAE', 'pearsonR', 'p_value'])] = [trainmae_all,
                                                                                                       trainpears_all[:,
                                                                                                       0],
                                                                                                       trainpears_all[:,
                                                                                                       1]]

        elif bunch.multiclass:
            preds, y_true, trainp, trainy = np.argmax(preds, 1), np.argmax(y_true, 1), \
                                            np.argmax(trainp, 1), np.argmax(trainy, 1)
            # print(preds, y_true)
            acc, trainacc = balanced_accuracy_score(y_true, preds), balanced_accuracy_score(trainy, trainp)
            print(f"{bunch.predicted_outcome}, Test accuracy : {acc:.03}, Train accuracy: {trainacc:.03}")

            performance.loc[dict(epoch=epoch, set="test", metrics=['accuracy'])] = acc
            performance.loc[dict(epoch=epoch, set="train", metrics=['accuracy'])] = trainacc

        elif not bunch.multi_outcome and not bunch.multiclass:
            mae_1, trainmae_1 = mae(preds, y_true), mae(trainp, trainy)
            pears_1, trainpears_1 = pearsonr(preds[:, 0], y_true[:, 0]), pearsonr(trainp[:, 0], trainy[:, 0])
            print(
                f"{bunch.predicted_outcome} : Test MAE : {mae_1:.03}, Test pearson R: {pears_1[0]:.03} (p = {pears_1[1]:.04})")

            performance.loc[dict(epoch=epoch, set="test", metrics=['MAE', 'pearsonR', 'p_value'])] = np.array(
                [mae_1, pears_1[0], pears_1[1]])[:, None]
            performance.loc[dict(epoch=epoch, set="train", metrics=['MAE', 'pearsonR', 'p_value'])] = np.array(
                [trainmae_1,
                 trainpears_1[0],
                 trainpears_1[1]])[:, None]

        # Checking every ep_int epochs. If there is no improvement on performance metrics, stop training early
        if bunch.early:
            if epoch > bunch.min_ep:
                if bunch.multi_outcome:  # if model stops learning on at least half of predicted outcomes, break
                    majority = int(np.ceil(bunch.num_outcome / 2))

                    stagnant_mae = (np.nanmean(
                        performance[epoch - bunch.ep_int:-1].loc[dict(set='test', metrics='MAE')],
                        axis=0) <=
                                    performance[epoch].loc[dict(set='test', metrics='MAE')]).sum() >= majority

                    stagnant_r = (np.nanmean(
                        np.abs(performance[epoch - bunch.ep_int:-1].loc[dict(set='test', metrics='pearsonR')]),
                        axis=0) <=
                                  np.abs(
                                      performance[epoch].loc[dict(set='test', metrics='pearsonR')])).sum() >= majority

                    if stagnant_mae or stagnant_r:
                        break

                elif bunch.multiclass:
                    if np.nanmean(performance[epoch - bunch.ep_int:-1].loc[dict(set='test', metrics='accuracy')]
                                  <= performance[epoch].loc[dict(set='test', metrics='accuracy')]):
                        break
                elif not bunch.multiclass and not bunch.multi_outcome:
                    stagnant_mae = np.nanmean(performance[epoch - bunch.ep_int:-1].loc[dict(set='test', metrics='MAE')],
                                              axis=0) <= performance[epoch].loc[dict(set='test', metrics='MAE')]
                    stagnant_r = np.nanmean(
                        performance[epoch - bunch.ep_int:-1].loc[dict(set='test', metrics='pearsonR')],
                        axis=0) <= performance[epoch].loc[dict(set='test', metrics='pearsonR')]
                    if stagnant_mae or stagnant_r:
                        break

    # # creating filename to save data
    rundate = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M")
    model_preamble = f"BNCNN_{bunch.architecture}_{bunch.po_str}_{bunch.cXdv_str}" \
                     f"_{bunch.transformations}_{bunch.deconfound_flavor}{bunch.scl}_{bunch.early_str}_" + rundate

    # Save trained model parameters
    filename_model = model_preamble + '_model.pt'
    torch.save(net, f'models/{filename_model}')

    # Save trained model performance
    performance = performance.assign_attrs(rundate=rundate, chosen_Xdatavars=bunch.cXdv_str,
                                           predicted_outcome=bunch.po_str, transformations=bunch.transformations,
                                           deconfound_flavor=bunch.deconfound_flavor, architecture=bunch.architecture,
                                           multiclass=bunch.multiclass, multi_outcome=bunch.multi_outcome)
    if bunch.early:
        performance = performance.assign_attrs(stop_int=epoch - bunch.ep_int)  # adding early stop epoch to xarray

    filename_performance = model_preamble + '_performance.nc'
    performance.name = filename_performance  # updating xarray name internally

    # determining best test results
    if bunch.multiclass:
        best_test_epoch = performance.loc[dict(set='test', metrics='accuracy')].argmax().values
    elif bunch.multi_outcome:  # best epoch has lowest mean error
        best_test_epoch = performance.loc[dict(set='test', metrics='MAE')].mean(
            axis=-1).argmin().values
    else:
        best_test_epoch = performance.loc[dict(set='test', metrics='MAE')].argmin().values

    performance = performance.assign_attrs(
        best_test_epoch=best_test_epoch)  # adding best test epoch

    performance.to_netcdf(f'performance/BNCNN/{filename_performance}')  # saving performance

    # Print best test-set results
    print(f'\nBest test performance'
          f'\ndataset: {bunch.cXdv_str}'
          f'\noutcome: {bunch.po_str}'
          f'\nepoch: {best_test_epoch}'
          f"\nMAE: {performance.loc[dict(set='test', metrics='MAE', epoch=best_test_epoch)].values.squeeze()}"
          f"\npearson R: {performance.loc[dict(set='test', metrics='pearsonR', epoch=best_test_epoch)].values.squeeze()}"
          f"\npearson p-value: {performance.loc[dict(set='test', metrics='p_value', epoch=best_test_epoch)].values.squeeze()}"
          f"\naccuracy: {performance.loc[dict(set='test', metrics='accuracy', epoch=best_test_epoch)].values.squeeze()}")

    return dict(performance=performance)

if __name__ == '__main__':
    main()
