import datetime

import numpy as np
import torch
import xarray as xr
from scipy.stats import pearsonr
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import mean_absolute_error as mae

from analysis.init_model import train, test, net
from analysis.load_model_data import multiclass, multi_outcome, num_outcome
from utils.degrees_of_freedom import nbepochs, predicted_outcome, chosen_Xdatavars, chosen_tasks, chosen_dir, \
    ep_int, min_ep, early, architecture, transformations, deconfound_flavor, scl
from utils.var_names import HCP268_tasks

global performance

# setting up xarray to hold performance metrics
sets = ['train', 'test']
metrics = ['loss', 'accuracy', 'MAE', 'pearsonR', 'p_value']
alloc_data = np.zeros((nbepochs, len(sets), len(metrics), num_outcome))
alloc_data[:] = np.nan

if multi_outcome:
    performance = xr.DataArray(alloc_data, coords=[np.arange(nbepochs), sets, metrics, predicted_outcome],
                               dims=['epoch', 'set', 'metrics', 'outcome'])
else:
    performance = xr.DataArray(alloc_data, coords=[np.arange(nbepochs), sets, metrics, predicted_outcome],
                               dims=['epoch', 'set', 'metrics', 'outcome'])

print('Using data: ', chosen_Xdatavars, '\n Predicting:', ", ".join(predicted_outcome))

# initial prediction from starting weights
preds, y_true, loss_test = test()

if multi_outcome:  # calculate predictive performance of multiple variables
    mae_all = np.array([mae(y_true[:, i], preds[:, i]) for i in range(len(predicted_outcome))])
    pears_all = np.array([list(pearsonr(y_true[:, i], preds[:, i])) for i in range(len(predicted_outcome))])
    print("Init Network")
    for i in range(len(predicted_outcome)):
        print(
            f"Test Set, {predicted_outcome[i]} : MAE : {100 * mae_all[i]:.02}, pearson R: {pears_all[i, 0]:.02}, p = {pears_all[i, 1]:.02}")

elif multiclass:  # calculate classification performance
    preds, y_true = np.argmax(preds, 1), np.argmax(y_true, 1)
    print(y_true, preds)
    acc_1 = balanced_accuracy_score(y_true, preds)
    print("Init Network")
    print(f"Test Set : Accuracy for Engagement : {100 * acc_1:.02}")


elif not multiclass and not multi_outcome:  # calculate predictive performance of 1 variable
    mae_1 = mae(preds[:, 0], y_true[:, 0])
    pears_1 = pearsonr(preds[:, 0], y_true[:, 0])
    print("Init Network")
    print(f"Test Set : MAE for Engagement : {100 * mae_1:.02}")
    print("Test Set : pearson R for Engagement : %0.02f, p = %0.4f" % (pears_1[0], pears_1[1]))

######################################
# # Run Epochs of training and testing
######################################

global epoch

for epoch in range(nbepochs):

    trainp, trainy, loss_train = train()
    preds, y_true, loss_test = test()

    performance.loc[dict(epoch=epoch, set="test", metrics='loss')] = [loss_test]
    performance.loc[dict(epoch=epoch, set="train", metrics='loss')] = [loss_train]

    print("\nEpoch %d" % epoch)

    if multi_outcome:
        mae_all, trainmae_all = np.array([mae(y_true[:, i], preds[:, i]) for i in range(len(predicted_outcome))]), \
                                np.array([mae(trainy[:, i], trainp[:, i]) for i in range(len(predicted_outcome))])
        pears_all, trainpears_all = np.array(
            [list(pearsonr(y_true[:, i], preds[:, i])) for i in range(len(predicted_outcome))]), \
                                    np.array([list(pearsonr(trainy[:, i], trainp[:, i])) for i in
                                              range(len(predicted_outcome))])

        for i in range(len(predicted_outcome)):
            print(
                f"{predicted_outcome[i]} : Test MAE : {mae_all[i]:.02}, pearson R: {pears_all[i, 0]:.02} (p = {pears_all[i, 1]:.02})")

        performance.loc[dict(epoch=epoch, set="test", metrics=['MAE', 'pearsonR', 'p_value'])] = [mae_all,
                                                                                                  pears_all[:, 0],
                                                                                                  pears_all[:, 1]]
        performance.loc[dict(epoch=epoch, set="train", metrics=['MAE', 'pearsonR', 'p_value'])] = [trainmae_all,
                                                                                                   trainpears_all[:,
                                                                                                   0],
                                                                                                   trainpears_all[:,
                                                                                                   1]]
    elif multiclass:
        preds, y_true, trainp, trainy = np.argmax(preds, 1), np.argmax(y_true, 1), \
                                        np.argmax(trainp, 1), np.argmax(trainy, 1)
        acc, trainacc = balanced_accuracy_score(y_true, preds), balanced_accuracy_score(trainy, trainp)

        print(f"{predicted_outcome}, Test accuracy : {acc:.02}")

        performance.loc[dict(epoch=epoch, set="test", metrics=['accuracy'])] = acc
        performance.loc[dict(epoch=epoch, set="train", metrics=['accuracy'])] = trainacc

    elif not multi_outcome and not multiclass:
        mae_1, trainmae_1 = mae(preds, y_true), mae(trainp, trainy)
        pears_1, trainpears_1 = pearsonr(preds[:, 0], y_true[:, 0]), pearsonr(trainp[:, 0], trainy[:, 0])
        print(
            f"{predicted_outcome} : Test MAE : {mae_1:.02}, Test pearson R: {pears_1[0]:.02} (p = {pears_1[1]:.04})")

        performance.loc[dict(epoch=epoch, set="test", metrics=['MAE', 'pearsonR', 'p_value'])] = np.array(
            [mae_1, pears_1[0], pears_1[1]])[:, None]
        performance.loc[dict(epoch=epoch, set="train", metrics=['MAE', 'pearsonR', 'p_value'])] = np.array(
            [trainmae_1,
             trainpears_1[0],
             trainpears_1[1]])[:, None]

    ####################u
    ## EARLY STOPPING ##
    ####################
    # Checking every ep_int epochs. If there is no improvement on performance metrics, stop training
    if (epoch > min_ep) and early:
        if multi_outcome:  # if model stops learning on at least half of predicted outcomes, break
            majority = int(np.ceil(num_outcome / 2))

            stagnant_mae = (np.nanmean(performance[epoch - ep_int:-1].loc[dict(set='test', metrics='MAE')],
                                       axis=0) <=
                            performance[epoch].loc[dict(set='test', metrics='MAE')]).sum() >= majority

            stagnant_r = (np.nanmean(
                np.abs(performance[epoch - ep_int:-1].loc[dict(set='test', metrics='pearsonR')]), axis=0) <=
                          np.abs(performance[epoch].loc[dict(set='test', metrics='pearsonR')])).sum() >= majority

            if stagnant_mae or stagnant_r:
                break

        elif multiclass:
            if np.nanmean(performance[epoch - ep_int:-1].loc[dict(set='test', metrics='accuracy')]
                          <= performance[epoch].loc[dict(set='test', metrics='accuracy')]):
                break
        elif not multiclass and not multi_outcome:
            stagnant_mae = np.nanmean(performance[epoch - ep_int:-1].loc[dict(set='test', metrics='MAE')],
                                      axis=0) <= performance[epoch].loc[dict(set='test', metrics='MAE')]
            stagnant_r = np.nanmean(performance[epoch - ep_int:-1].loc[dict(set='test', metrics='pearsonR')],
                                    axis=0) <= performance[epoch].loc[dict(set='test', metrics='pearsonR')]
            if stagnant_mae or stagnant_r:
                break

# # creating filename to save data
rundate = datetime.datetime.now().strftime("%m-%d-%H-%M")

po = '_'.join(predicted_outcome)
cXdv = '_'.join(chosen_Xdatavars)

if chosen_tasks == list(HCP268_tasks.keys())[:-1]:  # If all tasks, used simply name file with HCP_alltasks_268
    cXdv = f'{chosen_dir[0]}'
if predicted_outcome == [f'softcluster_{i}' for i in range(1, 14)]:  # if predicting on all clusters
    po = 'softcluster_all'

if early:
    early_str = f'es{ep_int}'
else:
    early_str = ''

model_preamble = f"BNCNN_{architecture}_{po}_{cXdv}" \
                 f"_{transformations}_{deconfound_flavor}{scl}_{early_str}_" + rundate

# Save trained model parameters
filename_model = model_preamble + '_model.pt'
torch.save(net, f'models/{filename_model}')

# get best test-set results
if multiclass:
    best_test_epoch = performance.loc[dict(set='test', metrics='accuracy')].argmax().values
elif multi_outcome:  # best epoch has lowest mean error
    best_test_epoch = performance.loc[dict(set='test', metrics='MAE')].mean(axis=-1).argmin().values
else:
    best_test_epoch = performance.loc[dict(set='test', metrics='MAE')].argmin().values

# Save trained model performance
performance = performance.assign_attrs(rundate=rundate, chosen_Xdatavars=cXdv, predicted_outcome=po,
                                       transformations=transformations, deconfound_flavor=deconfound_flavor,
                                       architecture=architecture, multiclass=multiclass,
                                       multi_outcome=multi_outcome, best_test_epoch=best_test_epoch)
if early:
    performance = performance.assign_attrs(stop_int=epoch - ep_int)  # adding early stop epoch to xarray

filename_performance = model_preamble + '_performance.nc'
performance.name = filename_performance  # updating xarray name internally

performance.to_netcdf(f'performance/{filename_performance}')  # saving performance

# print best results
print(f'\nBest test performance'
      f'\ndataset: {cXdv}'
      f'\noutcome: {po}'
      f'\nepoch: {best_test_epoch}'
      f"\nMAE: {performance.loc[dict(set='test', metrics='MAE', epoch=best_test_epoch)].values.squeeze()}"
      f"\npearson R: {performance.loc[dict(set='test', metrics='pearsonR', epoch=best_test_epoch)].values.squeeze()}"
      f"\npearson p-value: {performance.loc[dict(set='test', metrics='p_value', epoch=best_test_epoch)].values.squeeze()}"
      f"\naccuracy: {performance.loc[dict(set='test', metrics='accuracy', epoch=best_test_epoch)].values.squeeze()}")
