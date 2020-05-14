from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error as mae

from analysis.init_model import *

# setting up xarray to hold performance metrics
sets = ['train', 'test']
metrics = ['loss', 'accuracy', 'MAE', 'pearsonR', 'p_value']
alloc_data = np.zeros((nbepochs, len(sets), len(metrics), num_outcome))
alloc_data[:] = np.nan

global performance

if multi_outcome:
    performance = xr.DataArray(alloc_data, coords=[np.arange(nbepochs), sets, metrics, predicted_outcome],
                               dims=['epoch', 'set', 'metrics', 'outcome'])
else:
    performance = xr.DataArray(alloc_data, coords=[np.arange(nbepochs), sets, metrics, predicted_outcome],
                               dims=['epoch', 'set', 'metrics', 'outcome'])


def main():
    print('Using data: ', chosen_dir, '\n Predicting:', ", ".join(predicted_outcome))

    # # initializing weights
    # net.apply(init_weights_he)

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
        acc_1 = accuracy_score(preds, y_true)
        print("Init Network")
        print(f"Test Set : Accuracy for Engagement : {100 * acc_1:.2}")


    elif not multiclass and not multi_outcome:  # calculate predictive performance of 1 variable
        mae_1 = mae(preds[:, 0], y_true[:, 0])
        pears_1 = pearsonr(preds[:, 0], y_true[:, 0])
        print("Init Network")
        print(f"Test Set : MAE for Engagement : {100 * mae_1:.2}")
        print("Test Set : pearson R for Engagement : %0.2f, p = %0.4f" % (pears_1[0], pears_1[1]))

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
            acc, trainacc = accuracy_score(preds, y_true), accuracy_score(trainp, trainy)

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
                    break  # TODO: implement logic to break then run (1) save model, (2) plot model results, (3) run next model

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


if __name__ == "__main__":
    main()

import datetime

rundate = datetime.datetime.now().strftime("%m-%d-%H-%M")

model_preamble = f"BNCNN_{architecture}_{'_'.join(predicted_outcome)}_{'_'.join(chosen_Xdatavars)}" \
                 f"_{transformations}_{deconfound_flavor}{scl}__es{ep_int}_" + rundate

# Save trained model parameters
filename_model = model_preamble + '_model.pt'
torch.save(net, f'models/{filename_model}')

# Save trained model performance
performance = performance.assign_coords(stop_int=epoch - ep_int)  # adding early stop epoch to xarray
performance = performance.expand_dims('stop_int')
performance = performance.assign_coords(rundate=rundate)  # adding rundate to xarray
performance = performance.expand_dims('rundate')
filename_performance = model_preamble + '_performance.nc'
performance.name = filename_performance  # updating xarray name internally

performance.to_netcdf(f'performance/{filename_performance}')  # saving performance

# Print best test-set results
if multiclass:
    best_test_epoch = performance.loc[dict(set='test', metrics='accuracy')].argmax().values
else:
    best_test_epoch = performance.loc[dict(set='test', metrics='MAE')].argmin().values

print(f'\nBest test performance'
      f'\nepoch: {best_test_epoch}'
      f"\nMAE: {performance.loc[dict(set='test', metrics='MAE', epoch=best_test_epoch)].values.squeeze()}"
      f"\npearson R: {performance.loc[dict(set='test', metrics='pearsonR', epoch=best_test_epoch)].values.squeeze()}"
      f"\npearson p-value: {performance.loc[dict(set='test', metrics='p_value', epoch=best_test_epoch)].values.squeeze()}"
      f"\naccuracy: {performance.loc[dict(set='test', metrics='accuracy', epoch=best_test_epoch)].values.squeeze()}")
