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
    performance = xr.DataArray(alloc_data, coords=[np.arange(nbepochs), sets, metrics, outcome_names],
                               dims=['epoch', 'set', 'metrics', 'outcome'])
else:
    performance = xr.DataArray(alloc_data, coords=[np.arange(nbepochs), sets, metrics, [outcome_names]],
                               dims=['epoch', 'set', 'metrics', 'outcome'])


def main():
    print('Using data: ', chosen_dir, '\n')

    # # initializing weights
    # net.apply(init_weights_he)

    # initial prediction from starting weights
    preds, y_true, loss_test = test()

    if multi_outcome:  # calculate predictive performance of multiple variables
        mae_all = np.array([mae(y_true[:, i], preds[:, i]) for i in range(len(outcome_names))])
        pears_all = np.array([list(pearsonr(y_true[:, i], preds[:, i])) for i in range(len(outcome_names))])
        print("Init Network")
        for i in range(len(outcome_names)):
            print(
                f"Test Set, {outcome_names[i]} : MAE : {100 * mae_all[i]:.02}, pearson R: {pears_all[i, 0]:.02}, p = {pears_all[i, 1]:.02}")

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
            mae_all, trainmae_all = np.array([mae(y_true[:, i], preds[:, i]) for i in range(len(outcome_names))]), \
                                    np.array([mae(trainy[:, i], trainp[:, i]) for i in range(len(outcome_names))])
            pears_all, trainpears_all = np.array(
                [list(pearsonr(y_true[:, i], preds[:, i])) for i in range(len(outcome_names))]), \
                                        np.array([list(pearsonr(trainy[:, i], trainp[:, i])) for i in
                                                  range(len(outcome_names))])

            for i in range(len(outcome_names)):
                print(
                    f"{outcome_names[i]} : Test MAE : {mae_all[i]:.02}, pearson R: {pears_all[i, 0]:.02} (p = {pears_all[i, 1]:.02})")

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

            print(f"{outcome_names}, Test accuracy : {acc:.02}")

            performance.loc[dict(epoch=epoch, set="test", metrics=['accuracy'])] = acc
            performance.loc[dict(epoch=epoch, set="train", metrics=['accuracy'])] = trainacc

        elif not multi_outcome and not multiclass:
            mae_1, trainmae_1 = mae(preds, y_true), mae(trainp, trainy)
            pears_1, trainpears_1 = pearsonr(preds[:, 0], y_true[:, 0]), pearsonr(trainp[:, 0], trainy[:, 0])
            print(f"{outcome_names} : Test MAE : {mae_1:.02}, Test pearson R: {pears_1[0]:.02} (p = {pears_1[1]:.04})")

            performance.loc[dict(epoch=epoch, set="test", metrics=['MAE', 'pearsonR', 'p_value'])] = [mae_1,
                                                                                                      pears_1[0],
                                                                                                      pears_1[1]]
            performance.loc[dict(epoch=epoch, set="train", metrics=['MAE', 'pearsonR', 'p_value'])] = [trainmae_1,
                                                                                                       trainpears_1[0],
                                                                                                       trainpears_1[1]]

        ####################
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

# adding stop epoch to xarray
performance = performance.assign_coords(stop_int=epoch - ep_int)
performance = performance.expand_dims('stop_int')
