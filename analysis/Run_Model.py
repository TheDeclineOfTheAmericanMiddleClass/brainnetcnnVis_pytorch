from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error as mae

from analysis.Init_model import *
# from analysis.Load_model_data import *
# from analysis.Define_model import *
from preprocessing.Model_DOF import *

# initializing weights
net.apply(init_weights_he)

# initial prediction from starting weights
preds, y_true, loss_val = test()

if multi_outcome:
    # prediciton of all variables, collapsed over mae
    mae_all = np.array([mae(y_true[:, i], preds[:, i]) for i in range(len(outcome_names))])
    pears_all = np.array([list(pearsonr(y_true[:, i], preds[:, i])) for i in range(len(outcome_names))])
    print("Init Network")
    for i in range(len(outcome_names)):
        print(
            f"Test Set, {outcome_names[i]} : MAE : {100 * mae_all[i]:.02}, pearson R: {pears_all[i, 0]:.02}, p = {pears_all[i, 1]:.02}")
else:
    # prediction of 1 variable
    mae_1 = mae(preds[:, 0], y_true[:, 0])
    pears_1 = pearsonr(preds[:, 0], y_true[:, 0])
    print("Init Network")
    print(f"Test Set : MAE for Engagement : {100 * mae_1:.2}")
    print("Test Set : pearson R for Engagement : %0.2f, p = %0.4f" % (pears_1[0], pears_1[1]))

######################################
# # Run Epochs of training and testing
######################################

nbepochs = 300

allloss_train = []
allloss_test = []
allmae_test1 = []
allpears_test1 = []
allpval_test1 = []

for epoch in range(nbepochs):
    loss_train = train(epoch)

    allloss_train.append(loss_train)

    preds, y_true, loss_val = test()

    allloss_test.append(loss_val)

    print("\nEpoch %d" % epoch)

    if multi_outcome:
        mae_all = np.array(
            [mae(y_true[:, i], preds[:, i]) for i in range(len(outcome_names))])  # num_outcomes-sized array
        pears_all = np.array(
            [list(pearsonr(y_true[:, i], preds[:, i])) for i in
             range(len(outcome_names))])  # 2 x num_outcomes-sized array
        for i in range(len(outcome_names)):
            print(
                f"Test Set, {outcome_names[i]} : MAE : {mae_all[i]:.02}, pearson R: {pears_all[i, 0]:.02}, p = {pears_all[i, 1]:.02}")  # deleted 100 * factors

        allmae_test1.append(list(mae_all))
        allpears_test1.append(list(pears_all[:, 0]))
        allpval_test1.append(list(pears_all[:, 1]))

    else:
        mae_1 = mae(preds, y_true)
        pears_1 = pearsonr(preds[:, 0], y_true[:, 0])  # NOTE: pearsonr only takes 1-dim arrays
        print(f"Test Set, {outcome_names} : MAE : {mae_1:.02}, pearson R: {pears_1[0]:.02}, p = {pears_1[1]:.04}")

        allmae_test1.append(mae_1)
        allpears_test1.append(pears_1[0])
        allpval_test1.append(pears_1[1])

    # # EARLY STOPPING
    # Checking every ep_int epochs. If there is no improvement on avg MAE or Pearson r, stop training
    if (epoch > 0) & (epoch % ep_int == 0):
        # if model stops learning on at least half of predicted outcomes, break
        if multi_outcome:
            majority = int(np.ceil(len(allmae_test1[epoch]) / 2))
            stagnant_mae = (np.nanmean(allmae_test1[epoch - ep_int:-1], axis=0) <= allmae_test1[
                epoch]).sum() >= majority
            stagnant_r = (np.nanmean(np.abs(allpears_test1[epoch - ep_int:-1]), axis=0) <= np.abs(
                allpears_test1[epoch])).sum() >= majority
            if stagnant_mae and stagnant_r:
                break  # TODO: implement logic to break then run (1) save model, (2) plot model results, (3) run next model
        else:
            stagnant_mae = np.nanmean(allmae_test1[epoch - ep_int:-1], axis=0) <= allmae_test1[epoch]
            stagnant_r = np.nanmean(allpears_test1[epoch - ep_int:-1], axis=0) <= allpears_test1[epoch]
            if stagnant_mae and stagnant_r:
                break


# take only values of MAE in epochs before the one that triggered early stopping
# ... OR if no early stopping, take values that came ep_int epochs before final one
losses_train = allloss_train[:-ep_int]
losses_test = allloss_test[:-ep_int]
maes_test = allmae_test1[:-ep_int]
pvals_test = allpval_test1[:-ep_int]
pears_test = allpears_test1[:-ep_int]

# the model's final performance
final_mae = maes_test[-1]
final_pears = (pears_test[-1], pvals_test[-1])
