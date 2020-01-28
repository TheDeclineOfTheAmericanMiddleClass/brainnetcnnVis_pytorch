# importing our training, testing functions and weight initializations
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error as mae
# from analysis.Load_model_data import *
# from analysis.Define_model import *
from analysis.Init_model import *

# setting early stopping interval between epochs to check for changes in mae
ep_int = 8

# initializing weights
net.apply(init_weights_he)

preds, y_true, loss_val = test()

mae_1 = mae(preds[:, 0], y_true[:, 0])
pears_1 = pearsonr(preds[:, 0], y_true[:, 0])
print("Init Network")
# print("Test Set : MAE for Engagement : %0.2f %%" % (100 * mae_1))
print(f"Test Set : MAE for Engagement : {100 * mae_1:.2}")  # TODO: INCLUDE % IF ERRORS OCCUR
print("Test Set : pearson R for Engagement : %0.2f, p = %0.4f" % (pears_1[0], pears_1[1]))

# # when choose to detect sex and age at the same time
# mae_2 = mae(preds[:, 1], y_true[:, 1])
# pears_2 = pearsonr(preds[:, 1], y_true[:, 1])
# print("Test Set : MAE for Training : %0.2f %%" % (100 * mae_2))
# print("Test Set : pearson R for Training : %0.2f, p = %0.4f" % (pears_2[0], pears_2[1]))

# Run Epochs of training and testing

nbepochs = 300
allloss_train = []
allloss_test = []

allmae_test1 = []
allpears_test1 = []
allpval_test1 = []

# allmae_test2 = []
# allpears_test2 = []


for epoch in range(nbepochs):
    loss_train = train(epoch)

    allloss_train.append(loss_train)

    preds, y_true, loss_val = test()

    allloss_test.append(loss_val)

    print("Epoch %d" % epoch)
    mae_1 = mae(preds[:, 0], y_true[:, 0])
    pears_1 = pearsonr(preds[:, 0], y_true[:, 0])

    allmae_test1.append(mae_1)
    allpears_test1.append(pears_1[0])
    allpval_test1.append(pears_1[1])

    # print("Test Set : MAE for Engagement : %0.2f %%" % (100 * mae_1)) # why was this multiplied by 100 and
    # expressed as percentage?
    print("\nTest Set : MAE for Engagement : %0.2f %%" % mae_1)
    print("Test Set : pearson R for Engagement : %0.2f, p = %0.4f" % (pears_1[0], pears_1[1]))

    # TODO: implement early stopping using relative tolerance tof test set
    # Checking every ep_int epochs. If there is no improvement on avg test error, stop training
    if (epoch > 0) & (epoch % ep_int == 0):
        if (np.mean(allmae_test1[epoch - ep_int:-1]) <= allmae_test1[epoch]):
            break

    # # PREDICTION OF 2nd VALUE
    # mae_2 = mae(preds[:, 1], y_true[:, 1])
    # pears_2 = pearsonr(preds[:, 1], y_true[:, 1])
    #
    # allmae_test2.append(mae_2)
    # allpears_test2.append(pears_2[0])
    #
    # print("Test Set : MAE for Training : %0.2f %%" % (100 * mae_2))
    # print("Test Set : pearson R for Training : %0.2f, p = %0.4f" % (pears_2[0], pears_2[1]))

# take only values of MAE in epochs before the one that triggered early stopping
stoploss_train = allloss_train[:-ep_int]
stoploss_test = allloss_test[:-ep_int]
stopmae_test1 = allmae_test1[:-ep_int]
stoppval_test1 = allpval_test1[:-ep_int]
stoppears_test1 = allpears_test1[:-ep_int]

stopmae_1 = stopmae_test1[-1]
stoppears_1 = (stoppears_test1[-1], stoppval_test1[-1])
