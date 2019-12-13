# importing our training, testing functions and weight initializations
from analysis.Init_model import *
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error as mae
from analysis.Load_model_data import *

# initializing weights
net.apply(init_weights_he)

preds, y_true, loss_val = test()

mae_1 = mae(preds[:, 0], y_true[:, 0])
pears_1 = pearsonr(preds[:, 0], y_true[:, 0])
print("Init Network")
print("Test Set : MAE for Engagement : %0.2f %%" % (100 * mae_1))
print("Test Set : pearson R for Engagement : %0.2f, p = %0.4f" % (pears_1[0], pears_1[1]))

# TODO: figure out why 2 values were being predicted
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

    print("Validation Set : MAE for Engagement : %0.2f %%" % (100 * mae_1))
    print("Validation Set : pearson R for Engagement : %0.2f, p = %0.4f" % (pears_1[0], pears_1[1]))

    # TODO: figure out why 2 values were being predicted
    # mae_2 = mae(preds[:, 1], y_true[:, 1])
    # pears_2 = pearsonr(preds[:, 1], y_true[:, 1])
    #
    # allmae_test2.append(mae_2)
    # allpears_test2.append(pears_2[0])
    #
    # print("Test Set : MAE for Training : %0.2f %%" % (100 * mae_2))
    # print("Test Set : pearson R for Training : %0.2f, p = %0.4f" % (pears_2[0], pears_2[1]))
