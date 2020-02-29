import time

from scipy.stats import pearsonr
from sklearn import neural_network
# from preprocessing.Model_DOF import
from sklearn.linear_model import ElasticNet
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error as mae

from analysis.Load_model_data import *

# SETTING UP DATA TO BE TRAINED
Xtrain_corr = X[train_ind]
Y_train = Y[train_ind]  # Y_train

Xtest_corr = X[test_ind]
Y_test = Y[test_ind]  # Y_test

Xval_corr = X[val_ind]
Y_val = Y[val_ind]


# for deconfounded data...
shallowX_train = np.array([Xtrain_corr[j][np.triu_indices(len(Xtrain_corr[0]), k=1)] for j in range(len(Xtrain_corr))])
shallowY_train = np.array(list(Y_train))
shallowX_test = np.array([Xtest_corr[j][np.triu_indices(len(Xtest_corr[0]), k=1)] for j in range(len(Xtest_corr))])
shallowY_test = np.array(list(Y_test))


#########################
### ElasticNet model ####
#########################

def train_ElasticNet():
    print('Training ElasticNet...')

    # TODO: implement elastic net cross-validation?
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html#sklearn.linear_model.ElasticNetCV

    regr = ElasticNet(random_state=1234)
    # regr = ElasticNet(random_state=1234, l1_ratio=0)  # initializing with default weights
    # regr = ElasticNet(random_state=1234, l1_ratio=1, precompute=True)
    # regr = ElasticNet(random_state=1234, l1_ratio=1, normalize=True)
    # regr = ElasticNet(random_state=1234, l1_ratio=1, precompute=True, normalize=True)
    # regr = ElasticNet(random_state=1234, l1_ratio=1, precompute=True, normalize=True, selection='random')
    # regr = ElasticNet(random_state=1234, l1_ratio=.7, precompute=True, normalize=True, selection='random')

    regr.fit(shallowX_train, shallowY_train)
    y_elasticPred = regr.predict(shallowX_test)
    print('...training complete!\n')

    # print(regr.coef_)
    # print(regr.intercept_)

    # elasticNet metrics
    elastic_r, elastic_p = pearsonr(shallowY_test, y_elasticPred)
    elastic_mae = mae(shallowY_test, y_elasticPred)

    return elastic_r, elastic_p, elastic_mae, regr


# elastic_r, elastic_p, elastic_mae, trained_elastic = train_ElasticNet()

###############################
### Support Vector Machine ####
###############################
from sklearn import svm


def train_SVM():
    print('Training SVM...')

    if not multi_outcome:
        clf = svm.SVC(kernel='linear', gamma='scale', verbose=True)
        clf.fit(shallowX_train, shallowY_train)

        y_svmPred = clf.predict(shallowX_test)

        svm_r, svm_p = pearsonr(shallowY_test, y_svmPred)
        svm_mae = mae(shallowY_test, y_svmPred)
        print('...training complete!\n')

    else:
        svm_r = svm_p = svm_mae = np.nan

    return svm_r, svm_p, svm_mae, clf


# svm_r, svm_p, svm_mae, trained_svm = train_SVM()

###################################
### Fully Connected Network/MLP ###
###################################

# TODO: once BrainNetCNN fixed for multiclass classification, fix hyperparameter-tuned Yeo FC90Net for sex classifciaton
# class FC90Net_YeoSex(torch.nn.Module):
#     def __init__(self, example):  # removed num_classes=10
#         super(FC90Net_YeoSex, self).__init__()
#         self.dense1 = torch.nn.Linear(1, 3, example)
#         self.dense2 = torch.nn.Linear(3, 2)
#         # self.dense3 = torch.nn.Linear(2,num_outcome)
#
#     def forward(self, x):
#         out = F.dropout(F.leaky_relu(self.dense1(x), negative_slope=.33), p=.00275)
#         out = out.view(out.size(0), -1)
#         out = F.leaky_relu(self.dense2(out), negative_slope=.33)
#         # out = F.leaky_relu(self.dense3(out), negative_slope=.33)
#
#         return out

def train_FC90net():
    print('Training FC90Net...')
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=10)

    # NOTE: dropout no possible for sklearn MLP
    FC90Net = neural_network.MLPClassifier(hidden_layer_sizes=(3, 2),
                                           # age (9, 1), sex (3, 2), ? (shallowX_train.shape[-1], 90)
                                           max_iter=500,
                                           solver='sgd',
                                           learning_rate='constant',
                                           learning_rate_init=lr,
                                           momentum=momentum,
                                           activation='relu',
                                           verbose=True,
                                           early_stopping=False,
                                           tol=1e-4,
                                           n_iter_no_change=10,
                                           random_state=1234)

    FC90Net.fit(shallowX_train, shallowY_train)
    print(f"time elapsed: {time.time() - start_time:.2f}s")

    if multiclass:
        fc_metrics = accuracy_score(shallowY_test, y_fcPred)

    else:
        fc_r, fc_p = pearsonr(shallowY_test, y_fcPred)
        fc_mae = mae(shallowY_test, y_fcPred)
        fc_metrics = {'pearson R': fc_r, 'p-value': fc_p, 'MAE': fc_mae}

    return fc_metrics, FC90Net

start_time = time.time()
fc_metrics, FC90Net = train_FC90net()
y_fcPred = FC90Net.predict(shallowX_test)
print(f'Yeo_FC90net **{predicted_outcome}** test accuracy: {fc_metrics * 100:.4} %')


# def print_shallow_results():
#     # PRINTING THE SHALLOW RESULTS
#     s_colhs = ['ElasticNet', 'SVM']
#     s_rowhs = ['pearson r', 'mean absolute error']
#     s_tabel = np.array([[f'{elastic_r:.3}, p-value: {elastic_p:.3}',
#                          f'{svm_r:.3}, p-value: {svm_p:.3}'],
#                         [f'{elastic_mae:.3}', f'{svm_mae:.3}']])
#
#     for i, x in enumerate(s_colhs):
#         for j, y in enumerate(s_rowhs):
#             print(f'{x} {y}: {s_tabel[j, i]}')
#         print('')
