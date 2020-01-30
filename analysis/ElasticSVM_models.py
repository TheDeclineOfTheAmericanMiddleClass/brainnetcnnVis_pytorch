from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error as mae

from analysis.Load_model_data import *

# # reshape subject's data into vector
# svmdata = []
# for i, x in enumerate(cdata):
#     svmdata.append(np.array(x[np.triu_indices(len(data[0]), k=1)]))
#
# shallowX_train = [svmdata[j] for j in list(train_ind)]  # cov matrices for training
# shallowY_train = [ages[j] for j in list(train_ind)]  # training ages
# shallowY_test = [ages[j] for j in list(test_ind)]  # test ages

# SETTING UP DATA TO BE TRAINED ON
X = tdata
Y = ages

Xtrain_corr = X[tr_i]
Y_train = Y[tr_i]  # Y_train

Xtest_corr = X[te_i]
Y_test = Y[te_i]  # Y_test

Xval_corr = X[v_i]
Y_val = Y[v_i]


# for deconfounded data...
shallowX_train = np.array([Xtrain_corr[j][np.triu_indices(len(Xtrain_corr[0]), k=1)] for j in range(len(Xtrain_corr))])
shallowY_train = Y_train
shallowX_test = np.array([Xtest_corr[j][np.triu_indices(len(Xtest_corr[0]), k=1)] for j in range(len(Xtest_corr))])
shallowY_test = Y_test

print('Training ElasticNet...')
########## training elasticNet model ######################
from sklearn.linear_model import ElasticNet
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

print('Training SVM...')
######## training SVM ###############
from sklearn import svm

clf = svm.SVC(gamma='scale', verbose=True)
clf.fit(shallowX_train, shallowY_train)

y_svmPred = clf.predict(shallowX_test)

svm_r, svm_p = pearsonr(shallowY_test, y_svmPred)
svm_mae = mae(shallowY_test, y_svmPred)
print('...training complete!\n')

# PRINTING THE SHALLOW RESULTS
s_colhs = ['ElasticNet', 'SVM']
s_rowhs = ['pearson r', 'mean absolute error']
s_tabel = np.array([[f'{elastic_r:.2}, p-value: {elastic_p:.2}',
                     f'{svm_r:.2}, p-value: {svm_p:.2}'],
                    [f'{elastic_mae:.2}', f'{svm_mae:.2}']])

for i, x in enumerate(s_colhs):
    for j, y in enumerate(s_rowhs):
        print(f'{x} {y}: {s_tabel[j, i]}')
    print('')
