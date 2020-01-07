from analysis.Load_model_data import *
from preprocessing.Main_preproc import *

from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error as mae

# reshape subject's data into vector
svmdata = []
for i, x in enumerate(cdata):
    svmdata.append(np.array(x[np.triu_indices(len(data[0]), k=1)]))

shallowX = [svmdata[j] for j in list(train_ind)]  # training cov matrices
shallowY = [ages[j] for j in list(train_ind)]  # training ages

y_testTrue = [ages[j] for j in list(test_ind)]  # test ages

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


regr.fit(shallowX, shallowY)
y_elasticPred = regr.predict([svmdata[j] for j in list(test_ind)])
print('...training complete!\n')

# print(regr.coef_)
# print(regr.intercept_)

# elasticNet metrics
elastic_r, elastic_p = pearsonr(y_testTrue, y_elasticPred)
elastic_mae = mae(y_testTrue, y_elasticPred)

print('Training SVM...')
######## training SVM ###############
from sklearn import svm

clf = svm.SVC(gamma='scale', verbose=True)
clf.fit(shallowX, shallowY)

y_svmPred = clf.predict([svmdata[j] for j in list(test_ind)])

svm_r, svm_p = pearsonr(y_testTrue, y_svmPred)
svm_mae = mae(y_testTrue, y_svmPred)
print('...training complete!\n')
