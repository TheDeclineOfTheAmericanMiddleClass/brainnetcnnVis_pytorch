from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error as mae
from preprocessing.Main_preproc import *
from analysis.Load_model_data import *

# reshape subject's data into vector
svmdata = []
for i, x in enumerate(data):
    svmdata.append(np.array(x[np.triu_indices(300, k=1)]))

ages = np.array(AiY)

# concatenate train and validation sets for shallow networks
shallowX = [svmdata[j] for j in list(train_ind) + list(val_ind)]  # training cov matrices
shallowY = [ages[j] for j in list(train_ind) + list(val_ind)]  # training ages

y_testTrue = [ages[j] for j in list(test_ind)]  # test ages

########## training elasticNet model ######################
from sklearn.linear_model import ElasticNet
# TODO: implement elastic net cross-validation?
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html#sklearn.linear_model.ElasticNetCV

regr = ElasticNet(random_state=0)  # initializing with default weights
regr.fit(shallowX, shallowY)
y_elasticPred = regr.predict([svmdata[j] for j in list(test_ind)])

# print(regr.coef_)
# print(regr.intercept_)

# elasticNet metrics
elastic_r, elastic_p = pearsonr(y_testTrue, y_elasticPred)
elastic_mae = mae(y_testTrue, y_elasticPred)

######## training SVM ###############
from sklearn import svm

clf = svm.SVC(gamma='auto')
clf.fit(shallowX, shallowY)

y_svmPred = clf.predict([svmdata[j] for j in list(test_ind)])

svm_r, svm_p = pearsonr(y_testTrue, y_svmPred)
svm_mae = mae(y_testTrue, y_svmPred)
