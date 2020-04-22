from sklearn import neural_network, svm
from sklearn.linear_model import ElasticNet, SGDClassifier, MultiTaskElasticNet
from sklearn.model_selection import cross_validate

from analysis.load_model_data import *
from preprocessing.preproc_funcs import *

# creating data arrays to be trained on
if data_are_matrices:
    shallowX_train = np.concatenate([X[var].sel(dict(subject=train_subs)).values[:,
                                     np.triu_indices_from(X[var][0], k=1)[0], np.triu_indices_from(X[var][0], k=1)[1]]
                                     for var in X.keys()], axis=1)

elif not data_are_matrices:
    shallowX_train = X[list(X.data_vars)[0]][train_ind].values

shallowY_train = Y[train_ind]
shallowY_test = Y[test_ind]

if multiclass:  # transforming one_hot encoded Y-data back into multiclass
    shallowY_train = onehot_to_multiclass(shallowY_train)
    shallowY_test = onehot_to_multiclass(shallowY_test)

# defining regression scoring methods
# scoring = dict(zip(['mae','r2'], [make_scorer(mean_absolute_error, multioutput='raw_values'), 'r2']))
scoring = ['neg_mean_absolute_error', 'r2']


# defining training on SVM, FC90 (MLP), and ElasticNet
def train_SVM():
    print('Training SVM...')

    if multi_outcome:
        print('SVM cannot handle multi-outcome problems. Sorry!\n')
        return

    elif multiclass:
        # clf = SGDClassifier(verbose=True, random_state=1234)
        clf = svm.SVC(kernel='linear', gamma='scale', verbose=True, random_state=1234)
        cv_results = cross_validate(clf, shallowX_train, shallowY_train, cv=5,
                                    scoring=['balanced_accuracy'],
                                    verbose=True)

    else:
        clf = svm.SVR(kernel='linear', gamma='scale',
                      verbose=True)  # TODO: look deeper into bad input shape for multioutcome data (i.e. personality)
        cv_results = cross_validate(clf, shallowX_train, shallowY_train, cv=5,
                                    scoring=scoring,
                                    verbose=True)


    print('...training complete!\n')

    return cv_results


def train_FC90net():
    print('Training FC90Net...')
    # kf = KFold(n_splits=10)

    if multi_outcome:
        fl = num_outcome
    elif multiclass:
        fl = num_classes
    else:
        fl = 1  # TODO: see later that this works

    if multiclass:
        FC90Net = neural_network.MLPClassifier(hidden_layer_sizes=(3, fl),
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
        cv_results = cross_validate(FC90Net, shallowX_train, shallowY_train, cv=5,
                                    scoring=['balanced_accuracy'],
                                    verbose=True)

    else:  # TODO: make sure this works for single-class, single-outcome
        FC90Net = neural_network.MLPRegressor(verbose=True, random_state=1234)
        cv_results = cross_validate(FC90Net, shallowX_train, shallowY_train, cv=5,
                                    scoring=scoring,
                                    verbose=True)

    print('...training complete!\n')

    return cv_results


def train_ElasticNet():
    print('Training ElasticNet...')

    if multiclass:
        regr = SGDClassifier(penalty='elasticnet', l1_ratio=.5,  # logistic regression with even L1/L2 penalty
                             random_state=1234)
        cv_results = cross_validate(regr, shallowX_train, shallowY_train, cv=5,
                                    scoring=['balanced_accuracy'],
                                    verbose=True)

    # TODO: see if I should use multitaskCV here for multioutput problems
    elif multi_outcome:
        regr = MultiTaskElasticNet(random_state=1234)
        cv_results = cross_validate(regr, shallowX_train, shallowY_train, cv=5,
                                    scoring=scoring,
                                    verbose=True)
    else:
        regr = ElasticNet(random_state=1234)
        cv_results = cross_validate(regr, shallowX_train, shallowY_train, cv=5,
                                    scoring=scoring,
                                    verbose=True)

    print('...training complete!\n')

    return cv_results


# Training Networks
print(f'Training shallow networks to predict {predicted_outcome}, from data in {list(X.keys())}...\n')

# SVM_cv_results = train_SVM()

Elastic_cv_results = train_ElasticNet()

# FC90_cv_results = train_FC90net()
