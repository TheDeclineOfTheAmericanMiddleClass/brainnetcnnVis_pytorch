from sklearn import neural_network, svm
from sklearn.linear_model import ElasticNet, SGDClassifier
from sklearn.model_selection import cross_validate

from analysis.load_model_data import *
from preprocessing.preproc_funcs import *

print(f'Training shallow networks to predict {predicted_outcome}, from data in {list(X.keys())}...\n')

# creating data arrays to be trained on
if not multi_input:  # TODO: adjust this for xarray
    shallowX_train = np.array([sample[np.triu_indices(len(sample), k=1)] for j, sample in enumerate(X[train_ind])])
    shallowX_test = np.array([sample[np.triu_indices(len(sample), k=1)] for j, sample in enumerate(X[test_ind])])

elif multi_input:
    # concatenate the upper triangle of of each variable in the X xarray dataset
    shallowX_train = np.concatenate([X[var].isel({'subject': train_ind}).values[:,
                                     np.triu_indices_from(X[var][0], k=1)[0], np.triu_indices_from(X[var][0], k=1)[1]]
                                     for i, var in enumerate(X.keys())], axis=1)
    shallowX_test = np.concatenate([X[var].isel({'subject': test_ind}).values[:,
                                    np.triu_indices_from(X[var][0], k=1)[0], np.triu_indices_from(X[var][0], k=1)[1]]
                                    for i, var in enumerate(X.keys())], axis=1)

shallowY_train = Y[train_ind]
shallowY_test = Y[test_ind]

if multiclass:  # transforming one_hot encoded Y-data back into multiclass
    shallowY_train = onehot_to_multiclass(shallowY_train)
    shallowY_test = onehot_to_multiclass(shallowY_test)


# defining training on SVM, FC90 (MLP), and ElasticNet
def train_SVM():
    print('Training SVM...')

    if not multi_outcome and not multiclass:
        clf = svm.SVC(kernel='linear', gamma='scale', verbose=True, random_state=1234)
    else:
        clf = SGDClassifier(verbose=True, random_state=1234)

    # Cross-validation score
    cv_results = cross_validate(clf, shallowX_train, shallowY_train, cv=5,
                                scoring=['neg_mean_absolute_error', 'r2'],
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

    # NOTE: dropout no possible for sklearn MLP
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

    if multiclass:
        cv_results = cross_validate(FC90Net, shallowX_train, shallowY_train, cv=5,
                                    scoring=['balanced_accuracy'],
                                    verbose=True)

    else:  # TODO: make sure this works for single-class, single-outcome
        cv_results = cross_validate(FC90Net, shallowX_train, shallowY_train, cv=5,
                                    scoring=['neg_mean_absolute_error', 'r2'],
                                    verbose=True)

    print('...training complete!\n')

    return cv_results


def train_ElasticNet():
    print('Training ElasticNet...')

    if multiclass:
        regr = SGDClassifier(penalty='elasticnet', l1_ratio=.5, alpha=1,  # logistic regression with even L1/L2 penalty
                             random_state=1234)
        cv_results = cross_validate(regr, shallowX_train, shallowY_train, cv=5,
                                    scoring=['balanced_accuracy'],
                                    verbose=True)

    else:  # TODO: make sure this works for single-class, single-outcome
        regr = ElasticNet(random_state=1234)
        cv_results = cross_validate(regr, shallowX_train, shallowY_train, cv=5,
                                    scoring=['neg_mean_absolute_error', 'r2'],
                                    verbose=True)

    print('...training complete!\n')

    return cv_results


elastic_cv_results_ = train_ElasticNet()  # linear regression

SVM_cv_results = train_SVM()

FC90_cv_results = train_FC90net()
