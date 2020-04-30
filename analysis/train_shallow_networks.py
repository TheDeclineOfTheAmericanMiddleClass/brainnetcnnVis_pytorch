from sklearn import neural_network, svm
from sklearn.linear_model import ElasticNet, SGDClassifier, MultiTaskElasticNet
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MaxAbsScaler

from analysis.load_model_data import *
from preprocessing.preproc_funcs import *

scale_features = True

subs = train_subs.tolist() + test_subs.tolist() + val_subs.tolist()
inds = train_ind.tolist() + test_ind.tolist() + val_ind.tolist()

# creating data arrays to be trained on
if data_are_matrices:
    shallowX_train = np.concatenate([X[var].sel(dict(subject=subs)).values[:,
                                     np.triu_indices_from(X[var][0], k=1)[0], np.triu_indices_from(X[var][0], k=1)[1]]
                                     for var in chosen_Xdatavars], axis=1)

elif not data_are_matrices:
    shallowX_train = X[chosen_Xdatavars[0]][inds].values

    if scale_features:  # TODO: fix so only training data used to fit scaler in CV loop
        scaler = MaxAbsScaler().fit(X[chosen_Xdatavars[0]][train_ind.tolist()].values)
        shallowX_train = scaler.transform(shallowX_train)

shallowY_train = Y[inds]

if multiclass:  # transforming one_hot encoded Y-data back into multiclass
    shallowY_train = onehot_to_multiclass(shallowY_train)

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
        cv_results = cross_validate(clf, shallowX_train, shallowY_train, cv=cv_folds,
                                    scoring=['balanced_accuracy'],
                                    verbose=True)

    else:
        clf = svm.SVR(kernel='linear', gamma='scale',
                      verbose=True)
        cv_results = cross_validate(clf, shallowX_train, shallowY_train, cv=cv_folds,
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
        if predicted_outcome == ['Gender']:
            hl_sizes = (3, fl)

        FC90Net = neural_network.MLPClassifier(hidden_layer_sizes=hl_sizes,
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
        cv_results = cross_validate(FC90Net, shallowX_train, shallowY_train, cv=cv_folds,
                                    scoring=['balanced_accuracy'],
                                    verbose=True)

    else:
        hl_sizes = (9, fl)
        # if np.any([po in ['NEOFAC_O', 'NEOFAC_C', 'NEOFAC_E', 'NEOFAC_A', 'NEOFAC_N'] for po in predicted_outcome]):
        #     hl_sizes = (223, 128, 192, fl)  # per He et al. 2019, 58 behavior prediction
        if predicted_outcome == ['Age_in_Yrs']:
            hl_sizes = (9, fl)  # per He et al. 2019, age prediction

        FC90Net = neural_network.MLPRegressor(  # hidden_layer_sizes=hl_sizes,
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

        cv_results = cross_validate(FC90Net, shallowX_train, shallowY_train, cv=cv_folds,
                                    scoring=scoring,
                                    verbose=True)

    print('...training complete!\n')

    return cv_results


def train_ElasticNet():
    print('Training ElasticNet...')

    if multiclass:
        regr = SGDClassifier(penalty='elasticnet', l1_ratio=.5,  # logistic regression with even L1/L2 penalty
                             random_state=1234)
        cv_results = cross_validate(regr, shallowX_train, shallowY_train, cv=cv_folds,
                                    scoring=['balanced_accuracy'],
                                    verbose=True)

    # TODO: see if I should use multitaskCV here for multioutput problems
    elif multi_outcome:
        regr = MultiTaskElasticNet(random_state=1234)
        cv_results = cross_validate(regr, shallowX_train, shallowY_train, cv=cv_folds,
                                    scoring=scoring,
                                    verbose=True)
    else:
        regr = ElasticNet(random_state=1234)
        cv_results = cross_validate(regr, shallowX_train, shallowY_train, cv=cv_folds,
                                    scoring=scoring,
                                    verbose=True)

    print('...training complete!\n')

    return cv_results


# Training Networks
print(f'Training shallow networks to predict {", ".join(predicted_outcome)}, from data in {chosen_Xdatavars}...\n')

FC90_cv_results = train_FC90net()

SVM_cv_results = train_SVM()

Elastic_cv_results = train_ElasticNet()

# formatting a printing results
float_formatter = "{:.3f}".format
np.set_printoptions(formatter={'float_kind': float_formatter})

for results in [FC90_cv_results, Elastic_cv_results, SVM_cv_results]:
    if multiclass:
        print(-np.mean(results['balanced_accuracy']))
    else:
        print(-np.mean(results['test_neg_mean_absolute_error']), np.mean(results['test_r2']))
