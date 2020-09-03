import datetime
import itertools
import os
import pickle
import random

import numpy as np
import xarray as xr
from scipy.stats import pearsonr
from sklearn import neural_network, svm
from sklearn.linear_model import ElasticNet, SGDClassifier, MultiTaskElasticNet
from sklearn.metrics import balanced_accuracy_score, r2_score, mean_absolute_error
from sklearn.model_selection import cross_validate

from utils.util_funcs import Bunch, create_cv_folds, create_shallow_array


def main(args):
    bunch = Bunch(args)
    seed = 1234  # setting random seed

    # TODO: note all functions dependent on bunch
    def train_SVM(X_in, Y_in, scoring=None, sklearn_cv=False,
                  X_test=None, Y_test=None, results=None):

        if scoring is None:
            scoring = ['neg_mean_absolute_error', 'r2']
        print('Training SVM...')

        if bunch.multi_outcome:
            print('SVM cannot handle multi-outcome problems. Sorry!\n')
            return

        elif bunch.multiclass:
            clf = svm.SVC(kernel='linear', gamma='scale', verbose=True, random_state=seed,
                          class_weight='balanced')  # default reg. param C = 1.0
            if sklearn_cv:
                results = cross_validate(clf, X_in, Y_in, cv=bunch.cv_folds, scoring=['balanced_accuracy'],
                                         verbose=True)
            else:
                clf.fit(X_in, Y_in)
                prediction = clf.predict(X_test)
                t_bacc = balanced_accuracy_score(Y_test, prediction)

                results['test_balanced_accuracy'].append(t_bacc)

        else:
            clf = svm.SVR(kernel='linear', gamma='scale', verbose=True)  # default reg. param C = 1.0
            if sklearn_cv:
                results = cross_validate(clf, X_in, Y_in, cv=bunch.cv_folds, scoring=scoring, verbose=True)
            else:
                clf.fit(X_in, Y_in)
                prediction = clf.predict(X_test)
                t_r2 = r2_score(Y_test, prediction)
                t_mae = mean_absolute_error(Y_test, prediction)

                results['test_r2'].append(t_r2)
                results['test_mean_absolute_error'].append(t_mae)

        print('...training complete!\n')

        return results

    def train_FC90net(X_in, Y_in, scoring=None, sklearn_cv=False,
                      X_test=None, Y_test=None, results=None):

        if scoring is None:
            scoring = ['neg_mean_absolute_error', 'r2']
        print('Training FC90Net...')
        # kf = KFold(n_splits=10)

        if bunch.multi_outcome:
            fl = bunch.num_outcome
        elif bunch.multiclass:
            fl = bunch.num_classes
        else:
            fl = 1  # TODO: see later that this works

        if bunch.multiclass:
            if bunch.predicted_outcome == ['Gender']:
                hl_sizes = (3, fl)

            print(f'Hidden layer size: {hl_sizes}')
            FC90Net = neural_network.MLPClassifier(hidden_layer_sizes=hl_sizes,
                                                   max_iter=500,
                                                   solver='sgd',
                                                   # learning_rate='constant',
                                                   learning_rate='adaptive',
                                                   # learning_rate_init=lr,
                                                   momentum=bunch.momentum,
                                                   activation='relu',
                                                   verbose=True,
                                                   early_stopping=False,
                                                   random_state=seed)

            if sklearn_cv:
                results = cross_validate(FC90Net, X_in, Y_in, cv=bunch.cv_folds,
                                         scoring=['balanced_accuracy'],
                                         verbose=True)
            else:
                FC90Net.fit(X_in, Y_in)
                prediction = FC90Net.predict(X_test)
                t_bacc = balanced_accuracy_score(Y_test, prediction)

                results['test_balanced_accuracy'].append(t_bacc)

        else:
            hl_sizes = (9,)
            # if np.any([po in ['NEOFAC_O', 'NEOFAC_C', 'NEOFAC_E', 'NEOFAC_A', 'NEOFAC_N'] for po in predicted_outcome]):
            #     hl_sizes = (223, 128, 192, fl)  # per He et al. 2019, 58 behavior prediction
            if bunch.predicted_outcome == ['Age_in_Yrs']:
                hl_sizes = (9, fl)  # per He et al. 2019, age prediction

            print(f'Hidden layer size: {hl_sizes}')
            FC90Net = neural_network.MLPRegressor(hidden_layer_sizes=hl_sizes,
                                                  max_iter=500,
                                                  solver='sgd',
                                                  # learning_rate='constant',
                                                  learning_rate='adaptive',
                                                  # learning_rate_init=lr,
                                                  momentum=bunch.momentum,
                                                  activation='relu',
                                                  verbose=True,
                                                  early_stopping=False,
                                                  # tol=1e-4,
                                                  # n_iter_no_change=10,
                                                  random_state=seed)

            if sklearn_cv:
                results = cross_validate(FC90Net, X_in, Y_in, cv=bunch.cv_folds, scoring=scoring, verbose=True)
            else:
                FC90Net.fit(X_in, Y_in)
                prediction = FC90Net.predict(X_test)
                t_r2 = r2_score(Y_test, prediction)
                t_mae = mean_absolute_error(Y_test, prediction)

                results['test_r2'].append(t_r2)
                results['test_mean_absolute_error'].append(t_mae)

        print('...training complete!\n')

        return results

    def train_ElasticNet(X_in, Y_in, scoring=None, sklearn_cv=False,
                         X_test=None, Y_test=None, results=None):

        print('Training ElasticNet...')

        if bunch.multiclass:
            regr = SGDClassifier(penalty='elasticnet', l1_ratio=.5,  # logistic regression with even L1/L2 penalty
                                 random_state=seed)

            if sklearn_cv:
                results = cross_validate(regr, X_in, Y_in, cv=bunch.cv_folds,
                                         scoring=['balanced_accuracy'],
                                         verbose=True)
            else:
                regr.fit(X_in, Y_in)
                prediction = regr.predict(X_test)
                t_bacc = balanced_accuracy_score(Y_test, prediction)

                results['test_balanced_accuracy'].append(t_bacc)

        # TODO: see if I should use multitaskCV here for multioutput problems
        elif bunch.multi_outcome:
            regr = MultiTaskElasticNet(random_state=seed)
            if sklearn_cv:
                results = cross_validate(regr, X_in, Y_in, cv=bunch.cv_folds,
                                         scoring=scoring,
                                         verbose=True)
            else:
                regr.fit(X_in, Y_in)
                prediction = regr.predict(X_test)
                t_r2 = r2_score(Y_test, prediction)
                t_mae = mean_absolute_error(Y_test, prediction)

                results['test_r2'].append(t_r2)
                results['test_mean_absolute_error'].append(t_mae)

        else:
            regr = ElasticNet(random_state=seed)
            if sklearn_cv:
                results = cross_validate(regr, X_in, Y_in, cv=bunch.cv_folds,
                                         scoring=scoring,
                                         verbose=True)
            else:
                regr.fit(X_in, Y_in)
                prediction = regr.predict(X_test)
                t_r2 = r2_score(Y_test, prediction)
                t_mae = mean_absolute_error(Y_test, prediction)

                results['test_r2'].append(t_r2)
                results['test_mean_absolute_error'].append(t_mae)

        print('...training complete!\n')

        return results

    def train_CV(n_folds, train_func, scoring=None, sklearn_cv=False):
        """ Trains shallow models with n_fold cross validation

        :param sklearn_cv: (bool) whether to use sklearn's cross validation implementation
        :param n_folds: (int) number of cross-validation folds to train over
        :param train_func: (function) shallow model training function in {train_SVM, train_FC90net, train_ElasticNet}
        :return:
        """

        partition_keys = ['train_subs', 'test_subs', 'val_subs']
        features_keys = ['sig_features']
        all_keys = scoring + partition_keys + features_keys

        # creating partitions for test, train, validate
        subs_by_fold, inds_by_fold = create_cv_folds(bunch.X, n_folds=n_folds + 1, seed=seed,
                                                     separate_families=False, shuffle=True)

        # allocating dictionary for results
        cv_results = dict(zip(all_keys, [[] for _ in range(len(all_keys))]))

        # for each of n_folds model trainings, pick an N choose k combination at random
        N = n_folds + 1  # total number of splits
        k = n_folds - 1  # number of folds in train set
        train_fold_combos = np.random.permutation([x for x in itertools.permutations(range(N), k)])[:n_folds]

        binary = [0, 1]  # indices for train and validation fold

        # run loop over training folds
        for i, train_folds in enumerate(train_fold_combos):

            print('\npartitioning data into arrays to feed to model...')
            # discerning test and validation fold from remaining N choose (N-k) folds
            random.shuffle(binary)
            non_train_folds = list(set(range(N)) - set(train_folds))
            test_fold = non_train_folds[binary[0]]
            val_fold = non_train_folds[binary[1]]

            # indices for each fold in the outcome
            train_inds = np.hstack(inds_by_fold[train_folds])
            test_inds = np.hstack(inds_by_fold[test_fold])
            val_inds = np.hstack(inds_by_fold[val_fold])

            # subject numbers for each fold in the outcome
            train_subs = np.hstack(subs_by_fold[train_folds])
            test_subs = np.hstack(subs_by_fold[test_fold])
            val_subs = np.hstack(subs_by_fold[val_fold])

            cv_results['train_subs'].append(train_subs)  # saving to results
            cv_results['test_subs'].append(test_subs)
            cv_results['val_subs'].append(val_subs)

            # unravel each subject's data into 1D arrays, for each partition
            shallow_X_train, shallow_Y_train = create_shallow_array(X=bunch.X, Y=bunch.Y,
                                                                    X_is_matrix=bunch.data_are_matrices,
                                                                    chosen_Xdatavars=bunch.chosen_Xdatavars,
                                                                    subs=train_subs, inds=train_inds,
                                                                    multiclass=bunch.multiclass,
                                                                    train_inds=train_inds)

            shallow_X_test, shallow_Y_test = create_shallow_array(X=bunch.X, Y=bunch.Y,
                                                                  X_is_matrix=bunch.data_are_matrices,
                                                                  chosen_Xdatavars=bunch.chosen_Xdatavars,
                                                                  subs=test_subs, inds=test_inds,
                                                                  multiclass=bunch.multiclass,
                                                                  train_inds=train_inds)

            shallow_X_train = shallow_X_train.reshape(len(train_subs), -1)  # reshaping to (subjects x features)
            n_features = shallow_X_train.shape[-1]  # number of features
            p_thresh = .001  # p-value threshold

            # determining significant features
            print('determining significant features...')
            sig_features = np.zeros(n_features, dtype=bool)
            for feature in range(n_features):
                _, p = pearsonr(shallow_X_train[:, feature], shallow_Y_train)
                sig_features[feature] = bool(p < p_thresh)
            assert sum(sig_features) > 0, 'no significant features detected...'

            # TODO: add new loop here for more conservative feature choice (i.e. only sig_features over all n_folds)

            cv_results['sig_features'].append(np.argwhere(sig_features).squeeze().tolist())  # saving to results

            shallow_X_train = shallow_X_train[:, sig_features]  # pruning the X array
            shallow_X_test = shallow_X_test[:, sig_features]

            cv_results = train_func(shallow_X_train, shallow_Y_train,
                                    scoring=['neg_mean_absolute_error', 'r2'], sklearn_cv=sklearn_cv,
                                    X_test=shallow_X_test, Y_test=shallow_Y_test, results=cv_results)

        return cv_results

    def save_CV_results(cv_results, cv_folds, model_name, scoring=None):
        """ Saves results of cross validated shallow network resultss

        :param scoring: results keys for performance metrics
        :param cv_results: the output of sklearn.model_selection.cross_validate()
        :param cv_folds: number of cross validation folds
        :param model_name: the name of the model trained
        :return: None
        """

        rundate = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M")  # current time
        save_dir = os.path.join('performance', model_name)  # save directory
        model_preamble = f"{model_name}_{bunch.po_str}_{bunch.cXdv_str}_{bunch.transformations}_{bunch.deconfound_flavor}{bunch.scl}_" + rundate

        # separating info into dictionaries
        scoring_dict = {k: v for k, v in cv_results.items() if v and k in scoring}
        assert not not scoring_dict, 'results must have some keys in scoring'
        non_scoring_dict = {k: v for k, v in cv_results.items() if v and k not in scoring}

        metrics = list(scoring_dict.keys())  # performance metrics keys
        results_data = np.array(list(scoring_dict.values())).squeeze()  # performance metrics data

        performance = xr.DataArray(results_data, coords=[metrics, range(cv_folds)],
                                   dims=['metrics', 'cv_fold'], name=model_preamble)

        # Save trained model performance #
        performance = performance.assign_attrs(rundate=rundate, chosen_Xdatavars=bunch.cXdv_str,
                                               outcome=bunch.po_str, transformations=bunch.transformations,
                                               deconfound_flavor=bunch.deconfound_flavor)

        # saving performance + partition and feature info
        performance.to_netcdf(os.path.join(save_dir, model_preamble + '_performance.nc'))
        pickle.dump(non_scoring_dict, open(os.path.join(save_dir, model_preamble + '_PFinfo.pkl'), "wb"))

    # all possible scoring methods for all problems (i.e. multiclass, multioutcome continuous, etc.)
    scoring_keys = ['test_neg_mean_absolute_error', 'test_mean_absolute_error', 'test_r2', 'test_balanced_accuracy']

    if 'SVM' in bunch.model:
        train_func = train_SVM
    elif 'FC90' in bunch.model:
        train_func = train_FC90net
    elif 'ElasticNet' in bunch.model:
        train_func = train_ElasticNet

    print(f'Training {bunch.model} to predict {", ".join(bunch.predicted_outcome)}, from data in'
          f' {bunch.chosen_Xdatavars} over {bunch.cv_folds} folds...\n')
    cv_results = train_CV(n_folds=bunch.cv_folds, train_func=train_func, scoring=scoring_keys, sklearn_cv=False)

    # saving results after all n_fold training
    print(f'saving all {bunch.cv_folds}-fold results...\n')
    save_CV_results(cv_results, bunch.cv_folds, bunch.model[0], scoring=scoring_keys)

    # # printing results
    np.set_printoptions(precision=4)
    print(f'Results from training {bunch.model[0]} to predict {bunch.predicted_outcome} from {bunch.chosen_Xdatavars}'
          f' over {bunch.cv_folds} cross-validated folds:\n')
    for key, value in list(cv_results.items()):
        if value and key in scoring_keys:
            print(f'{key}: {cv_results[key]}')


if __name__ == '__main__':
    main()
