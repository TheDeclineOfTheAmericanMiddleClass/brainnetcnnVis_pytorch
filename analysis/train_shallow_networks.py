import datetime

import numpy as np
import xarray as xr
from sklearn import neural_network, svm
from sklearn.linear_model import ElasticNet, SGDClassifier, MultiTaskElasticNet
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MaxAbsScaler

from utils.util_funcs import Bunch, onehot_to_multiclass, namestr


def main(args):
    bunch = Bunch(args)

    # formatting a printing results
    float_formatter = "{:.3f}".format
    np.set_printoptions(formatter={'float_kind': float_formatter})

    # setting data transformation parameters, performance metrics
    scale_features = True
    scoring = ['neg_mean_absolute_error', 'r2']

    # training on all data, without segregation by twin status
    subs = bunch.partition_subs["train"].tolist() + bunch.partition_subs["test"].tolist() + bunch.partition_subs[
        "val"].tolist()
    inds = bunch.partition_inds["train"].tolist() + bunch.partition_inds["test"].tolist() + bunch.partition_inds[
        "val"].tolist()

    # creating data arrays to be trained on
    if bunch.data_are_matrices:
        shallowX_train = np.concatenate([bunch.X[var].sel(dict(subject=subs)).values[:,
                                         np.triu_indices_from(bunch.X[var][0], k=1)[0],
                                         np.triu_indices_from(bunch.X[var][0], k=1)[1]]
                                         for var in bunch.chosen_Xdatavars], axis=1)

    elif not bunch.data_are_matrices:
        shallowX_train = bunch.X[bunch.chosen_Xdatavars[0]][inds].values

        if scale_features:
            scaler = MaxAbsScaler().fit(
                bunch.X[bunch.chosen_Xdatavars[0]][bunch.partition_inds["train"].tolist()].values)
            shallowX_train = scaler.transform(shallowX_train)

    shallowY_train = bunch.Y[inds]

    if bunch.multiclass:  # transforming one_hot encoded Y-data back into multiclass
        shallowY_train = onehot_to_multiclass(shallowY_train)

    # function to save results of sklearn cross validation
    def save_CV_results(cv_results, cv_folds, model_name):
        """
        Saves results of sklearn cross validation
        :param cv_results: the output of sklearn.model_selection.cross_validate()
        :param cv_folds: number of cross validation folds
        :param model_name: the name of the model trained
        :return: None
        """
        # setting up xarray to hold performance metrics
        metrics = list(cv_results.keys())
        alloc_data = np.array(list(cv_results.values())).squeeze()
        performance = xr.DataArray(alloc_data, coords=[metrics, range(cv_folds)],
                                   dims=['metrics', 'cv_fold'])

        # specifying filename
        rundate = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M")
        model_preamble = f"SVM_{bunch.po_str}_{bunch.cXdv_str}_{bunch.transformations}_{bunch.deconfound_flavor}{bunch.scl}_" + rundate
        filename_performance = model_preamble + '_performance.nc'
        performance.name = filename_performance  # updating xarray name internally

        # Save trained model performance #
        performance = performance.assign_attrs(rundate=rundate, chosen_Xdatavars=bunch.cXdv_str,
                                               outcome=bunch.po_str, transformations=bunch.transformations,
                                               deconfound_flavor=bunch.deconfound_flavor)

        performance.to_netcdf(f'performance/{model_name}/{filename_performance}')  # saving performance

    # defining training on SVM, FC90 (MLP), and ElasticNet
    def train_SVM(X_in, Y_in, scoring):
        print('Training SVM...')

        if bunch.multi_outcome:
            print('SVM cannot handle multi-outcome problems. Sorry!\n')
            return

        elif bunch.multiclass:
            # clf = SGDClassifier(verbose=True, random_state=1234)
            clf = svm.SVC(kernel='linear', gamma='scale', verbose=True, random_state=1234, class_weight='balanced')
            cv_results = cross_validate(clf, X_in, Y_in, cv=bunch.cv_folds,
                                        scoring=['balanced_accuracy'],
                                        verbose=True)

        else:
            clf = svm.SVR(kernel='linear', gamma='scale',
                          verbose=True)
            cv_results = cross_validate(clf, X_in, Y_in, cv=bunch.cv_folds,
                                        scoring=scoring,
                                        verbose=True)

        print('...training complete!\n')
        save_CV_results(cv_results, bunch.cv_folds, 'SVM')

        return cv_results

    def train_FC90net(X_in, Y_in, scoring):
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
                                                   random_state=1234)

            cv_results = cross_validate(FC90Net, X_in, Y_in, cv=bunch.cv_folds,
                                        scoring=['balanced_accuracy'],
                                        verbose=True)

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
                                                  random_state=1234)

            cv_results = cross_validate(FC90Net, X_in, Y_in, cv=bunch.cv_folds, scoring=scoring, verbose=True)

        print('...training complete!\n')
        save_CV_results(cv_results, bunch.cv_folds, 'FC90Net')

        return cv_results

    def train_ElasticNet(X_in, Y_in, scoring):
        print('Training ElasticNet...')

        if bunch.multiclass:
            regr = SGDClassifier(penalty='elasticnet', l1_ratio=.5,  # logistic regression with even L1/L2 penalty
                                 random_state=1234)
            cv_results = cross_validate(regr, X_in, Y_in, cv=bunch.cv_folds,
                                        scoring=['balanced_accuracy'],
                                        verbose=True)

        # TODO: see if I should use multitaskCV here for multioutput problems
        elif bunch.multi_outcome:
            regr = MultiTaskElasticNet(random_state=1234)
            cv_results = cross_validate(regr, X_in, Y_in, cv=bunch.cv_folds,
                                        scoring=scoring,
                                        verbose=True)
        else:
            regr = ElasticNet(random_state=1234)
            cv_results = cross_validate(regr, X_in, Y_in, cv=bunch.cv_folds,
                                        scoring=scoring,
                                        verbose=True)

        print('...training complete!\n')
        save_CV_results(cv_results, bunch.cv_folds, 'ElasticNet')

        return cv_results

    # Training Networks
    print(
        f'Training {bunch.model} to predict {", ".join(bunch.predicted_outcome)}, from data in {bunch.chosen_Xdatavars}...\n')

    if 'SVM' in bunch.model:
        model_results = train_SVM(shallowX_train, shallowY_train, scoring)
    elif 'FC90' in bunch.model:
        model_results = train_FC90net(shallowX_train, shallowY_train, scoring)
    elif 'ElasticNet' in bunch.model:
        model_results = train_ElasticNet(shallowX_train, shallowY_train, scoring)

    # # Printing results
    for results in [model_results]:
        print(namestr(results, locals()), f'dataset(s): {bunch.chosen_Xdatavars}')

        if bunch.multiclass:
            print('balanced_accuracy: ', np.mean(results['test_balanced_accuracy']), '\n')
        else:
            print('MAE: ', -np.mean(results['test_neg_mean_absolute_error']), 'r^2: ', np.mean(results['test_r2']),
                  '\n')


if __name__ == '__main__':
    main()
