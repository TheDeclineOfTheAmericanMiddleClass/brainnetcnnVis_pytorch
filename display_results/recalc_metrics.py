# # Calculates new performance metrics on trained models, saves them to performance results xarray
import os

import numpy as np
import torch
import xarray as xr
from scipy.stats import spearmanr, pearsonr

from analysis import load_model_data, define_models, init_model
from preprocessing import read_data
from utils.util_funcs import namestr
from utils.var_names import HCP268_tasks


def recalc_metrics(metrics, model_dir, perf_dir):
    """Calculates new performance metrics from saved BNCNN CV-fold trained models
    Saves results to a new performance file.

    :param metrics: (list of functions) correlation functions with input (predicted, true) and output (r, p)
    :return: None

    Note: Only implemented for (a) single tasks or (b) all tasks from HCP_alltasks_268, using default hyperparameters
        in dof_parser.py
    """

    performance_results = [x for x in os.listdir(perf_dir) if x.endswith('.nc')]  # performance xarray filenames

    # TODO: calculate below params from exhaustive perf_dir xarray attributes
    # fixed
    chosen_dir = ['HCP_alltasks_268']  # TODO: parsing from other chosen_dir
    deconfound_flavor = 'X1Y0'
    architecture = 'usama'
    multiclass = 0
    ordered_confound_names = ['Gender', 'Age_in_Yrs']

    # variable
    possible_transforms = ['tangent', 'untransformed']
    possible_datavars = [[y] for y in
                         [*[f'{chosen_dir[0]}_{x}' for x in list(HCP268_tasks.keys())[:-1]], 'HCP_alltasks_268_all']]
    possible_tasks = [*[[x] for x in list(HCP268_tasks.keys())[:-1]], list(HCP268_tasks.keys())[:-1]]
    possible_outcomes = [[f'NEOFAC_{x}'] for x in 'OCEAN']
    possible_folds = range(5)

    for transformation in possible_transforms:  # all possible transformations
        for j, datavar in enumerate(possible_datavars):  # all possible tasks

            pargs = dict()
            # # find transformation-task specific performance results
            # result_by_task_transform = [x for x in performance_results if x.find(transformation) >= 0 and
            #                             x.find(*datavar) >= 0]
            # base_filenames = [x[:-33] for x in result_by_task_transform]  # discarding date info
            #
            # # parsing args from sample results file
            # sample_result_path = os.path.join(perf_dir, result_by_task_transform[0])  # performance xarray filepath
            # sample_performance = xr.load_dataarray(sample_result_path)  # loading performance xarray
            # pargs = sample_performance.attrs.copy()
            #
            # predicted_outcome = [pout for pout in predict_choices if pargs['predicted_outcome'].find(pout) >= 0]
            #
            # # recovering confound names
            # confound_idxs = np.argsort([pargs['confound_names'].find(confound) for confound in confounds if confound
            #                             and pargs['confound_names'].find(confound) >= 0])
            # confound_names = [confound for confound in confounds if confound and
            #                   pargs['confound_names'].find(confound) >= 0]  # splitting '_' concatenated confound names
            # ordered_confound_names = [confound_names[i] for i in confound_idxs]

            chosen_tasks = possible_tasks[j]
            num_input = len(chosen_tasks)
            multi_input = num_input > 1
            cv_folds = len(possible_folds)
            multi_outcome = len(possible_tasks) > 1

            # updating passed arguments
            pargs.update(dict(confound_names=ordered_confound_names, chosen_dir=chosen_dir,
                              data_are_matrices=True, tan_mean='euclidean', scale_confounds=False,
                              optimizer='sgd', lr=1e-5, momentum=.9, wd=5e-4, max_norm=1.5,
                              transformations=transformation, chosen_tasks=chosen_tasks, chosen_Xdatavars=datavar,
                              cv_folds=cv_folds, num_input=num_input, multi_input=multi_input,
                              multi_outcome=multi_outcome))

            pargs.update(read_data.main(pargs))  # task-, transformation-specific
            pargs.update(dict(use_cuda=False, device='cpu'))  # calculating using CPU

            for outcome in possible_outcomes:

                perf_name = [x for x in os.listdir(perf_dir) if x.find(transformation) >= 0 and
                             x.find(*outcome) >= 0 and x.find(*datavar) >= 0]
                perf_path = os.path.join(perf_dir, perf_name[0])  # calculates from only one xarray fitting combination
                performance = xr.load_dataarray(perf_path)
                model_names = [x for x in os.listdir(model_dir) if x.startswith(perf_name[0][:-18])]

                for model_name in model_names:

                    fold = int(model_name[model_name.find('fold-') + len('fold-')])
                    pargs.update({f'best_test_epoch_fold_{fold}': performance.attrs[f'best_test_epoch_fold_{fold}']})
                    pargs.update(dict(predicted_outcome=outcome, fold=fold, deconfound_flavor=deconfound_flavor,
                                      architecture=architecture, multiclass=multiclass, chosen_Xdatavars=datavar))

                    # loading data and architecture
                    pargs.update(load_model_data.main(pargs))  # fold-, outcome-specific
                    pargs.update(define_models.main(pargs))
                    pargs.update(init_model.main(pargs))
                    print('data and architecture loaded...')

                    # loading model parameters
                    model_path = os.path.join(model_dir, model_name)
                    try:  # if model state_dict saved
                        model = pargs['net']  # getting model class
                        model.load_state_dict(torch.load(model_path))
                        print('model parameters loaded...')
                        model.eval()
                    except AttributeError:  # if model saved
                        model = torch.load(model_path)
                        print('model loaded...')
                        model.eval()

                    # note inconsistent prediction for test and val is due to dropout
                    test, val = pargs['test'], pargs['val']

                    # calculating metrics and updating xarray
                    print(f'\nRecalculating metrics for model {model_name}')
                    for metric in metrics:
                        for eval_func in [test, val]:
                            predicted, true, _ = eval_func()  # get prediction and ground truth

                            for out_n, out_x in enumerate(outcome):
                                r, p = metric(predicted[:, out_n], true[:, out_n])  # calculate metric
                                mname_r = 'recalc_' + namestr(metric, globals())  # get metric names
                                mname_p = mname_r[:-1] + 'p'

                                # add coordinates to performance xarray
                                for met_name in [mname_r, mname_p]:
                                    if met_name not in performance.metrics:
                                        # create nan-fill to broadcast into xarray
                                        null_fill = np.ones_like(performance.loc[dict(set=namestr(eval_func, locals()),
                                                                                      cv_fold=fold,
                                                                                      outcome=out_x,
                                                                                      metrics=performance.metrics[
                                                                                          0])]) * np.nan

                                        null_fill = xr.DataArray(null_fill[:, np.newaxis], dims=['epoch', 'metrics'],
                                                                 coords=dict(metrics=[met_name],
                                                                             epoch=range(len(performance.epoch))))

                                        performance, _ = xr.broadcast(performance,
                                                                      null_fill)  # add new metric name to xarray

                                        # met_name = np.array([met_name])
                                        # performance = performance.assign_coords(met_name=("metrics", met_name))

                                # add metrics to performance xarray
                                performance.loc[dict(set=namestr(eval_func, locals()), cv_fold=fold,
                                                     epoch=pargs[f'best_test_epoch_fold_{fold}'], outcome=out_x,
                                                     metrics=[mname_r, mname_p])] = [r, p]

                    save_path = os.path.join(perf_dir, 'recalc', 'recalc_' + perf_name[0])
                    print(f'\nSaving {save_path}')
                    # TODO: debug saving error, issue with xr.broadcast
                    performance.to_netcdf(save_path)  # saving recalculated metrics to new file


model_dir = 'models'
perf_dir = 'performance/BNCNN/personality_single_outcome'
metrics = [spearmanr, pearsonr]

recalc_metrics(metrics=metrics, model_dir=model_dir, perf_dir=perf_dir)
