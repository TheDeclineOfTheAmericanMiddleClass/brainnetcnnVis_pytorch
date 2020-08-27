import argparse

from utils.var_names import HCP268_tasks, data_directories, predict_choices

# adding name to parser
parser = argparse.ArgumentParser(description="train multiple personality-predicting models on HCP data")

parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
parser.add_argument('-pp', "--plot_performance", help='plot results of BNCNN training', action='store_true')

# degrees of freedom in the model input/output
in_out = parser.add_argument_group('in_out', 'model I/O params')
in_out.add_argument("-po", "--predicted_outcome", required=True, choices=predict_choices, type=str, action='append',
                    help="the outcome to predict")
in_out.add_argument("-cd", "--chosen_dir", required=True, choices=list(data_directories.keys()), nargs=1,
                    help='the data directories')
in_out.add_argument("-ct", "--chosen_tasks", required=True, choices=list(HCP268_tasks.keys()), type=str,
                    action='append', help="the HCP268_tasks to train on")
in_out.add_argument("-mo", "--model", required=True, choices=['BNCNN', 'SVM', 'Gerlach_cluster', 'FC90', 'ElasticNet'],
                    type=str,
                    help='the model to use', nargs=1)

in_out.add_argument('--architecture',
                    choices=['usama', 'yeo_sex', 'kawahara', 'usama', 'parvathy_v2', 'FC90Net', 'yeo_58'],
                    default='usama', help='brainnetcnn architecture', nargs='?')

# data transformation args
transforms = parser.add_argument_group('transforms', 'data transformation params')
transforms.add_argument('--transformations', choices=['positive definite', 'untransformed', 'tangent'],
                        default='untransformed', help='data transformations to apply', nargs='?')
transforms.add_argument('--deconfound_flavor', choices=['X1Y0', 'X1Y1', 'X0Y0'], default='X0Y0',
                        help='deconfounding method',
                        nargs='?')
transforms.add_argument("-sc", "--scale_confounds", action='store_true', default=False,
                        help='apply uniform scaling to confounds before deconfounding')

# optional training hyperparameters
hyper = parser.add_argument_group('hyper', 'model training hyperparams')
hyper.add_argument('--optimizer', choices=['sgd', 'adam'], default='sgd', help='optimization strategy', nargs='?')
hyper.add_argument('--momentum', default=0.9, type=float, help='momentum', nargs='?')
hyper.add_argument('--lr', default=0.00001, type=float, help='learning rate',
                   nargs='?')  # TODO: note learning rate .9 caused gradient issues
hyper.add_argument('--wd', default=.0005, type=float, help=' weight decay', nargs='?')
hyper.add_argument('--max_norm', default=1.5, type=float,
                   help='maximum value of normed gradients, to prevent explosion/vanishing', nargs='?')

# params for epochs to train over
epochs = parser.add_argument_group('epochs', 'training iterations params')
epochs.add_argument('--nbepochs', default=300, type=int, help='max epochs to train BNCNN over', nargs='?')
epochs.add_argument('-ea', '--early', action='store_true', help='early stopping')
epochs.add_argument('--cv_folds', default=5, type=int, help='cross validation folds for SVM', nargs='?')

# personality clustering parameters
clustering = parser.add_argument_group('clustering', 'Gerlach personality clustering params')
clustering.add_argument('--Q', default=5, type=int, help='number of feature used to fit clustering GMM', nargs='?')
clustering.add_argument('--dataset_to_cluster', choices=['IMAGEN', 'HCP'], default='HCP',
                        help='the NEO-FFI dataset from generate clusters from', nargs='?')

# # SETTING CONDITIONAL VARIABLES
uncond_args = parser.parse_args()  # parsing unconditional variables first
logic = parser.add_argument_group('pipeline_logic', 'necessary vars for pipeline logic')

if uncond_args.model in [['BNCNN'], ['SVM']]:
    if uncond_args.transformations == 'tangent':
        transforms.add_argument('--tan_mean', choices=['euclidean', 'harmonic'], default='euclidean', nargs='?')

    if uncond_args.early:
        epochs.add_argument('--ep_int', type=int, default=5, help='if no improvement after {ep_int} epochs, stop early',
                            nargs='?')
        epochs.add_argument('--min_ep', default=50, type=int, help='mininmum epochs to train before early stopping',
                            nargs='?')
        # note: f'{}' string formatting doesn't work with every terminal. ?must be python 3 compatible
        epochs.add_argument('early_str', action='store_const', const=f'es{epochs.ep_int}')
    else:
        epochs.add_argument('early_str', action='store_const', const='')

    if uncond_args.deconfound_flavor != 'X0Y0':
        transforms.add_argument('--confound_names',
                                choices=['Weight', 'Height', 'Handedness', 'Age_in_Yrs', 'PSQI_Score', None],
                                required=True, type=str, help='confounds to regress out of outcome', nargs='+')

    if uncond_args.chosen_dir == 'Johann_mega_graph':
        in_out.add_argument("-eb", "--edge_betweenness", action='store_true',
                            help="if reading in data_directories['Johann_mega_graph'], include edgebetweeness vector")
        logic.add_argument('data_are_matrices', action='store_const', const=False, help='bool for transformations')
    else:
        logic.add_argument('data_are_matrices', action='store_const', const=True, help='bool for transformations')

    if uncond_args.scale_confounds:
        transforms.add_argument('scl', action='store_const', const='scaled')
    else:
        transforms.add_argument('scl', action='store_const', const='')

    if uncond_args.predicted_outcome == [f'softcluster_{i}' for i in range(1, 14)]:  # if predicting on all clusters
        in_out.add_argument('po_str', action='store_const', const='softcluster_all')
    elif all(
            [x in uncond_args.predicted_outcome for x in ['NEOFAC_O', 'NEOFAC_C', 'NEOFAC_E', 'NEOFAC_A', 'NEOFAC_N']]):
        print('Using all NEOFAC dimensions...')
        in_out.add_argument('po_str', action='store_const', const='NEOFAC_all')
    else:
        in_out.add_argument('po_str', action='store_const', const='_'.join(uncond_args.predicted_outcome))

# logic for accurate calculation of multi_input
try:
    logic.add_argument('num_input', action='store_const',
                       const=len(uncond_args.chosen_dir) - 1 + len(uncond_args.chosen_tasks),
                       help='number of input matrices to train on')
except TypeError:
    parser.exit('Chosen tasks (-ct) and/or chosen directory (-cd) must not be None.')

if (len(uncond_args.chosen_dir) - 1 + len(uncond_args.chosen_tasks)) == 1:
    logic.add_argument('multi_input', action='store_const', const=False, help='bool for multi input training')
else:
    logic.add_argument('multi_input', action='store_const', const=True, help='bool for multi input training')


# names for xarray data variables
chosen_Xdatavars = uncond_args.chosen_dir.copy()
if 'HCP_alltasks_268' in chosen_Xdatavars:
    chosen_Xdatavars.remove('HCP_alltasks_268')
    for task in uncond_args.chosen_tasks:
        datavar = f'HCP_alltasks_268_{task}'
        chosen_Xdatavars.append(datavar)
logic.add_argument('chosen_Xdatavars', action='store_const', const=chosen_Xdatavars,
                   help='names for xarray data variables')

# # 2nd round of parsing, to set variable conditional on the conditional variables
cond_args = parser.parse_args()

# If all tasks, use simplified name HCP_alltasks_268
if all([x in cond_args.chosen_tasks for x in list(HCP268_tasks.keys())[:-1]]):
    print('Using all HCP 268 tasks...')
    in_out.add_argument('cXdv_str', action='store_const', const='HCP_alltasks_268_all')
else:
    in_out.add_argument('cXdv_str', action='store_const', const='_'.join(cond_args.chosen_Xdatavars))

# exit logic for mutually exclusive args
if 'HCP_alltasks_268' in uncond_args.chosen_dir and 'NA' in uncond_args.chosen_tasks:
    parser.exit('Please choose at least non-NA task(s) for HCP_alltasks_268')

if 'HCP_alltasks_268' not in uncond_args.chosen_dir and 'NA' not in uncond_args.chosen_tasks:
    parser.exit('Please set variable chosen_tasks to [\'NA\'] if chosen_dir != HCP_alltasks_268.')

if uncond_args.model == 'BNCNN' and 'Johann_mega_graph' in uncond_args.chosen_dir:
    parser.exit('BNCNN cannot train on non-matrix data (e.g. Johann_mega_graph)')

args = parser.parse_args()  # parsing unconditional and conditional args
pargs = vars(args)  # dict of passed args


# # printing i/o of model
if args.verbose:
    print(f"\nTraining {args.model} to predict {args.predicted_outcome} from "
          f"{args.chosen_dir} directory, with tasks: {args.chosen_tasks} data...\n")
    print(pargs, '\n')


# # training models
if args.model == ['BNCNN']:
    from preprocessing import read_data

    print('reading data...')
    pargs.update(read_data.main(pargs))

    from analysis import load_model_data, init_model, train_model

    print('loading model data...')
    pargs.update(load_model_data.main(pargs))

    from analysis import define_models  # import must directly precede define_models.main()

    print('defining models...')
    pargs.update(define_models.main(pargs))
    print('initializing model...')
    pargs.update(init_model.main(pargs))
    print('training model...')
    pargs.update(train_model.main(pargs))

    print('\nBNCNN training done!\n')

    if args.plot_performance:
        from display_results import plot_model_results
        plot_model_results.main(args)

elif args.model == ['SVM']:

    from preprocessing import read_data

    print('reading data...')
    pargs.update(read_data.main(pargs))

    from analysis import load_model_data, train_shallow_networks

    print('loading model data...')
    pargs.update(load_model_data.main(pargs))
    print('training SVM...')
    train_shallow_networks.main(pargs)
    print('\nSVM training done!\n')

elif args.model == ['Gerlach_cluster']:
    from preprocessing import gerlach_clustering

    print('clustering data...')
    gerlach_clustering.main(pargs)
