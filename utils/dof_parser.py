import argparse

from utils.var_names import HCP268_tasks, data_directories, predict_choices

# adding name to parser
parser = argparse.ArgumentParser(description="train multiple personality-predicting models on HCP data")

parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")

# degrees of freedom in the model input/output
in_out = parser.add_argument_group('in_out', 'model I/O params')
in_out.add_argument("-po", "--predicted_outcome", choices=predict_choices, type=str, action='append',
                    help="the outcome to predict")
in_out.add_argument("-cd", "--chosen_dir", choices=list(data_directories.keys()), default=['HCP_alltasks_268'], nargs=1,
                    help='the data directories')
in_out.add_argument("-ct", "--chosen_tasks", type=str, choices=list(HCP268_tasks.keys()),
                    action='append', help="the HCP268_tasks to train on")
in_out.add_argument("-mo", "--model", type=str, choices=['BNCNN', 'SVM'], default='BNCNN', help='the model to use',
                    nargs=1)

in_out.add_argument('--architecture',
                    choices=['usama', 'yeo_sex', 'kawahara', 'usama', 'parvathy_v2', 'FC90Net', 'yeo_58'],
                    default='usama', help='brainnetcnn architecture', nargs=1)

# data transformation args
transforms = parser.add_argument_group('transforms', 'data transformation params')
transforms.add_argument('--transformations', choices=['positive definite', 'untransformed', 'tangent'],
                        default='untransformed', help='data transformations to apply', nargs=1)
transforms.add_argument('--deconfound_flavor', choices=['X1Y0', 'X1Y1', 'X0Y0'], default='X0Y0',
                        help='deconfounding method',
                        nargs=1)
transforms.add_argument("-sc", "--scale_confounds", action='store_true', default=False,
                        help='apply uniform scaling to confounds before deconfounding')

# personality clustering parameters
clustering = parser.add_argument_group('clustering', 'Gerlach personality clustering params')
clustering.add_argument('--Q', default=5, type=int, help='number of feature used to fit clustering GMM', nargs=1)
clustering.add_argument('--dataset_to_cluster', choices=['IMAGEN', 'HCP'], default='HCP',
                        help='the NEO-FFI dataset from generate clusters from', nargs=1)

# optional training hyperparameters
hyper = parser.add_argument_group('hyper', 'model training hyperparams')
hyper.add_argument('--optimizer', choices=['sgd', 'adam'], default='sgd', help='optimization strategy', nargs=1)
hyper.add_argument('--momentum', default=0.9, type=float, help='momentum', nargs=1)
hyper.add_argument('--lr', default=0.9, type=float, help='learning rate', nargs=1)
hyper.add_argument('--wd', default=.0005, type=float, help=' weight decay', nargs=1)
hyper.add_argument('--max_norm', default=1.5, type=float,
                   help='maximum value of normed gradients, to prevent explosion/vanishing', nargs=1)

# params for epochs to train over
epochs = parser.add_argument_group('epochs', 'training iterations params')
epochs.add_argument('--nbepochs', default=300, type=int, help='max epochs to train BNCNN over', nargs=1)
epochs.add_argument('-ea', '--early', action='store_true', help='early stopping')
epochs.add_argument('--cv_folds', default=5, type=int, help='cross validation folds for SVM', nargs=1)

# # SETTING CONDITIONAL VARIABLES
uncond_args = parser.parse_args()  # parsing unconditional variables first
logic = parser.add_argument_group('pipeline_logic', 'necessary vars for pipeline logic')

if uncond_args.transformations == 'tangent':
    transforms.add_argument('--tan_mean', choices=['euclidean', 'harmonic'], default='euclidean', nargs=1)
if uncond_args.early:
    epochs.add_argument('--ep_int', type=int, default=5, help='if no improvement after {ep_int} epochs, stop early',
                        nargs=1)
    epochs.add_argument('--min_ep', default=50, type=int, help='mininmum epochs to train before early stopping',
                        nargs=1)
if uncond_args.deconfound_flavor != 'X0Y0':
    transforms.add_argument('--confound_names',
                            choices=['Weight', 'Height', 'Handedness', 'Age_in_Yrs', 'PSQI_Score', None],
                            required=True, type=str, help='confounds to regress out of outcome', nargs='+')

if uncond_args.chosen_dir == 'Johann_mega_graph':  # TODO: make exclusive of BNCNN
    eb = in_out.add_argument("-eb", "--edge_betweenness", action='store_true',
                             help="if reading in data_directories['Johann_mega_graph'], include edgebetweeness vector")
    logic.add_argument('data_are_matrices', action='store_const', const=False, help='bool for transformations')
else:
    logic.add_argument('data_are_matrices', action='store_const', const=True, help='bool for transformations')

if uncond_args.scale_confounds:
    transforms.add_argument('scl', action='store_const', const='_scaled')
else:
    transforms.add_argument('scl', action='store_const', const='')

# logic for accurate calculation of multi_input
logic.add_argument('num_input', action='store_const',
                   const=len(uncond_args.chosen_dir) - 1 + len(uncond_args.chosen_tasks),
                   help='number of input matrices to train on')

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

# exit logic for mutually exclusive args
if 'HCP_alltasks_268' in uncond_args.chosen_dir and 'NA' in uncond_args.chosen_tasks:
    parser.exit('Please choose at least non-NA task(s) for HCP_alltasks_268')

if 'HCP_alltasks_268' not in uncond_args.chosen_dir and 'NA' not in uncond_args.chosen_tasks:
    parser.exit('Please set variable chosen_tasks to [\'NA\'] if chosen_dir != HCP_alltasks_268.')

if uncond_args.model == 'BNCNN' and 'Johann_mega_graph' in uncond_args.chosen_dir:
    parser.exit('BNCNN cannot train on non-matrix data (e.g. Johann_mega_graph)')

args = parser.parse_args()  # parsing unconditional and conditional args

print(args)

# # training models below
if args.verbose:
    print(
        f"Training {args.model} to predict {args.predicted_outcome} from {args.chosen_dir}_{args.chosen_tasks} data...")

# if args.model == 'BNCNN':
#     from preprocessing import read_data
#     from analysis import load_model_data, define_models, init_model, train_model
#
#
# elif args.model == 'SVM':
#     from analysis import train_shallow_networks
#
#     train_shallow_networks.main()  # TODO: have shallow_models save results
