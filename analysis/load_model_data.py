import os

from preprocessing.degrees_of_freedom import *
from preprocessing.preproc_funcs import *
from preprocessing.read_data import subnums, cdata

cdata['Gender'] = pd.get_dummies(cdata.Gender.values).F.to_xarray()  # transforming 'gender' to dummy variables
outcome = np.array([cdata[x].values for x in predicted_outcome],
                   dtype=float).squeeze().T  # Setting variable for network to predict

###############################################################
# Defining train, test, validaton sets with Parvathy's partitions
###############################################################
final_test_list = np.loadtxt('Subject_Splits/final_test_list.txt')
final_train_list = np.loadtxt('Subject_Splits/final_train_list.txt')
final_val_list = np.loadtxt('Subject_Splits/final_val_list.txt')
train_subs, train_ind, _ = np.intersect1d(subnums, final_train_list, return_indices=True)
val_subs, val_ind, _ = np.intersect1d(subnums, final_val_list, return_indices=True)
test_subs, test_ind, _ = np.intersect1d(subnums, final_test_list, return_indices=True)

print(f'{train_ind.shape + test_ind.shape + val_ind.shape} subjects total included in test-train-validation sets '
      f'({len(train_ind) + len(test_ind) + len(val_ind)} total)...\n')

###############################################################
# # Deconfounding X and Y for data classes
###############################################################
dir_str = '_'.join(chosen_dir)
saved_dc_x = f'data/transformed_data/deconfounded/{dir_str}{scl}_{deconfound_flavor}_{"_".join(predicted_outcome)}_x.npy'
saved_dc_y = f'data/transformed_data/deconfounded/{dir_str}{scl}_{deconfound_flavor}_{"_".join(predicted_outcome)}_y.npy'

if deconfound_flavor == 'X1Y1' or deconfound_flavor == 'X1Y0':  # If we have data to deconfound...

    # Extracting confounds
    if not not confound_names:
        for c in confound_names:  # removing nan-valued confounds
            cdata = cdata.dropna(dim=c)
        confounds = cdata[confound_names]

        if scale_confounds:  # scaling confounds ONLY according to train set
            confounds = [x / np.max(np.abs(x[train_ind])) for x in confounds]

    if os.path.isfile(saved_dc_x):  # if data has already been deconfounded, load it
        print('Loading saved deconfounded matrices ...\n')
        dcdata = np.load(saved_dc_x)
        # train_ind, test_ind, val_ind = np.load(saved_dc_inds, allow_pickle=True)

        if os.path.isfile(saved_dc_y):
            Y = np.load(saved_dc_y)
        else:
            Y = outcome

    else:  # if data hasn't been deconfounded, deconfound it
        print('Deconfounding data ...\n')
        X_corr, Y_corr, nan_ind = deconfound_dataset(data=cdata, confounds=confounds,  # TODO: update cdata as an array
                                                     set_ind=train_ind, outcome=outcome)

        dcdata = X_corr  # DeConfounded DATA
        np.save(saved_dc_x, dcdata)  # saving deconfounded data

        if deconfound_flavor == 'X1Y1':  # load deconfounded Y data
            Y = Y_corr
            np.save(saved_dc_y, Y)

        elif deconfound_flavor == 'X1Y0':
            Y = outcome

    cdata = dcdata  # using deconfounded data for further analyses # TODO: assign dcddata to cdata xarray

elif deconfound_flavor == 'X0Y0':  # no changes to X data
    Y = outcome
    # np.save(saved_dc_x, cdata)
    # np.save(saved_dc_y, Y)

# Setting up multiclass classification with one-hot encoding
if multiclass and one_hot:
    Y = multiclass_to_onehot(Y).astype(float)  # ensures Y is not of type object

###################################################################
# # Projecting matrices into positive definite
###################################################################
# TODO: update for cdata as an xarray
if data_to_use == 'positive definite' or data_to_use == 'tangent':
    # Test all matrices for positive definiteness
    num_notPD, which = areNotPD(cdata)
    print(f'There are {num_notPD} non-PD matrices in {dir_str}...\n')

    # If data set has non-PD matrices, convert to closest PD matrix
    if num_notPD != 0:
        saved_pd = f'data/transformed_data/positive_definite/{dir_str}{scl}_PD.npy'
        if os.path.isfile(saved_pd):
            print('Loading saved positive definite matrices ...\n')
            pddata = np.load(saved_pd)
        else:
            print('Transforming non-PD matrices to nearest PD neighbor ...')
            pdmats = PD_transform(cdata[which])
            pddata = cdata.copy()
            pddata[which] = pdmats
            np.save(saved_pd, pddata)
    else:
        pddata = cdata

###################################################################
# # Projecting matrices into tangent space
###################################################################

if data_to_use == 'tangent':
    # If data set non-existent, projecting matrices into tangent space
    saved_tan = f'data/transformed_data/tangent/{dir_str}{scl}_tangent.npy'
    if os.path.isfile(saved_tan):
        print('Loading saved tangent space matrices ...\n')
        tdata = np.load(saved_tan)
    else:
        print('Transforming all matrices into tangent space ...')
        tdata = tangent_transform(pddata[train_ind], pddata, ref=tan_mean)
        np.save(saved_tan, tdata)

####################################
# CHOOSING WHICH TYPE OF DATA TO USE
####################################

if data_to_use == 'tangent':
    X = tdata  # Y already chosen by deconfound_flavor
    del (cdata, pddata, tdata)
elif data_to_use == 'untransformed':
    X = cdata
    del cdata
elif data_to_use == 'positive definite':
    X = pddata
    del (cdata, pddata)

############################################################################################
# # TODO: implement shrinking of tangent space data, ?implement optimal shrinkage parameter
############################################################################################
# from sklearn.covariance import LedoitWolf
# cov = LedoitWolf().fit()  # must be fit with saamples x features
# shrink = .7 # arbitrary
# regcov = (1-shrink) * cov.covariance_ + shrink * np.trace(cov)/len(cov) * np.identity(len(cov) # regularized covariance
# stdata =

