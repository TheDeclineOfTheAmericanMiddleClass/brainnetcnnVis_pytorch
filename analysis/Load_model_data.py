import os

import numpy as np
import pandas as pd

from preprocessing.Analyze_raw_data import PD_transform
from preprocessing.CVCR_Deconfounding import deconfound_matrix
from preprocessing.Main_preproc import restricted, behavioral, subnums, cdata, toi, dataDir
from preprocessing.Preproc_funcs import reshape_deconfounded_matrix, areNotPD
from preprocessing.Tangent_transform import tangent_transform

# Parvathy's partitions
final_test_list = np.loadtxt('Subject_Splits/final_test_list.txt')
final_train_list = np.loadtxt('Subject_Splits/final_train_list.txt')
final_val_list = np.loadtxt('Subject_Splits/final_val_list.txt')
# print(len(final_train_list) + len(final_test_list) + len(final_val_list))

###############################################################
# # Reading in HCP interview data
###############################################################

# Info for all subjects (age, family number, subjectID)
Family_ID = restricted['Family_ID']  # for partitioning
Subject = np.array(restricted['Subject'])
ages = np.array(restricted['Age_in_Yrs'])

# Personality info
ffi_O = np.array(behavioral['NEOFAC_O'])
ffi_C = np.array(behavioral['NEOFAC_C'])
ffi_E = np.array(behavioral['NEOFAC_E'])
ffi_A = np.array(behavioral['NEOFAC_A'])
ffi_N = np.array(behavioral['NEOFAC_N'])

# # finding indices of patients without FFI data
ffi_labels = ['O', 'C', 'E', 'A', 'N']
ffis = [ffi_O, ffi_C, ffi_E, ffi_A, ffi_N]
ffi_nansubs = []
for i, x in enumerate(ffis):
    ffi_nansubs.append(np.where(np.isnan(x))[0])
ffis = np.array(ffis).T

# Confound info
weight = np.array(restricted['Weight'])
height = np.array(restricted['Height'])
sleep_quality = np.array(behavioral['PSQI_Score'])  # larger score indicates worse quality of sleep
handedness = np.array(restricted['Handedness'])  # larger score indicates worse quality of sleep
gender = np.array(behavioral['Gender'])

# transforming gender to dummy variables
dummyv = [-1, 1]  # dummy variables
for i, x in enumerate(np.unique(gender)):  # quantifying gender
    gen = np.where(gender == x)[0]
    gender[gen] = dummyv[i]

# confounds = [gender, weight, height, sleep_quality, handedness]
confounds = [ages, gender, weight, height, sleep_quality, handedness]
scaled_confounds = [x / np.max(x) for _, x in enumerate(confounds)]
confounds = scaled_confounds

# finding indices of patients without confound data
con_nansubs = []
for i, x in enumerate(confounds):
    con_nansubs.append(np.where(pd.isnull(x))[0])

## Defining train, test, validaton sets
# 70-15-15 train test validation split
train_ind = np.where(np.isin(subnums, final_train_list))[0]
test_ind = np.where(np.isin(subnums, final_test_list))[0]
val_ind = np.where(np.isin(subnums, final_val_list))[0]
print(f'{train_ind.shape + test_ind.shape + val_ind.shape} subjects total included in test-train-validation sets...\n')


###############################################################
# # Deconfounding X and Y for data classes
###############################################################
# TODO: Test data is actually deconfounded.
#  Lack of SVM learning about age with age deconfounded is necessary but not sufficient result

def deconfound_all(data,
                   tbd_ind,
                   confounds,
                   d_ind=train_ind,
                   outcome=ages):  # TODO: change as necessary for outcome of interest
    """
    Takes input of a dataset, its confounds, and returns the deconfounded dataset
    :param outcome: ground truth value to be deconfounded, per Y1
    :param cdata: Samples x symmetric matrices (row x column) to be deconfounded, per X1
    :param confounds: Confounds x samples, to be factored out of cdata
    :param d_ind: data indices of cdata from which deconfounding parameters will be calculated
    :return: List of deconfounded
    """

    X_corr = []
    Y_corr = []

    # confound parameter estimation for X and Y
    _, C_pi, b_hat_X, nan_ind = deconfound_matrix(data, confounds, set_ind=d_ind)
    Y_c = np.delete(outcome[d_ind], nan_ind, axis=0)  # Y confound
    b_hat_Y = C_pi @ Y_c  # the confound parameter estimation

    for i, x in enumerate(tbd_ind):
        C_tbd = np.vstack(confounds).astype(float).T[tbd_ind[i]]
        Xtbd_corr = cdata[tbd_ind[i]] - reshape_deconfounded_matrix(C_tbd @ b_hat_X, new_size=len(cdata[0]))

        Y_tbd = outcome[tbd_ind[i]]

        Ytbd_corr = Y_tbd - C_tbd @ b_hat_Y

        X_corr.extend(Xtbd_corr)
        Y_corr.extend(Ytbd_corr)

    train_ind = np.arange(len(tbd_ind[0]))
    test_ind = np.arange(len(tbd_ind[1])) + len(tbd_ind[0])
    val_ind = np.arange(len(tbd_ind[2])) + len(tbd_ind[0]) + len(tbd_ind[1])

    return np.array(X_corr), np.array(Y_corr), train_ind, test_ind, val_ind


X_corr, Y_corr, tr_i, te_i, v_i = deconfound_all(cdata, [train_ind, test_ind, val_ind],  # TODO: change cdata
                                                 confounds=confounds, d_ind=train_ind,
                                                 outcome=ffis)  # TODO: change this for different prediction

###################################################################
# # Projecting matrices into positive definite, then tangent space
###################################################################

# Test all matrices for positive definiteness
num_notPD, which = areNotPD(X_corr)
print(f'There are {num_notPD} non-PD matrices in {dataDir}...\n')

# If data set has non-PD matrices, convert to closest PD matrix
if num_notPD != 0:
    saved_pd = f'data/transformed_data/positive_definite/scaled_deconfounded_ages_pd_{toi}_{dataDir[27:]}.npy'  # TODO: change this on running new datasets
    if os.path.isfile(saved_pd):
        print('Loading saved positive definite matrices ...\n')
        pddata = np.load(saved_pd)
    else:
        print('Transforming non-PD matrices to nearest PD neighbor ...\n')
        pdmats = PD_transform(cdata[which])
        pddata = cdata.copy()
        pddata[which] = pdmats
        np.save(saved_pd, pddata)
else:
    pddata = X_corr

# If data set non-existent, projecting matrices into tangent space
saved_tan = f'data/transformed_data/tangent/scaled_deconfounded_ages_tan_{toi}_{dataDir[27:]}.npy'  # TODO: change this on running new datasets
if os.path.isfile(saved_tan):
    print('Loading saved tangent space matrices ...\n')
    tdata = np.load(saved_tan)
else:
    print('Transforming all matrices into tangent space ...\n')
    tdata = tangent_transform(pddata, ref='euclidean')
    np.save(saved_tan, tdata)

data = tdata
del (cdata, pddata)

# # File path code
# e = everything
# eba = everything but age as deconfounder


############################################################################################
# # TODO: implement shrinking of tangent space data, ?implement optimal shrinkage parameter
############################################################################################
# from sklearn.covariance import LedoitWolf
# cov = LedoitWolf().fit()  # must be fit with saamples x features
# shrink = .7 # arbitrary
# regcov = (1-shrink) * cov.covariance_ + shrink * np.trace(cov)/len(cov) * np.identity(len(cov) # regularized covariance
# stdata =
