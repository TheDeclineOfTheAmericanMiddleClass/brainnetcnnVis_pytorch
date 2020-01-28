import torch
import numpy as np
from preprocessing.Main_preproc import restricted, behavioral, subnums, data
from preprocessing.Preproc_funcs import reshape_deconfounded_matrix
from preprocessing.CVCR_Deconfounding import deconfound_matrix

# Parvathy's partitions
final_test_list = np.loadtxt('Subject_Splits/final_test_list.txt')
final_train_list = np.loadtxt('Subject_Splits/final_train_list.txt')
final_val_list = np.loadtxt('Subject_Splits/final_val_list.txt')

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

# Confound info
weight = np.array(restricted['Weight'])
height = np.array(restricted['Height'])
sleep_quality = np.array(behavioral['PSQI_Score'])  # larger score indicates worse quality of sleep
handedness = np.array(restricted['Handedness'])  # larger score indicates worse quality of sleep
gender = np.array(behavioral['Gender'])

dummyv = [-1, 1]  # dummy variables
for i, x in enumerate(np.unique(gender)):  # quantifying gender
    gen = np.where(gender == x)[0]
    gender[gen] = dummyv[i]

confounds = [gender, weight, height, sleep_quality, handedness]
# confounds = [ages, gender, weight, height, sleep_quality, handedness]


## Defining train, test, validaton sets
# 70-15-15 train test validation split
train_ind = np.where(np.isin(subnums, final_train_list))[0]
test_ind = np.where(np.isin(subnums, final_test_list))[0]
val_ind = np.where(np.isin(subnums, final_val_list))[0]

###############################################################
# # Deconfounding X and Y for data classes
###############################################################

# for training
C_train, C_pi_train, b_hat_trainX, nan_train = deconfound_matrix(data, confounds,
                                                                 set_ind=train_ind)
Xtrain_corr = np.delete(data[train_ind], nan_train, axis=0) - reshape_deconfounded_matrix(C_train @ b_hat_trainX,
                                                                                          new_size=len(data[0]))
Y_train = np.delete(ages[train_ind], nan_train, axis=0)
b_hat_trainY = C_pi_train @ Y_train  # the confound parameter estimation
Ytrain_corr = Y_train - C_train @ b_hat_trainY

# for testing
C_test, _, _, nan_test = deconfound_matrix(data, confounds,
                                           set_ind=test_ind)  # only using C_test, full deconfound function not necessary
Xtest_corr = np.delete(data[test_ind], nan_test, axis=0) - reshape_deconfounded_matrix(C_test @ b_hat_trainX,
                                                                                       new_size=len(data[0]))
Y_test = np.delete(ages[test_ind], nan_test, axis=0)
Ytest_corr = Y_test - C_test @ b_hat_trainY

# TODO: Test data is actually deconfounded
# In theory, if SVM doesn't learn anything about age with age deconfounded, this a necessary but not sufficient result

# # Choose ages as indices of parvathy's 1003 list that are in the current subject list
# pruned_ind = np.where(np.isin(np.array(Subject), subnums))[
#     0]  # pruned_ind redundant given import of restrictred/behavioral[subnums]
#
# ages = np.array(AiY)[pruned_ind]
