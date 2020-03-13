import os

from preprocessing.degrees_of_freedom import *
from preprocessing.preproc_funcs import *
from preprocessing.read_data import restricted, behavioral, subnums, cdata

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

# Age, Sex, and Confound info
weight = np.array(restricted['Weight'])
height = np.array(restricted['Height'])
sleep_quality = np.array(behavioral['PSQI_Score'])  # larger score indicates worse quality of sleep
handedness = np.array(restricted['Handedness'])  # larger score indicates worse quality of sleep
gender = np.array(behavioral['Gender'])
ages = np.array(restricted['Age_in_Yrs'])

# Personality info
ffi_O = np.array(behavioral['NEOFAC_O'])
ffi_C = np.array(behavioral['NEOFAC_C'])
ffi_E = np.array(behavioral['NEOFAC_E'])
ffi_A = np.array(behavioral['NEOFAC_A'])
ffi_N = np.array(behavioral['NEOFAC_N'])

# Finding indices of patients without FFI data
ffi_labels = ['O', 'C', 'E', 'A', 'N']
ffis = [ffi_O, ffi_C, ffi_E, ffi_A, ffi_N]
ffi_nansubs = []
for i, x in enumerate(ffis):
    ffi_nansubs.append(np.where(np.isnan(x))[0])
ffi_nansubs = np.unique(ffi_nansubs)
ffis = np.array(ffis).T

# transforming 'gender' to dummy variables
dummyv = [0, 1]  # dummy variables
for i, x in enumerate(np.unique(gender)):  # quantifying gender
    gen = np.where(gender == x)[0]
    gender[gen] = dummyv[i]


## Defining train, test, validaton sets
# 70-15-15 train test validation split
train_ind = np.where(np.isin(subnums, final_train_list))[0]
test_ind = np.where(np.isin(subnums, final_test_list))[0]
val_ind = np.where(np.isin(subnums, final_val_list))[0]
print(
    f'{train_ind.shape + test_ind.shape + val_ind.shape} subjects total included in test-train-validation sets ({len(train_ind) + len(test_ind) + len(val_ind)} total)...\n')

confounds = [ages, weight, height, sleep_quality, handedness]  # defining confounds

# scaling confounds ONLY according to train set
if scaled:
    confounds = [x / np.max(np.abs(x[train_ind])) for _, x in enumerate(confounds)] # TODO: note this won't make sense for multiclass w/ 3+ classes

# finding indices of patients without confound data
con_nansubs = []
for i, x in enumerate(confounds):
    con_nansubs.append(np.where(pd.isnull(x))[0])

# Setting variable for network to predict
if predicted_outcome == 'neuro':
    outcome = ffi_N
elif predicted_outcome == 'open':
    outcome = ffi_O
elif predicted_outcome == 'allFFI':
    outcome = ffis
elif predicted_outcome == 'age':
    outcome = ages
elif predicted_outcome == 'sex':
    outcome = gender

###############################################################
# # Deconfounding X and Y for data classes
###############################################################
# TODO: Test data is actually deconfounded.
#  Lack of SVM learning about age with age deconfounded is necessary but not sufficient result

saved_dc_x = f'data/transformed_data/deconfounded/{list(dataDirs.keys())[list(dataDirs.values()).index(dataDir)]}{scl}_{deconfound_flavor}_{predicted_outcome}_x.npy'
saved_dc_y = f'data/transformed_data/deconfounded/{list(dataDirs.keys())[list(dataDirs.values()).index(dataDir)]}{scl}_{deconfound_flavor}_{predicted_outcome}_y.npy'

if deconfound_flavor == 'X1Y1' or deconfound_flavor == 'X1Y0':  # If we have data to deconfound...
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
        X_corr, Y_corr, nan_ind = deconfound_dataset(data=cdata, confounds=confounds,
                                                     set_ind=train_ind, outcome=outcome)

        dcdata = X_corr  # DeConfounded DATA
        # np.save(saved_dc_x, dcdata)  # saving deconfounded data

        if deconfound_flavor == 'X1Y1':  # load deconfounded Y data
            Y = Y_corr
            # np.save(saved_dc_y, Y)

        elif deconfound_flavor == 'X1Y0':
            Y = outcome

    cdata = dcdata  # using deconfounded data for further analyses

elif deconfound_flavor == 'X0Y0':  # no changes to X data
    Y = outcome
    # np.save(saved_dc_x, cdata)
    # np.save(saved_dc_y, Y)

##########################################
## Setting up multiclass classification ##
##########################################

if multiclass and one_hot:  # Sets multiclass outcome as one-hot encoded targets
    Y_classes = np.zeros((Y.squeeze().shape[0], len(np.unique(Y))))
    for i, x in enumerate(np.unique(Y)):
        Y_classes[[np.where(Y == x)[0]], i] = 1
    Y = Y_classes

Y = Y.astype(float)  # ensuring Y is not of type object
###################################################################
# # Projecting matrices into positive definite
###################################################################

if data_to_use == 'positive definite' or data_to_use == 'tangent':
    # Test all matrices for positive definiteness
    num_notPD, which = areNotPD(cdata)
    print(f'There are {num_notPD} non-PD matrices in {dataDir}...\n')

    # If data set has non-PD matrices, convert to closest PD matrix
    if num_notPD != 0:
        saved_pd = f'data/transformed_data/positive_definite/{list(dataDirs.keys())[list(dataDirs.values()).index(dataDir)]}{scl}_PD.npy'
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
    saved_tan = f'data/transformed_data/tangent/{list(dataDirs.keys())[list(dataDirs.values()).index(dataDir)]}{scl}_tangent.npy'
    if os.path.isfile(saved_tan):
        print('Loading saved tangent space matrices ...\n')
        tdata = np.load(saved_tan)
    else:
        print('Transforming all matrices into tangent space ...')
        tdata = tangent_transform(pddata, ref=tan_mean)
        np.save(saved_tan, tdata)

####################################
# CHOOSING WHICH TYPE OF DATA TO USE
####################################

if data_to_use == 'tangent':
    X = tdata  # Y already chosen by deconfound_flavor
    del (cdata, pddata, tdata)
elif data_to_use == 'untransformed':
    X = cdata
elif data_to_use == 'positive definite':
    X = pddata
    del (cdata, pddata)

# # exporting outcome as .xlsx file for parvathy's code
# import pandas as pd
# df = pd.DataFrame({'Subject': Subject.astype(int), 'Gender':Y[:,0]})
# filepath = f'parvathy/Code/Data_labels/{list(dataDirs.keys())[list(dataDirs.values()).index(dataDir)]}{scl}_{deconfound_flavor}_{predicted_outcome}_y.xlsx'
# df.to_excel(filepath, index=False)

############################################################################################
# # TODO: implement shrinking of tangent space data, ?implement optimal shrinkage parameter
############################################################################################
# from sklearn.covariance import LedoitWolf
# cov = LedoitWolf().fit()  # must be fit with saamples x features
# shrink = .7 # arbitrary
# regcov = (1-shrink) * cov.covariance_ + shrink * np.trace(cov)/len(cov) * np.identity(len(cov) # regularized covariance
# stdata =

