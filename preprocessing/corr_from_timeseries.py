###################################################################
# # Testing if the ridge regression data truly were from HCP or Lea
###################################################################

# import h5py
# import numpy as np
# testy = h5py.File('data/3T_HCP1200_MSMAll_d300_ts2_RIDGE/100206.mat')["CorrMatrix"][:]
# testy2 = np.loadtxt('data/HCP_created_ICA300_mats/pcorr/100206.txt')
# np.all(testy == testy2) # looks like from HCP
# isPD(testy)
#
# np.fill_diagonal(testy,1)
# isPD(testy)

########################################################
# # Creating new correlation matrices from HCP time series
########################################################


# # Creating correaltion matrices anew
# dataDir = 'data/HCP1200_ICA300_timeseries'
# saveDir = 'data/self_created_HCP_mats/corr.npy'
# c_type = 'corr'
#
# cdata, not_PD = create_connectivity(saveDir=saveDir,
#                              dataDir=dataDir, c_type=c_type)

# # Creating partial correaltion matrices with chosen rho
# dataDir = 'data/HCP1200_ICA300_timeseries'
# saveDir = 'data/self_created_HCP_mats/pcorr.npy'
# c_type = 'pcorr'
# rho = .5
#
# cdata, not_PD = create_connectivity(saveDir=saveDir,
#                              dataDir=dataDir, c_type=c_type, rho=rho)

# # Saving freshly calculated matrices as .txt files
# saveDir = 'data/self_created_HCP_mats/corr.npy'
# data = np.load(saveDir)
# psaveDir = 'data/self_created_HCP_mats/pcorr.npy'
# pdata = np.load(psaveDir)
#
# for i, x in enumerate(data):
#     np.savetxt(f'data/self_created_HCP_mats/ICA300_rho50_pcorr/{subnums[i]}.txt', pdata[i])
#     np.savetxt(f'data/self_created_HCP_mats/ICA300_corr/{subnums[i]}.txt', x)

###############################################
# # Deconfounding time series data
###############################################
#
# # Data and important indices
# dataDir = 'data/self_created_HCP_mats/ICA300_corr'
# # dataDir = 'data/self_created_HCP_mats/ICA300_rho50_pcorr' # 3 non-PD matrices
# cdata, subnums = read_mat_data(dataDir)
# train_subnums = np.loadtxt('subject_splits/train_subnums.txt')
# train_ind = np.where(np.isin(subnums, train_subnums))[0]
#
# # Confound info
# restricted, behavioral = read_dem_data(subnums)
#
# ages = np.array(restricted['Age_in_Yrs'])
# weight = np.array(restricted['Weight'])
# height = np.array(restricted['Height'])
# sleep_quality = np.array(behavioral['PSQI_Score'])  # larger score indicates worse quality of sleep
# handedness = np.array(restricted['Handedness'])  # larger score indicates worse quality of sleep
# gender = np.array(behavioral['Gender'])
#
# dummyv = [-1, 1]  # dummy variables
# for i, x in enumerate(np.unique(gender)):  # quantifying gender
#     gen = np.where(gender == x)[0]
#     gender[gen] = dummyv[i]
#
# confounds = [ages, gender, weight, height, sleep_quality, handedness]
#
# C, C_pi, b_hatX, nan_ind = deconfound_matrix(est_data=cdata, confounds=confounds, set_ind=train_ind)

#################################################
# # Consolidating each face task into .txt files TODO: OUTDATED! Please delete upon confirmation
#################################################
#
# import shutil
# import zipfile
# from os import listdir
# import nibabel as nib
#
# # Identifiying face task .zips
# filedir = 'data/HCP.zips'
# face_files = []
# for i, fil in enumerate(listdir(filedir)):
#     if fil.endswith('3T_tfMRI_EMOTION_preproc.zip'):
#         face_files.append(fil)
# face_files.sort()
#
# # releavant data_directories
# graddir = ['LR', 'RL']
# LRgz = 'MNINonLinear/Results/tfMRI_EMOTION_LR/tfMRI_EMOTION_LR.nii.gz'
# RLgz = 'MNINonLinear/Results/tfMRI_EMOTION_RL/tfMRI_EMOTION_RL.nii.gz'
# ts = [LRgz, RLgz]  # time series
#
# # Extracting face timeseries, concatenating LR/RL and writing to .txt file
# for i, fil in enumerate(face_files[0:1]):  # for each subject's face zip
#     subfilePath = f'{filedir}/{fil}'
#     with zipfile.ZipFile(subfilePath, 'r') as zip_ref:  # takes care of closing zip file
#         zip_ref.extractall('data/tmp')  # TODO: change so extracted to actual temp file
#
#         # allocate empty array to concatenate
#         for j, grad in enumerate(ts):
#             some_grad = f'data/tmp/{fil[0:6]}/{grad}'
#             img = nib.load(some_grad)
#             print(img.shape)
#             # with gzip.open(, 'rb') as f_in:
#             #     with open(f'data/timeseries/raw/face/{graddir[j]}/{fil[0:6]}.txt', 'wb') as f_out:
#             #         shutil.copyfileobj(f_in, f_out)
#
#         # concatenate LR ans RL nii timeseries
#         shutil.rmtree(f'data/tmp/{fil[0:6]}')  # delete temp file
#

import matplotlib.pyplot as plt


def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")

# epi_img_data = img.get_fdata()
# slice_0 = epi_img_data[45, :, :, 0]
# slice_1 = epi_img_data[:, 55, :, 0]
# slice_2 = epi_img_data[:, :, 45, 0]
# show_slices([slice_0, slice_1, slice_2])
# plt.suptitle("Center slices for EPI image")  # doctest: +SKIP
