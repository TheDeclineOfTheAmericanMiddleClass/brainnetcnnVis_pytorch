import os
import zipfile
from os import listdir

import nibabel as nib
import numpy as np

float_formatter = "{:.3f}".format
np.set_printoptions(formatter={'float_kind': float_formatter})


# reading HCP subject-specific CIFTI files
def read_CIFTI():
    datapath = '/raid/projects/Adu/brainnetcnnVis_pytorch/data/HCP_S1200_PTNmaps_d300'
    cift_dir = listdir(datapath)
    img = nib.load(datapath + '/' + cift_dir[0])
    hdr = img.header

    test = nib.Cifti2Image(img)
    test.shape  # (300 x 91282)


# tty = np.loadtxt('/raid/projects/Adu/brainnetcnnVis_pytorch/data/NodeTimeseries_3T_HCP1200_MSMAll_d300_ts2/100206.txt')
# tty.shape  # (4800 x 300), rfMRI runs(4800 total timepoints).

# unpacking .nii.gz files from .zips
def unpack_HCP_zips(zipsfolder, LR_path, RL_path):
    """
    Unpacks HCP emotion_face NIFTI files to new folders for each scan gradient, for each subject in zipsfolder.
    :param zipsfolder: folder containing zipped HCP task data
    :param LR_path: path to save LR gradient data
    :param RL_path: path to save RL gradient data
    :return:
    """
    listing = os.listdir(zipsfolder)
    for files in listing:
        if files.endswith('_3T_tfMRI_EMOTION_preproc.zip'):
            with open(zipsfolder + files, 'rb') as fileobj:
                with zipfile.ZipFile(fileobj) as z:
                    subject = z.filename[len(zipsfolder):int(len(zipsfolder) + 6)]
                    for zip_info in z.infolist():
                        if zip_info.filename[-22:].endswith('fMRI_EMOTION_LR.nii.gz'):
                            zip_info.filename = subject + '_' + os.path.basename(zip_info.filename)
                            z.extract(zip_info, path=LR_path)
                            print(f'Subject {subject} LR.nii.gz data extracted...')
                        if zip_info.filename[-22:].endswith('fMRI_EMOTION_RL.nii.gz'):
                            zip_info.filename = subject + '_' + os.path.basename(zip_info.filename)
                            z.extract(zip_info, path=RL_path)
                            print(f'Subject {subject} RL.nii.gz data extracted...')
            fileobj.close()


emo_folder = 'data/HCP.zips/'
emo_LR_nii = 'data/timeseries/raw/face/LR'
emo_RL_nii = 'data/timeseries/raw/face/RL'
emo_both = 'data/timeseries/raw/face/both'


# unpack_HCP_zips(emo_folder, LR_path=emo_LR_nii, RL_path=emo_RL_nii)


# read in and concatenate LR-RL NIFTI files
def read_NIFTI():
    global img

    gz_listing = os.listdir(emo_LR_nii)
    for files in gz_listing[0:1]:
        img = nib.load('/'.join([emo_LR_nii, files]))  # (91, 109, 91, 176)


read_NIFTI()
