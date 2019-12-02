from __future__ import print_function
import h5py
from os import listdir
from os.path import isfile, join
import pandas as pd
import torch
import numpy as np

################# Reading in the data ##################
########################################################

# Everything to be put on a GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# setting data directory
# TODO: figure out whether ICA300 dataset created by Lea or HCP
dataDir = 'data/3T_HCP1200_MSMAll_d300_ts2_RIDGE'  # ICA FIX 300 from ?HCP/LEA
# dataDir = 'data/POWER_264_FIXEXTENDED_RIDGEP'

# Names of only the files in the folder
filenames = [f for f in listdir(dataDir) if isfile(join(dataDir, f))]

# Associated subject numbers
subnums = []
for i, filename in enumerate(filenames):
    subnums.append(int(filename[:-4]))

# Allocating list for connectivity matrices
data = []

# Reading in data from all 1003 subjects
for i, filename in enumerate(filenames):
    hf = h5py.File(f'{dataDir}/{filename}', 'r')
    # n1 = torch.tensor(hf["CorrMatrix"][:], device=device) # maybe just push to tensor after all tangent space transforms complete
    n1 = np.array(hf["CorrMatrix"][:])
    data.append(n1)

# Data as a numpy array
data = np.array(data)

# Demographic data
behavioral = pd.read_csv('data/hcp/behavioral.csv')
restricted = pd.read_csv('data/hcp/restricted.csv')

# Specifying indices of overlapping subjects in 1206 and 1003 subject datasets
subInd = np.where(np.isin(restricted["Subject"], subnums))[0]

# Only using data from 1003
restricted = restricted.reindex(subInd)
