from __future__ import print_function
import h5py
from os import listdir
from os.path import isfile, join
import pandas as pd
import re
import numpy as np
import scipy
import torch


def read_raw_data(dataDir):

    data = []  # Allocating list for connectivity matrices
    subnums = []  # subject IDs

    # Names of only the connectivity matrix files in the folder

    eb = 'data/edge_betweenness'
    if dataDir == eb:
        filenames = []

        for i, subdir in enumerate(listdir(f'{eb}')):
            subfold = listdir(f'{eb}/{subdir}')  # subfolders
            filtnames = list(filter(re.compile('HCP').match, subfold))
            filenames.extend(filtnames)
            # Reading in data from all 1003 subjects
            for _, filename in enumerate(filtnames):
                # print(filename)
                n1 = scipy.io.loadmat(f'{eb}/{subdir}/{filename}')['mat']
                data.append(n1)

        s = 13  # start index of subject ID in filename

    else:
        filenames = [f for f in listdir(dataDir) if isfile(join(dataDir, f))]
        s = 0  # start index of subject ID in filename

        # Reading in data from all 1003 subjects
        for i, filename in enumerate(filenames):
            hf = h5py.File(f'{dataDir}/{filename}', 'r')
            n1 = np.array(hf["CorrMatrix"][:])
            data.append(n1)

    # Associated subject numbers
    for i, filename in enumerate(filenames):
        subnums.append(int(filename[s:-4]))

    # Data as a numpy array
    data = np.array(data)

    # Demographic data
    behavioral = pd.read_csv('data/hcp/behavioral.csv')
    restricted = pd.read_csv('data/hcp/restricted.csv')

    # Specifying indices of overlapping subjects in 1206 and 1003 subject datasets
    subInd = np.where(np.isin(restricted["Subject"], subnums))[0]

    # Only using data from 1003
    restricted = restricted.reindex(subInd)
    behavioral = behavioral.reindex(subInd)

    print('Raw data processed...')

    return data, restricted, behavioral, subnums