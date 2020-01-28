from __future__ import print_function
import h5py
from os import listdir
from os.path import isfile, join
import re
import numpy as np
import scipy


def read_raw_data(dataDir):
    data = []  # Allocating list for connectivity matrices
    subnums = []  # subject IDs

    # Names of only the connectivity matrix files in the folder

    eb = 'data/edge_betweenness'
    if dataDir == eb:
        filenames = []  # TODO: ensure these filenames get sorted

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
        filenames.sort()
        s = 0  # start index of subject ID in filename

        for i, filename in enumerate(filenames):
            if filenames[0].endswith('.npy'):
                data = list(np.load(f'{dataDir}/{filename}'))

            elif filenames[0].endswith('.txt'):  # for .txt file
                tf = np.loadtxt(f'{dataDir}/{filename}')
                data.append(tf)

            else:  # for .h5py files
                hf = h5py.File(f'{dataDir}/{filename}', 'r')
                n1 = np.array(hf["CorrMatrix"][:])
                data.append(n1)

        # # for netmats1.txt processing...
        # # changing filenames for 1003 subject numbers to readable
        # if data.shape[0] == 1003:
        #     dataDir = 'data/3T_HCP1200_MSMAll_d300_ts2_RIDGE'
        #     filenames = [f for f in listdir(dataDir) if isfile(join(dataDir, f))]


    # Associated subject numbers
    for i, filename in enumerate(filenames):
        subnums.append(int(filename[s:-4]))

    # Data as a numpy array
    data = np.array(data)

    print('Raw data processed...\n')

    return data, subnums
