from __future__ import print_function

import glob
import re
from os import listdir
from os.path import isfile, join

import h5py
import numpy as np
from scipy import io


def read_raw_data(dataDir, actually_read=True):  # TODO: remove actually_read if it serves no purpose
    data = []  # Allocating list for connectivity matrices
    subnums = []  # subject IDs

    # Names of only the connectivity matrix files in the folder
    eb = 'data/edge_betweenness'
    if dataDir == eb:  # for .mat file
        filenames = []  # TODO: ensure these filenames get sorted
        if not not glob.glob(f'{dataDir}/{"*.npy"}'):
            for file in glob.glob(f'{dataDir}/{"*.npy"}'):
                if file == f'{dataDir}/eb_data.npy':
                    data = list(np.load(file))
                elif file == f'{dataDir}/eb_subnums.npy':
                    subnums = list(np.load(file))

        else:
            for i, subdir in enumerate(listdir(f'{eb}')):  # for every subdirectory
                subfold = listdir(f'{eb}/{subdir}')
                filtnames = list(filter(re.compile('HCP').match, subfold))  # index HCP matrices
                filtnames.sort()
                filenames.extend(filtnames)  # add them to list for later

                if actually_read:
                    # Reading in data from all 1003 subjects
                    for _, file in enumerate(filtnames):
                        n1 = io.loadmat(f'{eb}/{subdir}/{file}')['mat']
                        data.append(n1)

                        sn = re.findall(r'\d+', file)[-1][-6:]  # subject ID
                        subnums.append(sn)

    else:
        filenames = [f for f in listdir(dataDir) if isfile(join(dataDir, f))]
        filenames.sort()
        s = 0  # start index of subject ID in filename
        if actually_read:

            if not not glob.glob(f'{dataDir}/{"*.npy"}'):  # if numpy file of consolidated data saved, use that
                for file in glob.glob(f'{dataDir}/{"*.npy"}'):
                    data = list(np.load(file))

            elif not not glob.glob(f'{dataDir}/{"*.txt"}'):  # otherwise use .txt files
                for file in glob.glob(f'{dataDir}/{"*.txt"}'):
                    data.append(np.loadtxt(file))

            elif not not glob.glob(f'{dataDir}/{"*.mat"}'):  # otherwise use .h5py  (aka .mat) files
                for file in glob.glob(f'{dataDir}/{"*.mat"}'):
                    hf = h5py.File(file, 'r')

                    if np.any(np.isin(list(hf.keys()), "CorrMatrix")):  # HCP ICA300 ridge data
                        n1 = np.array(hf["CorrMatrix"][:])

                    elif np.any(np.isin(list(hf.keys()), "CORR")):  # Lea's HCP face data
                        n1 = np.array(hf["CORR"][:])
                        sn = np.array(hf['IDS'][:]).astype(int)
                        subnums.append(sn)  # taking subnums from file

                    data.append(n1)

            # for i, filename in enumerate(filenames):
            #     if filenames[0].endswith('.npy'):  # for .npy files, given priority
            #         data = list(np.load(f'{dataDir}/{filename}'))
            #
            #     elif filenames[0].endswith('.txt'):  # for .txt file
            #         # with open(f'{dataDir}/{filename}', 'rb') as f:
            #         #     tf = f.read().decode(errors='replace')
            #         tf = np.loadtxt(f'{dataDir}/{filename}')
            #         data.append(tf)
            #     else:  # for .h5py files
            #         hf = h5py.File(f'{dataDir}/{filename}', 'r')
            #         n1 = np.array(hf["CorrMatrix"][:])
            #         data.append(n1)

    # # for netmats1.txt processing...
        # # changing filenames for 1003 subject numbers to readable
        # if data.shape[0] == 1003:
        #     dataDir = 'data/3T_HCP1200_MSMAll_d300_ts2_RIDGE'
        #     filenames = [f for f in listdir(dataDir) if isfile(join(dataDir, f))]

    if not subnums:  # if subnums still an empty list
        for i, filename in enumerate(filenames):  # reading in subnums
            if filename.endswith(".txt") or filename.endswith(".mat"):
                num = re.findall(r'\d+', filename)  # find digits in filenames
                # print(num)
                subnums.append(num)

    subnums.sort()
    subnums = np.array(subnums).astype(float).squeeze()  # necessary for comparison to train-test-split partitions

    if actually_read:
        # Data as a numpy array
        data = np.array(data)
    else:
        data = []

    print('Raw data processed...\n')

    return data, subnums
