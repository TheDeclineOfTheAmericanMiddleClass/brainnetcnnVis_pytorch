from __future__ import print_function

import glob
import inspect
import pickle
import re
import sys
from os import listdir
from os.path import isfile, join

import h5py
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
from numpy import linalg as la
from scipy import io
from scipy.linalg import logm, inv


# read subjects' demographic data
def read_dem_data(subnums):
    # Demographic data
    behavioral = pd.read_csv('data/hcp/behavioral.csv')
    restricted = pd.read_csv('data/hcp/restricted.csv')

    # Specifying indices of overlapping subjects with subnums
    subInd = np.where(np.isin(restricted["Subject"], subnums))[0]

    # Only using data from subnums
    restricted = restricted.reindex(subInd)
    behavioral = behavioral.reindex(subInd)

    return restricted, behavioral


# read face-emotional data from HCP900 data
def read_HCP900_data():
    filepath = 'data/cfHCP900_FSL_GM/cfHCP900_FSL_GM.mat'
    taskdata = {}
    f = h5py.File(filepath, 'r')
    for k, v in f.items():
        taskdata[k] = np.array(v)  # dim: 268 x 268 x 9 , TODO: only read task of interest

    # making task name accessible
    for j in range(len(f['SESSIONS'])):
        st = f['SESSIONS'][j][0]
        obj = f[st]
        name = ''.join(chr(i) for i in obj[:])
        taskdata['SESSIONS'][j] = name

    # reading necessary data from hd5f file
    taskIDs = taskdata['IDS'].reshape(-1).astype(int)
    taskCorr = taskdata['CORR']
    taskNames = taskdata['SESSIONS']

    # closing hdf5 file to avoid errors later if attempting to open on a new thread
    import gc

    for obj in gc.get_objects():  # Browse through ALL objects
        if isinstance(obj, h5py.File):  # Just HDF5 files
            try:
                obj.close()  # close  the file
            except:
                pass  # Was already closed

    return taskIDs, taskCorr, taskNames, taskdata


# read connectivity matrices from .txt, .npy, .mat files
def read_mat_data(dataDir, toi=[]):
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

                # Reading in data from all 1003 subjects
                for _, file in enumerate(filtnames):
                    n1 = io.loadmat(f'{eb}/{subdir}/{file}')['mat']
                    data.append(n1)

                    sn = re.findall(r'\d+', file)[-1][-6:]  # subject ID
                    subnums.append(sn)

    elif dataDir == 'data/cfHCP900_FSL_GM':
        taskIDs, taskCorr, taskNames, taskdata = read_HCP900_data()
        data = arr2mat_HCP900(taskCorr, toi, taskdata, r_transform=False)
        subnums = list(taskIDs)

    elif dataDir == 'data/Send_to_Tim/HCP_IMAGEN_ID_mega_file.txt':
        mega_vars = np.loadtxt(dataDir, delimiter=',', dtype=str, max_rows=1)
        mega_subs = np.loadtxt(dataDir, delimiter=',', dtype=str, usecols=0, skiprows=1)
        mega_hcp_inds = np.argwhere([file.startswith('HCP') for file in mega_subs]).squeeze()
        data = np.loadtxt(dataDir, delimiter=',', skiprows=1, usecols=range(1, len(mega_vars) - 1))[
            mega_hcp_inds]  # TODO: add option to use IMAGEN data later
        subnums = [int(name[-6:]) for name in mega_subs[mega_hcp_inds]]

    else:
        filenames = [f for f in listdir(dataDir) if isfile(join(dataDir, f))]
        filenames.sort()

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

    if not subnums:  # if subnums still an empty list
        for i, filename in enumerate(filenames):  # reading in subnums
            if filename.endswith(".txt") or filename.endswith(".mat"):
                num = re.findall(r'\d+', filename)  # find digits in filenames
                subnums.append(num)

    subnums.sort()
    subnums = np.array(subnums).astype(float).squeeze()  # necessary for comparison to train-test-split partitions

    data = np.array(data, dtype=float)

    print(f'Success! {dataDir} {toi} read in.\n')

    return data, subnums


# tests random matrices in loaded data for positive definiteness and size
def test_mat_data(data, nMat=1):
    print('Running positive definite test...\n...')
    # Testing arbitrarily chosen matrices for positive definiteness
    testind = np.random.randint(0, len(data), nMat)
    # r_testmat = np.empty([testmat.size, data[0].shape[0], data[0].shape[0]])

    for i, x in enumerate(testind):
        # r_testmat[i] = z2r(data[x]) + np.eye(data[x].shape[0])  # for z-scores
        # r_testmat[i] = data[x] + np.eye(data[x].shape[0]) # for all else
        #
        # assert r_testmat[i].shape == (data[0].shape[0], data[0].shape[0])
        # assert r_testmat[i].max() == 1.0

        if not isPD(data[x]):
            print(f"Subject {testind[i]}/{len(data)}'s matrix of size {data[x].shape} is not positive definite!")
        else:
            print(f"Success! Subject {testind[i]}/{len(data)}'s matrix of size {data[x].shape} is positive definite!")
    print('\n')


# plots loaded in data for qualitative examination
def plot_mat_data(data, dataDir, nMat=1):
    '''Takes input of data matrices and name of directory, and number of matrices to test.'''

    testmat = np.random.randint(0, len(data), nMat)  # arbitrary subjects matrices

    plt.rcParams.update({'font.size': 6})
    fig, axs = plt.subplots(nMat, 2, figsize=(8, 5))
    fig.subplots_adjust(hspace=.3, wspace=.05)
    axs = axs.ravel()

    c = 0  # setting counter

    # Plotting arbitrary subject partial correlation matrix (should be sparse)
    for i, x in enumerate(testmat):
        if nMat > 1:  # case: nMat > 1
            # Histogram
            lt = np.ravel(np.tril(data[x]))
            nonZel = lt != 0  # non zero elements
            axs[c].hist(lt[nonZel], bins=500)
            axs[c].set_title(f'Histogram of lower triangle entries\nsubject {testmat[i]} in {dataDir}')

            # Connectivity matrix
            im = axs[c + 1].imshow(data[x])
            axs[c + 1].set_title(f'Connectivity Matrix for subject {testmat[i]}')
            fig.colorbar(im, ax=axs[c + 1])
            im.set_clim(lt.min(), lt.max())
            c += 2

        else:  # case: nMat = 1
            # Plotting histogram of connectivity matrix values to see if they are gaussian-distributed
            lt = np.ravel(np.tril(data[x]))
            nonZel = lt != 0  # non zero elements
            axs[0].hist(lt[nonZel], bins=500)
            axs[0].set_title(f'Histogram of lower triangle entries\nsubject {testmat[0]} in {dataDir}')

            # Plotting connectivity matrix
            im = axs[1].imshow(data[x])
            axs[1].set_title(f'Connectivity Matrix for subject {testmat[0]}')
            fig.colorbar(im, ax=axs[1])
            im.set_clim(lt.min(), lt.max())

    fig.show()

    # # exporting to interactive html file
    # mpld3.save_html(fig, 'figures/edge_betwenness_matrices_1.7.19')


# fisher z-scores to correlation coefficient r
def z2r(x):
    return np.tanh(x)


# Transforming into pearson correlations
def R_transform(data):
    """Transforms an array/list of matrices from fisher z-score to pearson R correlation."""
    rdata = np.empty_like(data)
    npd_count = 0
    for i, x in enumerate(data):
        rdata[i] = z2r(x)
        if not isPD(rdata[i]):
            npd_count += 1
    print(f'R_transform returned {npd_count} non-positive definite matrices')
    return rdata


# testing multiple matrices for positive definiteness
def areNotPD(manyB):
    """
    Script to test many matrices for positive definiteness
    :param manyB: array of matrices
    :return: number of matrices that aren't PD, and their indices in manyB
    """
    howmany = 0
    which = []
    for i, B in enumerate(manyB):
        if not isPD(B):
            howmany += 1
            which.append(i)

    return howmany, which


# Defining positive definite trasnformation of matrices
def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k ** 2 + spacing)
        k += 1

    return A3


# Transforming matrices into nearest positive definite matrix
def PD_transform(datamats):
    pddata = np.empty_like(datamats)
    npd_count = 0
    for i, x in enumerate(datamats):
        pddata[i] = nearestPD(x)
        if not isPD(pddata[i]):
            print(f'Matrix {i} is not positive definite!')
            npd_count += 1
        if i % 200 == 0:
            print(f'Making making matrix {i}/{len(pddata)} positive definite...')
    print(f'PD_transform successfully transformed {len(datamats) - npd_count} matrices\n')
    return pddata


# Determining positive definite matrix by cholesky decompositon
def isPD(B):
    """Returns true when input is positive-definite, via Cholesky
    credit: https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False


# create correlation matrix from time series
def CORR(A):
    """
    :param A: a timeseries text file
    :return: Correlation matrix
    """
    TS = np.loadtxt(A)

    TS -= numpy.nanmean(TS, axis=0)

    ST = numpy.nanstd(TS, axis=0)
    ST[ST == 0] = 1
    TS /= ST

    CO = numpy.cov(TS.T)

    numpy.fill_diagonal(CO, 0)
    return CO


# create partial correlation matrix from time series
def ICORR(A, RHO):
    """
    :param A: a time series text file
    :param RHO: L2 regularization term (larger means more regularization)
    :return: Partial correlation matrix
    """
    TS = np.loadtxt(A)

    CO = numpy.cov(TS.T)
    CB = CO / numpy.sqrt(numpy.mean(numpy.diag(CO) ** 2))
    IC = -numpy.linalg.inv(CB + RHO * numpy.eye(CB.shape[0]))
    DV = numpy.sqrt(numpy.abs(numpy.diag(IC)))

    CR = (IC / DV[:, None]) / DV[None, :]
    numpy.fill_diagonal(CR, 0)
    return CR


# code for whitening data.
def whiten(X, fudge=1E-18):
    """
    :param X: covariance matrix
    :param fudge: insurance that eigenvectors with small eigvenvalues aren't overamplified
    :return: whitnend matrix X_white, and whitening matrix W
    """
    # eigenvalue decomposition of the covariance matrix
    d, V = np.linalg.eigh(X)

    # a fudge factor can be used so that eigenvectors associated with
    # small eigenvalues do not get overamplified.
    D = np.diag(1. / np.sqrt(d + fudge))

    # whitening matrix
    W = np.dot(np.dot(V, D), V.T)

    # multiply by the whitening matrix
    X_white = np.dot(X, W)

    return X_white, W


# checking if matrices are symmetric
def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


# TODO: finish implementation of twins
# partition data so twins are not separated between test-train-validation sets
def partition(restricted, Family_ID):
    '''Partitioning data so one families twins remain in test/validation/training sets'''

    ZygositySR = restricted["ZygositySR"]
    ZygosityGT = restricted['ZygosityGT']
    HasGT = restricted['HasGT']

    GTyes = np.where(np.isin(HasGT, True))[0]  # subjects with genetic tests
    GTno = np.where(np.isin(HasGT, False))[0]  # subject wo/ genetic tests
    assert len(GTyes) <= 1142
    assert len(GTno) <= 64

    srMZ = np.where(np.isin(ZygositySR, "MZ"))[0]  # self-reported monozygotic
    srNotMZ = np.where(np.isin(ZygositySR, "NotMZ"))[0]  # self-reported non-monozygotic
    srNotTwin = np.where(np.isin(ZygositySR, "NotTwin"))[0]  # self-reported not twin
    srBlank = np.where(np.isin(ZygositySR, " "))[0]  # no self-report on twin status
    assert len(srMZ) + len(srNotMZ) + len(srNotTwin) + len(srBlank) == 1003

    gcMZ = np.where(np.isin(ZygosityGT, "MZ"))[0]  # genetically confirmed monozygotic
    gcDZ = np.where(np.isin(ZygosityGT, "DZ"))[0]  # genetically confirmed dizygotic
    gcBlank = np.where(np.isin(ZygosityGT, " "))[0]  # not genetically confirmed
    assert len(gcMZ) + len(gcDZ) + len(gcBlank) == 1003

    gcMZTwins = set(gcMZ) & set(GTyes)  # genetically confirmed MZ twins
    assert len(gcMZTwins) <= 298  # 298 MZ in 1200 data

    gcDZTwins = set(gcDZ) & set(GTyes)  # genetically confirmed DZ twins
    assert len(gcDZTwins) <= 188  # 188 DZ twins in 1200 data

    # p5 and p6 refer to points 5 and 6 of pg 89 in HCP release reference manual
    # https://www.humanconnectome.org/storage/app/media/documentation/s1200/HCP_S1200_Release_Reference_Manual.pdf

    p5 = set(srMZ) & set(gcBlank)  # point 5 of pg 89
    assert len(p5) <= 66  # 66 subjects with ZygositySR=MZ, but ZygosityGT=Blank in 1200 data

    p6 = set(srNotMZ) & set(gcBlank)
    assert len(p6) <= 65  # 65 subjects with ZygositySR=NotMZ, but ZygosityGT=Blank in 1200 data

    # subjects whose putative twin is not part of the 1206 released study subjects.
    nsMZ = p5 & set(GTno)  # subjects whose putative MZ twin IS  part of the 1206, but HasGT=FALSE for one of the pair,
    nsDZ = p6 & set(
        GTno)  # subjects whose putative DZ twin IS  part of the 1206, but HasGT=FALSE for one or both of the pair
    noGTTwins = nsMZ | nsDZ
    assert len(noGTTwins) <= 56, 'More non-singular twins than expected. Should be fewer than 56.'

    # Creating full list of non-singular twins (adding genetically confirmed twins)
    nsTwins = noGTTwins | gcMZTwins | gcDZTwins

    # TODO: figure out if problem in finding twins from nsTwinFams or familymems or Family_ID
    # family IDs of families with self-reported, non-singular twins
    nsTwinfams = list(np.unique(list(Family_ID.iloc[list(nsTwins)])))
    assert len(nsTwinfams) <= len(nsTwins) / 2

    # Sanity check, confirming self-reported but not-genetically confirmed twins are of the same family
    sib1 = int(np.where(Family_ID == nsTwinfams[0])[0][0])
    sib2 = int(np.where(Family_ID == nsTwinfams[0])[0][1])
    assert Family_ID.iloc[sib1] == Family_ID.iloc[sib2]

    # Grouping twins together
    twingroups = []  # groups of subjects who are twins

    for i, x in enumerate(nsTwinfams):  # For each family with twins...
        familymems = list(np.where(Family_ID == x)[0])  # list indices of all members
        # print(familymems)
        for j, y in enumerate([noGTTwins, gcMZTwins, gcDZTwins]):
            twins = set(familymems) & y  # find if/which family members are twins...
            if len(twins) > 2:  # (if more than two twins, tell us, but still add them all)
                print(f'family found with {len(twins)} twins')
            elif len(twins) < 2:
                print('only one twin in family!')
            if twins:
                twingroups.append(twins)  # and add them to our list

    # TODO: Random shuffling of standalone participants and twins into 70-15-15 train-validation-test split

    return  # test, train, validation # subject IDs


# DIY tangent space transformation
def tangent_transform(refmats, projectmats, ref='euclidean'):
    """
    Projects array of matrices (projectmats) into tangent space, using the mean of another array (refmats) as reference.
    Implementation from dadi et al., 2019. Source: https://hal.inria.fr/hal-01824205v3
    Calculation of reference means from Pervaiz et al., 2019. https://www.biorxiv.org/content/10.1101/741595v2.full.pdf

    :param refmats: positive definite covariance matrices (samples x rows x columns), from which mean is calculated
    :param projectmats: positive definite matrices to be projected into tangent space
    :param ref: reference mean to use (i.e. euclidean, harmonic, log euclidean, riemannian, kullback)
    :return: tangent-projected matrices
    """
    if ref == 'harmonic':  # use harmonic mean
        Ch = 0
        for i, x in enumerate(refmats):
            Ch += inv(x)
        Ch *= 1 / len(refmats)
        refMean = inv(Ch)

    elif ref == 'euclidean':  # use euclidean mean
        refMean = 1 / len(refmats) * np.mean(refmats, axis=0)

    else:
        raise ValueError(f'Tangent transform not implemented for {ref} yet!')
        return

    d, V = np.linalg.eigh(refMean)  # EVD on reference mean covariance matrix
    fudge = 1E-18  # ensures our eigenvectors don't explode
    wsStar = V.T @ np.diag(1 / np.sqrt(d + fudge)) @ V
    mag = len(projectmats[1])  # matrix length magnitude

    tmats = np.zeros_like(projectmats)
    for i, x in enumerate(projectmats):
        m = np.dot(wsStar, x).dot(wsStar)
        m = m.reshape(mag, mag)
        if i % 199 == 0:
            print(f'Projecting matrix {i}/{len(tmats)} into tangent space...')
        tmats[i] = logm(m)

    return tmats


# create connectivity matrices from time series data
def create_connectivity(dataDir='data/HCP_created_ICA300_timeseries', rho=.5,
                        saveDir='data/self_created_HCP_mats/ICA300_corr', c_type='corr'):
    """
    Script to create correlation matrices from time series data
    :param c_type: the type of correlation matrix to create
    :param dataDir: Directory of time series .txt files
    :param rho: regularization term
    :param saveDir: Directory to save partial correlation mats
    :return: matrices that are not positive definite, despite regularization
    """
    not_PD = []
    PD_testy3 = 0
    all_mat = []

    filenames = [f for f in listdir(dataDir) if isfile(join(dataDir, f))]
    filenames.sort()
    print(filenames)

    if filenames[0].endswith('.npy'):
        bigD = np.load(f'{dataDir}/{filenames[0]}')
        for i, x in enumerate(bigD):
            if c_type == 'pcorr':
                testy3 = ICORR(x, RHO=rho)
            elif c_type == 'corr':
                testy3 = CORR(x)

            np.fill_diagonal(testy3, 1)
            all_mat.append(testy3)

            if isPD(testy3):
                PD_testy3 += 1
            else:
                not_PD.append(i)
                print(f'{x} not PD!')

    elif filenames[0].endswith('.txt'):
        for i, x in enumerate(filenames):
            if c_type == 'pcorr':
                testy3 = ICORR(f'{dataDir}/{x}', RHO=rho)
            elif c_type == 'corr':
                testy3 = CORR(f'{dataDir}/{x}')

            np.fill_diagonal(testy3, 1)
            all_mat.append(testy3)

            if isPD(testy3):
                PD_testy3 += 1
            else:
                not_PD.append(i)
                print(f'{x} not PD!')

    all_mat = np.array(all_mat)
    np.save(saveDir, all_mat)

    print(f'{PD_testy3} positive-definite matrices returned in {c_type} calculations...')
    if c_type == 'pcorr':
        print(f'rho = {rho}\n')

    return all_mat, not_PD


# get parameters necessary for deconfounding
def get_confound_parameters(est_data, confounds, set_ind=None):
    """Takes array of square matrices (samples x matrices) and returns confound signals, the parameter.

    est_data: full data from which the confound parameters are estimated
    set_ind: indices of the est_data from which the confound parameters will be estimated
    confounds: list of confounds, each containing same number of samples as est_data
    data_tbd: data to be deconfounded

    return:
        nan_ind: the indices (out of the set_ind) that have any confound == nan
        C: the nan-removed confound matrix
        C_pi: pseudoinverse of confounds
        b_hatX: deconfounded X

    Calculations based off equations (2) - (4):
    https://www.sciencedirect.com/science/article/pii/S1053811918319463?via%3Dihub#sec2
    """

    # vectorizing matrix and subtracting mean
    t = np.array([x[np.triu_indices(len(x), k=1)] for x in est_data])
    t -= np.mean(t, axis=0)

    est_array = np.array([t[j] for j in list(set_ind)])  # specifying arrays from which we'll deconfound

    # creating confound matrix
    C = np.vstack(confounds).astype(float).T[set_ind]

    # identifying nan values in confounds
    nan_ind = np.unique(np.argwhere(np.isnan(C)).squeeze())

    # deleting samples that have confounds with NaN values
    C = np.delete(C, nan_ind, axis=0)
    X = np.delete(est_array, nan_ind, axis=0)

    # regressing out confounds
    C_pi = np.linalg.pinv(C)  # moore-penrose pseudoinverse
    b_hatX = C_pi @ X  # confound parameter estimate

    return C_pi, b_hatX, nan_ind


# Reshapes task array (i.e. after deconfounding) into symmetric matrices, optionally does z-score to R transformation
def arr2mat_HCP900(taskCorr, task, taskdata, r_transform=False, is_task=True, new_size=268):
    taskCorrMat = []

    for i in range(taskCorr.shape[1]):  # reshaping array into symmetric matrix
        out = np.zeros((new_size, new_size))

        if is_task:  # adding to allow for reshaping of confound-corrected data array
            try:
                taskind = np.where(np.array(taskdata['SESSIONS']) == task)[0][0]
            except IndexError:
                print(f'\'{task}\' is not a valid task. Please provide a valid task to be read in.')
                sys.exit()

        uinds = np.triu_indices(len(out), k=1)
        out[uinds] = taskCorr[:, i, taskind]
        out = np.triu(out, 1) + out.T

        # TODO: decide if to delete NaN or just set to zero
        where_are_NaNs = np.isnan(out)  # changing error-prone NaN values to zero
        out[where_are_NaNs] = 0

        taskCorrMat.append(out)

    taskCorrMat = np.array(taskCorrMat)  # setting as array

    if r_transform:
        taskCorrMat = R_transform(taskCorrMat)  # transforming to pearson R data

    for i, x in enumerate(taskCorrMat):
        np.fill_diagonal(x, 1)  # on z-scored data

    # if check_symmetric(taskCorrMat[0]):
    #     print('Matrix 0 is symmetric! Assumming all matrices are...\n')
    # else:
    #     print('Matrix 0 is not symmetric. Something went wrong...\n')

    return taskCorrMat


# reshapes array, sets NaNs to zero,
def array2matrix(samples, mat_size=300):
    """
    :param samples: samples x upper triangular array of a matrix
    :param mat_size: determined size of newly shaped matrix
    :return: mat_size x mat_size symmetric matrix
    """
    d_mats = []

    for i in range(len(samples)):  # reshaping array into symmetric matrix
        out = np.zeros((mat_size, mat_size))

        uinds = np.triu_indices(len(out), k=1)
        out[uinds] = samples[i]
        out = np.triu(out, 1) + out.T

        where_are_NaNs = np.isnan(out)  # changing error-prone NaN values to zero
        if np.any(where_are_NaNs):
            print(f'Setting NaNs to zero in matrix {i}...')
            out[where_are_NaNs] = 0  # sets NaNs to zero

        d_mats.append(out)

    d_mats = np.array(d_mats)  # setting as array

    return d_mats


def deconfound_dataset(data, confounds, set_ind, outcome):
    """
    Takes input of a data, its confounds. Deletes samples with nan-valued Y entries.
     Returns the deconfounded data.

    :param outcome: ground truth value to be deconfounded, per Y1
    :param data: Samples x symmetric matrices (row x column) to be deconfounded, per X1
    :param confounds: Confounds x samples, to be factored out of cdata
    :param set_ind: sample indices of data from which deconfounding parameters will be calculated
    :return: List of deconfounded X, Y as well as new train-test-validation indices
    """

    # confound parameter estimation for X
    C_pi, b_hat_X, nan_ind = get_confound_parameters(data, confounds, set_ind=set_ind)

    # ...and Y, with nans removed
    Y_c = np.delete(outcome[set_ind], nan_ind, axis=0)
    b_hat_Y = C_pi @ Y_c  # Y confound parameter estimation

    # takes all data as an array, removes need for tbd_ind
    C_tbd = np.vstack(confounds).astype(float).T

    X_corr = data - array2matrix(C_tbd @ b_hat_X, mat_size=data.shape[-1])
    Y_corr = outcome - C_tbd @ b_hat_Y

    # TODO: return explained variance from decconfounds
    return np.array(X_corr), np.array(Y_corr), nan_ind


# get name of variable as a string
def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


def multiclass_to_onehot(Y):
    Y_classes = np.zeros((Y.squeeze().shape[0], len(np.unique(Y))))
    for i, x in enumerate(np.unique(Y)):
        Y_classes[[np.where(Y == x)[0]], i] = 1
    return Y_classes


def onehot_to_multiclass(a):
    return np.array([np.where(r == 1)[0][0] for r in a])


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj][0]


def NEOFFIdomain_latent_transform(data, dataset='HCP', Q=5):
    """
    Calculating factor-transformed, varimax-rotated latent dimensions of personality from NEO-FFI domain data.
    Saves .npy file with transformed data.

    :param data: pandas-like DataFrame/xarray-like DataArray with subject data. NEO-FFI data must be in columns of headers starting with 'NEO'.
    :param Q: number of features form which latent dimensions are calculated
    :return: None
    """

    # read in personality data
    NEO_keys = list(filter(lambda x: x.startswith('NEO'), list(data.keys())))
    try:  # data as xarray DA
        feature_info = data[NEO_keys].to_array().values
    except AttributeError:  # data as pandas DF
        feature_info = data[NEO_keys].values

    # ensure correct dims
    feature_info = feature_info.reshape(-1, Q)

    # dictionary of data necessary to transform HCP data
    mvtr = pickle.load(open(f'personality-types/data_filter/ipip{Q}-mvtr-1.pkl', "rb"))

    # z-score acc. to the Gerlach mean/std
    z_muvar = np.load(f'personality-types/data_filter/ipip{Q}-pre_cluster_zscore_mu_var-1.npy')
    z_mu, z_var = z_muvar[0], z_muvar[1]

    # transforming HCP data
    latent_data = (feature_info - mvtr['mu']) @ mvtr['trans_mat']  # applying scaling & factor analysis fit-transform
    latent_data = (mvtr['rot_mat'] @ latent_data.T).T  # varimax rotation
    latent_data = (latent_data - z_mu) / z_var  # z-scoring

    # saving as file, to be run through soft-cluster anaylsis
    np.save(f'personality-types/data_filter/{dataset}_ipip{Q}_domain_latent_transform.npy', latent_data)


def derive_HCP_NEOFFI60_scores():
    # # Reading in HCP NEO-FFI60 raw item responses
    unrestricted = pd.read_csv('data/unrestricted_adrymoat_6_30_2020_0_54_27.csv')
    NEORAW_keys = list(filter(lambda x: x.startswith('NEORAW'), list(unrestricted.keys())))
    NEORAW_keys.insert(0, 'Subject')
    HCP_NEORAW = unrestricted[NEORAW_keys].dropna()

    # coding the items per dom_key
    neuroticism_items = [1, 11, 16, 31, 46, 6, 21, 26, 36, 41, 51, 56]
    extraversion_items = [7, 12, 37, 42, 2, 17, 27, 57, 22, 32, 47, 52]
    openness_items = [13, 23, 43, 48, 53, 58, 3, 8, 18, 38]
    agreeableness_items = [9, 14, 19, 24, 29, 44, 54, 59, 4, 34, 39, 49]
    conscientiousness_items = [5, 10, 15, 30, 55, 25, 35, 60, 20, 40, 45, 50]

    # coding forward or reverse scoring
    n_keying = dict(forward=[11, 6, 21, 26, 36, 41, 51, 56], reverse=[1, 16, 31, 46])
    e_keying = dict(forward=[7, 37, 2, 17, 22, 32, 47, 52], reverse=[12, 42, 27, 57])
    o_keying = dict(forward=[13, 43, 53, 58], reverse=[23, 48, 3, 8, 18, 38])
    a_keying = dict(forward=[19, 4, 34, 49], reverse=[9, 14, 24, 29, 44, 54, 59, 39])
    c_keying = dict(forward=[5, 10, 25, 35, 60, 20, 40, 50], reverse=[15, 30, 55, 45])
    dom_names = ['NEOFAC_N', 'NEOFAC_E', 'NEOFAC_O', 'NEOFAC_A', 'NEOFAC_C']

    # deriving scores (1-5) for each item and domains
    forward_score = dict(SD=1, D=2, N=3, A=4, SA=5)  # assuming 'strongly agree/disagree' is the abbrev.
    reverse_score = dict(SD=5, D=4, N=3, A=2, SA=1)

    HCP_NEOscored = HCP_NEORAW.copy()

    for i, dom_key in enumerate([n_keying, e_keying, o_keying, a_keying, c_keying]):
        for_items = ['NEORAW_' + (('0' + str(x))[-2:]) for x in dom_key['forward']]
        rev_items = ['NEORAW_' + (('0' + str(x))[-2:]) for x in dom_key['reverse']]
        HCP_NEOscored[for_items] = HCP_NEOscored[for_items].replace(forward_score)
        HCP_NEOscored[rev_items] = HCP_NEOscored[rev_items].replace(reverse_score)

        # deriving domain score
        HCP_NEOscored[dom_names[i]] = HCP_NEOscored[for_items + rev_items].sum(axis=1)

    return HCP_NEOscored


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)
