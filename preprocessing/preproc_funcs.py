from __future__ import print_function

import glob
import inspect
import re
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

    # Specifying indices of overlapping subjects in 1206 and 1003 subject datasets
    subInd = np.where(np.isin(restricted["Subject"], subnums))[0]

    # Only using data from 1003
    restricted = restricted.reindex(subInd)
    behavioral = behavioral.reindex(subInd)

    return restricted, behavioral


# read face-emotional data from HCP900 cohort
def read_face900_data():
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


# tests random matrices in loaded data for positive definiteness and size
def test_raw_data(data, nMat=1):
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
def plot_raw_data(data, dataDir, nMat=1):
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
        if i % 199 == 0:
            print(f'Attempting to make {i}/{len(pddata)} matrices positive definite...')
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


# partition dataset so twins are not separated between test-train-validation sets
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
    assert len(gcMZTwins) <= 298  # 298 MZ in 1200 dataset

    gcDZTwins = set(gcDZ) & set(GTyes)  # genetically confirmed DZ twins
    assert len(gcDZTwins) <= 188  # 188 DZ twins in 1200 dataset

    # p5 and p6 refer to points 5 and 6 of pg 89 in HCP release reference manual
    # https://www.humanconnectome.org/storage/app/media/documentation/s1200/HCP_S1200_Release_Reference_Manual.pdf

    p5 = set(srMZ) & set(gcBlank)  # point 5 of pg 89
    assert len(p5) <= 66  # 66 subjects with ZygositySR=MZ, but ZygosityGT=Blank in 1200 dataset

    p6 = set(srNotMZ) & set(gcBlank)
    assert len(p6) <= 65  # 65 subjects with ZygositySR=NotMZ, but ZygosityGT=Blank in 1200 dataset

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
def tangent_transform(pdmats, ref='euclidean'):
    """
    Takes array of positive definite matrices and returns their projection into tangent space.
    Implementation from dadi et al., 2019. Source: https://hal.inria.fr/hal-01824205v3
    Calculation of reference means from Pervaiz et al., 2019. https://www.biorxiv.org/content/10.1101/741595v2.full.pdf
    :param pdmats: positive definite covariance matrices (samples x rows x columns)
    :param ref: reference mean to use (i.e. euclidean, harmonic, log euclidean, riemannian, kullback)
    :return:
    """
    if ref == 'harmonic':  # use harmonic mean
        Ch = 0
        for i, x in enumerate(pdmats):
            Ch += inv(x)
        Ch *= 1 / len(pdmats)
        refMean = inv(Ch)

    elif ref == 'euclidean':  # use euclidean mean
        refMean = 1 / len(pdmats) * np.mean(pdmats, axis=0)

    else:
        raise ValueError(f'Tangent transform not implemented for {ref} yet!')
        return

    d, V = np.linalg.eigh(refMean)  # EVD on reference mean covariance matrix
    fudge = 1E-18  # ensures our eigenvectors don't explode

    wsStar = V.T @ np.diag(1 / np.sqrt(d + fudge)) @ V

    tmats = np.zeros_like(pdmats)
    for i, x in enumerate(pdmats):
        m = np.dot(wsStar, x).dot(wsStar)
        m = m.reshape(len(pdmats[1]), len(pdmats[1]))
        if i % 199 == 0:
            print(f'Projecting {i}/{len(tmats)} matrices into tangent space...')
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

    est_data: full dataset from which the confound parameters are estimated
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
def array2matrix_face900(taskCorr, task, taskdata, r_transform=False, is_task=True, new_size=268):
    taskCorrMat = []

    for i in range(taskCorr.shape[1]):  # reshaping array into symmetric matrix
        out = np.zeros((new_size, new_size))

        if is_task:  # adding to allow for reshaping of confound-corrected data array
            taskind = np.where(np.array(taskdata['SESSIONS']) == task)[0][0]

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

    if check_symmetric(taskCorrMat[0]):
        print('Matrix 0 is symmetric! Assumming all matrices are...\n')
    else:
        print('Matrix 0 is not symmetric. Something went wrong...\n')

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


# deconfounding entire dataset from
def deconfound_dataset(data, confounds, set_ind, outcome):
    """
    Takes input of a dataset, its confounds. Deletes samples with nan-valued Y entries.
     Returns the deconfounded dataset.

    :param outcome: ground truth value to be deconfounded, per Y1
    :param data: Samples x symmetric matrices (row x column) to be deconfounded, per X1
    :param confounds: Confounds x samples, to be factored out of cdata
    :param set_ind: sample indices of dataset from which deconfounding parameters will be calculated
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
