import numpy
import numpy as np
from numpy import linalg as la


# torch tensor to numpy array
def t2n(x):
    return x.cpu().numpy()


# fisher z-scores to correlation coefficient r
def z2r(x):
    return np.tanh(x)


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


# Determining positive definite matrix by cholesky decompositon
def isPD(B):
    """Returns true when input is positive-definite, via Cholesky
    credit: https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False


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


# def is_pos_def(x): # not using due to purported np.all errors
#     return np.all(np.linalg.eigvals(x) > 0)

# Transforming into pearson correlations
def R_transform(data):
    """Transforms an array/list of matrices from fisher z-score to pearson R correlation."""
    rdata = np.empty_like(data)
    npd_count = 0
    for i, x in enumerate(data):
        rdata[i] = z2r(x)
        if isPD(rdata[i]) == False:
            npd_count += 1
    print(f'R_transform returned {npd_count} non-positive definite matrices')
    return rdata


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


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def reshape_task_matrix(taskCorr, task, taskdata, r_transform=True, is_task=True, new_size=268):
    taskCorrMat = []

    for i in range(taskCorr.shape[1]):  # reshaping array into symmetric matrix
        out = np.zeros((new_size, new_size))

        if is_task == True:  # adding to allow for reshaping of confound-corrected data array
            taskind = np.where(np.array(taskdata['SESSIONS']) == task)[0][0]

        uinds = np.triu_indices(len(out), k=1)
        # linds = np.tril_indices(len(out), k=-1)

        out[uinds] = taskCorr[:, i, taskind]
        # out[linds] = taskCorr[:, i, taskind]

        out = np.triu(out, 1) + out.T
        # out = np.tril(out, 1) + out.T

        # TODO: decide if to delete NaN or just set to zero
        where_are_NaNs = np.isnan(out)  # changing error-prone NaN values to zero
        out[where_are_NaNs] = 0

        taskCorrMat.append(out)

    taskCorrMat = np.array(taskCorrMat)  # setting as array

    if r_transform == True:
        taskCorrMat = R_transform(taskCorrMat)  # transforming to pearson R data

    for i, x in enumerate(taskCorrMat):
        np.fill_diagonal(x, 1)  # on z-scored data

    if check_symmetric(taskCorrMat[0]):
        print('Matrix 0 is symmetric! Assumming all matrices are...\n')
    else:
        print('Matrix 0 is not symmetric. Something went wrong...\n')

    return taskCorrMat


def reshape_deconfounded_matrix(samples, new_size=300):
    """
    :param samples: matrix of samples x upper triangular array
    :param new_size: determined size of newly shaped matrix
    :return: new_size x new_size symmetric matrix
    """
    d_mats = []

    for i in range(len(samples)):  # reshaping array into symmetric matrix
        out = np.zeros((new_size, new_size))

        uinds = np.triu_indices(len(out), k=1)
        out[uinds] = samples[i]
        out = np.triu(out, 1) + out.T

        # TODO: decide if to delete NaN or just set to zero
        where_are_NaNs = np.isnan(out)  # changing error-prone NaN values to zero
        out[where_are_NaNs] = 0

        d_mats.append(out)

    d_mats = np.array(d_mats)  # setting as array

    return d_mats


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


import inspect


def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]
