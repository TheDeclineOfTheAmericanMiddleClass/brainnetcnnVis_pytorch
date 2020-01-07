from numpy import linalg as la
import numpy as np

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
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3



# determining positive definite matrix by cholesky decompositon
# credit: https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite
def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False

# def is_pos_def(x): # not using due to purported np.all errors
#     return np.all(np.linalg.eigvals(x) > 0)

# Transforming into pearson correlations
def R_transform(data):
    rdata = np.empty_like(data)
    npd_count = 0
    for i, x in enumerate(data):
        rdata[i] = z2r(x)
        if isPD(rdata[i]) == False:
            npd_count += 1
    print(f'R_transform returned {npd_count} non-positive definite matrices')

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
    print(f'PD_transform returned {npd_count} non-positive definite matrices')
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
    D = np.diag(1. / np.sqrt(d+fudge))

    # whitening matrix
    W = np.dot(np.dot(V, D), V.T)

    # multiply by the whitening matrix
    X_white = np.dot(X, W)

    return X_white, W
