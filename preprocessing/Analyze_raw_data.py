import numpy as np
import matplotlib.pyplot as plt
from preprocessing.Preproc_funcs import *

def test_raw_data(data, nMat=1):
    # Testing arbitrarily chosen matrices for positive definiteness
    nMat = 1  # number of matrices
    testmat = np.random.randint(0, len(data), nMat)
    r_testmat = np.empty([testmat.size, data[0].shape[0], data[0].shape[0]])

    for i, x in enumerate(testmat):
        r_testmat[i] = z2r(data[x]) + np.eye(data[x].shape[0])  # for z-scores
        # r_testmat[i] = data[x] + np.eye(data[x].shape[0]) # for all else

        assert r_testmat[i].shape == (data[0].shape[0], data[0].shape[0])
        assert r_testmat[i].max() == 1.0

        if not isPD(r_testmat[i]):
            print(f"Subject {testmat[0]}'s matrix is not positive definite!")
        else:
            print(f"Success! Subject {testmat[0]}'s matrix is positive definite!")


def plot_raw_data(data, dataDir,nMat=1):
    '''Takes input of data matrices and name of directory, and number of matrices to test.'''

    testmat = np.random.randint(0, len(data), nMat)  # arbitrary subjects matrices

    fig, axs = plt.subplots(nMat, 2, figsize=(8, 5))
    fig.subplots_adjust(hspace=.2, wspace=.2)
    axs = axs.ravel()

    # Plotting arbitrary subject partial correlation matrix (should be sparse)
    for i, x in enumerate(testmat):
        if nMat > 1:  # case: nMat > 1
            c = 0  # setting counter

            # Histogram
            lt = np.ravel(np.tril(data[x]))
            nonZel = lt != 0  # non zero elements
            axs[c].hist(lt[nonZel], bins=500)
            axs[c].set_title(f'Histogram of lower triangle entries\nsubject {testmat[0]} in {dataDir}')

            # Connectivity matrix
            axs[c+1].imshow(data[x])
            axs[c+1].set_title(f'Connectivity Matrix for subject {testmat}')

            c += 2
        else:  # case: nMat = 1
            # Plotting histogram of connectivity matrix values to see if they are gaussian-distributed
            lt = np.ravel(np.tril(data[x]))
            nonZel = lt != 0  # non zero elements
            axs[0].hist(lt[nonZel], bins=500)
            axs[0].set_title(f'Histogram of lower triangle entries\nsubject {testmat[0]} in {dataDir}')

            # Plotting connectivity matrix
            axs[1].imshow(data[x])
            axs[1].set_title(f'Connectivity Matrix for subject {testmat[0]}')

    fig.show()