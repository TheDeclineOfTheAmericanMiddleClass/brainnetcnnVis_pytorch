import numpy as np
import matplotlib.pyplot as plt
import pyriemann.tangentspace as pyr


####### Defining Functions ##########
# # torch tensor to numpy array
# def t2n(x):
#     return x.cpu().numpy()

# fisher z-scores to correlation coefficient r
def z2r(x):
    return np.tanh(x)


# ensuring positive definite matrix
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


# Testing arbitrarily chose matrices for positive definiteness
nMat = 1  # number of matrices
testmat = np.random.randint(0, len(data), nMat)
r_testmat = np.empty([testmat.size, data[0].shape[0], data[0].shape[0]])

for i, x in enumerate(testmat):
    r_testmat[i] = z2r(data[x]) + np.eye(data[x].shape[0])  # for z-scores
    # r_testmat[i] = data[x] + np.eye(data[x].shape[0]) # for all else

    assert r_testmat[i].shape == (data[0].shape[0], data[0].shape[0])
    assert r_testmat[i].max() == 1.0
    assert is_pos_def(r_testmat[i]), "Matrix is not positive definite!"

# ####### Ensuring Gaussianity of Untransoformed Partial Correlations #########
# # Checking if partial correlation matrix is truly gaussian-distributed
# fig, axs = plt.subplots(1, nMat, figsize=(8, 5))
# fig.subplots_adjust(hspace=.2, wspace=.1)
#
# lt = np.ravel(np.tril(t2n(data[0])))
# nonZel = lt != 0  # non zero elements
# axs.hist(lt[nonZel], bins=500)
# axs.set_title(f'Histogram of lower triangle entries for subject {testmat[0]} in {dataDir}')

###### Transforming z-scored data in pearson correlations ######
restricted = np.empty_like(data)
npd_count = 0
for i, x in enumerate(data):
    restricted[i] = z2r(x)
    if is_pos_def(restricted[i]) == False:
        print(f'Matrix {i} is not positive definite!')
        npd_count += 1

# ########## Plotting arbitrary subject partial correlation (should be sparse) ##############
# fig, axs = plt.subplots(1, nMat, figsize=(10, 4))
# fig.subplots_adjust(hspace=.2, wspace=.1)
#
# if nMat > 1:  # case: nMat > 1
#     axs = axs.ravel()
#     for i, x in enumerate(nMat):
#         axs[i].imshow(r_testmat[i])
#         print(f' Is matrix {x} positive definite? {is_pos_def(r_testmat[i])}')
#
# else:  # case: nMat = 1
#     axs.imshow(r_testmat[0])
#     print(f'Matrix {nMat} positive definite? {is_pos_def(r_testmat[0])}')
#
# fig.show()

##### Transforming into tangent space ############
# pyr.TangentSpace()
