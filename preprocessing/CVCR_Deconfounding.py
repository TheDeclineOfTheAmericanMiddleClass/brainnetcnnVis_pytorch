import numpy as np


def deconfound_matrix(est_data, confounds, set_ind=None):
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
    b_hatX = C_pi @ X

    return C_pi, b_hatX, nan_ind
