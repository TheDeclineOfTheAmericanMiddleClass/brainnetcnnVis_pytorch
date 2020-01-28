import numpy as np


def deconfound_matrix(est_data, confounds, set_ind=None):
    """Takes array of square matrices (samples x matrices) and returns confound signals, the parameter.

    est_data: full dataset from which the confound parameters are estimated
    set_ind: indices of the est_data from which the confound parameters will be estimated
    confounds: list of confounds, each containing same number of samples as est_data
    data_tbd: data to be deconfounded

    return:
        nan_ind: the indices (out of the set_ind) that have any confound == nan

    Calculations based off equations (2) - (4):
    https://www.sciencedirect.com/science/article/pii/S1053811918319463?via%3Dihub#sec2
    """

    # reshaping matrix data to array
    t = []
    for i, x in enumerate(est_data):
        t.append(np.array(x[np.triu_indices(len(est_data[0]), k=1)]))
    est_array = np.array([t[j] for j in list(set_ind)])  # training arrays

    # creating confound matrix
    C = np.vstack(confounds).astype(float).T[set_ind]  # confounds

    # deleting samples that have confounds with NaN values
    nan_ind = np.unique(np.where(np.isnan(C) == True)[0])
    C = np.delete(C, nan_ind, axis=0)
    X = np.delete(est_array, nan_ind, axis=0)

    # regressing out confounds
    C_pi = np.linalg.pinv(C)
    b_hatX = C_pi @ X

    return C, C_pi, b_hatX, nan_ind
