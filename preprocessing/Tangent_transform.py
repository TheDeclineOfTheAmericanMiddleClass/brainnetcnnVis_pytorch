import pyriemann as pyr
import torch
import numpy as np
from scipy.linalg import logm, expm, inv

vlogm = np.vectorize(logm)  # vectorized logm

# defining DIY tangent space transformation
def tangent_transform(pdmats, ref='euclidean'):
    """
    Implementation from dadi et al., 2019. Source: https://hal.inria.fr/hal-01824205v3
    Calculation of reference means from Pervaiz et al., 2019. https://www.biorxiv.org/content/10.1101/741595v2.full.pdf
    :param data: covariance matrices (samples x rows x columns)
    :param ref: reference mean to use (i.e. euclidean, harmonic, log euclidean, riemannian, kullback)
    :return:
    """
    if ref == 'harmonic':  # use harmonic mean
        Ch = 0
        for i, x in enumerate(pdmats):
            Ch += inv(x)
        Ch *= 1 / len(pdmats)
        refMean = inv(Ch)

    else:  # use euclidean mean
        refMean = 1 / len(pdmats) * np.mean(pdmats, axis=0)

    d, V = np.linalg.eigh(refMean)  # EVD on reference mean covariance matrix
    fudge = 1E-18  # ensures our eigenvectors don't explode

    wsStar = V.T @ np.diag(1 / np.sqrt(d + fudge)) @ V

    tmats = np.zeros_like(pdmats)
    for i, x in enumerate(pdmats):
        tmats = np.dot(wsStar, x).dot(wsStar)
        tmats = tmats.reshape(-1, len(pdmats[1]), len(pdmats[1]))
        tmats[i] = logm(tmats)

    return tmats

# TODO: get to source of pyriemman error
# def tangent_transform(data):
#     """ Takes ndarray covariance matrices and transforms data into tangent space. """
#
#     # ts = pyr.estimation.Covariances(data)
#     ts = pyr.tangentspace.TangentSpace()
#     transformed_data = ts.fit_transform(data)
#
#     return transformed_data

# trying alternative implementation
# import cmath
# csqrt = np.vectorize(cmath.sqrt)
# def tangent_transform(data):
#     Ch = np.zeros_like(data[0])
#     for i, x in enumerate(data):
#         Ch += inv(x)
#     Ch *= 1/len(data)
#     Ch = inv(Ch)
#     C = logm(np.real(inv(csqrt(data)) @ data @ inv(csqrt(data))))
#     return C
#
