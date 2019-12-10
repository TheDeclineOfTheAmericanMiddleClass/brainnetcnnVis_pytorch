import pyriemann as pyr
import torch
import numpy as np


def tangent_transform(data):
    """ Takes ndarray covariance matrices and transforms data into tangent space. """

    # ts = pyr.estimation.Covariances(data)
    ts = pyr.tangentspace.TangentSpace()
    transformed_data = ts.fit_transform(data)

    return torch.tensor(transformed_data, device=device)
