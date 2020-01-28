import pandas as pd
import numpy as np


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
