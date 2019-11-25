import h5py
from os import listdir
from os.path import isfile, join

# setting data directory
dataDir = '3T_HCP1200_MSMAll_d300_ts2_RIDGE'

# names of only the files in the folder
filenames = [f for f in listdir(dataDir) if isfile(join(dataDir, f))]

# allocating list for matrices
data = []

# reading in data
for i, filename in enumerate(filenames):
    with h5py.File(f'{dataDir}/{filename}', 'r') as f:
        # List all groups
        # print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        data.append(f[a_group_key])

del filename, f, a_group_key, i, dataDir
