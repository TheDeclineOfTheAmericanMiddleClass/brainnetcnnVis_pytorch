import h5py
import numpy as np
from preprocessing.Preproc_funcs import R_transform

filepath = 'data/cfHCP900_FSL_GM/cfHCP900_FSL_GM.mat'
taskdata = {}
f = h5py.File(filepath, 'r')
for k, v in f.items():
    taskdata[k] = np.array(v)  # dim: 268 x 268 x 9 , TODO: only read task of interest

# making task name accessible
for j in range(len(f['SESSIONS'])):
    st = f['SESSIONS'][j][0]
    obj = f[st]
    name = ''.join(chr(i) for i in obj[:])
    taskdata['SESSIONS'][j] = name

# reading necessary data from hd5f file
taskIDs = taskdata['IDS'].reshape(-1).astype(int)
taskCorr = taskdata['CORR']
taskNames = taskdata['SESSIONS']

# closing hdf5 file to avoid errors later if attempting to open on a new thread
import gc

for obj in gc.get_objects():  # Browse through ALL objects
    if isinstance(obj, h5py.File):  # Just HDF5 files
        try:
            obj.close()
        except:
            pass  # Was already closed
