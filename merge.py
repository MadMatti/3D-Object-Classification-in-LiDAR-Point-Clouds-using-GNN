import h5py
import numpy as np

# ModelNet10
DATASET_PATH_1 = '/tmp_workspace/3d/modelnet10_hdf5_2048/train0.h5'
DATASET_PATH_2 = '/tmp_workspace/3d/modelnet10_hdf5_2048/train1.h5'

# Load 1st Part of the Dataset
f1 = h5py.File(DATASET_PATH_1, 'r')

# Load 2nd Part of the Dataset
f2 = h5py.File(DATASET_PATH_2, 'r')

# Merge the two datasets
data = np.concatenate((f1['data'][:], f2['data'][:]), axis=0)
label = np.concatenate((f1['label'][:], f2['label'][:]), axis=0)

# Close the files
f1.close()
f2.close()

# Save the merged dataset
f = h5py.File(DATASET_PATH_1.replace('train0', 'train'), 'w')
f.create_dataset('data', data=data)
f.create_dataset('label', data=label)
f.close()
