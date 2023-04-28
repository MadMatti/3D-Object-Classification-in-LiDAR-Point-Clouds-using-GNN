import h5py

# ModelNet10
DATASET_PATH = '/tmp_workspace/3d/modelnet10_hdf5_2048'

# Load Part of the Dataset
f = h5py.File(DATASET_PATH + '/train0.h5', 'r')

# List all groups
print("Keys: %s" % f.keys())

# Get the data
data = f['data'][:]

# Get the labels
label = f['label'][:]

# Close the file
f.close()

# Plot the data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

item = data[0]

# Downsample the point cloud
item = item[::4, :]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the point cloud
ax.scatter(item[:,0], item[:,1], item[:,2])

plt.show()

# Convert the point cloud to a graph where each node represents a point and each edge represents the distance between two points
import networkx as nx
from scipy.spatial.distance import pdist, squareform

def knn_graph(data, k):
    """
    Construct a kNN graph from the given data
    :param data: point cloud data
    :param k: number of nearest neighbors
    :return: networkx graph
    """
    # Compute pairwise distance matrix
    D = squareform(pdist(data))

    # Sort distance matrix in ascending order and get indices of points
    idx = np.argsort(D, axis=1)

    # Construct kNN graph
    G = nx.Graph()
    for i in range(data.shape[0]):
        for j in idx[i, 1:k+1]:
            G.add_edge(i, j, weight=D[i, j])

    return G

# Plot the graph
G = knn_graph(item, 20)
nx.draw(G)
plt.show()

# Plot the adjacency matrix
A = nx.adjacency_matrix(G).todense()
plt.imshow(A)
plt.show()


