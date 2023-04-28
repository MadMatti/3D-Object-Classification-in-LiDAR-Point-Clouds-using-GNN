import h5py

from torch.utils.data import Dataset as TorchDataset
import torch

import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist, squareform
def knn_graph(data, k):
    """
    Construct a kNN graph from the given data
    :param data: point cloud data
    :param k: number of nearest neighbors
    :return: networkx graph
    """
    # Compute pairwise distance matrix
    D = pdist(data)
    #D = np.exp(-D)
    D = 1/(1+D)
    D = squareform(D)

    # Sort distance matrix in ascending order and get indices of points
    idx = np.argsort(D, axis=1)

    # Construct kNN graph
    G = nx.Graph()
    for i in range(data.shape[0]):
        for j in idx[i, 1:k+1]:
            G.add_edge(i, j, weight=D[i, j])

    return G

class Dataset(TorchDataset):
    def __init__(self, path=None):
        """
        Initializes the dataset
        Parameters
        ----------
        data : torch.Tensor
            The data of the dataset
        target : torch.Tensor
            The target of the dataset
        """
        self.data = []
        self.label = []

        if path is None:
            return

        # Check if cache exists
        import os
        if os.path.exists(path + '.cache'):
            # Load from cache
            print('Loading from cache...')
            import pickle
            with open(path + '.cache', 'rb') as f:
                self.data, self.label = pickle.load(f)
        else:
            # Load from file
            f = h5py.File(path, 'r')

            # convert to adjacency matrix
            for i in range(len(f['data'])):
                item = f['data'][i]

                # Downsample the point cloud
                item = item[::4, :]

                # Convert the point cloud to a graph where each node represents a point and each edge represents the distance between two points
                G = knn_graph(item, 20)

                # Convert the graph to an adjacency matrix
                A = nx.adjacency_matrix(G).todense().astype(np.float32)

                # Add 1 dimension for the channel
                A = np.expand_dims(A, axis=0)

                # Plot the adjacency matrix
                #import matplotlib.pyplot as plt
                #plt.imshow(A)
                #plt.title(f['label'][i])
                #plt.show()

                self.data.append(A)

                # Convert the label to a one-hot vector
                label = np.zeros(10)
                label[f['label'][i]] = 1
                self.label.append(label)

            f.close()

            # Save to cache
            import pickle
            with open(path + '.cache', 'wb') as f:
                pickle.dump((self.data, self.label), f)

        # Print the number of items
        print('Number of items:', len(self.data))

        # Print the class distribution
        print('Class distribution:', np.sum(self.label, axis=0))

    def __len__(self):
        return len(self.data)

    def split(self, ratio):
        """
        Create two new datasets with the given ratio
        Parameters
        ----------
        ratio : float
            The ratio of the new datasets
        Returns
        -------
        Dataset
            The first dataset
        Dataset
            The second dataset
        """
        left = self.__class__(None)
        left.data = self.data[:int(len(self.data) * ratio)]
        left.label = self.label[:int(len(self.data) * ratio)]

        right = self.__class__(None)
        right.data = self.data[int(len(self.data) * ratio):]
        right.label = self.label[int(len(self.data) * ratio):]

        return left, right

    def __next__(self):
        return self.__getitem__(self.__iter__().__next__())
    
    def __getitem__(self, index):
        """
        Get the item at the given index
        Parameters
        ----------
        index : int
            The index of the item
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The data and the target of the item
        """
        return self.data[index], self.label[index]

    def get_class_weights(self):
        """
        Get the weights for each class
        Returns
        -------
        torch.Tensor
            The weights for each class
        """

        # Compute the weights
        weights = 1 / np.sum(self.label, axis=0)

        # Normalize the weights
        weights = weights / np.sum(weights)

        # Convert to tensor
        weights = torch.Tensor(weights)

        return weights