from typing import List, Tuple, Union
import h5py

from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Dataset as GeometricDataset, Data
import torch
import torch_geometric.utils as utils

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

    # Construct kNN graph, use 3D points as node features
    G = nx.Graph()
    for i in range(data.shape[0]):
        for j in idx[i, 1:k+1]:
            G.add_edge(i, j, weight=D[i, j])
    for i in range(data.shape[0]):
        G.nodes[i]['x'] = data[i]

    return G

class Dataset(GeometricDataset):
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
        self.path = path
        self.data = []
        self.label = []
        
        super(Dataset, self).__init__(path)
    
    def processed_file_names(self) -> str | List[str] | Tuple:
        return ['data.pt', 'label.pt']

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
    
    def process(self):
        if self.path is None:
            return
        
        path = self.path + '/train0.h5'
        
        import os
        if os.path.exists(path + '.cache'):
            # Load from cache
            print('Loading from cache...')
            import pickle
            with open(path + '.cache', 'rb') as f:
                self.data, self.label = pickle.load(f)
            
            # Load from file
            f = h5py.File(path, 'r')

            # convert to adjacency matrix
            for i in range(len(f['data'])):
                item = f['data'][i]

                # Downsample the point cloud
                item = item[::4, :]

                # Convert the point cloud to a graph where each node represents a point and each edge represents the distance between two points
                G = knn_graph(item, 1)

                # Convert the graph to an adjacency matrix
                #A = nx.adjacency_matrix(G).todense().astype(np.float32)

                # Add 1 dimension for the channel
                #A = np.expand_dims(A, axis=0)

                # Convert to torch geometric data
                A = utils.from_networkx(G)

                # Add label
                A.y = torch.tensor(f['label'][i], dtype=torch.long)

                # Convert the label to a one-hot vector
                label = np.zeros(10)
                label[f['label'][i]] = 1
                self.label.append(label)

                # Plot the adjacency matrix
                #import matplotlib.pyplot as plt
                #plt.imshow(A)
                #plt.title(f['label'][i])
                #plt.show()

                self.data.append(A)

            f.close()

            # Save to cache
            import pickle
            with open(path + '.cache', 'wb') as f:
                pickle.dump((self.data, self.label), f)

        # Print the number of items
        print('Number of items:', len(self.data))

        # Print the class distribution
        print('Class distribution:', np.sum(self.label, axis=0))

    
    def len(self):
        return len(self.data)
    
    def get(self, idx):
        return self.data[idx]