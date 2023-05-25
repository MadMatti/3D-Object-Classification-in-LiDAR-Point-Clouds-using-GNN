from typing import List, Tuple, Union
import h5py

from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Dataset as GeometricDataset, Data
import torch
import torch_geometric.utils as utils
from ordered_set import OrderedSet

import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist, squareform
from utils import knn_graph, resample_point_cloud
import torch_geometric.data as pyg

from tqdm import tqdm

CLASS_NAME_TO_ID = {
    'bathtub': 0,
    'bed': 1,
    'chair': 2,
    'desk': 3,
    'dresser': 4,
    'monitor': 5,
    'night_stand': 6,
    'sofa': 7,
    'table': 8,
    'toilet': 9
}

ID_TO_CLASS_NAME = {
    0: 'bathtub',
    1: 'bed',
    2: 'chair',
    3: 'desk',
    4: 'dresser',
    5: 'monitor',
    6: 'night_stand',
    7: 'sofa',
    8: 'table',
    9: 'toilet'
}

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
        self.classes = OrderedSet()
        
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
        print("Processing dataset")
        if self.path is None:
            return
        
        path = self.path + '/train.h5'
        
        import os
        if os.path.exists(path + '.cache'):
            # Load from cache
            print("Cache found!, loading from cache")
            import pickle
            with open(path + '.cache', 'rb') as f:
                self.data, self.label = pickle.load(f)
        
        else:
            print("Cache not found")

            # Load from file
            f = h5py.File(path, 'r')

            # convert to adjacency matrix
            for i in tqdm(range(len(f['data'])), desc='Progress'):
                item = f['data'][i]
                label_id = f['label'][i][0]

                # Resample the point cloud to 500 points
                item = resample_point_cloud(item, 500)

                # Cast to float64
                item = item.astype(np.float32)

                # Convert the point cloud to a graph where each node represents a point and each edge represents the distance between two points
                G = knn_graph(item, 5)

                # Node features
                x = torch.tensor([features['x'] for _, features in G.nodes(data=True)], dtype=torch.float32)

                # Edge features
                edge_attr = torch.tensor([features['weight'] for _, _, features in G.edges(data=True)], dtype=torch.float32)

                # Edges
                edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous().view(2, -1)

                # Label
                y = torch.tensor([label_id], dtype=torch.long)

                # Create a PyTorch Geometric data object.
                A = pyg.Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

                self.label.append(ID_TO_CLASS_NAME[label_id])
                self.data.append(A)

            f.close()

            # Save to cache
            import pickle
            with open(path + '.cache', 'wb') as f:
                pickle.dump((self.data, self.label), f)

        # Create classes set
        for l in self.label:
            self.classes.add(l)

        # Convert labels to one-hot
        new_labels = []
        for l in self.label:
            new_label = np.zeros(len(self.classes))
            new_label[self.classes.get_loc(l)] = 1

            new_labels.append(new_label)
        self.label = new_labels

        # Print the number of items
        print('Number of items:', len(self.data))

        # Print the class distribution
        print('Class distribution:', np.sum(self.label, axis=0))

    
    def len(self):
        return len(self.data)
    
    def get(self, idx):
        return self.data[idx]