from typing import List, Tuple, Union

from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Dataset as GeometricDataset, Data
import torch
import torch_geometric.utils as utils

import numpy as np
import pickle

from preprocess import kitti as kitti_preprocess

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
        
        import os
        if os.path.exists(self.path + '.cache'):
            # Load from cache
            print('Loading from cache...')
            with open(self.path + '.cache', 'rb') as f:
                self.data, self.label = pickle.load(f)
        
        else:
            # Get class mapping
            class_id_to_names = kitti_preprocess.CLASS_IDS_TO_NAMES
            class_names_to_id = kitti_preprocess.CLASS_NAMES_TO_IDS
            n_classes = len(class_id_to_names.items())

            # List all files in the directory
            graph_files = os.listdir(os.path.join(self.path, 'X'))
            label_files = os.listdir(os.path.join(self.path, 'y'))

            # Sort the files
            graph_files.sort()
            label_files.sort()

            # Load nx graphs from pkls
            for graph_file, label_file in zip(graph_files, label_files):
                # Load the graph
                G = pickle.load(open(os.path.join(self.path, 'X', graph_file), 'rb'))

                # Load the label
                with open(os.path.join(self.path, 'y', label_file), 'r') as f:
                    label = f.read()
                    label = label.strip()
                label_id = class_names_to_id[label]

                # Convert the graph to pytorch geometric data
                A = utils.from_networkx(G)

                # Add label
                A.y = torch.tensor(label_id, dtype=torch.long)

                # Convert the label to a one-hot vector
                label = np.zeros(n_classes)
                label[label_id] = 1
                
                self.label.append(label)

                self.data.append(A)

            # Save to cache
            with open(self.path + '.cache', 'wb') as f:
                pickle.dump((self.data, self.label), f)

        # Print the number of items
        print('Number of items:', len(self.data))

        # Print the class distribution
        print('Class distribution:', np.sum(self.label, axis=0))

    
    def len(self):
        return len(self.data)
    
    def get(self, idx):
        return self.data[idx]