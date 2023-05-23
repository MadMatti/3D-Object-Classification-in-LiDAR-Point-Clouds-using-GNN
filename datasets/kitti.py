from typing import List, Tuple, Union

from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Dataset as GeometricDataset, Data
import torch
import torch_geometric.utils as utils

from ordered_set import OrderedSet
import numpy as np
import pickle
from tqdm import tqdm
import multiprocessing
import os

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
    
    def process_sample(self, graph_file, label_file):
        # Load the graph
        with open(graph_file, "rb") as f:
            G = pickle.load(f)

        # Load the label
        with open(label_file, 'r') as f:
            label = f.read()
            label = label.strip()

        label_id = self.classes.add(label)

        # Convert the graph to pytorch geometric data
        A = utils.from_networkx(G)

        # Add label
        A.y = torch.tensor(label_id, dtype=torch.long)

        # Convert the label to a one-hot vector
        #label = np.zeros(len(class_names_to_id.items()))
        #label[label_id] = 1

        return A, label

    def process(self):
        print("Processing dataset")
        if self.path is None:
            return

        if os.path.exists(self.path + ".cache"):
            # Load from cache
            print("Cache found!, loading from cache")
            with open(self.path + '.cache', 'rb') as f:
                self.data, self.label = pickle.load(f)

        else:
            print("Cache not found")

            # List all files in the directory
            graph_files = [os.path.join(self.path, 'X', x) for x in os.listdir(os.path.join(self.path, 'X'))]
            label_files = [os.path.join(self.path, 'y', x) for x in os.listdir(os.path.join(self.path, 'y'))]

            # Sort the files
            graph_files.sort()
            label_files.sort()

            # Parallelize the loop using multiprocessing
            pool = multiprocessing.Pool(processes=30)
            results = []
            for graph_file, label_file in zip(graph_files, label_files):
                results.append(pool.apply_async(self.process_sample, (graph_file, label_file)))

            for result in tqdm(results, desc="Progress", total=len(results)):
                A, label = result.get()
                self.data.append(A)
                self.label.append(label)
                self.classes.add(label)

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