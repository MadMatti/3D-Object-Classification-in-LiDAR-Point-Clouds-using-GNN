from typing import List, Tuple, Union

from torch_geometric.data import Dataset as GeometricDataset, Data
import torch
import torch_geometric.utils as utils

from ordered_set import OrderedSet
import numpy as np
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import networkx as nx
import torch_geometric.data as pyg
import concurrent.futures

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

        # Node features
        x = torch.tensor([features['x'] for _, features in G.nodes(data=True)], dtype=torch.float32)

        # Edge features
        edge_attr = torch.tensor([features['weight'] for _, _, features in G.edges(data=True)], dtype=torch.float32)

        # Edges
        edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous().view(2, -1)

        # Label
        y = torch.tensor([label_id], dtype=torch.long)

        # Create a PyTorch Geometric data object.
        data = pyg.Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        return data, label

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

            # Create a ThreadPoolExecutor with the desired number of threads
            with ThreadPoolExecutor(max_workers=5) as executor:
                # Create a list to store the futures
                futures = []

                # Submit tasks to the executor for each pair of graph_file and label_file
                for graph_file, label_file in zip(graph_files, label_files):
                    future = executor.submit(self.process_sample, graph_file, label_file)
                    futures.append(future)

                for future in tqdm(as_completed(futures), desc="Progress", total=len(futures)):
                    A, label = future.result()
                    self.data.append(A)
                    self.label.append(label)

            # Save to cache
            with open(self.path + '.cache', 'wb') as f:
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