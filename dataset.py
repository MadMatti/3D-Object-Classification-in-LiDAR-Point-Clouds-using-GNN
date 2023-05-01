import os 
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
import networkx as nx
import matplotlib.pyplot as plt

path_dataset = '/Users/mattiaevangelisti/Documents/KITTI'

def load_point_cloud(filename):
    point_cloud = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    return point_cloud[:, 0:3]

def create_knn_graph(point_cloud, k=10):
    point_cloud_tensor = torch.tensor(point_cloud, dtype=torch.float)
    edge_index, edge_weight = knn_graph(point_cloud_tensor, k=k, batch=None, loop=False)
    edge_weight = torch.sqrt(edge_weight)

    graph_data = Data(x=point_cloud_tensor, edge_index=edge_index, edge_attr=edge_weight)

    return graph_data
    
def main():
    kitti_graphs = []
    point_cloud_path = os.path.join(path_dataset, 'data_object_velodyne/training/velodyne')
    point_cloud_files = [os.path.join(point_cloud_path, f) for f in os.listdir(point_cloud_path) if f.endswith('.bin')]
    i = 0
    for file in point_cloud_files:
        point_cloud = load_point_cloud(file)
        graph_data = create_knn_graph(point_cloud, k=10)
        kitti_graphs.append(graph_data)
        i += 1
        if i == 10:
            break

    graph_index = 0
    graph_data = kitti_graphs[graph_index]
    

if __name__ == "__main__":
    main()

    