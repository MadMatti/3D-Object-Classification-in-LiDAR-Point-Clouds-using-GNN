import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph


path_dataset = '/Users/mattiaevangelisti/Documents/KITTI'
save_path = '/Users/mattiaevangelisti/Documents/KITTI/KITTI_graph'
os.makedirs(save_path, exist_ok=True)


def load_point_cloud(filename):
    point_cloud = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    return point_cloud[:, :3]

def create_knn_graph(point_cloud, k=10):
    point_cloud_tensor = torch.tensor(point_cloud, dtype=torch.float)
    edge_index, edge_weight = knn_graph(point_cloud_tensor, k=k, batch=None, loop=False)
    edge_weight = torch.sqrt(edge_weight)

    graph_data = Data(x=point_cloud_tensor, edge_index=edge_index, edge_attr=edge_weight)

    return graph_data

def process_and_save_graphs(path_dataset, save_path, k=10):
    point_cloud_path = os.path.join(path_dataset, 'data_object_velodyne/training/velodyne')
    point_cloud_files = [os.path.join(point_cloud_path, f) for f in os.listdir(point_cloud_path) if f.endswith('.bin')]

    for i, file in enumerate(point_cloud_files):
        point_cloud = load_point_cloud(file)
        graph_data = create_knn_graph(point_cloud, k=k)

        save_file_path = os.path.join(save_path, f'kitti_graph_{i}.pt')
        torch.save(graph_data, save_file_path)

    print(f"Processed and saved {len(point_cloud_files)} graphs to {save_path}")

def load_kitti_graphs(save_path):
    graph_files = [os.path.join(save_path, f) for f in os.listdir(save_path) if f.endswith('.pt')]
    kitti_graphs = [torch.load(file) for file in graph_files]
    return kitti_graphs


def main():
    process_and_save_graphs(path_dataset, save_path, k=10)


if __name__ == '__main__':
    main()
    

    