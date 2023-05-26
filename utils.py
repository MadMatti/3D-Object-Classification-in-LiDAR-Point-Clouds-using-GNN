import numpy as np
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import torch_geometric.data as pyg
import torch

def ry_to_rz(ry):
    """
    param ry (float): yaw angle in cam coordinate system
    return: (flaot): yaw angle in velodyne coordinate system
    """
    angle = -ry - np.pi / 2
    angle = np.where(np.greater_equal(angle, np.pi), angle - np.pi, angle)
    angle = np.where(np.less(angle, np.pi), 2 * np.pi + angle, angle)
    return angle


def get_bbox3d(obj_xyz_cam, rot_y, dimensions, tr_velo_to_cam, R_cam_to_rect):
    """returns 3D object location center (x, y, z)"""
    length = dimensions[2]
    width = dimensions[1]
    height = dimensions[0]
    rot_z = ry_to_rz(rot_y)

    # projection from camera coordinates to lidar coordinates
    obj_xyz_cam = np.vstack((obj_xyz_cam.reshape(3,1), [1]))
    rot_mat = np.linalg.inv(R_cam_to_rect @ tr_velo_to_cam)
    obj_xyz_lidar = rot_mat @ obj_xyz_cam
    obj_x = obj_xyz_lidar[0][0]
    obj_y = obj_xyz_lidar[1][0]
    obj_z = obj_xyz_lidar[2][0]

    return np.array([obj_x, obj_y, obj_z, length, width, height, rot_z])

def get_point_cloud_in_bbox3d(point_cloud, bbox):
    """
    Get the point cloud that is inside the bounding box
    """

    x, y, z, w, l, h, rz = bbox
    
    # Rotate the point cloud to make it parallel to the axes
    rotation_matrix = np.array([[np.cos(rz), -np.sin(rz), 0],
                                [np.sin(rz), np.cos(rz), 0],
                                [0, 0, 1]])

    rotated_point_cloud = np.dot(point_cloud - np.array([x, y, z]), rotation_matrix.T)

    # Define the boundaries of the bounding box
    x_min = -w / 2
    x_max = w / 2
    y_min = -l / 2
    y_max = l / 2
    z_min = 0
    z_max = h

    # Filter the points within the bounding box
    mask = (rotated_point_cloud[:, 0] >= x_min) & (rotated_point_cloud[:, 0] <= x_max) \
           & (rotated_point_cloud[:, 1] >= y_min) & (rotated_point_cloud[:, 1] <= y_max) \
           & (rotated_point_cloud[:, 2] >= z_min) & (rotated_point_cloud[:, 2] <= z_max)

    filtered_point_cloud = rotated_point_cloud[mask]

    return filtered_point_cloud

def get_bbox3d_corners(bbox):
    """
    Get the 3D bounding box corners
    """

    x, y, z, w, l, h, rz = bbox
    # x: object center x
    # y: object center y
    # z: object bottom center z
    # w: object width
    # h: object height
    # l: object length
    # rz: object rotation around center z axis
    
    # Calculate the rotation matrix
    rotation_matrix = np.array([[np.cos(rz), -np.sin(rz), 0],
                                [np.sin(rz), np.cos(rz), 0],
                                [0, 0, 1]])
    
    # Calculate the half-dimensions of the box
    half_h = h / 2
    half_w = w / 2
    half_l = l / 2
    
    # Define the eight corners of the box
    corners = np.array([[-half_w, -half_l, 0],
                        [half_w, -half_l, 0],
                        [half_w, half_l, 0],
                        [-half_w, half_l, 0],
                        [-half_w, -half_l, h],
                        [half_w, -half_l, h],
                        [half_w, half_l, h],
                        [-half_w, half_l, h]])
    
    # Rotate and translate the corners
    rotated_corners = np.dot(corners, rotation_matrix.T)
    translated_corners = rotated_corners + np.array([x, y, z])
    
    return translated_corners

def nx_to_torch_geometric(graph, label):
    """
    Convert a networkx graph to torch geometric data format
    """

    # Node features
    x = torch.tensor([features['x'] for _, features in graph.nodes(data=True)], dtype=torch.float32)

    # Edge features
    edge_attr = torch.tensor([features['weight'] for _, _, features in G.edges(data=True)], dtype=torch.float32)

    # Edges
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous().view(2, -1)

    # Label
    y = torch.tensor([label], dtype=torch.long)

    # Create a PyTorch Geometric data object.
    data = pyg.Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    return data

def point_cloud_to_torch_geometric(point_cloud, label, k):
    """ Convert point cloud to torch geometric data format """

    # Resample the point cloud
    point_cloud = resample_point_cloud(point_cloud, k)

    # Construct kNN graph
    graph = knn_graph(point_cloud, k)

    # Convert the graph to torch geometric data format
    data = nx_to_torch_geometric(graph, label)

    return data


def resample_point_cloud(point_cloud, k):
    """
    Resample the point cloud to have a fixed number of points
    """
    # Get the number of points in the point cloud
    num_points = point_cloud.shape[0]

    # If the number of points is less than the number of samples, repeat the points
    if num_points < k:
        indices = np.random.choice(num_points, k - num_points)
        point_cloud = np.vstack((point_cloud, point_cloud[indices]))

    # If the number of points is greater than the number of samples, sample the points
    elif num_points > k:
        indices = np.random.choice(num_points, k)
        point_cloud = point_cloud[indices]

    return point_cloud

def knn_graph_old(data, k):
    """
    Construct a NetworkX graph from the given data
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

    # Construct kNN graph, use 3D coordinates as node features
    G = nx.Graph()
    for i in range(data.shape[0]):
        for j in idx[i, 1:k+1]:
            G.add_edge(i, j, weight=D[i, j])
    for i in range(data.shape[0]):
        G.nodes[i]['x'] = data[i]

    return G

def knn_graph(data, label, k):
    """
    Construct a kNN graph from the given data
    :param data: point cloud data
    :param label: label of the point cloud
    :param k: degree of the graph
    :return: pytorch geometric data object
    """
    
    # Create x from data
    x = np.copy(data)

    # Initialize edge_index
    edge_index = np.empty((2, k*data.shape[0]), dtype=np.int64)

    # Initialize edge_attr
    edge_attr = np.empty((k*data.shape[0]), dtype=np.float32)

    # Compute pairwise distance matrix
    D = pdist(data)
    #D = np.exp(-D)
    D = 1/(1+D)
    D = squareform(D)

    # Sort distance matrix in ascending order and get indices of points
    idx = np.argsort(D, axis=1)

    # Construct kNN graph
    for i in range(data.shape[0]):
        for j in range(k):
            edge_index[0, i*k+j] = i
            edge_index[1, i*k+j] = idx[i, j]
            edge_attr[i*k+j] = D[i, idx[i, j]]
    
    # Convert to torch tensors
    x = torch.tensor(x, dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    y = torch.tensor([label], dtype=torch.long)

    # Create a PyTorch Geometric data object.
    data = pyg.Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    return data 