import numpy as np
from scipy.spatial.distance import pdist, squareform
import networkx as nx

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

def get_point_cloud_in_bbox3d(point_cloud, box):
    """
    Get the point cloud that is inside the bounding box
    """
    x, y, z, h, w, l, ry = box

    # Extract the coordinates from the point cloud
    x_pc = point_cloud[:, 0]
    y_pc = point_cloud[:, 1]
    z_pc = point_cloud[:, 2]

    # Define the boundaries of the bounding box
    x_min = x - l / 2
    x_max = x + l / 2
    y_min = y - w / 2
    y_max = y + w / 2
    z_min = z - h / 2
    z_max = z + h / 2

    # Find the indices of the points within the bounding box
    indices = (x_pc >= x_min) & (x_pc <= x_max) & (y_pc >= y_min) & (y_pc <= y_max) & (z_pc >= z_min) & (z_pc <= z_max)

    # Get the point cloud inside the bounding box
    point_cloud_in_box = point_cloud[indices]

    return point_cloud_in_box

def get_bbox3d_corners(box3d):
    """
    Get the 3D bounding box corners
    """
    x, y, z, h, w, l, ry = box3d

    # Calculate half dimensions
    l2 = l / 2
    w2 = w / 2
    h2 = h / 2

    # Define corner coordinates
    corners = [
        [x + l2 * np.cos(ry) + w2 * np.sin(ry), y + h2, z - l2 * np.sin(ry) + w2 * np.cos(ry)],
        [x + l2 * np.cos(ry) - w2 * np.sin(ry), y + h2, z - l2 * np.sin(ry) - w2 * np.cos(ry)],
        [x - l2 * np.cos(ry) - w2 * np.sin(ry), y + h2, z + l2 * np.sin(ry) - w2 * np.cos(ry)],
        [x - l2 * np.cos(ry) + w2 * np.sin(ry), y + h2, z + l2 * np.sin(ry) + w2 * np.cos(ry)],
        [x + l2 * np.cos(ry) + w2 * np.sin(ry), y - h2, z - l2 * np.sin(ry) + w2 * np.cos(ry)],
        [x + l2 * np.cos(ry) - w2 * np.sin(ry), y - h2, z - l2 * np.sin(ry) - w2 * np.cos(ry)],
        [x - l2 * np.cos(ry) - w2 * np.sin(ry), y - h2, z + l2 * np.sin(ry) - w2 * np.cos(ry)],
        [x - l2 * np.cos(ry) + w2 * np.sin(ry), y - h2, z + l2 * np.sin(ry) + w2 * np.cos(ry)]
    ]

    return np.array(corners)

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