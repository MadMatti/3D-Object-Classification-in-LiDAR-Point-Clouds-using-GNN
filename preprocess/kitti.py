import os
import numpy as np
import torch
import utils
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

DEBUG = False

DATASET_PATH = "/tmp_workspace/KITTI/"
DATASET_TRAIN_PATH = os.path.join(DATASET_PATH, "training")
DATASET_TEST_PATH = os.path.join(DATASET_PATH, "testing")
SAVE_PATH = os.path.join(DATASET_PATH, "processed")

CLASS_NAMES_TO_IDS = {"Car": 1, "Pedestrian": 2, "Cyclist": 4}
CLASS_IDS_TO_NAMES = {1: "Car", 2: "Pedestrian", 4: "Cyclist"}

NUM_VERTEXES_PER_SAMPLE = 100
NUM_EDGES_PER_VERTEX = 5

os.makedirs(SAVE_PATH, exist_ok=True)

def load_velodyne(bin_path):
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    # ignore reflectivity info
    return obj[:,:3]

def load_labels(file_path, tr_velo_to_cam, matrix_rectification):
    """Extracts relevant information from label file
    0     -> Object type
    1     -> from 0 (non-truncated) to 1 (truncated), where truncated
             refers to the object leaving image boundaries
    2     -> 0 = fully visible, 1 = partly occluded,
             2 = largely occluded, 3 = unknown
    3     -> Observation angle of object [-pi..pi]
    4:7   -> 2D bounding box of object in the image,
             contains left, top, right, bottom pixel coordinates
    8:10  -> 3D object dimensions: height, width, length (in meters)
    11:13 -> The bottom center location x, y, z of the 3D object
             in camera coordinates (in meters)
    14    -> Rotation ry around Y-axis in camera coordinates [-pi..pi]

    Creates 3D bounding box label which contains
    [center (x, y, z), length, width, height, heading]
    """

    with open(file_path) as f:
        lines = f.readlines()

    class_ids_list = []
    truncations_list = []
    occlusions_list = []
    observation_angles_list = []
    bboxs_list = []
    bboxs3D_list = []
    dimensions_list = []
    centers_cam_list = []
    rotations_list = []

    for line in lines:
        obj_label = line.strip().split()
        obj_type = obj_label[0]

        if obj_type in CLASS_NAMES_TO_IDS:
            class_id = CLASS_NAMES_TO_IDS[obj_type]
            truncated = float(obj_label[1])
            occluded = int(obj_label[2])
            alpha = float(obj_label[3])
            bbox_coords = np.array(
                [obj_label[4], obj_label[5], obj_label[6], obj_label[7]]
            ).astype(float)
            dimension = np.array([obj_label[8], obj_label[9], obj_label[10]]).astype(
                float
            )
            center_cam = np.array([obj_label[11], obj_label[12], obj_label[13]]).astype(
                float
            )
            rotation = float(obj_label[14])

            bbox_3d_lidar = utils.get_bbox3d(center_cam, rotation, dimension, tr_velo_to_cam, matrix_rectification)

            class_ids_list.append(class_id)
            truncations_list.append(truncated)
            occlusions_list.append(occluded)
            observation_angles_list.append(alpha)
            bboxs_list.append(bbox_coords)
            bboxs3D_list.append(bbox_3d_lidar)
            dimensions_list.append(dimension)
            centers_cam_list.append(center_cam)
            rotations_list.append(rotation)

    labels_feature_dict = {
        "num_valid_labels": len(class_ids_list),
        "num_obj": len(lines),
        "class_ids": class_ids_list,
        "obj_truncated": truncations_list,
        "obj_occluded": occlusions_list,
        "obj_alpha": observation_angles_list,
        "obj_bbox": bboxs_list,
        "box_3d": bboxs3D_list,
        "obj_dimensions": dimensions_list,
        "obj_center_cam": centers_cam_list,
        "obj_rotation_y": rotations_list,
    }

    return labels_feature_dict

def parse_calib(file_path):
    with open(file_path) as f:
        lines = f.readlines()

    matrix_proj_0 = np.array(lines[0].strip().split(":")[1].split(), dtype=np.float32)
    matrix_proj_1 = np.array(lines[1].strip().split(":")[1].split(), dtype=np.float32)
    matrix_proj_2 = np.array(lines[2].strip().split(":")[1].split(), dtype=np.float32)
    matrix_proj_3 = np.array(lines[3].strip().split(":")[1].split(), dtype=np.float32)
    matrix_rectification = np.array(lines[4].strip().split(":")[1].split(), dtype=np.float32)
    matrix_tr_velo_to_cam = np.array(lines[5].strip().split(":")[1].split(), dtype=np.float32)
    matrix_tr_imu_to_velo = np.array(lines[6].strip().split(":")[1].split(), dtype=np.float32)

    matrix_tr_velo_to_cam = np.vstack((matrix_tr_velo_to_cam.reshape(3,4), [0., 0., 0., 1.]))
    matrix_proj_2 = np.vstack((matrix_proj_2.reshape(3, 4), [0., 0., 0., 0.]))
    R_cam_to_rect = np.eye(4)
    R_cam_to_rect[:3, :3] = matrix_rectification.reshape(3, 3)

    calib_feature_dict = {
        "calib/matrix_proj_0": matrix_proj_0.flatten().tolist(),
        "calib/matrix_proj_1": matrix_proj_1.flatten().tolist(),
        "calib/matrix_proj_2": matrix_proj_2.flatten().tolist(),
        "calib/matrix_proj_3": matrix_proj_3.flatten().tolist(),
        "calib/matrix_rectification": R_cam_to_rect.flatten().tolist(),
        "calib/matrix_tr_velo_to_cam": matrix_tr_velo_to_cam.flatten().tolist(),
        "calib/matrix_tr_imu_to_velo": matrix_tr_imu_to_velo.flatten().tolist(),
    }

    return calib_feature_dict, matrix_tr_velo_to_cam, R_cam_to_rect

def process_and_save_graphs(path_dataset, save_path, k=10):
    POINT_CLOUDS_PATH = os.path.join(path_dataset, "velodyne")
    LABELS_PATH = os.path.join(path_dataset, "label_2")
    CALIB_PATH = os.path.join(path_dataset, "calib")

    num_points_list = []

    # Create the save path
    if not os.path.exists(os.path.join(save_path, "X")):
        os.makedirs(os.path.join(save_path, "X"))
    if not os.path.exists(os.path.join(save_path, "y")):
        os.makedirs(os.path.join(save_path, "y"))

    # List all the files 
    point_clouds_files = os.listdir(POINT_CLOUDS_PATH)
    labels_files = os.listdir(LABELS_PATH)
    calib_files = os.listdir(CALIB_PATH)

    # Sort the files
    point_clouds_files.sort()
    labels_files.sort()
    calib_files.sort()

    # Check if the number of files is the same
    assert len(point_clouds_files) == len(labels_files) == len(calib_files)

    # Create the dataset
    dataset = []

    # Iterate over all the files
    for i in range(len(point_clouds_files)):
        # Get the file name
        point_cloud_file = point_clouds_files[i]
        label_file = labels_files[i]
        calib_file = calib_files[i]

        # Get the path
        point_cloud_path = os.path.join(POINT_CLOUDS_PATH, point_cloud_file)
        label_path = os.path.join(LABELS_PATH, label_file)
        calib_path = os.path.join(CALIB_PATH, calib_file)

        # Load calibration
        calib_feature_dict, matrix_tr_velo_to_cam, R_cam_to_rect = parse_calib(calib_path)

        # Load the point cloud
        point_cloud = load_velodyne(point_cloud_path)

        # Load the 3d object labels
        objects = load_labels(label_path, matrix_tr_velo_to_cam, R_cam_to_rect)
        object_classes = objects["class_ids"]
        object_boxes = objects["box_3d"]

        def draw_box_3d(ax, bbox):
            """
            Draw the 3D bounding box
            """
            # Get the corners
            corners = utils.get_bbox3d_corners(bbox)

            # Extract corner coordinates
            x = [corner[0] for corner in corners]
            y = [corner[1] for corner in corners]
            z = [corner[2] for corner in corners]

            # Connect the corners to form the box
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom rectangle
                (4, 5), (5, 6), (6, 7), (7, 4),  # Top rectangle
                (0, 4), (1, 5), (2, 6), (3, 7)   # Connect top and bottom rectangles
            ]

            for edge in edges:
                ax.plot([x[edge[0]], x[edge[1]]], [y[edge[0]], y[edge[1]]], [z[edge[0]], z[edge[1]]], 'r')


        # For each object in the point cloud
        for i in range(len(object_classes)):
            # Get the class id and name
            class_id = object_classes[i]
            class_name = CLASS_IDS_TO_NAMES[class_id]

            # Get the 3D bounding box
            bbox_3d = object_boxes[i]

            # Get the 3D bounding box corners
            corners = utils.get_bbox3d_corners(bbox_3d)

            # Get the point cloud inside the bounding box
            point_cloud_in_box = utils.get_point_cloud_in_bbox3d(point_cloud, bbox_3d)

            # Discard the bounding boxes with less than 500 points
            num_points = point_cloud_in_box.shape[0]
            num_points_list.append(num_points)
        
            if num_points < 300:
                continue
            
            # Resample the point cloud to have the same number of points
            point_cloud_in_box = utils.resample_point_cloud(point_cloud_in_box, k=NUM_VERTEXES_PER_SAMPLE)

            # Plot the point cloud in 3D
            if DEBUG:          
                print("Class: {}, Number of points: {}".format(class_name, num_points))

                fig = plt.figure()
                
                # Plot the point cloud and the smaller object point cloud in two separate figures
                ax = fig.add_subplot(121, projection='3d')
                plt.title("Point cloud (downsampled 1/5)")
                plt.xlabel("X")
                plt.ylabel("Y")
                ax.set_zlabel("Z")
                ax.scatter(point_cloud[::5,0], point_cloud[::5,1], point_cloud[::5,2], s=0.1)
                # Draw the 3D bounding box
                draw_box_3d(ax, bbox_3d)

                ax = fig.add_subplot(122, projection='3d')
                plt.title("Isolated object (" + class_name + ")")
                plt.xlabel("X")
                plt.ylabel("Y")
                ax.set_zlabel("Z")
                ax.scatter(point_cloud_in_box[:,0], point_cloud_in_box[:,1], point_cloud_in_box[:,2], s=0.5)

                plt.show()

            # Create the graph
            graph = utils.knn_graph(point_cloud_in_box, k=NUM_EDGES_PER_VERTEX)

            # Save the graph
            with open(os.path.join(SAVE_PATH, "X", "graph_{}.pkl".format(i)), "wb") as f:
                pickle.dump(graph, f)

            # Save the label
            with open(os.path.join(SAVE_PATH, "y", "label_{}.txt".format(i)), "w") as f:
                f.write(class_name)

def main():
    # Process the training dataset
    process_and_save_graphs(DATASET_TRAIN_PATH, SAVE_PATH)


if __name__ == '__main__':
    main()
    

    