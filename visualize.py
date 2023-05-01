import numpy as np
import os
import open3d as o3d
import cv2
import matplotlib.pyplot as plt

path_dataset = '/Users/mattiaevangelisti/Documents/KITTI'

def load_velodyne_points(filename):
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    return points

def load_calib(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    calib = {}
    for line in lines:
        if ':' not in line:
            continue
        key, value = line.split(':', 1)
        calib[key] = np.array([float(x) for x in value.split()])
    return calib


def read_labels(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    objects = [line.strip().split(' ') for line in lines]
    return objects

def draw_bounding_boxes(image, objects, calib):
    for obj in objects:
        label = obj[0]
        if label not in ['Car', 'Pedestrian', 'Cyclist']:
            continue
        box = np.array([float(x) for x in obj[4:8]]).astype(np.int32)
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

def main():
    frame = 100

    velo_file = os.path.join(path_dataset, 'data_object_velodyne/training/velodyne', '%06d.bin' % frame)
    img_file = os.path.join(path_dataset, 'data_object_image_2/training/image_2', '%06d.png' % frame)
    calib_file = os.path.join(path_dataset, 'data_object_calib/training/calib', '%06d.txt' % frame)
    label_file = os.path.join(path_dataset, 'training/label_2', '%06d.txt' % frame)

    points = load_velodyne_points(velo_file)
    img = cv2.imread(img_file)
    calib = load_calib(calib_file)
    objects = read_labels(label_file)

    # Draw 2D bounding boxes
    draw_bounding_boxes(img, objects, calib)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, 0:3])
    pcd.colors = o3d.utility.Vector3dVector(points[:, 0:3] / 255)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()

    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

if __name__ == '__main__':
    main()