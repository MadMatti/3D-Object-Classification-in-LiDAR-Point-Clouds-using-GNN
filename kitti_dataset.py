import numpy as np
import mayavi.mlab as mlab

def load_velodyne_file(file_path):
    """Load and parse a velodyne binary file."""
    assert file_path.endswith('.bin'), 'Invalid file format'
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32).reshape(-1, 4)
    return data



# Load data
data = load_velodyne_file('/Volumes/Z8/data_object_velodyne/testing/velodyne/000001.bin')

# Plot data
fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 480))
mlab.points3d(data[:, 0], data[:, 1], data[:, 2], data[:, 2], mode='point', colormap='gnuplot', figure=fig)

# Show plot
mlab.show()
