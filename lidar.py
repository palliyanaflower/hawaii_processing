import numpy as np
import open3d as o3d
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from pathlib import Path

def create_grid(x_range=10, y_range=10, step=1.0):
    """
    Creates a grid on the XY plane from -x_range..x_range and -y_range..y_range.
    step: distance between lines
    """
    lines = []
    points = []

    # vertical lines (constant x)
    x_vals = np.arange(-x_range, x_range + step, step)
    y_vals = np.arange(-y_range, y_range + step, step)

    for x in x_vals:
        points.append([x, -y_range, 0])
        points.append([x, y_range, 0])
        lines.append([len(points)-2, len(points)-1])

    # horizontal lines (constant y)
    for y in y_vals:
        points.append([-x_range, y, 0])
        points.append([x_range, y, 0])
        lines.append([len(points)-2, len(points)-1])

    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(points)
    grid.lines = o3d.utility.Vector2iVector(lines)
    grid.colors = o3d.utility.Vector3dVector([[0.7,0.7,0.7] for _ in lines])  # light gray
    return grid


bag_path = Path("lidar_data/bag_lidar_2025_08_14-00_56_40_11")
pcd = o3d.geometry.PointCloud()

storage_options = StorageOptions(uri=str(bag_path), storage_id="sqlite3")
converter_options = ConverterOptions("", "")
reader = SequentialReader()
reader.open(storage_options, converter_options)

all_points = []  # accumulator
messages_per_display = 10  # change this to control how often to update display
voxel_size = None          # set to None to skip downsampling
counter = 0

while reader.has_next():
    topic, data, t = reader.read_next()
    if topic != "/ouster/points":
        continue

    msg = deserialize_message(data, PointCloud2)

    # Read points as structured array
    structured_points = np.array(list(pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True)))
    points = np.vstack([structured_points['x'], structured_points['y'], structured_points['z']]).T.astype(np.float32)



    # Optional: downsample the current scan
    if voxel_size is not None:
        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(points)
        temp_pcd = temp_pcd.voxel_down_sample(voxel_size=voxel_size)
        points = np.asarray(temp_pcd.points)

    all_points.append(points)
    counter += 1

    # Display every N messages
    if counter % messages_per_display == 0:
        combined_points = np.vstack(all_points)
        pcd.points = o3d.utility.Vector3dVector(combined_points)

        # Avg distance of point from robot
        distances = np.linalg.norm(combined_points, axis=1)
        avg_distance = np.mean(distances)
        print("Avg dist", avg_distance)

        # Color by height
        colors = (combined_points[:,2] - combined_points[:,2].min()) / (combined_points[:,2].max() - combined_points[:,2].min())
        pcd.colors = o3d.utility.Vector3dVector(np.vstack([colors, colors, colors]).T)

        o3d.visualization.draw_geometries([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)])
        exit()

# Final display of all accumulated points
if all_points:
    combined_points = np.vstack(all_points)
    pcd.points = o3d.utility.Vector3dVector(combined_points)
    colors = (combined_points[:,2] - combined_points[:,2].min()) / (combined_points[:,2].max() - combined_points[:,2].min())
    pcd.colors = o3d.utility.Vector3dVector(np.vstack([colors, colors, colors]).T)
    o3d.visualization.draw_geometries([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)])
