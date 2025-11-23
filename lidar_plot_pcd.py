import numpy as np
import open3d as o3d
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
        lines.append([len(points) - 2, len(points) - 1])

    # horizontal lines (constant y)
    for y in y_vals:
        points.append([-x_range, y, 0])
        points.append([x_range, y, 0])
        lines.append([len(points) - 2, len(points) - 1])

    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(points)
    grid.lines = o3d.utility.Vector2iVector(lines)
    grid.colors = o3d.utility.Vector3dVector([[0.7, 0.7, 0.7] for _ in lines])  # light gray
    return grid


# --- Load PCD file ---
pcd_path = Path("data/cb/processed_data/lidar/pcd/99.pcd")  # <-- change this path
# pcd_path = Path("/home/kalliyanlay/Documents/BYU/research/camera_lidar_calibration/data/multicam_lidar_calib_data/lidar/pcd/0003.pcd")  # <-- change this path

if not pcd_path.exists():
    raise FileNotFoundError(f"PCD file not found: {pcd_path}")

pcd = o3d.io.read_point_cloud(str(pcd_path))
points = np.asarray(pcd.points)

if points.size == 0:
    raise RuntimeError("No points found in PCD file.")

# Optional: downsample (set voxel_size to None to disable)
voxel_size = None  # e.g., 0.1 for downsampling
if voxel_size is not None:
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    points = np.asarray(pcd.points)

# Compute average distance from origin
distances = np.linalg.norm(points, axis=1)
avg_distance = np.mean(distances)
print(f"Avg distance from origin: {avg_distance:.3f} m")

# Color points by height (Z value)
z_vals = points[:, 2]
colors = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min())
pcd.colors = o3d.utility.Vector3dVector(np.vstack([colors, colors, colors]).T)

# Visualize with grid and coordinate frame
grid = create_grid(x_range=10, y_range=10, step=1.0)
o3d.visualization.draw_geometries([pcd, grid, o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)])