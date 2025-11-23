import numpy as np
import open3d as o3d
from pathlib import Path
import time


def create_grid(x_range=10, y_range=10, step=1.0):
    """
    Creates a grid on the XY plane from -x_range..x_range and -y_range..y_range.
    """
    lines = []
    points = []

    x_vals = np.arange(-x_range, x_range + step, step)
    y_vals = np.arange(-y_range, y_range + step, step)

    # Vertical lines
    for x in x_vals:
        points.append([x, -y_range, 0])
        points.append([x, y_range, 0])
        lines.append([len(points) - 2, len(points) - 1])

    # Horizontal lines
    for y in y_vals:
        points.append([-x_range, y, 0])
        points.append([x_range, y, 0])
        lines.append([len(points) - 2, len(points) - 1])

    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(points)
    grid.lines = o3d.utility.Vector2iVector(lines)
    grid.colors = o3d.utility.Vector3dVector([[0.7, 0.7, 0.7] for _ in lines])
    return grid


def load_pcd_sequence(pcd_dir):
    """
    Loads all PCD files from a directory, sorted by filename.
    """
    files = sorted(Path(pcd_dir).glob("*.pcd"))
    if not files:
        raise FileNotFoundError(f"No PCD files found in {pcd_dir}")
    return files


# --- Settings ---
pcd_dir = Path("data/cb/processed_data/lidar/pcd")  # <-- change this to your folder
voxel_size = 0.1  # optional downsampling
frame_size = 0.5  # robot frame axis size
pause_time = 0.1  # seconds between frames

# --- Load all PCD files ---
pcd_files = load_pcd_sequence(pcd_dir)

# --- Create grid and initialize visualizer ---
grid = create_grid(x_range=15, y_range=15, step=1.0)
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Map Building", width=1280, height=720)
vis.add_geometry(grid)

# --- Prepare empty map ---
map_pcd = o3d.geometry.PointCloud()
vis.add_geometry(map_pcd)

# --- Create initial robot frame ---
robot_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
vis.add_geometry(robot_frame)

# --- Simulate or load poses ---
# You can replace this with actual robot poses (Nx4x4 matrices)
num_frames = len(pcd_files)
poses = []
for i in range(num_frames):
    pose = np.eye(4)
    pose[0, 3] = i * 0.5  # move forward in x
    poses.append(pose)

# --- Main loop: update map and robot frame ---
for i, (pcd_path, pose) in enumerate(zip(pcd_files, poses)):
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    if voxel_size:
        pcd = pcd.voxel_down_sample(voxel_size)
    pcd.transform(pose)  # transform to global frame

    # Update global map
    map_pcd += pcd
    map_pcd = map_pcd.voxel_down_sample(voxel_size=0.1)

    # Update robot frame position
    robot_frame.transform(np.linalg.inv(robot_frame.get_rotation_matrix_from_xyz([0, 0, 0])))  # reset rotation
    robot_frame.translate(-np.array(robot_frame.get_center()), relative=False)  # reset position
    robot_frame.transform(pose)

    # Update visualization
    vis.update_geometry(map_pcd)
    vis.update_geometry(robot_frame)
    vis.poll_events()
    vis.update_renderer()
    print(f"Frame {i+1}/{num_frames}: Added {pcd_path.name}")
    time.sleep(pause_time)

print("✅ Finished map building visualization.")
vis.run()
vis.destroy_window()
