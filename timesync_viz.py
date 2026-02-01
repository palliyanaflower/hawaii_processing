import matplotlib.pyplot as plt
from pathlib import Path
import json
import numpy as np

# --- Paths ---
# base_dir = Path("data/haleiwa_neighborhood/processed_lidar_cam_gps/cam1/bag_camera_1_2025_08_11-22_11_16_0")
# base_dir = Path("data/makalii_point/processed_lidar_cam_gps/cam3/bag_camera_3_2025_08_13-01_35_58_48")
base_dir = Path("data/makalii_point/processed_lidar_cam_gps/cam2/bag_camera_2_2025_08_13-01_35_58_12")
timestamps_file = base_dir / "metadata/timestamps.json"

# --- Load timestamps.json ---
with open(timestamps_file, "r") as f:
    timestamps = json.load(f)

# --- Prepare folder names and y positions ---
# folders = ["camera/left_cam", "camera/right_cam", "lidar/img", "lidar/range", "lidar/xyz", "lidar/pcd", "nav/gps"]
folders = ["camera/rgb", "camera/right_cam", "camera/left_cam", "lidar/pcd", "nav/gps"]
y_positions = {name: i + 1 for i, name in enumerate(folders)}

# --- Extract times (convert ns -> s, relative to first LiDAR points timestamp) ---
lidar_ref_times = np.array([item["t"] for item in timestamps["lidar/pcd"]], dtype=np.float64)
cam_left_ref_times = np.array([item["t"] for item in timestamps["camera/left_cam"]], dtype=np.float64)

start_time = lidar_ref_times[0]

times_dict = {}
for folder in folders:
    # Some folders may not exist in timestamps
    if folder not in timestamps:
        continue
    times_dict[folder] = np.array([item["t"] for item in timestamps[folder]], dtype=np.float64)
    times_dict[folder] = (times_dict[folder] - start_time) * 1e-9  # ns -> s

# --- Plot ---
plt.figure(figsize=(12, 6))

# Plot scatter for each folder
for folder in folders:
    if folder not in times_dict:
        continue
    ts = times_dict[folder]
    plt.scatter(ts, np.ones_like(ts) * y_positions[folder], s=20, label=folder)

# Draw lines from LiDAR points (reference) to other folders
for i in range(len(lidar_ref_times)):
    y0 = y_positions["lidar/pcd"]
    t_ref = (lidar_ref_times[i] - start_time) * 1e-9

    for folder in folders:
        if folder == "lidar/pcd" or folder == "camera/right_cam" or folder not in times_dict:
            continue
        if i >= len(times_dict[folder]):  # skip if folder has fewer matches
            continue
        t_other = times_dict[folder][i]
        plt.plot([t_ref, t_other], [y0, y_positions[folder]], color="gray", alpha=0.5)

# Draw lines from right cam to left cam
for i in range(len(cam_left_ref_times)):
    t_ref = (cam_left_ref_times[i] - start_time) * 1e-9
    t_other = times_dict["camera/right_cam"][i]
    plt.plot([t_ref, t_other], [y_positions["camera/left_cam"], y_positions["camera/right_cam"]], color="gray", alpha=0.5)


plt.yticks(list(y_positions.values()), list(y_positions.keys()))
plt.xlabel("Time [s]")
plt.title("Matched timestamps per folder (LiDAR points as reference)")
plt.legend()
plt.tight_layout()
plt.show()
