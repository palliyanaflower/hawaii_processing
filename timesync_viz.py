import matplotlib.pyplot as plt
from pathlib import Path
import json
import numpy as np

# --- Paths ---
base_dir = Path("data/haleiwa_neighborhood/processed_data")
timestamps_file = base_dir / "metadata/timestamps.json"

# --- Load timestamps.json ---
with open(timestamps_file, "r") as f:
    timestamps = json.load(f)

# --- Prepare folder names and y positions ---
folders = ["camera", "lidar/img", "lidar/range", "lidar/xyz", "lidar/pcd", "nav/gps"]
y_positions = {name: i + 1 for i, name in enumerate(folders)}

# --- Extract times (convert ns -> s, relative to first LiDAR points timestamp) ---
lidar_ref_times = np.array([item["t"] for item in timestamps["lidar/pcd"]], dtype=np.float64)
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
        if folder == "lidar/pcd" or folder not in times_dict:
            continue
        if i >= len(times_dict[folder]):  # skip if folder has fewer matches
            continue
        t_other = times_dict[folder][i]
        plt.plot([t_ref, t_other], [y0, y_positions[folder]], color="gray", alpha=0.5)

plt.yticks(list(y_positions.values()), list(y_positions.keys()))
plt.xlabel("Time [s]")
plt.title("Matched timestamps per folder (LiDAR points as reference)")
plt.legend()
plt.tight_layout()
plt.show()
