# import json
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path

# def plot_matches(json_path):

#     with open(json_path, "r") as f:
#         data = json.load(f)

#     # Extract ordered arrays
#     t_cam = np.array([d["t_camera"] for d in data], dtype=np.float64)
#     t_gps = np.array([d["t_gps"] for d in data], dtype=np.float64)
#     dt_us = np.abs(t_cam - t_gps)

#     # Convert to seconds for readability
#     t_cam_s = (t_cam - t_cam[0]) / 1e6
#     t_gps_s = (t_gps - t_gps[0]) / 1e6
#     dt_ms = dt_us / 1000.0

#     # ---- Plot Camera vs GPS time ----
#     plt.figure(figsize=(12,6))
#     plt.plot(t_cam_s, label="Camera timestamps (s)")
#     plt.plot(t_gps_s, label="GPS timestamps (s)")
#     plt.title("Camera vs GPS timestamps (relative)")
#     plt.xlabel("Frame index")
#     plt.ylabel("Time (s, relative to first frame)")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

#     # ---- Plot time difference ----
#     plt.figure(figsize=(12,6))
#     plt.plot(dt_ms, label="|t_cam - t_gps| (ms)")
#     plt.title("Camera ↔ GPS Timestamp Difference")
#     plt.xlabel("Frame index")
#     plt.ylabel("Difference (ms)")
#     plt.grid(True)
#     plt.legend()
#     plt.show()

#     print("Mean time diff (ms):", np.mean(dt_ms))
#     print("Max time diff  (ms):", np.max(dt_ms))
#     print("Min time diff  (ms):", np.min(dt_ms))


# # Example usage
# plot_matches("data/makalii_point/processed_data_imggps/matches.json")

import json
import numpy as np
import matplotlib.pyplot as plt

def plot_pair_matches(json_path):

    # ---- Load JSON ----
    with open(json_path, "r") as f:
        data = json.load(f)

    # Extract timestamps
    t_cam = np.array([d["t_camera"] for d in data], dtype=np.float64)
    t_gps = np.array([d["t_gps"] for d in data], dtype=np.float64)

    # Convert to seconds (relative)
    start_time = min(t_cam[0], t_gps[0])
    t_cam_s = (t_cam - start_time) * 1e-6
    t_gps_s = (t_gps - start_time) * 1e-6

    # ---- Prepare plot layers ----
    folders = ["camera", "gps"]
    y_positions = {"camera": 0, "gps": 1}

    times_dict = {
        "camera": t_cam_s,
        "gps": t_gps_s,
    }

    # ---- Plot ----
    plt.figure(figsize=(12, 6))

    # Scatter timestamps
    for folder in folders:
        ts = times_dict[folder]
        plt.scatter(ts, np.ones_like(ts) * y_positions[folder],
                    s=20, label=folder)

    # Draw connecting lines between matched camera–gps pairs
    for i in range(len(t_cam_s)):
        t0 = t_cam_s[i]
        t1 = t_gps_s[i]

        plt.plot([t0, t1],
                 [y_positions["camera"], y_positions["gps"]],
                 color="gray", alpha=0.5)

    plt.yticks([0, 1], ["camera", "gps"])
    plt.xlabel("Time [s]")
    plt.title("Matched Camera ↔ GPS Timestamps")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("Δt mean (ms):", np.mean(np.abs(t_cam - t_gps)) / 1000)
    print("Δt max  (ms):", np.max(np.abs(t_cam - t_gps)) / 1000)


# Example usage
plot_pair_matches("data/makalii_point/processed_data_imggps/matches.json")
