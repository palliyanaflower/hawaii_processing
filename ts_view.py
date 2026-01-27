#!/usr/bin/env python3
import rosbag2_py
from rclpy.serialization import deserialize_message
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# --- CONFIG ---
camera_bag = Path("data/makalii_point/cam2/bag_camera_2_2025_08_13-01_35_58_12")
lidar_bag  = Path("data/makalii_point/lidar/bag_lidar_2025_08_13-01_35_58_12")
nav_bag    = Path("data/makalii_point/nav/bag_navigation_sensors_2025_08_13-01_35_58")

duration_limit_s = 10.0  # plot first X seconds

bags_and_topics = {
    camera_bag: [
        # "/oak_d_lr_2/rgb/image_rect",
        # "/oak_d_lr_2/stereo/image_raw",
        "/oak_d_lr_2/right/image_raw",
        "/oak_d_lr_2/left/image_raw"
    ],
    lidar_bag: [
        "/ouster/points",
        # "/ouster/range_image",
        # "/ouster/xyz_image",
    ],
    nav_bag: [
        "/imu/nav_sat_fix",
        # "/sbg/ekf_nav",
        # "/sbg/imu_data",
        # "/sbg/ekf_quat",
    ],
}

# --- Helper to extract timestamps from one bag ---
def get_rosbag_timestamps(bag_path, topics, limit_s):
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    all_topics = reader.get_all_topics_and_types()
    topic_names = {t.name for t in all_topics}

    timestamps = {topic: [] for topic in topics if topic in topic_names}
    first_time = None

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic not in timestamps:
            continue

        if first_time is None:
            first_time = t
        dt = (t - first_time) * 1e-9  # ns → s

        if dt > limit_s:
            break

        timestamps[topic].append(dt)

    return timestamps


# --- Collect timestamps from all bags ---
all_timestamps = {}
for bag_path, topics in bags_and_topics.items():
    print(f"Reading {bag_path} ...")
    ts = get_rosbag_timestamps(bag_path, topics, duration_limit_s)
    all_timestamps.update(ts)


# --- Plot ---
plt.figure(figsize=(14, 7))
y_positions = {topic: i + 1 for i, topic in enumerate(all_timestamps.keys())}

for topic, times in all_timestamps.items():
    if len(times) == 0:
        continue
    plt.scatter(times, np.ones_like(times) * y_positions[topic], s=10, label=topic)

plt.yticks(list(y_positions.values()), list(y_positions.keys()))
plt.xlabel("Time [s]")
plt.ylabel("Topic")
plt.title(f"Raw message timestamps from all rosbags (first {duration_limit_s} s)")
# plt.legend(markerscale=2)
plt.tight_layout()
plt.show()
