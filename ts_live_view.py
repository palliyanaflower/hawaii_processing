#!/usr/bin/env python3
import rosbag2_py
from rclpy.serialization import deserialize_message
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.cm as cm
from TimeSyncClass import TimeSync

      
# --- CONFIG ---
camera_bag = Path("data/haleiwa_neighborhood/cam1/bag_camera_1_2025_08_11-22_11_16_0")
lidar_bag  = Path("data/haleiwa_neighborhood/lidar/bag_lidar_2025_08_11-22_11_16_0")
nav_bag    = Path("data/haleiwa_neighborhood/nav/bag_navigation_sensors_2025_08_11-22_11_16")

duration_limit_s = 1.0  # Only visualize first X seconds

bags_and_topics = {
    camera_bag: [
        # "/oak_d_lr_1/rgb/image_rect",
        "/oak_d_lr_1/stereo/image_raw",
        # "/oak_d_lr_1/right/image_raw",
    ],
    lidar_bag: [
        "/ouster/points",
        # "/ouster/range_image",
        # "/ouster/xyz_image",
    ],
    nav_bag: [
        # "/imu/nav_sat_fix",
        "/sbg/ekf_nav",
        # "/sbg/imu_data",
        # "/sbg/ekf_quat",
    ],
}

backbone_topic =  "/oak_d_lr_1/stereo/image_raw"

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


# --- Collect timestamps ---
all_timestamps = {}
for bag_path, topics in bags_and_topics.items():
    print(f"Reading {bag_path} ...")
    ts = get_rosbag_timestamps(bag_path, topics, duration_limit_s)
    all_timestamps.update(ts)

# Flatten and sort all timestamps globally for sequential visualization
timeline = []
for topic, times in all_timestamps.items():
    for t in times:
        timeline.append((t, topic))
timeline.sort(key=lambda x: x[0])

# --- Plot interactive timeline ---
plt.ion()
fig, ax = plt.subplots(figsize=(12, 6))
y_positions = {topic: i + 1 for i, topic in enumerate(all_timestamps.keys())}

ax.set_yticks(list(y_positions.values()))
ax.set_yticklabels(list(y_positions.keys()))
ax.set_xlabel("Time [s]")
ax.set_ylabel("Topic")
ax.set_title(f"Step-through visualization (first {duration_limit_s} s)")
ax.set_xlim(0, duration_limit_s)
ax.set_ylim(0, len(y_positions) + 1)

print("\nPress ENTER to advance to the next timestamp. Close window or press Ctrl+C to quit.\n")

# --- Assign each topic a distinct color ---
unique_topics = list(y_positions.keys())
cmap = cm.get_cmap("tab10", len(unique_topics))  # categorical palette
topic_colors = {topic: cmap(i) for i, topic in enumerate(unique_topics)}

# --- Get matches ---
# matcher = TimeSync()
# matched_ts = {}
# for topic in all_timestamps:
#     if topic != backbone_topic:
#         matched_ts[topic] = matcher.get_time_match(all_timestamps[topic], all_timestamps[backbone_topic])

# queues_all = {}
# ts_backbone = []
# key = 1
# # --- Interactive plotting loop ---
# for i, (t, topic) in enumerate(timeline):
#     color = topic_colors[topic]
#     ax.scatter(t, y_positions[topic], s=25, color=color, label=topic)

#     # Add to timesync queue
#     if topic not in queues_all:
#         queues_all[topic] = TimeSync()
#     queues_all[topic].add_to_ts_queue(t)

#     # Update plot
#     plt.pause(0.05)

#     # Only stop for nav_sat_fix (or remove this condition to stop every time)
#     if topic == backbone_topic:
#         ts_backbone.append(t)

#         # Incremental matcher
#         for topic_temp in unique_topics:
#             if topic_temp != backbone_topic:
#                 print("Len queue", topic_temp,  len(queues_all[topic_temp].ts_queue))
                
#                 idx_matched_bb, ts_matched_query = queues_all[topic_temp].get_ts_match(key, ts_backbone)
#                 if idx_matched_bb is not None:
#                     key = idx_matched_bb

#                     if ts_matched_query is not None:
#                         ts_matched_bb = ts_backbone[idx_matched_bb]
#                         ax.plot([ts_matched_bb, ts_matched_query], [y_positions[backbone_topic], y_positions[topic_temp]], color=color)
#         # # Batch matcher
#         # for topic in matched_ts:
#         #     if t in matched_ts[topic]:
#         #         ax.plot([t, matched_ts[topic][t]], [y_positions[backbone_topic], y_positions[topic]], color=color)
#         input(f"[{i+1}/{len(timeline)}] Time: {t:.3f}s | Topic: {topic} — Press ENTER...")


queues_all = {}
ts_backbone = []
key = 0  # start index for backbone timestamps

for i, (t, topic) in enumerate(timeline):
    color = topic_colors[topic]
    ax.scatter(t, y_positions[topic], s=25, color=color, label=topic)

    # Add timestamp to appropriate queue
    if topic not in queues_all:
        queues_all[topic] = TimeSync()
    queues_all[topic].add_to_ts_queue(t)

    plt.pause(0.05)

    if topic == backbone_topic:
        ts_backbone.append(t)

        # Try matching every other topic to this backbone timestamp
        for topic_temp in unique_topics:
            if topic_temp == backbone_topic or topic_temp not in queues_all:
                continue

            print("Len queue", topic_temp, len(queues_all[topic_temp].ts_queue))
            idx_matched_bb, ts_matched_query = queues_all[topic_temp].get_ts_match(key, ts_backbone)

            if ts_matched_query is not None:
                ts_matched_bb = ts_backbone[idx_matched_bb]
                ax.plot(
                    [ts_matched_bb, ts_matched_query],
                    [y_positions[backbone_topic], y_positions[topic_temp]],
                    color=topic_colors[topic_temp],
                )

        key = min(key + 1, len(ts_backbone) - 1)
        # input(f"[{i+1}/{len(timeline)}] Time: {t:.3f}s | Topic: {topic} — Press ENTER...")

plt.ioff()
plt.show()
