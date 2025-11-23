from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage
import numpy as np
import cv2, json
from pathlib import Path
from tqdm import tqdm
import open3d as o3d
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from rclpy.serialization import deserialize_message


class TimeSyncQueue:
    """A queue for incremental timestamp-based matching."""
    def __init__(self, max_len=1000):
        self.queue = []
        self.max_len = max_len

    def add(self, t, msg, raw=None, msgtype=None):
        """Add message to queue with timestamp."""
        self.queue.append({"t": t, "msg": msg, "raw": raw, "msgtype": msgtype})
        if len(self.queue) > self.max_len:
            self.queue.pop(0)

    def get_closest(self, t_query, max_diff_ms=200.0):
        """Return and remove the closest message within max_diff_ms to t_query."""
        if not self.queue:
            return None

        best_idx = None
        best_dt = float("inf")
        for i, entry in enumerate(self.queue):
            dt = abs(entry["t"] - t_query) / 1e6
            if dt < best_dt:
                best_dt = dt
                best_idx = i

        if best_dt <= max_diff_ms:
            return self.queue.pop(best_idx)
        else:
            return None


def process_bag_timesync_reference(camera_bag, lidar_bag, output_dir, max_time_diff_ms=100.0):
    output_dir = Path(output_dir)
    (output_dir / "camera").mkdir(parents=True, exist_ok=True)
    for sub in ["img", "pcd", "range", "xyz"]:
        (output_dir / f"lidar/{sub}").mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata").mkdir(parents=True, exist_ok=True)

    # --- Prepare queues ---
    camera_queue = TimeSyncQueue()
    lidar_reflec_queue = TimeSyncQueue()
    lidar_range_queue = TimeSyncQueue()
    
    # This will store mapping from numeric filename -> original timestamp
    matched_timestamps = {
        "camera": [],
        "lidar/img": [],
        "lidar/range": [],
        "lidar/xyz": [],
        "lidar/pcd": []
    }

    file_idx = 0  # numeric filenames

    # --- Open both readers ---
    with AnyReader([Path(camera_bag)]) as reader_cam, AnyReader([Path(lidar_bag)]) as reader_lidar:

        # --- Fill camera and LiDAR queues ---
        print("Reading camera bag into queue...")
        for conn, ts, raw in tqdm(reader_cam.messages(), desc="Camera bag"):
            if conn.topic == "/oak_d_lr_1/rgb/image_raw":
                camera_queue.add(ts, None, raw, conn.msgtype)

        print("Reading lidar bag into queues...")
        for conn, ts, raw in tqdm(reader_lidar.messages(), desc="Lidar bag"):
            if conn.topic == "/ouster/reflec_image":
                lidar_reflec_queue.add(ts, None, raw, conn.msgtype)
            elif conn.topic == "/ouster/range_image":
                lidar_range_queue.add(ts, None, raw, conn.msgtype)
            elif conn.topic == "/ouster/points":
                # LiDAR points are reference timestamps
                cam_entry = camera_queue.get_closest(ts, max_diff_ms=max_time_diff_ms)
                reflec_entry = lidar_reflec_queue.get_closest(ts, max_diff_ms=max_time_diff_ms)
                range_entry = lidar_range_queue.get_closest(ts, max_diff_ms=max_time_diff_ms)

                if cam_entry is None or reflec_entry is None or range_entry is None:
                    continue

                # --- Deserialize messages ---
                cam_msg = reader_cam.deserialize(cam_entry["raw"], cam_entry["msgtype"])
                reflec_msg = reader_lidar.deserialize(reflec_entry["raw"], reflec_entry["msgtype"])
                range_msg = reader_lidar.deserialize(range_entry["raw"], range_entry["msgtype"])
                points_msg = reader_lidar.deserialize(raw, conn.msgtype) # sensor_msgs/msg/PointCloud2
                points_msg = deserialize_message(raw, PointCloud2)
                print("\nlidar type", conn.msgtype)

                # --- Save camera image ---
                cam_img = message_to_cvimage(cam_msg)
                cam_fname = f"{file_idx}.png"
                cv2.imwrite(str(output_dir / "camera" / cam_fname), cam_img)
                matched_timestamps["camera"].append({"t": cam_entry["t"], "file": cam_fname})

                # --- Save LiDAR images ---
                reflec_img = message_to_cvimage(reflec_msg)
                reflec_fname = f"{file_idx}.png"
                cv2.imwrite(str(output_dir / "lidar/img" / reflec_fname), reflec_img)
                matched_timestamps["lidar/img"].append({"t": reflec_entry["t"], "file": reflec_fname})

                range_img = message_to_cvimage(range_msg)
                range_fname = f"{file_idx}.png"
                cv2.imwrite(str(output_dir / "lidar/range" / range_fname), range_img)
                matched_timestamps["lidar/range"].append({"t": range_entry["t"], "file": range_fname})

                # --- Save LiDAR points ---
                gen = pc2.read_points(points_msg, field_names=("x", "y", "z"), skip_nans=True)
                points = np.array([ [x, y, z] for x, y, z in gen ], dtype=np.float32) # Unpack tuples
                xyz = points[:, :3]
 
                xyz_fname = f"{file_idx}.npy"
                np.save(output_dir / "lidar/xyz" / xyz_fname, xyz)
                matched_timestamps["lidar/xyz"].append({"t": ts, "file": xyz_fname})

                pcd_fname = f"{file_idx}.pcd"
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz)
                o3d.io.write_point_cloud(str(output_dir / "lidar/pcd" / pcd_fname), pcd)
                matched_timestamps["lidar/pcd"].append({"t": ts, "file": pcd_fname})

                file_idx += 1  # increment numeric filename

    # --- Save metadata ---
    with open(output_dir / "metadata/timestamps.json", "w") as f:
        json.dump(matched_timestamps, f, indent=2)

    print(f"✅ Fully matched export complete with numeric filenames: {output_dir}")


# Example usage
process_bag_timesync_reference(
    camera_bag="data/haleiwa_neighborhood/cam1/bag_camera_1_2025_08_11-22_11_16_0",
    lidar_bag="data/haleiwa_neighborhood/lidar/bag_lidar_2025_08_11-22_11_16_0",
    output_dir="data/haleiwa_neighborhood/processed_data",
    max_time_diff_ms=100.0
)

# from rosbags.highlevel import AnyReader
# from rosbags.image import message_to_cvimage
# import numpy as np
# import cv2, json
# from pathlib import Path
# from tqdm import tqdm
# import open3d as o3d


# class TimeSyncQueue:
#     """A queue for incremental timestamp-based matching."""
#     def __init__(self, max_len=1000):
#         self.queue = []
#         self.max_len = max_len

#     def add(self, t, msg, raw=None, msgtype=None):
#         """Add message to queue with timestamp."""
#         self.queue.append({"t": t, "msg": msg, "raw": raw, "msgtype": msgtype})
#         if len(self.queue) > self.max_len:
#             self.queue.pop(0)

#     def get_closest(self, t_query, max_diff_ms=200.0):
#         """Return and remove the closest message within max_diff_ms to t_query."""
#         if not self.queue:
#             return None

#         best_idx = None
#         best_dt = float("inf")
#         for i, entry in enumerate(self.queue):
#             dt = abs(entry["t"] - t_query) / 1e6
#             if dt < best_dt:
#                 best_dt = dt
#                 best_idx = i

#         if best_dt <= max_diff_ms:
#             return self.queue.pop(best_idx)
#         else:
#             return None


# def process_bag_timesync_reference(camera_bag, lidar_bag, output_dir, max_time_diff_ms=100.0):
#     output_dir = Path(output_dir)
#     (output_dir / "camera").mkdir(parents=True, exist_ok=True)
#     for sub in ["img", "pcd", "range", "xyz"]:
#         (output_dir / f"lidar/{sub}").mkdir(parents=True, exist_ok=True)
#     (output_dir / "metadata").mkdir(parents=True, exist_ok=True)

#     # --- Prepare queues ---
#     camera_queue = TimeSyncQueue()
#     lidar_reflec_queue = TimeSyncQueue()
#     lidar_range_queue = TimeSyncQueue()
    
#     matched_timestamps = {"camera": [], "lidar": []}

#     # --- Open both readers ---
#     with AnyReader([Path(camera_bag)]) as reader_cam, AnyReader([Path(lidar_bag)]) as reader_lidar:

#         # --- Fill camera and LiDAR queues ---
#         print("Reading camera bag into queue...")
#         for conn, ts, raw in tqdm(reader_cam.messages(), desc="Camera bag"):
#             if conn.topic == "/oak_d_lr_1/rgb/image_raw":
#                 camera_queue.add(ts, None, raw, conn.msgtype)

#         print("Reading lidar bag into queues...")
#         for conn, ts, raw in tqdm(reader_lidar.messages(), desc="Lidar bag"):
#             if conn.topic == "/ouster/reflec_image":
#                 lidar_reflec_queue.add(ts, None, raw, conn.msgtype)
#             elif conn.topic == "/ouster/range_image":
#                 lidar_range_queue.add(ts, None, raw, conn.msgtype)
#             elif conn.topic == "/ouster/points":
#                 # LiDAR points are reference timestamps
#                 # Match camera and other lidar topics to this
#                 cam_entry = camera_queue.get_closest(ts, max_diff_ms=max_time_diff_ms)
#                 reflec_entry = lidar_reflec_queue.get_closest(ts, max_diff_ms=max_time_diff_ms)
#                 range_entry = lidar_range_queue.get_closest(ts, max_diff_ms=max_time_diff_ms)

#                 # Skip if any topic is missing
#                 if cam_entry is None or reflec_entry is None or range_entry is None:
#                     continue

#                 # --- Deserialize messages ---
#                 cam_msg = reader_cam.deserialize(cam_entry["raw"], cam_entry["msgtype"])
#                 reflec_msg = reader_lidar.deserialize(reflec_entry["raw"], reflec_entry["msgtype"])
#                 range_msg = reader_lidar.deserialize(range_entry["raw"], range_entry["msgtype"])
#                 points_msg = reader_lidar.deserialize(raw, conn.msgtype)

#                 # --- Save camera image ---
#                 cam_img = message_to_cvimage(cam_msg)
#                 cam_fname = f"{cam_entry['t']}.png"
#                 cv2.imwrite(str(output_dir / "camera" / cam_fname), cam_img)
#                 matched_timestamps["camera"].append({"t": cam_entry["t"], "file": cam_fname})

#                 # --- Save LiDAR images ---
#                 reflec_img = message_to_cvimage(reflec_msg)
#                 cv2.imwrite(str(output_dir / "lidar/img" / f"{reflec_entry['t']}.png"), reflec_img)

#                 range_img = message_to_cvimage(range_msg)
#                 cv2.imwrite(str(output_dir / "lidar/range" / f"{range_entry['t']}.png"), range_img)

#                 # --- Save LiDAR points ---
#                 arr = np.frombuffer(points_msg.data, dtype=np.float32).reshape(-1, int(points_msg.point_step / 4))
#                 xyz = arr[:, :3]

#                 np.save(output_dir / "lidar/xyz" / f"{ts}.npy", xyz)

#                 pcd = o3d.geometry.PointCloud()
#                 pcd.points = o3d.utility.Vector3dVector(xyz)
#                 o3d.io.write_point_cloud(str(output_dir / "lidar/pcd" / f"{ts}.pcd"), pcd)

#                 matched_timestamps["lidar"].append({"t": ts, "file": f"{ts}.pcd"})

#     # --- Save metadata ---
#     with open(output_dir / "metadata/timestamps.json", "w") as f:
#         json.dump(matched_timestamps, f, indent=2)

#     print(f"✅ Fully matched export complete: {output_dir}")


# # Example usage
# process_bag_timesync_reference(
#     camera_bag="data/haleiwa_neighborhood/cam1/bag_camera_1_2025_08_11-22_11_16_0",
#     lidar_bag="data/haleiwa_neighborhood/lidar/bag_lidar_2025_08_11-22_11_16_0",
#     output_dir="data/haleiwa_neighborhood/processed_data_timesync_reference",
#     max_time_diff_ms=100.0
# )



# from rosbags.highlevel import AnyReader
# from rosbags.image import message_to_cvimage
# import numpy as np
# import cv2, json
# from pathlib import Path
# from tqdm import tqdm
# import open3d as o3d
# import yaml


# def process_bag(camera_bag, lidar_bag, output_dir):
#     output_dir = Path(output_dir)
#     (output_dir / "camera").mkdir(parents=True, exist_ok=True)
#     for sub in ["img", "pcd", "range", "xyz"]:
#         (output_dir / f"lidar/{sub}").mkdir(parents=True, exist_ok=True)
#     (output_dir / "metadata").mkdir(parents=True, exist_ok=True)

#     timestamps = {"camera": [], "lidar": []}

#     # ---- Camera bag ----
#     with AnyReader([Path(camera_bag)]) as reader:
#         for conn, ts, raw in tqdm(reader.messages(), desc="Camera bag"):
#             if conn.topic == "/oak_d_lr_1/rgb/image_raw":
#                 msg = reader.deserialize(raw, conn.msgtype)
#                 img = message_to_cvimage(msg)
#                 fname = f"{ts}.png"
#                 cv2.imwrite(str(output_dir / "camera" / fname), img)
#                 timestamps["camera"].append({"t": ts, "file": fname})

#     # ---- Lidar bag ----
#     with AnyReader([Path(lidar_bag)]) as reader:
#         for conn, ts, raw in tqdm(reader.messages(), desc="Lidar bag"):
#             if conn.topic == "/ouster/reflec_image":
#                 msg = reader.deserialize(raw, conn.msgtype)
#                 img = message_to_cvimage(msg)
#                 cv2.imwrite(str(output_dir / "lidar/img" / f"{ts}.png"), img)

#             elif conn.topic == "/ouster/range_image":
#                 msg = reader.deserialize(raw, conn.msgtype)
#                 img = message_to_cvimage(msg)
#                 cv2.imwrite(str(output_dir / "lidar/range" / f"{ts}.png"), img)

#             elif conn.topic == "/ouster/points":
#                 msg = reader.deserialize(raw, conn.msgtype)
#                 arr = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, int(msg.point_step / 4))
#                 xyz = arr[:, :3]

#                 # Save .npy
#                 np.save(output_dir / "lidar/xyz" / f"{ts}.npy", xyz)

#                 # Save .pcd
#                 pcd = o3d.geometry.PointCloud()
#                 pcd.points = o3d.utility.Vector3dVector(xyz)
#                 o3d.io.write_point_cloud(str(output_dir / "lidar/pcd" / f"{ts}.pcd"), pcd)

#                 timestamps["lidar"].append({"t": ts, "file": f"{ts}.pcd"})

#     # ---- Save metadata ----
#     with open(output_dir / "metadata/timestamps.json", "w") as f:
#         json.dump(timestamps, f, indent=2)

#     print(f"✅ Export complete: {output_dir}")


# # Example usage
# process_bag(
#     camera_bag="data/haleiwa_neighborhood/cam1/bag_camera_1_2025_08_11-22_11_16_0",
#     lidar_bag="data/haleiwa_neighborhood/lidar/bag_lidar_2025_08_11-22_11_16_0",
#     output_dir="data/haleiwa_neighborhood/processed_data"
# )


# def log_timestamps_yaml(data_dir: Path, prefix: str, output_file: Path):
#     entries = []
#     for file in sorted(data_dir.glob(f"{prefix}_*.json")):
#         timestamp = file.stem.split("_")[-1]
#         entries.append({"timestamp": timestamp, "filename": file.name})
