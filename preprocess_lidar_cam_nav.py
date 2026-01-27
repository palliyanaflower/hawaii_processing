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

import struct


def parse_pointcloud2(msg):
    """
    Manually parse a PointCloud2 message into XYZ + intensity NumPy array.
    """
    assert msg.point_step > 0
    assert msg.height >= 1

    # Build a lookup table for field offsets
    field_offsets = {f.name: (f.offset, f.datatype) for f in msg.fields}
    
    # Check required fields
    for field in ['x', 'y', 'z', 'intensity']:
        if field not in field_offsets:
            raise ValueError(f"Missing required field '{field}' in PointCloud2")

    # Data type mappings (ROS PointField types → struct format)
    DATATYPES = {
        1: ('B', 1),  # INT8
        2: ('b', 1),  # UINT8
        3: ('H', 2),  # INT16
        4: ('h', 2),  # UINT16
        5: ('I', 4),  # INT32
        6: ('i', 4),  # UINT32
        7: ('f', 4),  # FLOAT32
        8: ('d', 8),  # FLOAT64
    }

    unpack_fmts = {}
    for name, (offset, datatype) in field_offsets.items():
        fmt, size = DATATYPES[datatype]
        unpack_fmts[name] = (offset, fmt)

    points = []
    for i in range(0, len(msg.data), msg.point_step):
        pt_data = msg.data[i:i + msg.point_step]
        try:
            x = struct.unpack_from(unpack_fmts['x'][1], pt_data, unpack_fmts['x'][0])[0]
            y = struct.unpack_from(unpack_fmts['y'][1], pt_data, unpack_fmts['y'][0])[0]
            z = struct.unpack_from(unpack_fmts['z'][1], pt_data, unpack_fmts['z'][0])[0]
            intensity = struct.unpack_from(unpack_fmts['intensity'][1], pt_data, unpack_fmts['intensity'][0])[0]
            if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(z):
                continue
            points.append((x, y, z, intensity))
        except struct.error:
            continue  # Incomplete point

    return np.array(points, dtype=np.float32)  # shape: (N, 4)

class TimeSyncQueue:
    """A queue for incremental timestamp-based matching."""
    def __init__(self, max_len=1000):
        self.queue = []
        self.max_len = max_len

    def add(self, t, msg, raw=None, msgtype=None):
        """Add message to queue with timestamp."""
        self.queue.append({"t": t, "msg": msg, "raw": raw, "msgtype": msgtype})
        # if len(self.queue) > self.max_len:
        #     self.queue.pop(0)

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

def process_bag_timesync_reference(
    cam_num,
    camera_bag,
    lidar_bag,
    nav_bag,
    output_dir,
    max_time_diff_ms=100.0,
):

    output_dir = Path(output_dir)

    # ----------------- directories -----------------
    (output_dir / "camera/rgb").mkdir(parents=True, exist_ok=True)
    (output_dir / "camera/left_cam").mkdir(parents=True, exist_ok=True)
    (output_dir / "camera/right_cam").mkdir(parents=True, exist_ok=True)
    (output_dir / "lidar/xyz").mkdir(parents=True, exist_ok=True)
    (output_dir / "lidar/pcd").mkdir(parents=True, exist_ok=True)
    (output_dir / "nav/gps").mkdir(parents=True, exist_ok=True)
    (output_dir / "nav/gt").mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata").mkdir(parents=True, exist_ok=True)

    # ----------------- queues -----------------
    rgb_camera_queue = TimeSyncQueue()
    left_camera_queue = TimeSyncQueue()
    right_camera_queue = TimeSyncQueue()
    gps_queue = TimeSyncQueue()
    gt_queue = TimeSyncQueue()

    matched_timestamps = {
        "camera/rgb": [],
        "camera/left_cam": [],
        "camera/right_cam": [],
        "lidar/xyz": [],
        "lidar/pcd": [],
        "nav/gps": [],
        "nav/gt": [],
    }

    file_idx = 0
    lidar_ds_i = 0

    with AnyReader([Path(camera_bag)]) as reader_cam, \
         AnyReader([Path(lidar_bag)]) as reader_lidar, \
         AnyReader([Path(nav_bag)]) as reader_nav:

        # -------- camera --------
        for conn, ts, raw in tqdm(reader_cam.messages(), desc="Camera bag"):
            if conn.topic == f"/oak_d_lr_{cam_num}/rgb/image_raw":
                rgb_camera_queue.add(ts, None, raw, conn.msgtype)
            elif conn.topic == f"/oak_d_lr_{cam_num}/left/image_raw":
                left_camera_queue.add(ts, None, raw, conn.msgtype)
            elif conn.topic == f"/oak_d_lr_{cam_num}/right/image_raw":
                right_camera_queue.add(ts, None, raw, conn.msgtype)

        # -------- nav --------
        for conn, ts, raw in tqdm(reader_nav.messages(), desc="Nav bag"):
            if conn.topic == "/imu/nav_sat_fix":
                gps_queue.add(ts, None, raw, conn.msgtype)
            elif conn.topic == "/imu/odometry":
                gt_queue.add(ts, None, raw, conn.msgtype)

        # -------- lidar --------
        for conn, ts, raw in tqdm(reader_lidar.messages(), desc="LiDAR bag"):
            if conn.topic != "/ouster/points":
                continue

            if lidar_ds_i % 50 != 0:
                lidar_ds_i += 1
                continue

            left_entry = left_camera_queue.get_closest(ts, max_diff_ms=max_time_diff_ms)
            if left_entry is None:
                lidar_ds_i += 1
                continue
            right_entry = right_camera_queue.get_closest(left_entry["t"], max_diff_ms=5)
            rgb_entry = rgb_camera_queue.get_closest(ts, max_diff_ms=max_time_diff_ms)
            gps_entry = gps_queue.get_closest(ts, max_diff_ms=max_time_diff_ms)
            gt_entry = gt_queue.get_closest(ts, max_diff_ms=max_time_diff_ms)

            if rgb_entry is None or right_entry is None or gps_entry is None or gt_entry is None:
                lidar_ds_i += 1
                continue

            # -------- deserialize --------
            rgb_msg = reader_cam.deserialize(rgb_entry["raw"], rgb_entry["msgtype"])
            left_msg = reader_cam.deserialize(left_entry["raw"], left_entry["msgtype"])
            right_msg = reader_cam.deserialize(right_entry["raw"], right_entry["msgtype"])
            gps_msg = reader_nav.deserialize(gps_entry["raw"], gps_entry["msgtype"])
            gt_msg = reader_nav.deserialize(gt_entry["raw"], gt_entry["msgtype"])
            # points_msg = reader_lidar.deserialize(raw, conn.msgtype)
            points_msg = deserialize_message(raw, PointCloud2)

            # -------- save images --------
            cv2.imwrite(
                str(output_dir / "camera/rgb" / f"{file_idx}.png"),
                message_to_cvimage(rgb_msg),
            )
            cv2.imwrite(
                str(output_dir / "camera/left_cam" / f"{file_idx}.png"),
                message_to_cvimage(left_msg),
            )
            cv2.imwrite(
                str(output_dir / "camera/right_cam" / f"{file_idx}.png"),
                message_to_cvimage(right_msg),
            )

            matched_timestamps["camera/rgb"].append(
                {"t": rgb_entry["t"], "file": f"{file_idx}.png"}
            )
            matched_timestamps["camera/left_cam"].append(
                {"t": left_entry["t"], "file": f"{file_idx}.png"}
            )
            matched_timestamps["camera/right_cam"].append(
                {"t": right_entry["t"], "file": f"{file_idx}.png"}
            )

            # -------- save GPS --------
            gps_data = {
                "timestamp": gps_entry["t"],
                "lat": gps_msg.latitude,
                "lon": gps_msg.longitude,
                "alt": gps_msg.altitude,
            }

            with open(output_dir / "nav/gps" / f"{file_idx}.json", "w") as f:
                json.dump(gps_data, f, indent=2)

            matched_timestamps["nav/gps"].append(
                {"t": gps_entry["t"], "file": f"{file_idx}.json"}
            )

            # -------- save GT (odometry) --------
            pose = gt_msg.pose.pose
            gt_data = {
                "timestamp": gt_entry["t"],
                "position": {
                    "x": pose.position.x,
                    "y": pose.position.y,
                    "z": pose.position.z,
                },
                "orientation": {
                    "qx": pose.orientation.x,
                    "qy": pose.orientation.y,
                    "qz": pose.orientation.z,
                    "qw": pose.orientation.w,
                },
                "frame_id": gt_msg.header.frame_id,
                "child_frame_id": gt_msg.child_frame_id,
            }

            with open(output_dir / "nav/gt" / f"{file_idx}.json", "w") as f:
                json.dump(gt_data, f, indent=2)

            matched_timestamps["nav/gt"].append(
                {"t": gt_entry["t"], "file": f"{file_idx}.json"}
            )

            # -------- lidar points --------
            gen = pc2.read_points(points_msg, field_names=("x", "y", "z"), skip_nans=True)
            # xyz = np.array([[x, y, z] for x, y, z in gen], dtype=np.float32)
            points = np.array([ [x, y, z] for x, y, z in gen ], dtype=np.float32) # Unpack tuples
            xyz = points[:, :3]

            np.save(output_dir / "lidar/xyz" / f"{file_idx}.npy", xyz)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            o3d.io.write_point_cloud(
                str(output_dir / "lidar/pcd" / f"{file_idx}.pcd"), pcd
            )

            matched_timestamps["lidar/xyz"].append(
                {"t": ts, "file": f"{file_idx}.npy"}
            )
            matched_timestamps["lidar/pcd"].append(
                {"t": ts, "file": f"{file_idx}.pcd"}
            )

            # --- Save LiDAR points ---
            # gen = pc2.read_points(points_msg, field_names=("x", "y", "z"), skip_nans=True)
            # points = np.array([ [x, y, z] for x, y, z in gen ], dtype=np.float32) # Unpack tuples
            # xyz = points[:, :3]

            # xyz_fname = f"{file_idx}.npy"
            # np.save(output_dir / "lidar/xyz" / xyz_fname, xyz)
            # matched_timestamps["lidar/xyz"].append({"t": ts, "file": xyz_fname})

            pcd_fname = f"{file_idx}.pcd"
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(xyz)
            # o3d.io.write_point_cloud(str(output_dir / "lidar/pcd" / pcd_fname), pcd)
            # matched_timestamps["lidar/pcd"].append({"t": ts, "file": pcd_fname})

            # Parse manually -> X,Y,Z,I
            points_np = parse_pointcloud2(points_msg)  # shape: (N, 4)
            
            # Open3D with intensity
            pcd = o3d.t.geometry.PointCloud()
            pcd.point["positions"] = o3d.core.Tensor(points_np[:,:3], dtype=o3d.core.Dtype.Float32)
            pcd.point["intensity"] = o3d.core.Tensor(points_np[:,3].reshape(-1, 1), dtype=o3d.core.Dtype.Float32)
                
            o3d.t.io.write_point_cloud(str(output_dir / "lidar/pcd" / pcd_fname), pcd, write_ascii=True)

            file_idx += 1
            lidar_ds_i += 1


    with open(output_dir / "metadata/timestamps.json", "w") as f:
        json.dump(matched_timestamps, f, indent=2)

    print(f"✅ Export complete using /imu/odometry as ground truth ({file_idx} frames)")


###############################################
#      process ONE cam bag
################################################ process_bag_timesync_reference(
#     camera_bag="data/haleiwa_neighborhood/cam1/bag_camera_1_2025_08_11-22_11_16_0",
#     lidar_bag="data/haleiwa_neighborhood/lidar/bag_lidar_2025_08_11-22_11_16_0",
#     nav_bag="data/haleiwa_neighborhood/nav/bag_navigation_sensors_2025_08_11-22_11_16",
#     output_dir="data/haleiwa_neighborhood/processed_lidar_cam_gps/cam2",
#     max_time_diff_ms=100.0
# )


###############################################
#      process ALL cam bags
###############################################

from pathlib import Path
import re

cam_num = 3
data_root = Path("data/makalii_point")

cam_dir = data_root / f"cam{cam_num}"
lidar_dir = data_root / "lidar"
nav_dir = data_root / "nav"

def index_bags(bag_dir, pattern):
    """
    Returns {idx: Path} for bags matching pattern with trailing _<idx>
    """
    bags = {}
    for p in bag_dir.iterdir():
        if not p.is_dir():
            continue
        m = re.match(pattern, p.name)
        if m:
            idx = int(m.group(1))
            bags[idx] = p
    return bags

camera_bags = index_bags(
    cam_dir,
    rf"bag_camera_{cam_num}_.+_(\d+)$"
)

lidar_bags = index_bags(
    lidar_dir,
    r"bag_lidar_.+_(\d+)$"
)

nav_bag = Path("data/makalii_point/nav/bag_navigation_sensors_2025_08_13-01_35_58")

common_indices = sorted(
    set(camera_bags) & set(lidar_bags)# & set(nav_bags)
)

print(f"Found {len(common_indices)} synchronized bag sets:", common_indices)

for idx in common_indices:
    camera_bag = camera_bags[idx]
    lidar_bag = lidar_bags[idx]
    # nav_bag = nav_bags[idx]

    output_dir = (
        data_root
        / "processed_lidar_cam_gps"
        / f"cam{cam_num}"
        / camera_bag.stem
    )

    print(f"\nProcessing bag {idx}")
    print(f"  Camera: {camera_bag.name}")
    print(f"  LiDAR : {lidar_bag.name}")
    print(f"  Nav   : {nav_bag.name}")

    process_bag_timesync_reference(
        cam_num=cam_num,
        camera_bag=camera_bag,
        lidar_bag=lidar_bag,
        nav_bag=nav_bag,
        output_dir=output_dir,
    )
