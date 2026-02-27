from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
import open3d as o3d
import json
from bisect import bisect_left

import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from sbg_driver.msg import SbgEkfNav, SbgEkfQuat, SbgGpsPos

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

# def process_bag_timesync_reference(
#     cam_num,
#     camera_bag,
#     lidar_bag,
#     nav_bag,
#     output_dir,
#     max_time_diff_ms=100.0,
#     num_ds_frames = 1
# ):

#     output_dir = Path(output_dir)

#     # ----------------- directories -----------------
#     (output_dir / "camera/rgb").mkdir(parents=True, exist_ok=True)
#     (output_dir / "camera/left_cam").mkdir(parents=True, exist_ok=True)
#     (output_dir / "camera/right_cam").mkdir(parents=True, exist_ok=True)
#     (output_dir / "lidar/xyz").mkdir(parents=True, exist_ok=True)
#     (output_dir / "lidar/pcd").mkdir(parents=True, exist_ok=True)
#     (output_dir / "lidar/reflect").mkdir(parents=True, exist_ok=True)
#     (output_dir / "lidar/nearir").mkdir(parents=True, exist_ok=True)
#     (output_dir / "lidar/signal").mkdir(parents=True, exist_ok=True)
#     (output_dir / "lidar/range").mkdir(parents=True, exist_ok=True)
#     (output_dir / "nav/gps").mkdir(parents=True, exist_ok=True)
#     (output_dir / "nav/gt").mkdir(parents=True, exist_ok=True)
#     (output_dir / "metadata").mkdir(parents=True, exist_ok=True)

#     # ----------------- queues -----------------
#     rgb_camera_queue = TimeSyncQueue()
#     left_camera_queue = TimeSyncQueue()
#     right_camera_queue = TimeSyncQueue()
#     gps_queue = TimeSyncQueue()
#     gt_trn_queue = TimeSyncQueue()  # Translation component
#     gt_rot_queue = TimeSyncQueue()  # Rotation component
#     reflec_queue = TimeSyncQueue()
#     nearier_queue = TimeSyncQueue()
#     signal_queue = TimeSyncQueue()
#     range_queue = TimeSyncQueue()


#     matched_timestamps = {
#         "camera/rgb": [],
#         "camera/left_cam": [],
#         "camera/right_cam": [],
#         "lidar/xyz": [],
#         "lidar/pcd": [],
#         "lidar/reflec": [],
#         "lidar/nearir": [],
#         "lidar/signal": [],
#         "lidar/range": [],
#         "nav/gps": [],
#         "nav/gt_rot": [],
#         "nav/gt_trn": [],
#     }

#     file_idx = 0
#     lidar_ds_i = 0

#     with AnyReader([Path(camera_bag)]) as reader_cam, \
#          AnyReader([Path(lidar_bag)]) as reader_lidar, \
#          AnyReader([Path(nav_bag)]) as reader_nav:

#         # -------- camera --------
#         for conn, ts, raw in tqdm(reader_cam.messages(), desc="Camera bag"):
#             if conn.topic == f"/oak_d_lr_{cam_num}/rgb/image_raw":
#                 rgb_camera_queue.add(ts, None, raw, conn.msgtype)
#             elif conn.topic == f"/oak_d_lr_{cam_num}/left/image_raw":
#                 left_camera_queue.add(ts, None, raw, conn.msgtype)
#             elif conn.topic == f"/oak_d_lr_{cam_num}/right/image_raw":
#                 right_camera_queue.add(ts, None, raw, conn.msgtype)

#         # -------- nav --------
#         for conn, ts, raw in tqdm(reader_nav.messages(), desc="Nav bag"):
#             # if conn.topic == "/imu/nav_sat_fix":
#             if conn.topic == "/sbg/gps_pos":
#                 gps_queue.add(ts, None, raw, conn.msgtype)
#             elif conn.topic == "/sbg/ekf_nav":
#                 gt_trn_queue.add(ts, None, raw, conn.msgtype)
#             elif conn.topic == "/sbg/ekf_quat":
#                 gt_rot_queue.add(ts, None, raw, conn.msgtype)
       
#         # -------- lidar --------
#         for conn, ts, raw in tqdm(reader_lidar.messages(), desc="LiDAR bag"):
#             if conn.topic == "/ouster/reflec_image":
#                 reflec_queue.add(ts, None, raw, conn.msgtype)
#                 continue
#             elif conn.topic == "/ouster/nearir_image":
#                 nearier_queue.add(ts, None, raw, conn.msgtype)
#                 continue
#             elif conn.topic == "/ouster/signal_image":
#                 signal_queue.add(ts, None, raw, conn.msgtype)
#                 continue
#             elif conn.topic == "/ouster/range_image":
#                 range_queue.add(ts, None, raw, conn.msgtype)
#                 continue
#             elif conn.topic != "/ouster/points":
#                 continue

#             if lidar_ds_i % num_ds_frames != 0:
#                 lidar_ds_i += 1
#                 continue

#             left_entry = left_camera_queue.get_closest(ts, max_diff_ms=max_time_diff_ms)
#             if left_entry is None:
#                 lidar_ds_i += 1
#                 continue
#             right_entry = right_camera_queue.get_closest(left_entry["t"], max_diff_ms=5)
#             rgb_entry = rgb_camera_queue.get_closest(ts, max_diff_ms=max_time_diff_ms)
#             # gt_trn_entry = gps_queue.get_closest(ts, max_diff_ms=max_time_diff_ms)
#             gt_trn_entry = gt_trn_queue.get_closest(ts, max_diff_ms=max_time_diff_ms)
#             gt_rot_entry = gt_rot_queue.get_closest(ts, max_diff_ms=max_time_diff_ms)
#             reflec_entry= reflec_queue.get_closest(ts, max_diff_ms=max_time_diff_ms)
#             nearir_entry= nearier_queue.get_closest(ts, max_diff_ms=max_time_diff_ms)
#             signal_entry= signal_queue.get_closest(ts, max_diff_ms=max_time_diff_ms)
#             range_entry= range_queue.get_closest(ts, max_diff_ms=max_time_diff_ms)

#             # # View time diff
#             # best_dt = float("inf")
#             # for i, entry in enumerate(rgb_camera_queue.queue):
#             #     dt = abs(entry["t"] - ts) / 1e6
#             #     if dt < best_dt:
#             #         best_dt = dt
#             # print("rgb entry ms ", best_dt)

#             if rgb_entry is None or gt_trn_entry is None or gt_rot_entry is None:# or gps_entry is None:
#                 lidar_ds_i += 1
#                 continue

#             # -------- deserialize --------
#             rgb_msg = reader_cam.deserialize(rgb_entry["raw"], rgb_entry["msgtype"])
#             points_msg = deserialize_message(raw, PointCloud2)
#             gt_rot_msg = deserialize_message(gt_rot_entry["raw"], SbgEkfQuat)
#             gt_trn_msg = deserialize_message(gt_trn_entry["raw"], SbgEkfNav)
#             # gt_trn_msg = deserialize_message(gt_trn_entry["raw"], SbgGpsPos)
#             # gps_msg = reader_nav.deserialize(gps_entry["raw"], gps_entry["msgtype"])
#             if reflec_entry is not None: reflec_msg = reader_cam.deserialize(reflec_entry["raw"], reflec_entry["msgtype"])
#             if nearir_entry is not None: nearir_msg = reader_cam.deserialize(nearir_entry["raw"], nearir_entry["msgtype"])
#             if signal_entry is not None: signal_msg = reader_cam.deserialize(signal_entry["raw"], signal_entry["msgtype"])
#             if range_entry is not None: range_msg = reader_cam.deserialize(range_entry["raw"], range_entry["msgtype"])



#             # Optional
#             # left_msg = reader_cam.deserialize(left_entry["raw"], left_entry["msgtype"])
#             # right_msg = reader_cam.deserialize(right_entry["raw"], right_entry["msgtype"])
#             # points_msg = reader_lidar.deserialize(raw, conn.msgtype)

#             # -------- save images --------
#             cv2.imwrite(
#                 str(output_dir / "camera/rgb" / f"{file_idx}.png"),
#                 message_to_cvimage(rgb_msg),
#             )
#             cv2.imwrite(
#                 str(output_dir / "camera/reflec" / f"{file_idx}.png"),
#                 message_to_cvimage(reflec_msg),
#             )    
#             cv2.imwrite(
#                 str(output_dir / "camera/nearir" / f"{file_idx}.png"),
#                 message_to_cvimage(nearir_msg),
#             )
#             cv2.imwrite(
#                 str(output_dir / "camera/signal" / f"{file_idx}.png"),
#                 message_to_cvimage(signal_msg),
#             )
#             cv2.imwrite(
#                 str(output_dir / "camera/range" / f"{file_idx}.png"),
#                 message_to_cvimage(range_msg),
#             )

#             # cv2.imwrite(
#             #     str(output_dir / "camera/left_cam" / f"{file_idx}.png"),
#             #     message_to_cvimage(left_msg),
#             # )
#             # cv2.imwrite(
#             #     str(output_dir / "camera/right_cam" / f"{file_idx}.png"),
#             #     message_to_cvimage(right_msg),
#             # )

#             matched_timestamps["camera/rgb"].append(
#                 {"t": rgb_entry["t"], "file": f"{file_idx}.png"}
#             )
#             # matched_timestamps["camera/left_cam"].append(
#             #     {"t": left_entry["t"], "file": f"{file_idx}.png"}
#             # )
#             # matched_timestamps["camera/right_cam"].append(
#             #     {"t": right_entry["t"], "file": f"{file_idx}.png"}
#             # )

#             # # -------- save GPS --------
#             # gps_data = {
#             #     "timestamp": gps_entry["t"],
#             #     "lat": gps_msg.latitude,
#             #     "lon": gps_msg.longitude,
#             #     "alt": gps_msg.altitude,
#             # }

#             # with open(output_dir / "nav/gps" / f"{file_idx}.json", "w") as f:
#             #     json.dump(gps_data, f, indent=2)

#             # matched_timestamps["nav/gps"].append(
#             #     {"t": gps_entry["t"], "file": f"{file_idx}.json"}
#             # )

#             # -------- save GT (odometry) --------
#             # --- Position from SbgEkfNav ---
#             nav = gt_trn_msg

#             # Convert lat/lon to local ENU or UTM here if needed
#             gt_position = {
#                 "latitude": nav.latitude,
#                 "longitude": nav.longitude,
#                 "altitude": nav.altitude,
#             }


#             # --- Orientation from SbgEkfQuat ---
#             quat_msg = gt_rot_msg  # your SbgEkfQuat message

#             gt_orientation = {
#                 "qx": quat_msg.quaternion.x,
#                 "qy": quat_msg.quaternion.y,
#                 "qz": quat_msg.quaternion.z,
#                 "qw": quat_msg.quaternion.w,
#             }

#             gt_data = {
#                 "timestamp": gt_trn_entry["t"],
#                 "position": gt_position,
#                 "orientation": gt_orientation,
#                 "frame_id": nav.header.frame_id,
#             }

#             with open(output_dir / "nav/gt" / f"{file_idx}.json", "w") as f:
#                 json.dump(gt_data, f, indent=2)

#             matched_timestamps["nav/gt_trn"].append(
#                 {"t": gt_trn_entry["t"], "file": f"{file_idx}.json"}
#             )

#             matched_timestamps["nav/gt_rot"].append(
#                 {"t": gt_rot_entry["t"], "file": f"{file_idx}.json"}
#             )

#             # -------- lidar points --------
#             gen = pc2.read_points(points_msg, field_names=("x", "y", "z"), skip_nans=True)
#             # xyz = np.array([[x, y, z] for x, y, z in gen], dtype=np.float32)
#             points = np.array([ [x, y, z] for x, y, z in gen ], dtype=np.float32) # Unpack tuples
#             xyz = points[:, :3]

#             np.save(output_dir / "lidar/xyz" / f"{file_idx}.npy", xyz)

#             pcd = o3d.geometry.PointCloud()
#             pcd.points = o3d.utility.Vector3dVector(xyz)
#             o3d.io.write_point_cloud(
#                 str(output_dir / "lidar/pcd" / f"{file_idx}.pcd"), pcd
#             )

#             matched_timestamps["lidar/xyz"].append(
#                 {"t": ts, "file": f"{file_idx}.npy"}
#             )
#             matched_timestamps["lidar/pcd"].append(
#                 {"t": ts, "file": f"{file_idx}.pcd"}
#             )

#             # --- Save LiDAR points ---
#             # gen = pc2.read_points(points_msg, field_names=("x", "y", "z"), skip_nans=True)
#             # points = np.array([ [x, y, z] for x, y, z in gen ], dtype=np.float32) # Unpack tuples
#             # xyz = points[:, :3]

#             # xyz_fname = f"{file_idx}.npy"
#             # np.save(output_dir / "lidar/xyz" / xyz_fname, xyz)
#             # matched_timestamps["lidar/xyz"].append({"t": ts, "file": xyz_fname})

#             pcd_fname = f"{file_idx}.pcd"
#             # pcd = o3d.geometry.PointCloud()
#             # pcd.points = o3d.utility.Vector3dVector(xyz)
#             # o3d.io.write_point_cloud(str(output_dir / "lidar/pcd" / pcd_fname), pcd)
#             # matched_timestamps["lidar/pcd"].append({"t": ts, "file": pcd_fname})

#             # Parse manually -> X,Y,Z,I
#             points_np = parse_pointcloud2(points_msg)  # shape: (N, 4)
            
#             # Open3D with intensity
#             pcd = o3d.t.geometry.PointCloud()
#             pcd.point["positions"] = o3d.core.Tensor(points_np[:,:3], dtype=o3d.core.Dtype.Float32)
#             pcd.point["intensity"] = o3d.core.Tensor(points_np[:,3].reshape(-1, 1), dtype=o3d.core.Dtype.Float32)
                
#             o3d.t.io.write_point_cloud(str(output_dir / "lidar/pcd" / pcd_fname), pcd, write_ascii=True)

#             file_idx += 1
#             lidar_ds_i += 1


#     with open(output_dir / "metadata/timestamps.json", "w") as f:
#         json.dump(matched_timestamps, f, indent=2)

#     print(f"✅ Export complete using /imu/odometry as ground truth ({file_idx} frames)")

def find_nearest(entries, ts, max_diff_ms):
    if not entries:
        return None

    times = [e["t"] for e in entries]
    i = bisect_left(times, ts)  # binary search

    candidates = []
    if i < len(entries):
        candidates.append(entries[i])
    if i > 0:
        candidates.append(entries[i - 1])

    best = None
    best_dt = float("inf")

    for c in candidates:
        dt = abs(c["t"] - ts) / 1e6
        if dt < best_dt:
            best_dt = dt
            best = c

    return best if best_dt <= max_diff_ms else None

def process_bag_timesync_reference(
    cam_num,
    camera_bag,
    lidar_bag,
    nav_bag,
    output_dir,
    max_time_diff_ms=100.0,
    num_ds_frames=1,
):
    # =====================================================
    # DIRECTORIES
    # =====================================================
    output_dir = Path(output_dir)
    (output_dir / "metadata").mkdir(parents=True, exist_ok=True)

    (output_dir / "camera/rgb").mkdir(parents=True, exist_ok=True)
    (output_dir / "camera/left_cam").mkdir(parents=True, exist_ok=True)
    (output_dir / "camera/right_cam").mkdir(parents=True, exist_ok=True)
    (output_dir / "lidar/xyz").mkdir(parents=True, exist_ok=True)
    (output_dir / "lidar/pcd").mkdir(parents=True, exist_ok=True)
    (output_dir / "lidar/reflec").mkdir(parents=True, exist_ok=True)
    (output_dir / "lidar/nearir").mkdir(parents=True, exist_ok=True)
    (output_dir / "lidar/signal").mkdir(parents=True, exist_ok=True)
    (output_dir / "lidar/range").mkdir(parents=True, exist_ok=True)
    (output_dir / "nav/gps").mkdir(parents=True, exist_ok=True)
    (output_dir / "nav/gt").mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata").mkdir(parents=True, exist_ok=True)

    matched_timestamps = {
        "camera/rgb": [],
        "camera/left_cam": [],
        "camera/right_cam": [],
        "lidar/xyz": [],
        "lidar/pcd": [],
        "lidar/reflec": [],
        "lidar/nearir": [],
        "lidar/signal": [],
        "lidar/range": [],
        "nav/gps": [],
        "nav/gt": []
    }
    # =====================================================
    # PASS 1 — TIMESTAMP INDEXING ONLY
    # =====================================================
    print("PASS 1: indexing timestamps")

    rgb_msgs = []
    gt_trn_msgs = []
    gt_rot_msgs = []
    reflec_msgs = []
    nearir_msgs = []
    signal_msgs = []
    range_msgs = []
    lidar_msgs = []

    # ---------- CAMERA ----------
    with AnyReader([Path(camera_bag)]) as reader:
        for idx, (conn, ts, _) in enumerate(reader.messages()):

            entry = {"t": ts, "idx": idx}

            if conn.topic == f"/oak_d_lr_{cam_num}/rgb/image_raw":
                rgb_msgs.append(entry)

    # ---------- NAV ----------
    with AnyReader([Path(nav_bag)]) as reader:
        for idx, (conn, ts, _) in enumerate(reader.messages()):

            entry = {"t": ts, "idx": idx}

            if conn.topic == "/sbg/ekf_nav":
                gt_trn_msgs.append(entry)

            elif conn.topic == "/sbg/ekf_quat":
                gt_rot_msgs.append(entry)

    # ---------- LIDAR ----------
    with AnyReader([Path(lidar_bag)]) as reader:
        for idx, (conn, ts, _) in enumerate(reader.messages()):

            entry = {"t": ts, "idx": idx}

            if conn.topic == "/ouster/points":
                lidar_msgs.append(entry)

            elif conn.topic == "/ouster/reflec_image":
                reflec_msgs.append(entry)

            elif conn.topic == "/ouster/nearir_image":
                nearir_msgs.append(entry)

            elif conn.topic == "/ouster/signal_image":
                signal_msgs.append(entry)

            elif conn.topic == "/ouster/range_image":
                range_msgs.append(entry)

    for s in [
        rgb_msgs, gt_trn_msgs, gt_rot_msgs,
        reflec_msgs, nearir_msgs,
        signal_msgs, range_msgs,
        lidar_msgs,
    ]:
        s.sort(key=lambda x: x["t"])

    # =====================================================
    # PASS 2 — BUILD MATCH TABLE
    # =====================================================
    print("PASS 2: timestamp matching")

    matches = []

    for i, lidar in enumerate(tqdm(lidar_msgs)):

        if i % num_ds_frames != 0:
            continue

        ts = lidar["t"]

        rgb = find_nearest(rgb_msgs, ts, max_time_diff_ms)
        trn = find_nearest(gt_trn_msgs, ts, max_time_diff_ms)
        rot = find_nearest(gt_rot_msgs, ts, max_time_diff_ms)

        ref = find_nearest(reflec_msgs, ts, max_time_diff_ms)
        nir = find_nearest(nearir_msgs, ts, max_time_diff_ms)
        sig = find_nearest(signal_msgs, ts, max_time_diff_ms)
        rng = find_nearest(range_msgs, ts, max_time_diff_ms)

        if rgb and trn and rot and ref and nir and sig and rng:
            matches.append({
                "lidar": lidar,
                "rgb": rgb,
                "trn": trn,
                "rot": rot,
                "ref": ref,
                "nir": nir,
                "sig": sig,
                "rng": rng,
            })

    print(f"Matched frames: {len(matches)}")

    # =====================================================
    # PASS 3 — STREAM + EXPORT
    # =====================================================
    print("PASS 3: exporting")

    lidar_lookup = {m["lidar"]["idx"]: i
                    for i, m in enumerate(matches)}

    rgb_lookup = {m["rgb"]["idx"]: i
                  for i, m in enumerate(matches)}

    trn_lookup = {m["trn"]["idx"]: i
                  for i, m in enumerate(matches)}

    rot_lookup = {m["rot"]["idx"]: i
                  for i, m in enumerate(matches)}
    
    ref_lookup = {m["ref"]["idx"]: i
                  for i, m in enumerate(matches)}
    
    nir_lookup = {m["nir"]["idx"]: i
                  for i, m in enumerate(matches)}
    
    sig_lookup = {m["sig"]["idx"]: i
                  for i, m in enumerate(matches)}
    
    rng_lookup = {m["rng"]["idx"]: i
                  for i, m in enumerate(matches)}

    # ---------- CAMERA EXPORT ----------
    with AnyReader([Path(camera_bag)]) as reader:

        for idx, (conn, ts, raw) in enumerate(
                tqdm(reader.messages(), desc="Camera export")):

            if idx not in rgb_lookup:
                continue

            file_idx = rgb_lookup[idx]

            msg = reader.deserialize(raw, conn.msgtype)

            fname = f"{file_idx}.png"
            cv2.imwrite(
                str(output_dir / "camera/rgb" / fname),
                message_to_cvimage(msg),
            )

            matched_timestamps["camera/rgb"].append({"t": ts, "file": fname})

    # ---------- NAV EXPORT ----------
    nav_data = [{} for _ in matches]

    with AnyReader([Path(nav_bag)]) as reader:

        for idx, (conn, ts, raw) in enumerate(
            tqdm(reader.messages(), desc="Nav export")
        ):

            # ----------------------------
            # POSITION
            # ----------------------------
            if idx in trn_lookup:

                file_idx = trn_lookup[idx]

                nav = deserialize_message(raw, SbgEkfNav)

                nav_data[file_idx]["timestamp"] = ts
                nav_data[file_idx]["frame_id"] = nav.header.frame_id
                nav_data[file_idx]["position"] = {
                    "latitude": nav.latitude,
                    "longitude": nav.longitude,
                    "altitude": nav.altitude,
                }

            # ----------------------------
            # ORIENTATION
            # ----------------------------
            elif idx in rot_lookup:

                file_idx = rot_lookup[idx]

                quat = deserialize_message(raw, SbgEkfQuat)

                nav_data[file_idx]["orientation"] = {
                    "qx": quat.quaternion.x,
                    "qy": quat.quaternion.y,
                    "qz": quat.quaternion.z,
                    "qw": quat.quaternion.w,
                }

    for i, gt in enumerate(nav_data):

        if "position" not in gt or "orientation" not in gt:
            continue

        fname = f"{i}.json"
        ts = nav_data[i]["timestamp"]
        matched_timestamps["nav/gt"].append({"t": ts, "file": fname})

        with open(output_dir / "nav/gt" / fname, "w") as f:
            json.dump(gt, f, indent=2)

        

    # ---------- LIDAR EXPORT ----------
    with AnyReader([Path(lidar_bag)]) as reader:

        for idx, (conn, ts, raw) in enumerate(
            tqdm(reader.messages(), desc="LiDAR export")
        ):

            topic = conn.topic

            # =====================================================
            # POINT CLOUD
            # =====================================================
            if idx in lidar_lookup and topic == "/ouster/points":

                file_idx = lidar_lookup[idx]

                points_msg = deserialize_message(raw, PointCloud2)

                points_np = parse_pointcloud2(points_msg)

                pcd = o3d.t.geometry.PointCloud()
                pcd.point["positions"] = o3d.core.Tensor(
                    points_np[:, :3],
                    dtype=o3d.core.Dtype.Float32,
                )
                pcd.point["intensity"] = o3d.core.Tensor(
                    points_np[:, 3].reshape(-1, 1),
                    dtype=o3d.core.Dtype.Float32,
                )

                fname = f"{file_idx}.pcd"
                matched_timestamps["lidar/pcd"].append({"t": ts, "file": fname})

                o3d.t.io.write_point_cloud(
                    str(output_dir / "lidar/pcd" / fname),
                    pcd,
                    write_ascii=True,
                )


            # =====================================================
            # REFLECTIVITY IMAGE
            # =====================================================
            elif idx in ref_lookup and topic == "/ouster/reflec_image":

                file_idx = ref_lookup[idx]

                msg = reader.deserialize(raw, conn.msgtype)

                fname = f"{file_idx}.png"
                matched_timestamps["lidar/reflec"].append({"t": ts, "file": fname})

                cv2.imwrite(
                    str(output_dir / "lidar/reflec" / fname),
                    message_to_cvimage(msg),
                )

            # =====================================================
            # NEAR IR
            # =====================================================
            elif idx in nir_lookup and topic == "/ouster/nearir_image":

                file_idx = nir_lookup[idx]

                msg = reader.deserialize(raw, conn.msgtype)

                fname = f"{file_idx}.png"
                matched_timestamps["lidar/nearir"].append({"t": ts, "file": fname})

                cv2.imwrite(
                    str(output_dir / "lidar/nearir" / fname),
                    message_to_cvimage(msg),
                )

            # =====================================================
            # SIGNAL
            # =====================================================
            elif idx in sig_lookup and topic == "/ouster/signal_image":

                file_idx = sig_lookup[idx]

                msg = reader.deserialize(raw, conn.msgtype)

                fname = f"{file_idx}.png"
                matched_timestamps["lidar/signal"].append({"t": ts, "file": fname})

                cv2.imwrite(
                    str(output_dir / "lidar/signal" / fname),
                    message_to_cvimage(msg),
                )

            # =====================================================
            # RANGE IMAGE
            # =====================================================
            elif idx in rng_lookup and topic == "/ouster/range_image":

                file_idx = rng_lookup[idx]

                msg = reader.deserialize(raw, conn.msgtype)

                fname = f"{file_idx}.png"
                matched_timestamps["lidar/range"].append({"t": ts, "file": fname})

                cv2.imwrite(
                    str(output_dir / "lidar/range" / fname),
                    message_to_cvimage(msg),
                )

    with open(output_dir / "metadata/timestamps.json", "w") as f:
        json.dump(matched_timestamps, f, indent=2)

    print("✅ Export complete")

###############################################
#      process ONE cam bag
################################################ 
# process_bag_timesync_reference(
#     cam_num=2,
#     camera_bag="data/makalii_point/cam2/bag_camera_2_2025_08_13-01_35_58_11",
#     lidar_bag="data/makalii_point/lidar/bag_lidar_2025_08_13-01_35_58_11",
#     nav_bag="data/makalii_point/nav/bag_navigation_sensors_2025_08_13-01_35_58",
#     output_dir="data/makalii_point/processed_lidar_cam_gps/cam2/bag_camera_2_2025_08_13-01_35_58_11",
#     max_time_diff_ms=10
# )

# process_bag_timesync_reference(
#     cam_num=3,
#     camera_bag="data/makalii_point/cam3/bag_camera_3_2025_08_13-01_35_58_40",
#     lidar_bag="data/makalii_point/lidar/bag_lidar_2025_08_13-01_35_58_40",
#     nav_bag="data/makalii_point/nav/bag_navigation_sensors_2025_08_13-01_35_58",
#     output_dir="data/makalii_point/processed_lidar_cam_gps/cam3/bag_camera_3_2025_08_13-01_35_58_40",
# )

# ###############################################
# #      process ALL cam bags
# ###############################################

from pathlib import Path
import re

NUM_DS_FRAMES = 2
TIME_DIFF_MS = 10
cam_num = 2
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
        max_time_diff_ms=TIME_DIFF_MS,
        num_ds_frames=NUM_DS_FRAMES
    )
