# from rosbags.highlevel import AnyReader
# from rosbags.image import message_to_cvimage
# import numpy as np
# import cv2, json
# from pathlib import Path
# from tqdm import tqdm


# def load_topic_times(reader, topic_name):
#     """Return list of (timestamp, raw, msgtype)."""
#     out = []
#     for conn, ts, raw in reader.messages():
#         if conn.topic == topic_name:
#             out.append((ts, raw, conn.msgtype))
#     return out


# def match_closest(t_query, gps_times, max_diff_ms=50.0):
#     times = [t for t,_,_ in gps_times]

#     # nearest-neighbor search
#     idx = np.searchsorted(times, t_query)
#     candidates = []
#     if idx > 0:
#         candidates.append(idx - 1)
#     if idx < len(times):
#         candidates.append(idx)

#     # convert threshold ms → same units as timestamps (nanoseconds)
#     max_diff_ns = int(max_diff_ms * 1_000_000)  # ms → ns

#     best = None
#     best_dt = max_diff_ns

#     for i in candidates:
#         dt = abs(times[i] - t_query)
#         print(dt)
#         if dt < best_dt:
#             best_dt = dt
#             best = gps_times[i]

#     return best, best_dt


# def export_images_and_gps(camera_bag, gps_bag, output_dir,
#                           cam_topic="/oak_d_lr_3/rgb/image_raw",
#                           gps_topic="/imu/nav_sat_fix",
#                           N=5):

#     output_dir = Path(output_dir)
#     (output_dir / "camera").mkdir(parents=True, exist_ok=True)
#     (output_dir / "gps").mkdir(parents=True, exist_ok=True)

#     # ---- Load camera frames ----
#     print("Loading camera bag...")
#     with AnyReader([Path(camera_bag)]) as reader_cam:
#         cam_data = load_topic_times(reader_cam, cam_topic)

#     # ---- Load gps ----
#     print("Loading GPS bag...")
#     with AnyReader([Path(gps_bag)]) as reader_gps:
#         gps_data = load_topic_times(reader_gps, gps_topic)

#     # sort (they normally already are)
#     cam_data.sort(key=lambda x: x[0])
#     gps_data.sort(key=lambda x: x[0])

#     print(f"Loaded {len(cam_data)} images, {len(gps_data)} gps measurements")

#     matched = []

#     file_idx = 0
#     for i, (t_cam, raw_cam, msgtype_cam) in enumerate(tqdm(cam_data)):

#         # process every Nth image
#         if i % N != 0:
#             continue

#         # ---- find nearest GPS measurement ----
#         (gps_match, dt) = match_closest(t_cam, gps_data)

#         if gps_match is not None:
#             (t_gps, raw_gps, msgtype_gps) = gps_match

#             # deserialize msgs
#             with AnyReader([Path(camera_bag)]) as r1:
#                 cam_msg = r1.deserialize(raw_cam, msgtype_cam)
#             with AnyReader([Path(gps_bag)]) as r2:
#                 gps_msg = r2.deserialize(raw_gps, msgtype_gps)

#             # ---- Save image ----
#             cam_img = message_to_cvimage(cam_msg)
#             cam_fname = f"{file_idx}.png"
#             cv2.imwrite(str(output_dir / "camera" / cam_fname), cam_img)

#             # ---- Save GPS ----
#             gps_data_out = {
#                 "timestamp": t_gps,
#                 "lat": gps_msg.latitude,
#                 "lon": gps_msg.longitude,
#                 "alt": gps_msg.altitude,
#                 "timediff_us": int(abs(t_cam - t_gps))
#             }
#             gps_fname = f"{file_idx}.json"
#             with open(output_dir / "gps" / gps_fname, "w") as f:
#                 json.dump(gps_data_out, f, indent=2)

#             matched.append({
#                 "camera_file": cam_fname,
#                 "gps_file": gps_fname,
#                 "t_camera": t_cam,
#                 "t_gps": t_gps
#             })

#             file_idx += 1

#     # ---- Save metadata ----
#     with open(output_dir / "matches.json", "w") as f:
#         json.dump(matched, f, indent=2)

#     print(f"Done! Exported {file_idx} matched image+gps pairs into {output_dir}")


# # Example call
# export_images_and_gps(
#     camera_bag="data/makalii_point/cam2/bag_camera_2_2025_08_13-01_35_58_18",
#     gps_bag="data/makalii_point/nav/bag_navigation_sensors_2025_08_13-01_35_58",
#     output_dir="data/makalii_point/processed_data_imggps/cam2",
#     cam_topic="/oak_d_lr_2/rgb/image_raw",
#     gps_topic="/imu/nav_sat_fix",
#     N=30     # process every Nth image
# )

###############################################
#      🔥 NEW SECTION – process ALL cam3 bags
###############################################

from pathlib import Path
import numpy as np
from rosbags.highlevel import AnyReader
import cv2
import yaml
import os
import json

def load_topic_times(reader, topic_name):
    """Return list of (timestamp_ns, raw, msgtype)."""
    out = []
    for conn, ts, raw in reader.messages():
        if conn.topic == topic_name:
            out.append((ts, raw, conn.msgtype))
    return out


def match_closest(t_query, gps_times, max_diff_ms=100.0):
    times = [t for t,_,_ in gps_times]

    idx = np.searchsorted(times, t_query)
    candidates = []
    if idx > 0:
        candidates.append(idx - 1)
    if idx < len(times):
        candidates.append(idx)

    max_diff_ns = int(max_diff_ms * 1_000_000)  # ms→ns
    best, best_dt = None, max_diff_ns

    for i in candidates:
        dt = abs(times[i] - t_query)
        if dt < best_dt:
            best_dt = dt
            best = gps_times[i]

    return best, best_dt


def export_images_and_gps(camera_bag, gps_data, output_dir,
                          cam_topic="/oak_d_lr_3/rgb/image_raw",
                          gps_topic="/imu/nav_sat_fix",
                          N=5):

    output_dir = Path(output_dir)
    (output_dir / "camera").mkdir(parents=True, exist_ok=True)
    (output_dir / "gps").mkdir(parents=True, exist_ok=True)

    print(f"\n=== Processing {camera_bag} ===")

    # Load camera messages
    with AnyReader([camera_bag]) as reader_cam:
        cam_data = load_topic_times(reader_cam, cam_topic)

    cam_data.sort(key=lambda x: x[0])  # enforce ordering

    matched = []   # <-- NEW (same as old function)
    file_idx = 0

    # ---- Iterate over camera frames ----
    for i, (t_cam, raw_cam, msgtype) in enumerate(cam_data):
        if i % N != 0:
            continue

        match, dt = match_closest(t_cam, gps_data)
        if match is None:
            print(f"⚠ No GPS match for timestamp {t_cam}")
            continue

        # Deserialize msgs
        with AnyReader([camera_bag]) as r_cam:
            img_msg = r_cam.deserialize(raw_cam, msgtype)
        gps_msg = r_cam.deserialize(match[1], match[2]) # reusing same reader

        # ---- Save Image ----
        img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, -1)
        img_fname = f"{file_idx}.png"
        img_path = output_dir / "camera" / img_fname
        cv2.imwrite(str(img_path), img)

        # ---- Save GPS ----
        gps_out = {
            "timestamp_ns": int(match[0]),
            "lat": gps_msg.latitude,
            "lon": gps_msg.longitude,
            "alt": gps_msg.altitude,
            "offset_ns": int(dt)
        }
        gps_fname = f"{file_idx}.yaml"
        with open(output_dir / "gps" / gps_fname, "w") as f:
            yaml.dump(gps_out, f)

        # ---- Store match mapping (same structure style as old) ----
        matched.append({
            "camera_file": img_fname,
            "gps_file": gps_fname,
            "t_camera": int(t_cam),
            "t_gps": int(match[0]),
            "offset_ns": int(dt)
        })

        print(f"saved {img_fname}, gps={gps_fname}, offset={dt/1e6:.1f}ms")
        file_idx += 1

    # ---- Save summary file ----
    with open(output_dir / "matches.json", "w") as f:
        json.dump(matched, f, indent=2)

    print(f"\nDONE → {file_idx} matched image+gps pairs exported")
    print(f"matches file saved at: {output_dir/'matches.json'}")


###############################################
#      process ALL cam bags
###############################################

cam_num = 2
cam_root = Path("data/makalii_point/cam" + str(cam_num))
gps_bag = "data/makalii_point/nav/bag_navigation_sensors_2025_08_13-01_35_58"

# load GPS once (to avoid repeated parsing)
with AnyReader([Path(gps_bag)]) as reader_gps:
    gps_data = load_topic_times(reader_gps, "/imu/nav_sat_fix")
gps_data.sort(key=lambda x: x[0])

for camera_bag in sorted(cam_root.glob("bag_camera_*")):
    export_images_and_gps(
        camera_bag=camera_bag,
        cam_topic="/oak_d_lr_" + str(cam_num) +"/rgb/image_raw",
        gps_data=gps_data,
        output_dir="data/makalii_point/processed_data_imggps/cam" + str(cam_num) + f"/{camera_bag.stem}",
        N=100
    )
