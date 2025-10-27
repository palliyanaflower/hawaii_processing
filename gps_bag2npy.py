#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from sbg_driver.msg import SbgEkfNav
from rclpy.serialization import deserialize_message

# --- User settings ---
bag_dir = Path("kaneohe_coast/bag_navigation_sensors_2025_08_13-01_35_58")
topic = "/sbg/ekf_nav"
max_pos_error = 2.0  # optional filter
output_dir = bag_dir / "gps_arrays"
output_dir.mkdir(exist_ok=True)

# --- Helper function to read one db3 file ---
def extract_gps_from_db3(db3_path, topic, max_pos_error):
    """
    Extracts lat, lon, alt from one rosbag .db3 file for the specified topic.
    """
    lats, lons, alts = [], [], []

    storage_options = StorageOptions(uri=str(db3_path), storage_id="sqlite3")
    converter_options = ConverterOptions("", "")
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    topics_and_types = reader.get_all_topics_and_types()
    topic_types = {t.name: t.type for t in topics_and_types}
    if topic not in topic_types:
        print(f"Skipping {db3_path.name} — topic '{topic}' not found.")
        return None

    while reader.has_next():
        topic_name, data, t = reader.read_next()
        if topic_name != topic:
            continue

        msg: SbgEkfNav = deserialize_message(data, SbgEkfNav)
        pos_acc = getattr(msg, "position_accuracy", None)

        if pos_acc is None or pos_acc.x < max_pos_error:
            print("Start lat lon: ", msg.latitude, msg.longitude)
            exit()
            lats.append(msg.latitude)
            lons.append(msg.longitude)
            alts.append(msg.altitude)

    if not lats:
        print(f"No GPS data found in {db3_path.name}")
        return None

    return np.column_stack((lats, lons, alts))

# --- Process each db3 file ---
for db3_path in sorted(bag_dir.glob("*.db3")):
    print(f"Processing {db3_path.name} ...")
    latlon_alt = extract_gps_from_db3(db3_path, topic, max_pos_error)

    if latlon_alt is not None:
        out_path = output_dir / f"{db3_path.stem}_gps.npy"
        np.save(out_path, latlon_alt)
        print(f"✅ Saved GPS array to {out_path}")
    else:
        print(f"⚠️ No valid GPS data in {db3_path.name}")

print("\nDone! All GPS arrays saved in:", output_dir)