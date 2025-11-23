#!/usr/bin/env python3
import shutil
from pathlib import Path
import yaml

# -----------------------------
# USER INPUTS
# -----------------------------
camera_bag_folder = Path("cam_data/bag_camera_2_2025_08_14-00_56_40_2")
info_bag_folder = Path("caminfo_data/bag_camera_info_2025_08_14-00_56_41_2")
output_folder = Path("cammerged_data/bag_cammerged_2025_08_14-00_56_41_2")
merged_yaml_name = "metadata.yaml"

# -----------------------------
# Ensure output folder is clean
# -----------------------------
if output_folder.exists():
    shutil.rmtree(output_folder)
output_folder.mkdir(parents=True)

# -----------------------------
# Copy the .db3 files
# -----------------------------
camera_bag_file = camera_bag_folder / f"{camera_bag_folder.name}.db3"
info_bag_file = info_bag_folder / f"{info_bag_folder.name}.db3"

camera_bag_copy = output_folder / camera_bag_file.name
info_bag_copy = output_folder / info_bag_file.name

shutil.copy2(camera_bag_file, camera_bag_copy)
shutil.copy2(info_bag_file, info_bag_copy)

print(f"Copied bag files to {output_folder}")

# -----------------------------
# Merge the YAML files
# -----------------------------
camera_yaml_file = camera_bag_folder / "metadata.yaml"
info_yaml_file = info_bag_folder / "metadata.yaml"

with open(camera_yaml_file, "r") as f:
    camera_yaml = yaml.safe_load(f)

with open(info_yaml_file, "r") as f:
    info_yaml = yaml.safe_load(f)

# Get the info dicts
camera_info = camera_yaml["rosbag2_bagfile_information"]
info_info = info_yaml["rosbag2_bagfile_information"]

# Merge files and paths
camera_info["files"].extend(info_info["files"])
camera_info["relative_file_paths"].extend(info_info["relative_file_paths"])

# Merge message counts
camera_info["message_count"] += info_info["message_count"]

# Merge topics_with_message_count
if "topics_with_message_count" in camera_info and "topics_with_message_count" in info_info:
    topic_map = {t["topic_metadata"]["name"]: t for t in camera_info["topics_with_message_count"]}

    for t in info_info["topics_with_message_count"]:
        name = t["topic_metadata"]["name"]
        if name in topic_map:
            topic_map[name]["message_count"] += t["message_count"]
        else:
            topic_map[name] = t

    camera_info["topics_with_message_count"] = list(topic_map.values())

# Merge timing info
camera_info["starting_time"]["nanoseconds_since_epoch"] = min(
    camera_info["starting_time"]["nanoseconds_since_epoch"],
    info_info["starting_time"]["nanoseconds_since_epoch"]
)

camera_info["duration"]["nanoseconds"] = max(
    camera_info["duration"]["nanoseconds"],
    info_info["duration"]["nanoseconds"]
)

# -----------------------------
# Save merged YAML
# -----------------------------
merged_yaml_path = output_folder / merged_yaml_name
with open(merged_yaml_path, "w") as f:
    yaml.dump(camera_yaml, f, sort_keys=False)

print(f"Merged YAML saved to {merged_yaml_path}")
print("Done! The merged bag folder is ready for ROMAN.")
