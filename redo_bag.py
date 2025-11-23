#!/usr/bin/env python3
import yaml
from pathlib import Path

# Create a yaml file for each individual bags, so we don't need to download all db3 files
def create_yaml_for_each_bag(original_metadata_path, subset_files, output_root):
    with open(original_metadata_path, "r") as f:
        original_meta = yaml.safe_load(f)

    files_info = original_meta['rosbag2_bagfile_information']['files']
    rel_paths = original_meta['rosbag2_bagfile_information']['relative_file_paths']
    topics_info = original_meta['rosbag2_bagfile_information']['topics_with_message_count']

    for bag_file in subset_files:
        # Find the corresponding 'files' entry
        file_entry = next((f for f in files_info if Path(f['path']).name == bag_file), None)
        if not file_entry:
            print(f"Warning: {bag_file} not found in original metadata.yaml")
            continue

        # Build new metadata for this single bag
        new_meta = {
            'rosbag2_bagfile_information': {
                'version': original_meta['rosbag2_bagfile_information']['version'],
                'storage_identifier': original_meta['rosbag2_bagfile_information']['storage_identifier'],
                'duration': file_entry['duration'],
                'starting_time': file_entry['starting_time'],
                'message_count': file_entry['message_count'],
                'topics_with_message_count': topics_info,
                'compression_format': original_meta['rosbag2_bagfile_information'].get('compression_format', ''),
                'compression_mode': original_meta['rosbag2_bagfile_information'].get('compression_mode', ''),
                'relative_file_paths': [bag_file],
                'files': [file_entry]
            }
        }

        # Save to a folder named after the bag
        output_folder = Path(output_root) / bag_file.replace(".db3", "")
        output_folder.mkdir(parents=True, exist_ok=True)
        yaml_path = output_folder / "metadata.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(new_meta, f, sort_keys=False)
        print(f"Created {yaml_path}")

if __name__ == "__main__":
    original_metadata = "data/makalii_point/cam3/metadata.yaml"
    subset_files = [
        "bag_camera_3_2025_08_13-01_35_58_40.db3"
    ]
    output_root = "data/makalii_point/cam3"

    create_yaml_for_each_bag(original_metadata, subset_files, output_root)
