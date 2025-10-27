import yaml
from pathlib import Path

# Path to your metadata.yaml
metadata_file = Path("/home/kalliyanlay/Documents/BYU/research/hawaii_processing/metadata_cam2.yaml")

def list_rosbag_topics(metadata_path):
    with open(metadata_path, "r") as f:
        metadata = yaml.safe_load(f)

    print(f"\nBag: {metadata.get('rosbag2_bagfile_information', {}).get('relative_file_paths', ['?'])[0]}")
    print("="*60)

    topics = metadata["rosbag2_bagfile_information"]["topics_with_message_count"]
    for topic in topics:
        name = topic["topic_metadata"]["name"]
        msg_type = topic["topic_metadata"]["type"]
        count = topic["message_count"]
        print(f"{name:<40} {msg_type:<50} {count:>10}")

if __name__ == "__main__":
    list_rosbag_topics(metadata_file)
