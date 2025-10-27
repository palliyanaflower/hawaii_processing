#!/usr/bin/env python3
from rosbags.highlevel import AnyReader
from pathlib import Path


# Path to the folder containing the bag + metadata.yaml
bag_folder = "/home/kalliyanlay/Documents/BYU/research/hawaii_processing/cammerged_data/bag_cammerged_2025_08_14-00_56_41_10"

print(f"Attempting to open bag folder: {bag_folder}")

try:
    with AnyReader([Path(bag_folder)]) as reader:
        print("Bag opened successfully!")
        print("Connections found:")
        for conn in reader.connections:
            print(f"  Topic: {conn.topic}, Type: {conn.msgtype}")
        
        # Just read the first few messages
        print("\nReading first 5 messages:")
        for i, (connection, timestamp, rawdata) in enumerate(reader.messages()):
            print(f"{i+1}. Topic: {connection.topic}, Timestamp: {timestamp}")
            if i >= 4:
                break
except Exception as e:
    print("Error opening bag folder with AnyReader:")
    print(e)
