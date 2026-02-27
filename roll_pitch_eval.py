import json
import numpy as np
from scipy.spatial.transform import Rotation as R

def rpy_to_quaternion(roll, pitch, yaw, degrees=False):
    if degrees:
        roll  = np.deg2rad(roll)
        pitch = np.deg2rad(pitch)
        yaw   = np.deg2rad(yaw)

    # 'xyz' means roll-pitch-yaw about x-y-z
    r = R.from_euler('xyz', [roll, pitch, yaw])
    qx, qy, qz, qw = r.as_quat()  # scipy returns [x, y, z, w]
    return qx, qy, qz, qw

# ---- Parse JSON ----
json_path = "data/makalii_point/processed_lidar_cam_gps/cam2/bag_camera_2_2025_08_13-01_35_58_21/nav/gt/9.json"
# json_path = "data/makalii_point/processed_lidar_cam_gps/cam3/bag_camera_3_2025_08_13-01_35_58_35/nav/gt/4.json"
with open(json_path, 'r') as f:
    data = json.load(f)

qx = data["orientation"]["qx"]
qy = data["orientation"]["qy"]
qz = data["orientation"]["qz"]
qw = data["orientation"]["qw"]
print("qx", qx)
print("qy", qy)
print("qz", qz)
print("qw", qw)

# scipy expects [x, y, z, w]
quat = [qx, qy, qz, qw]

# Convert to Euler angles (roll, pitch, yaw)
rotation = R.from_quat(quat)
roll, pitch, yaw = rotation.as_euler('xyz', degrees=True)

print(f"\nRoll  (deg): {roll:.6f}")
print(f"Pitch (deg): {pitch:.6f}")
print(f"Yaw   (deg): {yaw:.6f}")

# qx, qy, qz, qw = rpy_to_quaternion(roll, pitch, yaw, degrees=True)
qx, qy, qz, qw = rpy_to_quaternion(roll=-2.4734015119991652, pitch=-2.902632713317871, yaw=159.17391967773438, degrees=True) # cam2
# qx, qy, qz, qw = rpy_to_quaternion(roll=6.116702882077823, pitch=-2.752687692642212, yaw=-68.125244140625, degrees=True) # cam3

print("\nqx", qx)
print("qy", qy)
print("qz", qz)
print("qw", qw)

# Idx 76
# Cam 3
# Vehicle estimated roll: 6.116702882077823 degrees
# Vehicle estimated pitch: -2.752687692642212 degrees
# Yaw: -68.125244140625
# {
#   "timestamp": 1755051086942102116,
#   "position": {
#     "latitude": 21.5757943266876,
#     "longitude": -157.87897810416712,
#     "altitude": 25.9531770102206
#   },
#   "orientation": {
#     "qx": -0.056024856433623184,
#     "qy": 0.06693979635551697,
#     "qz": 0.5311903978984271,
#     "qw": -0.8427437585275946
#   },
#   "frame_id": "imu_link_enu"
# }

# Cam 2 
# Estimated roll and pitch at: 1755050262937595114
# roll: -2.4734015119991652 degrees
# pitch: -2.902632713317871 degrees
# yaw: 159.17391967773438
# yaw: 
# {
#   "timestamp": 1755050262945242920,
#   "position": {
#     "latitude": 21.575863776629376,
#     "longitude": -157.87893821967705,
#     "altitude": 26.94032490859627
#   },
#   "orientation": {
#     "qx": -0.02439271182226813,
#     "qy": -0.015023136808953378,
#     "qz": -0.983141454215614,
#     "qw": -0.18058841412911367
#   },
#   "frame_id": "imu_link_enu"
# }