import numpy as np
import json

def intrinsics_and_distortion(json_path):
    with open(json_path, 'r') as f:
        calib = json.load(f)
    cam_K = np.array(np.array(calib['cam_K']['data'], dtype=np.float32))
    cam_dist = np.array(calib['cam_dist']['data'], dtype=np.float32)
    return cam_K, cam_dist