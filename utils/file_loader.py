import numpy as np
import json
import cv2
import open3d as o3d
from pathlib import Path
import matplotlib.pyplot as plt

def intrinsics_and_distortion(json_path):
    with open(json_path, 'r') as f:
        calib = json.load(f)
    cam_K = np.array(np.array(calib['cam_K']['data'], dtype=np.float32))
    cam_dist = np.array(calib['cam_dist']['data'], dtype=np.float32)
    return cam_K, cam_dist

def lidar_pcd(pcd_path):
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    points = np.asarray(pcd.points)

    # Load intensity for coloring
    pcd_t = o3d.t.io.read_point_cloud(str(pcd_path))
    intensity = pcd_t.point["intensity"].numpy().flatten()
    intensity_norm = (intensity - intensity.min()) / (np.ptp(intensity) + 1e-8)

    colors = plt.get_cmap("viridis")(intensity_norm)[:, :3] # drop alpha
    # colors = plt.get_cmap("viridis")(intensity_norm)[:, :3] # drop alpha

    return points, intensity, colors