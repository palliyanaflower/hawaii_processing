# Project lidar into camera
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
import matplotlib.pyplot as plt
# from calib_class import *
import json
import re
import utils.camera_geom as cg
import utils.file_loader as loader
from utils.lightglue_loader import LightGlueVisualizer
from argparse import ArgumentParser
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

if __name__ == "__main__":

    # ============================================================
    # Load Data
    # ============================================================
    camera_id = 3
    path_cam2 = "data/makalii_point/processed_lidar_cam_gps/cam2/bag_camera_2_2025_08_13-01_35_58_14/camera/rgb/0.png"
    path_cam3 = "data/makalii_point/processed_lidar_cam_gps/cam3/bag_camera_3_2025_08_13-01_35_58_46/camera/rgb/0.png"
    # path_cam2 = "data/makalii_point/processed_lidar_cam_gps/cam2/bag_camera_2_2025_08_13-01_35_58_12/camera/rgb/0.png"
    # path_cam3 = "data/makalii_point/processed_lidar_cam_gps/cam3/bag_camera_3_2025_08_13-01_35_58_48/camera/left_cam/3.png"
    if camera_id == 2:
        main_cam = path_cam2
        pcd_path = Path("data/makalii_point/processed_lidar_cam_gps/cam2/bag_camera_2_2025_08_13-01_35_58_14/lidar/pcd/0.pcd")    
        # pcd_path = Path("data/makalii_point/processed_lidar_cam_gps/cam2/bag_camera_2_2025_08_13-01_35_58_12/lidar/pcd/0.pcd")    
    else: 
        main_cam = path_cam3
        pcd_path = Path("data/makalii_point/processed_lidar_cam_gps/cam3/bag_camera_3_2025_08_13-01_35_58_46/lidar/pcd/0.pcd")
        # pcd_path = Path("data/makalii_point/processed_lidar_cam_gps/cam3/bag_camera_3_2025_08_13-01_35_58_48/lidar/pcd/3.pcd")

    # Camera intrinsics
    if camera_id == 2:
        K, dist = loader.intrinsics_and_distortion("data/calib_leftcam.json")
    else:
        K, dist = loader.intrinsics_and_distortion("data/calib_rightcam.json")
    K = K[:, :3]  # Extract intrinsics only

    # --- Load image ---
    img = cv2.imread(str(main_cam))
    h, w, _ = img.shape

    # Undistort image
    print("Undistorting image...")
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
    img = cv2.undistort(img, K, dist, None, new_K)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    K_proj = new_K


    # ============================================================
    # Initial guess transform (CAM2)
    # ============================================================
    T_cam2 = np.array([
        [-1, 0, 0,  0.02],
        [0, 0, -1, -0.07],
        [0, -1, 0,  -0.32],
        [0, 0, 0, 1]
    ])
    T_cam3 = np.array([
        [1, 0, 0, -0.02],
        [0, 0, -1, -0.07],
        [0, 1, 0, -0.33],
        [0, 0, 0, 1]
    ])

    if camera_id == 2:
        T_init = T_cam2
    else:
        T_init = T_cam3

    # ============================================================
    # LightGlue matches
    # ============================================================
    vis = LightGlueVisualizer()
    m_kpts_cam2, m_kpts_cam3 = vis.get_matched_keypoints(path_cam2, path_cam3)

    if camera_id == 2:
        m_kpts = m_kpts_cam2
    else:
        m_kpts = m_kpts_cam3

    vis.visualize_matches(path_cam2, path_cam3)

    # ============================================================
    # Load LiDAR and project to cam
    # ============================================================
    points, intensity, colors = loader.lidar_pcd(pcd_path)
 
    # Project LiDAR points
    u, v, pts_cam, lidar_img, mask_infront, mask_img = cg.filter_pts_to_image_plane(
        points, colors, T_init, K_proj, img.shape[1], img.shape[0], img
    )
    colors_masked = colors[mask_infront]
    colors_masked = colors_masked[mask_img]
    pts_infront = points[mask_infront]
    pts_infront = pts_infront[mask_img]

    # ============================================================
    # Plot image, LiDAR points, and matched keypoints
    # ============================================================
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    # plt.imshow(lidar_img)

    # Matched keypoints (red, CAM2)
    plt.scatter(
        m_kpts[:, 0],
        m_kpts[:, 1],
        s=25,
        c="red",
        label="Matched keypoints"
    )

    # ============================================================
    # Lidar-keypoint matches
    # ============================================================

    nn = cg.collect_lidar_neighbors_per_keypoint(
                                                    m_kpts,
                                                    u,
                                                    v,
                                                    pts_cam,
                                                    pts_infront,
                                                    pixel_radius=5,
                                                    remove_zero_points=True,
                                                )
    # Show points in 2D space
    plt.scatter(nn.pxs_all[:,0], nn.pxs_all[:,1], s=1, c=nn.group_ids, cmap="turbo", alpha=0.7, label='LiDAR')

    # Show points in 3D space
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter( # All lidar
        pts_infront[:, 0],
        pts_infront[:, 1],
        pts_infront[:, 2],
        s=1,
        c="gray", #colors_masked
        alpha=0.4,
        label="LiDAR"
    )

    ax.scatter( # Lidar neighbors
        nn.pts_all[:, 0],
        nn.pts_all[:, 1],
        nn.pts_all[:, 2],
        s=5,
        c=nn.group_ids,
        cmap="turbo",   # or tab20, hsv, turbo, viridis
        alpha=1.0,
    )

    num_with_neighbors = sum([len(d) > 0 for d in nn.depths_per_kp])
    print("Num keypoints with LiDAR neighbors:", num_with_neighbors)
    print("Num lidar neighbors: ", len(nn.pxs_all))

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=20, azim=-70)
    ax.set_aspect('equal', adjustable='datalim')

    plt.tight_layout()
    plt.show()