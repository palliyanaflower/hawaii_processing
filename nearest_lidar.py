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

if __name__ == "__main__":

    # ============================================================
    # Load Data
    # ============================================================
    camera_id = 3

    path_cam2 = "data/makalii_point/processed_lidar_cam_gps/cam2/bag_camera_2_2025_08_13-01_35_58_12/camera/left_cam/0.png"
    path_cam3 = "data/makalii_point/processed_lidar_cam_gps/cam3/bag_camera_3_2025_08_13-01_35_58_48/camera/left_cam/3.png"
    pcd_path = Path("data/makalii_point/processed_lidar_cam_gps/cam3/bag_camera_3_2025_08_13-01_35_58_48/lidar/pcd/3.pcd")
    main_cam = path_cam3

    # Camera intrinsics
    K, dist = loader.intrinsics_and_distortion("data/calib_leftcam.json")
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

    T_init = T_cam3

    # ============================================================
    # LightGlue matches
    # ============================================================
    vis = LightGlueVisualizer()
    m_kpts_cam2, m_kpts_cam3 = vis.get_matched_keypoints(path_cam2, path_cam3)
    m_kpts = m_kpts_cam3

    # ============================================================
    # Load LiDAR and project to cam
    # ============================================================
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    points = np.asarray(pcd.points)

    # Load intensity for coloring
    pcd_t = o3d.t.io.read_point_cloud(str(pcd_path))
    intensity = pcd_t.point["intensity"].numpy().flatten()
    intensity_norm = (intensity - intensity.min()) / (np.ptp(intensity) + 1e-8)
    colors = plt.get_cmap("turbo")(intensity_norm)[:, :3]  # drop alpha

    # Project LiDAR points
    u, v, pts_infront, pts_cam, lidar_img = cg.filter_pts_to_image_plane(
        points, colors, T_init, K_proj, img.shape[1], img.shape[0], img
    )

    # ============================================================
    # Plot image, LiDAR points, and matched keypoints
    # ============================================================
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.imshow(lidar_img)

    # # LiDAR points (blue)
    # plt.scatter(
    #     u, v,
    #     s=1,
    #     c="deepskyblue",
    #     alpha=0.7,
    #     label="LiDAR"
    # )

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

    pixel_radius = 4

    keypoint_lidar_depths = []
    lidar_nn_per_kp_px = []
    lidar_nn_per_kp = []

    for kp in m_kpts:
        x, y = kp[0], kp[1]  # keep as float

        # Find LiDAR points within window
        mask = (np.abs(u - x) <= pixel_radius) & (np.abs(v - y) <= pixel_radius)

        if np.any(mask):
            depths = pts_cam[mask, 2]  # z in camera frame

            # Stack u,v in shape (N,2) for pixel coordinates
            lidar_nn_per_kp_px.append(np.stack([u[mask], v[mask]], axis=1))
            lidar_nn_per_kp.append(pts_infront[mask, :])  # xyz in camera frame
        else:
            depths = np.array([])
            lidar_nn_per_kp_px.append(np.empty((0,2)))
            lidar_nn_per_kp.append(np.empty((0,3)))

        keypoint_lidar_depths.append(depths)

    # Check
    num_with_neighbors = sum([len(d) > 0 for d in keypoint_lidar_depths])
    print("Number of keypoints with LiDAR neighbors:", num_with_neighbors)

    # Show points in 2D space
    if num_with_neighbors > 0:
        all_lidar_neighbors_px = np.vstack([pts for pts in lidar_nn_per_kp_px if pts.size > 0])

        plt.scatter(all_lidar_neighbors_px[:,0], all_lidar_neighbors_px[:,1], s=1, c='deepskyblue', alpha=0.7, label='LiDAR')
        plt.legend()
        plt.axis('off')
    else:
        print("No LiDAR neighbors found!")

    # Show points in 3D space
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        pts_infront[:, 0],
        pts_infront[:, 1],
        pts_infront[:, 2],
        s=1,
        c="gray",
        # alpha=0.6,
        label="LiDAR"
    )

    # Keypoint-neighbor LiDAR points (blue)
    # Collect all neighbor points
    nn_pts_list = []
    group_ids = []

    for i, nn_pts in enumerate(lidar_nn_per_kp):
        if nn_pts is None or len(nn_pts) == 0:
            continue

        nn_pts = np.asarray(nn_pts)

        # Optional: remove all-zero points
        mask = ~np.all(np.isclose(nn_pts, 0.0), axis=1)
        nn_pts = nn_pts[mask]

        if nn_pts.shape[0] > 0:
            nn_pts_list.append(nn_pts)
            group_ids.append(np.full(nn_pts.shape[0], i))


    # Concatenate once
    if len(nn_pts_list) > 0:
        nn_pts_all = np.concatenate(nn_pts_list, axis=0)
        group_ids = np.concatenate(group_ids, axis=0)
    else:
        nn_pts_all = np.empty((0, 3))

    ax.scatter(
        nn_pts_all[:, 0],
        nn_pts_all[:, 1],
        nn_pts_all[:, 2],
        s=6,
        c=group_ids,
        cmap="tab20",   # or hsv, turbo, viridis
        alpha=0.9,
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=20, azim=-70)
    ax.set_aspect('equal', adjustable='datalim')

    plt.tight_layout()
    plt.show()