import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import cv2
import time
import open3d as o3d

from lightglue_loader import LightGlueVisualizer
import clipperpy

import camera_geom as cg
import file_loader as loader
from CameraLidar import CameraView
import viz
from KeypointNN import LidarKeypointNeighbors

# ============================================================
# Load Data
# ============================================================

# Large change in scale / perspective
# cam2_bagnum = 12
# cam2_imgnum = 0
# cam3_bagnum = 48
# cam3_imgnum = 3

# Lidar points not on building
# cam2_bagnum = 14
# cam2_imgnum = 0
# cam3_bagnum = 46
# cam3_imgnum = 0

# Best case scenario
cam2_bagnum = 18
cam2_imgnum = 3
cam3_bagnum = 40
cam3_imgnum = 26
cameras = [
    CameraView(
        name="cam2",
        image_path=Path(
            "data/makalii_point/processed_lidar_cam_gps/cam2/"
            f"bag_camera_2_2025_08_13-01_35_58_{cam2_bagnum}/camera/rgb/{cam2_imgnum}.png"
        ),
        lidar_pcd_path=Path(
            "data/makalii_point/processed_lidar_cam_gps/cam2/"
            f"bag_camera_2_2025_08_13-01_35_58_{cam2_bagnum}/lidar/pcd/{cam2_imgnum}.pcd"
        ),
        calib_path=Path("data/calib_leftcam.json"),
        T_lidar_cam=np.array([
            [-1, 0, 0,  0.02],
            [0, 0, -1, -0.07],
            [0, -1, 0, -0.32],
            [0, 0, 0, 1],
        ]),
    ),
    CameraView(
        name="cam3",
        image_path=Path(
            "data/makalii_point/processed_lidar_cam_gps/cam3/"
            f"bag_camera_3_2025_08_13-01_35_58_{cam3_bagnum}/camera/left_cam/{cam3_imgnum}.png"
        ),
        lidar_pcd_path=Path(
            "data/makalii_point/processed_lidar_cam_gps/cam3/"
            f"bag_camera_3_2025_08_13-01_35_58_{cam3_bagnum}/lidar/pcd/{cam3_imgnum}.pcd"
        ),
        calib_path=Path("data/calib_rightcam.json"),
        T_lidar_cam=np.array([
            [1, 0, 0, -0.02],
            [0, 0, -1, -0.07],
            [0, 1, 0, -0.33],
            [0, 0, 0, 1],
        ]),
    ),
]

for cam in cameras:
    cam.load_image_and_calib()
    cam.project_lidar()

# ============================================================
# LightGlue matches
# ============================================================
vis = LightGlueVisualizer()
m_kpts_cam2, m_kpts_cam3 = vis.get_matched_keypoints(
    cameras[0].image_path,
    cameras[1].image_path,
)

keypoints = {
    "cam2": m_kpts_cam2,
    "cam3": m_kpts_cam3,
}

# ============================================================
# Keypoint nearest neighbors
# ============================================================
nn = {}

for cam in cameras:
    nn[cam.name] = cg.collect_lidar_neighbors_per_keypoint(
        keypoints[cam.name],
        cam.u,
        cam.v,
        cam.pts_cam,
        cam.pts_infront,
        pixel_radius=5,
        remove_zero_points=True,
    )

    num_with_neighbors = sum([len(d) > 0 for d in nn[cam.name].pxs_per_kp])
    print("Num keypoints with neighbors:", num_with_neighbors)
    print("Num neighbors total:", len(nn[cam.name].pxs_all))

# ============================================================
# Plotting image, keypoints, lidar
# ============================================================

# Keypoint matches
vis.visualize_matches(cameras[0].image_path, cameras[1].image_path)
# viz.show_lidar_neighbors_2d(cameras, keypoints, nn)
# viz.show_lidar_neighbors_3d(cameras, nn)
plt.show()

# ============================================================
# Build consistency matrix
# ============================================================

#-----------------
# Make pointclouds
#-----------------
# Iterate through each keypoint match and gather all points in pointcloud 1 and 2
pc1 = []
pc2 = []
associations = [] 
kp_idxs_D1 = []    # original keypoint indices
nn_idxs_D1 = []    # nn indices for each keypoint
kp_idxs_D2 = []    # original keypoint indices
nn_idxs_D2 = []    # nn indices for each keypoint
for k_idx in range(len(nn["cam2"].pts_per_kp)): 
    # Get lidar neighbors around keypoint 
    nn_pts_i = nn["cam2"].pts_per_kp[k_idx] 
    nn_pts_j = nn["cam3"].pts_per_kp[k_idx] 
    
    # Skip if don't have lidar neighbors for both keypoints in keypoint match
    if len(nn_pts_i) == 0 or len(nn_pts_j) == 0: 
        continue 

    # -------------------------------
    # Make all putative associations 
    # -------------------------------
    for idx_i in range(len(nn_pts_i)): 
        for idx_j in range(len(nn_pts_j)): 
            associations.append([idx_i+len(pc1), idx_j+len(pc2)])

    # Gather points
    for idx_i in range(len(nn_pts_i)):
        pc1.append(nn_pts_i[idx_i])
        kp_idxs_D1.append(k_idx)
        nn_idxs_D1.append([idx_i])
    for idx_j in range(len(nn_pts_j)):
        pc2.append(nn_pts_j[idx_j])
        kp_idxs_D2.append(k_idx)
        nn_idxs_D2.append([idx_j])

pc1 = np.array(pc1)
pc2 = np.array(pc2)
associations = np.array(associations)
kp_idxs_D1 = np.array(kp_idxs_D1)
nn_idxs_D1 = np.array(nn_idxs_D1)
kp_idxs_D2 = np.array(kp_idxs_D2)
nn_idxs_D2 = np.array(nn_idxs_D2)
# print("\npc", pc1.shape, pc2.shape)

D1 = np.ascontiguousarray(
    np.vstack(pc1).T,   # stack FIRST, then transpose
    dtype=np.float64
)

D2 = np.ascontiguousarray(
    np.vstack(pc2).T,
    dtype=np.float64
)
A = np.ascontiguousarray(
    np.array(associations, dtype=np.int32)
)

superpoint_descs = []
semantic_descs = []
lidar_descs = []

# --------------------------------------------------
# Run CLIPPER
# --------------------------------------------------
print("\npc", D1.shape, D2.shape)
print("\nNum associations", associations.shape)
print(associations[:10])
print(associations[-10:])

iparams = clipperpy.invariants.EuclideanDistanceParams()
iparams.sigma = 0.04
iparams.epsilon = 0.06
invariant = clipperpy.invariants.EuclideanDistance(iparams)

params = clipperpy.Params()
params.rounding = clipperpy.Rounding.DSD_HEU
clipper = clipperpy.CLIPPER(invariant, params)
t0 = time.perf_counter()

clipper.score_pairwise_consistency(D1, D2, A)

t1 = time.perf_counter()
print(f"Affinity matrix creation took {t1-t0:.3f} seconds")

t0 = time.perf_counter()
clipper.solve()
t1 = time.perf_counter()

# A = clipper.get_initial_associations()
Ain = clipper.get_selected_associations()

print(f"CLIPPER selected {Ain.shape[0]} inliers in {t1-t0:.3f} s")

# --------------------------------------------------
# Select inlier points
# --------------------------------------------------
# Ain: (K, 2) int32 indices
idx1 = Ain[:, 0]
idx2 = Ain[:, 1]

inliers = {}
inliers["cam2"] = D1[:, idx1].T   # shape (K, 3)
inliers["cam3"] = D2[:, idx2].T   # shape (K, 3)

kp_idxs_inlier = {"cam2":kp_idxs_D1[idx1], "cam3":kp_idxs_D2[idx2]}
nn_idxs_inlier = {"cam2":nn_idxs_D1[idx1], "cam3":nn_idxs_D2[idx2]}

nn_inliers = {}

for cam_name in nn:

    depths_per_kp = []
    pxs_per_kp = []
    pts_per_kp = []

    pxs_all = []
    pts_all = []
    group_ids = []

    # Iterate over INLIER keypoints and their NN indices
    for new_kp_idx, (kp_idx, nn_idxs) in enumerate(
        zip(kp_idxs_inlier[cam_name], nn_idxs_inlier[cam_name])
    ):
        depths = []
        pxs = []
        pts = []

        for nn_idx in nn_idxs:
            depth = nn[cam_name].depths_per_kp[kp_idx][nn_idx]
            px    = nn[cam_name].pxs_per_kp[kp_idx][nn_idx]
            pt    = nn[cam_name].pts_per_kp[kp_idx][nn_idx]

            depths.append(depth)
            pxs.append(px)
            pts.append(pt)

            pxs_all.append(px)
            pts_all.append(pt)
            group_ids.append(new_kp_idx)

        depths_per_kp.append(np.asarray(depths))
        pxs_per_kp.append(np.asarray(pxs))
        pts_per_kp.append(np.asarray(pts))

    nn_inliers[cam_name] = LidarKeypointNeighbors(
        depths_per_kp=depths_per_kp,
        pxs_per_kp=pxs_per_kp,
        pts_per_kp=pts_per_kp,
        pts_all=np.asarray(pts_all),
        pxs_all=np.asarray(pxs_all),
        group_ids=np.asarray(group_ids),
    )


print("inlier points", inliers["cam2"].shape, inliers["cam3"].shape)

# # --------------------------------------------------
# # Plot CLIPPER results
# # --------------------------------------------------
# lidar_offset = np.zeros_like(cameras[0].pts_cam)
# lidar_offset[:,2] = 70
# lidar_offset[:,0] = 17
# model = o3d.geometry.PointCloud()
# model.points = o3d.utility.Vector3dVector(cameras[0].pts_cam + lidar_offset)
# model.paint_uniform_color(np.array([0,0,1.]))
# data = o3d.geometry.PointCloud()
# data.points = o3d.utility.Vector3dVector(cameras[1].pts_cam)
# data.paint_uniform_color(np.array([1.,0,0]))

# corr = o3d.geometry.LineSet.create_from_point_cloud_correspondences(model, data, Ain)
# o3d.visualization.draw_geometries([model, data, corr])

viz.show_lidar_neighbors_2d(cameras, keypoints, nn_inliers, kp_size=5, lidar_size=25)
viz.show_lidar_neighbors_3d(cameras, nn_inliers, neighbor_size=20)

plt.show()