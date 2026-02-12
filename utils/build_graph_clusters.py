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
cam3_bagnum = 51
cam3_imgnum = 0
cam2_bagnum = 10
cam2_imgnum = 3


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
        [-9.99925370e-01, -1.22168334e-02,  6.39677745e-05,
         2.00000000e-02],
       [-1.22298005e-16, -5.23596383e-03, -9.99986292e-01,
        -1.00000000e-01],
       [ 1.22170008e-02, -9.99911663e-01,  5.23557307e-03,
        -1.70000000e-01],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]
    ]) # left (cam2) - manual_undist_ssp_nocrop
    ),
    CameraView(
        name="cam3",
        image_path=Path(
            "data/makalii_point/processed_lidar_cam_gps/cam3/"
            f"bag_camera_3_2025_08_13-01_35_58_{cam3_bagnum}/camera/rgb/{cam3_imgnum}.png"
        ),
        lidar_pcd_path=Path(
            "data/makalii_point/processed_lidar_cam_gps/cam3/"
            f"bag_camera_3_2025_08_13-01_35_58_{cam3_bagnum}/lidar/pcd/{cam3_imgnum}.pcd"
        ),
        calib_path=Path("data/calib_rightcam.json"),
        T_lidar_cam=np.array([
        [ 0.99992385,  0.01225848, -0.00142493, -0.03      ],
        [-0.0017452 ,  0.02615559, -0.99965636, -0.11      ],
        [-0.012217  ,  0.99958272,  0.02617499, -0.12      ],
        [ 0.        ,  0.        ,  0.        ,  1.        ]
    ]) # right (cam3) - manual_undist_ssp_nocrop,
    ),
]

for cam in cameras:
    cam.load_image_and_calib()
    cam.project_lidar()

# ============================================================
# LightGlue matches
# ============================================================
print("\Paths")
print(cameras[1].image_path)
print(cameras[0].image_path)
vis = LightGlueVisualizer()
m_kpxs_cam3, m_kpxs_cam2 = vis.get_matched_keypoints(
    cameras[1].image_path,
    cameras[0].image_path,
)

print("Num LightGlue Matches", len(m_kpxs_cam2))
keypoints = {
    "cam2": m_kpxs_cam2,
    "cam3": m_kpxs_cam3,
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
        pixel_radius=10,
        remove_zero_points=True,
    )

    num_with_neighbors = sum([len(d) > 0 for d in nn[cam.name].pxs_per_kp])
    print("\nNum keypoints with neighbors:", num_with_neighbors)
    print("Num neighbors total:", len(nn[cam.name].pxs_all))


# ============================================================
# Plotting image, keypoints, lidar
# ============================================================

# Keypoint matches
vis.visualize_matches(
    cameras[1].image_path,  
    cameras[0].image_path,  
)
# viz.show_lidar_neighbors_2d(cameras, keypoints, nn)
# viz.show_lidar_neighbors_3d(cameras, nn)
# viz.show_lidar_2d(cameras) # all lidar points
# plt.show()

# ============================================================
# Get clusters
# ============================================================
from typing import List
from dataclasses import dataclass

@dataclass
class PointCloudClustered:
    
    kidxs: np.ndarray            # (M,) Keypoint index of each cluster
    pts: np.ndarray              # (M,3) 3D points of the cluster centroids
    pxs: np.ndarray              # (M,2) 2D pixels coords of the cluster centroids

# ---------------------------------------------------
# Get lidar foreground clusters around each keypoint 
# ---------------------------------------------------
pc_clustered = {}

BIN_WIDTH = 1.0        # meters
SIGNIF_THRESHOLD = 5  # min points in a depth bin

for cam in cameras:
    nn_info = nn[cam.name]

    centroid_kidxs = []
    centroid_pts = []
    centroid_pxs = []

    for k_idx in range(nn_info.num_kps):
        depths = np.asarray(nn_info.depths_per_kp[k_idx])
        pxs    = np.asarray(nn_info.pxs_per_kp[k_idx])
        pts    = np.asarray(nn_info.pts_per_kp[k_idx])

        # -----------------------
        # No lidar for keypoint
        # -----------------------
        if depths.size == 0:
            centroid_kidxs.append(-1)
            centroid_pts.append(np.zeros(3))
            centroid_pxs.append(np.zeros(2))
            continue

        # -----------------------
        # Depth histogram
        # -----------------------
        bins = np.arange(depths.min(), depths.max() + BIN_WIDTH, BIN_WIDTH)
        hist_counts, bin_edges = np.histogram(depths, bins=bins)

        # Find nearest significant depth bin
        signif_bins = np.where(hist_counts >= SIGNIF_THRESHOLD)[0]

        if len(signif_bins) == 0:
            centroid_kidxs.append(-1)
            centroid_pts.append(np.zeros(3))
            centroid_pxs.append(np.zeros(2))
            continue

        b = signif_bins[0]               # nearest (smallest depth)
        edge = bin_edges[b]

        # -----------------------
        # Keep lidar points in bin
        # -----------------------
        mask = (depths >= edge) & (depths < edge + BIN_WIDTH)

        pts_keep = pts[mask]
        pxs_keep = pxs[mask]

        if pts_keep.shape[0] == 0:
            centroid_kidxs.append(-1)
            centroid_pts.append(np.zeros(3))
            centroid_pxs.append(np.zeros(2))
            continue

        # -----------------------
        # Centroid
        # -----------------------
        centroid_pt = pts_keep.mean(axis=0)

        # Only for visualization
        centroid_px = cg.project_point_to_image_plane(
            centroid_pt,
            cam.T_lidar_cam,
            cam.K_proj
        )

        centroid_kidxs.append(k_idx)
        centroid_pts.append(centroid_pt)
        centroid_pxs.append(centroid_px)

    pc_clustered[cam.name] = PointCloudClustered(
        kidxs=np.asarray(centroid_kidxs),
        pts=np.asarray(centroid_pts),
        pxs=np.asarray(centroid_pxs)
    )



# ------------------------
# Get valid cluster pairs 
# ------------------------
centroid_kidxs = []

centroid_pts_cam2 = []
centroid_pts_cam3 = []

centroid_pxs_cam2 = []
centroid_pxs_cam3 = []

for k_idx in range(nn["cam2"].num_kps):

    k_idx_cam2 = pc_clustered["cam2"].kidxs[k_idx]
    k_idx_cam3 = pc_clustered["cam3"].kidxs[k_idx]

    # Keep if keypoint is present in both cluster clouds
    if k_idx_cam2!=-1 and k_idx_cam3!=-1:
        centroid_kidxs.append(k_idx)

        centroid_pts_cam2.append(pc_clustered["cam2"].pts[k_idx])
        centroid_pts_cam3.append(pc_clustered["cam3"].pts[k_idx])

        centroid_pxs_cam2.append(pc_clustered["cam2"].pxs[k_idx])
        centroid_pxs_cam3.append(pc_clustered["cam3"].pxs[k_idx])

pc_clustered_filtered = {}

# These keypoints should now be matched putative associations
# TODO have matches for different depth hypotheses
pc_clustered_filtered["cam2"] = PointCloudClustered(
                                                kidxs=np.array(centroid_kidxs),
                                                pts=np.array(centroid_pts_cam2),
                                                pxs=np.array(centroid_pxs_cam2)
                                            )
pc_clustered_filtered["cam3"] = PointCloudClustered(
                                                kidxs=np.array(centroid_kidxs),
                                                pts=np.array(centroid_pts_cam3),
                                                pxs=np.array(centroid_pxs_cam3)
                                            )
# viz.show_points_2d(cameras, keypoints, pc_clustered_filtered)
# viz.show_points_3d(cameras, pc_clustered_filtered)

# ============================================================
# Build consistency matrix
# ============================================================

# CLIPPER point clouds
D1 = pc_clustered_filtered["cam2"].pts.T.astype(np.float64)
D2 = pc_clustered_filtered["cam3"].pts.T.astype(np.float64)


# CLIPPER associations
n = D1.shape[1]
idx = np.arange(n, dtype=np.int32)
A = np.column_stack((idx, idx))

# --------------------------------------------------
# Run CLIPPER
# --------------------------------------------------
print("\npc", D1.shape, D2.shape)
print("\nLightglue matches:", len(m_kpxs_cam2))
print("Num associations:", A.shape[0])

iparams = clipperpy.invariants.EuclideanDistanceParams()
iparams.sigma = 0.4
iparams.epsilon = 0.6
invariant = clipperpy.invariants.EuclideanDistance(iparams)

params = clipperpy.Params()
params.rounding = clipperpy.Rounding.DSD_HEU
clipper = clipperpy.CLIPPER(invariant, params)
t0 = time.perf_counter()

clipper.score_pairwise_consistency(D1, D2, A)

t1 = time.perf_counter()
print(f"Affinity matrix creation took {t1-t0:.6f} s")

t0 = time.perf_counter()
clipper.solve()
t1 = time.perf_counter()

# A = clipper.get_initial_associations()
Ain = clipper.get_selected_associations()

print(f"CLIPPER selected {Ain.shape[0]} inliers in {t1-t0:.6f} s")

# --------------------------------------------------
# Select inlier points
# --------------------------------------------------
# Ain: (K, 2) int32 indices
idxs1 = Ain[:, 0]
idxs2 = Ain[:, 1]

inlier_pts_cam2 = []
inlier_pts_cam3 = []
inlier_pxs_cam2 = []
inlier_pxs_cam3 = []
for idx1, idx2 in zip(idxs1, idxs2):
    inlier_pts_cam2.append(pc_clustered_filtered["cam2"].pts[idx1])
    inlier_pts_cam3.append(pc_clustered_filtered["cam3"].pts[idx2])
    inlier_pxs_cam2.append(pc_clustered_filtered["cam2"].pxs[idx1])
    inlier_pxs_cam3.append(pc_clustered_filtered["cam3"].pxs[idx2])

pc_clustered_clipper = {}
pc_clustered_clipper["cam2"] = PointCloudClustered(
                                                kidxs=idxs1,
                                                pts=np.array(inlier_pts_cam2),
                                                pxs=np.array(inlier_pxs_cam2)
                                            )
pc_clustered_clipper["cam3"] = PointCloudClustered(
                                                kidxs=idxs2,
                                                pts=np.array(inlier_pts_cam3),
                                                pxs=np.array(inlier_pxs_cam3)
                                            )
# viz.show_points_2d(cameras, keypoints, pc_clustered_clipper, kp_size=35)
# viz.show_points_3d(cameras, pc_clustered_clipper, centroid_size=25)


# --------------------------------------------------
# Plot CLIPPER results
# --------------------------------------------------

# ------------------------
# Transformations for better viz
# ------------------------
Rz_180 = np.array([
    [-1,  0,  0],
    [ 0, -1,  0],
    [ 0,  0,  1],
])

x_offset = 0
y_offset = 100

lidar_offset_all = np.zeros_like(cameras[0].pts_all)
# lidar_offset_all[:, 2] = 70
lidar_offset_all[:, 1] = y_offset
# lidar_offset_all[:, 0] = x_offset

lidar_offset_inliers = np.zeros_like(D1.T)
lidar_offset_inliers[:, 1] = y_offset
# lidar_offset_inliers[:, 0] = x_offset

cam2_pts_rot = cameras[0].pts_all @ Rz_180.T + lidar_offset_all
D1_rot = D1.T @ Rz_180.T + lidar_offset_inliers

# ------------------------
# Full point clouds (context)
# ------------------------
model_all = o3d.geometry.PointCloud()
model_all.points = o3d.utility.Vector3dVector(
    cam2_pts_rot
)
model_all.paint_uniform_color([0.2, 0.2, 0.8])

data_all = o3d.geometry.PointCloud()
data_all.points = o3d.utility.Vector3dVector(
    cameras[1].pts_all
)
data_all.paint_uniform_color([0.8, 0.2, 0.2])

# ------------------------
# Inlier point clouds (only for correspondences)
# ------------------------


model = o3d.geometry.PointCloud()
model.points = o3d.utility.Vector3dVector(D1_rot)
model.paint_uniform_color(np.array([0,0,1.]))
data = o3d.geometry.PointCloud()
data.points = o3d.utility.Vector3dVector(D2.T)
data.paint_uniform_color(np.array([1.,0,0]))

corr = o3d.geometry.LineSet.create_from_point_cloud_correspondences(model, data, Ain)

# ------------------------
# Visualize
# ------------------------
o3d.visualization.draw_geometries([
    model_all,
    data_all,
    model,
    data,
    corr,
])

plt.show()