import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import cv2
import time
import open3d as o3d
import json
from scipy.spatial.transform import Rotation as R

import clipperpy

import viz
import camera_geom as cg
import file_loader as loader
from CameraLidar import CameraView
from KeypointNN import LidarKeypointNeighbors
from lightglue_loader import LightGlueVisualizer
from PoseEstimation import arun, weighted_umeyama

START_IDX = 0
gps_ref_path = "data/makalii_point/processed_lidar_cam_gps/cam2/bag_camera_2_2025_08_13-01_35_58_5/nav/gps/0.json"

# For clustering
BIN_WIDTH = 1.0        # meters
SIGNIF_THRESHOLD = 5  # min points in a depth bin

# For CLIPPER
SIGMA = 0.4
EPSILON = 0.6

# For relative pose estimation
MIN_NUM_INLIERS = 10

# ============================================================
# GPS Helper Functions
# ============================================================
def rotation_components_deg(R_pred, R_true):
    R_err = project_to_so3(R_pred) @ project_to_so3(R_true).T
    # xyz = roll, pitch, yaw
    return R.from_matrix(R_err).as_euler("xyz", degrees=True)

def project_to_so3(R):
    U, _, Vt = np.linalg.svd(R)
    R_proj = U @ Vt
    if np.linalg.det(R_proj) < 0:
        U[:, -1] *= -1
        R_proj = U @ Vt
    return R_proj


def rotation_error_deg(R_pred, R_true):
    R_true = project_to_so3(R_true)
    R_pred = project_to_so3(R_pred)

    # Relative rotation
    R_err = R_pred @ R_true.T

    # Numerical safety
    cos_theta = (np.trace(R_err) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Angle in radians → degrees
    theta = np.arccos(cos_theta)
    return np.degrees(theta)


def load_gps_from_json(gps_path):
    """
    Given a path to a GPS JSON file, return (lat, lon, alt) as a numpy array.
    
    Example JSON:
    {
      "timestamp": 1755051418122347538,
      "lat": 21.573429884,
      "lon": -157.87523150299998,
      "alt": 33.5156
    }
    """
    print("GPS path", gps_path)
    gps_path = Path(gps_path)
    if not gps_path.exists():
        raise FileNotFoundError(f"GPS JSON file not found: {gps_path}")

    with open(gps_path, "r") as f:
        data = json.load(f)

    lat = data["lat"]
    lon = data["lon"]
    alt = data["alt"]

    return lat, lon, alt

# Define reference origin (lat0, lon0, alt0)
lat0, lon0, alt0 = load_gps_from_json(gps_ref_path)

def gps_path_to_xyz(gps_path, lat0, lon0, alt0):
    """
    Convert lat/lon/alt to local x, y, z in meters relative to reference (lat0, lon0, alt0).
    Can handle scalar or arrays.
    
    Parameters
    ----------
    lat, lon, alt : scalar or array-like
        GPS coordinates
    lat0, lon0 : float
        Reference latitude and longitude
    alt0 : float, optional
        Reference altitude (default 0.0)
    
    Returns
    -------
    x, y, z : scalar or arrays
        Local coordinates in meters relative to (lat0, lon0, alt0)
    """
    lat, lon, alt = load_gps_from_json(gps_path)

    lat = np.asarray(lat)
    lon = np.asarray(lon)
    alt = np.asarray(alt)
    lat0 = float(lat0)
    lon0 = float(lon0)
    alt0 = float(alt0)

    R = 6378137.0  # WGS84 Earth radius in meters

    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    lat0_rad = np.radians(lat0)

    x = R * dlon * np.cos(lat0_rad)
    y = R * dlat
    z = alt - alt0

    return np.array([x, y, z])

# ============================================================
# Helper Functions
# ============================================================
def build_homogeneous(R, t):
    """
    Build a 4x4 homogeneous transformation matrix from rotation and translation.
    """
    H = np.eye(4, dtype=np.float64)
    H[:3, :3] = R
    H[:3, 3] = t.flatten()  # ensure t is 1D
    return H

def invert_H(H):
    R = H[:3, :3]
    t = H[:3, 3]
    H_inv = np.eye(4)
    H_inv[:3, :3] = R.T
    H_inv[:3, 3] = -R.T @ t
    return H_inv

def quaternion_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return np.arctan2(siny_cosp, cosy_cosp)

def get_gt_H(gt_path: Path) -> np.ndarray:
    """
    Load ground-truth pose from JSON and return 4x4 homogeneous transform
    from frame_id -> child_frame_id.
    """
    gt_path = Path(gt_path)

    with open(gt_path, "r") as f:
        gt = json.load(f)

    # Translation
    t = np.array([
        gt["position"]["x"],
        gt["position"]["y"],
        gt["position"]["z"],
    ])

    # Quaternion (x, y, z, w)
    q = [
        gt["orientation"]["qx"],
        gt["orientation"]["qy"],
        gt["orientation"]["qz"],
        gt["orientation"]["qw"],
    ]

    # Rotation matrix
    R_mat = R.from_quat(q).as_matrix()

    # Homogeneous transform
    H = np.eye(4)
    H[:3, :3] = R_mat
    H[:3, 3] = t

    return H

import numpy as np

def relative_cam2_to_cam3(H_imu_retr2quer, T_ic_cam2, T_ci_cam3):
    """
    H_imu_retr2quer: 4x4 numpy, imu_retr -> imu_quer
    T_ic_cam2: 4x4 cam2 -> imu0
    T_ci_cam3: 4x4 imu0 -> cam3
    """
    H_cam2_cam3 = T_ci_cam3 @ H_imu_retr2quer @ T_ic_cam2
    return H_cam2_cam3

# ============================================================
# Load Data
# ============================================================
def image_to_lidar_path(img_path: Path) -> Path:
    cam_imgnum = img_path.stem
    bag_dir = img_path.parents[2].name
    cam_root = img_path.parents[3]

    return (
        cam_root /
        bag_dir /
        "lidar/pcd" /
        f"{cam_imgnum}.pcd"
    )

def image_to_gt_path(img_path: Path) -> Path:
    img_path = Path(img_path)
    cam_imgnum = img_path.stem
    bag_dir = img_path.parents[2].name
    cam_root = img_path.parents[3]

    return (
        cam_root /
        bag_dir /
        "nav/gt" /
        f"{cam_imgnum}.json"
    )

def image_to_gps_path(img_path: Path) -> Path:
    cam_imgnum = img_path.stem
    bag_dir = img_path.parents[2].name
    cam_root = img_path.parents[3]

    return (
        cam_root /
        bag_dir /
        "nav/gps" /
        f"{cam_imgnum}.json"
    )

with open("matched_img_paths.json", "r") as f:
    data = json.load(f)

path_true_all = data["true"]
path_quer_all = data["queries"]
path_retr_all = data["retrieved"]

pred_cam3_gps_xy_aru = []    # Retr cam3 (retr) predicted positions
pred_cam3_gps_xy_pnp = []    # Retr cam3 (retr) predicted positions

errors_aru = []
errors_pnp = []


for path_idx in range(START_IDX, len(path_quer_all)):

    quer_cam_path = path_quer_all[path_idx]
    retr_cam_path = path_retr_all[path_idx][0]  # top K=1

    if len(retr_cam_path) == 0:
        continue

    print("\n---------------Idx", path_idx, "---------------")
    print("quer", quer_cam_path)
    print("retr", retr_cam_path)
    
    cameras = [
        CameraView(
            name="cam2",
            image_path=Path(retr_cam_path),
            lidar_pcd_path=Path(image_to_lidar_path(Path(retr_cam_path))),
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
            image_path=Path(quer_cam_path),
            lidar_pcd_path=Path(image_to_lidar_path(Path(quer_cam_path))),
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
        print(f"\n{cam.name} Num keypoints with neighbors:", num_with_neighbors)
        print(f"{cam.name} Num neighbors total:", len(nn[cam.name].pxs_all))


    # ============================================================
    # Plotting image, keypoints, lidar
    # ============================================================

    # # Keypoint matches
    # vis.visualize_matches(
    #     cameras[1].image_path,  
    #     cameras[0].image_path,  
    # )
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

    # ======================
    # Run CLIPPER
    # ======================
    print("\nLightglue matches:", len(m_kpxs_cam2))
    print("Num associations:", A.shape[0])

    iparams = clipperpy.invariants.EuclideanDistanceParams()
    iparams.sigma = SIGMA
    iparams.epsilon = EPSILON
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

    # ======================
    # Get relative pose estimate
    # ======================
    # if len(inlier_pts_cam2) > MIN_NUM_INLIERS:
    #     inlier_pts_cam2 = np.array(inlier_pts_cam2)
    #     inlier_pts_cam3 = np.array(inlier_pts_cam3)
    #     inlier_pxs_cam2 = np.array(inlier_pxs_cam2)
    #     inlier_pxs_cam3 = np.array(inlier_pxs_cam3)

    #     cam_retr = cameras[0] # cam2
    #     cam_quer = cameras[1] # cam3


    #     # GT query camera pose in global frame
    #     H_quer_global_true = get_gt_H(image_to_gt_path(cameras[1].image_path))
    #     H_retr_global_true = get_gt_H(image_to_gt_path(cameras[0].image_path))

    #     # Arun relative transform (retr→quer in cam3 frame)
    #     R_retr2quer_arun, t_retr2quer_arun = arun(inlier_pts_cam2.T, inlier_pts_cam3.T)
    #     # R_retr2quer_arun, t_retr2quer_arun = weighted_umeyama(inlier_pts_cam2.T, inlier_pts_cam3.T)
    #     H_retr2quer_cam3_arun = build_homogeneous(R_retr2quer_arun, t_retr2quer_arun)
    #     H_retr_global_pred_arun = H_quer_global_true @ H_retr2quer_cam3_arun
    #     H_quer_global_pred_arun = invert_H(H_retr2quer_cam3_arun) @ H_retr_global_true

    #     # PnP relative transform (retr→quer in cam3 frame?)
    #     ret,R_retr2quer_pnp, t_retr2quer_pnp = cv2.solvePnP(inlier_pts_cam2, inlier_pxs_cam3, 
    #                                                         cam_quer.K, 
    #                                                         cam_quer.dist)        
    #     H_retr2quer_cam3_pnp = build_homogeneous(R_retr2quer_pnp, t_retr2quer_pnp)
    #     H_retr_global_pred_pnp = H_quer_global_true @ H_retr2quer_cam3_pnp
    #     H_quer_global_pred_pnp = invert_H(H_retr2quer_cam3_pnp) @ H_retr_global_true

    #     # Translation part
    #     t_retr_global_true = H_retr_global_true[:3, 3]
    #     t_retr_global_arun = H_retr_global_pred_arun[:3, 3]
    #     t_retr_global_pnp = H_retr_global_pred_pnp[:3, 3]

    #     t_quer_global_true = H_quer_global_true[:3, 3]
    #     t_quer_global_arun = H_quer_global_pred_arun[:3, 3]
    #     t_quer_global_pnp = H_quer_global_pred_pnp[:3, 3]

    #     print("True retr trans gps: ", t_retr_global_true)
    #     print("Pred retr trans aru: ", t_retr_global_arun)
    #     print("Pred retr trans pnp: ", t_retr_global_pnp)

    #     print("True quer trans gps: ", t_quer_global_true)
    #     print("Pred quer trans aru: ", t_quer_global_arun)
    #     print("Pred quer trans pnp: ", t_quer_global_pnp)

    #     # Translational error
    #     trans_err_aru = t_quer_global_arun - t_quer_global_true
    #     trans_err_pnp = t_quer_global_pnp  - t_quer_global_true

    #     print("\nArun translational error (m):", trans_err_aru)
    #     print("PnP translational error  (m):", trans_err_pnp)

    #     # Rotation part
    #     R_retr_global_true = H_retr_global_true[:3, :3]
    #     R_retr_global_arun = H_retr_global_pred_arun[:3, :3]
    #     R_retr_global_pnp  = H_retr_global_pred_pnp[:3, :3]

    #     R_quer_global_true = H_quer_global_true[:3, :3]
    #     R_quer_global_arun = H_quer_global_pred_arun[:3, :3]
    #     R_quer_global_pnp  = H_quer_global_pred_pnp[:3, :3]

    #     rot_err_aru = rotation_components_deg(R_quer_global_arun, R_quer_global_true)
    #     rot_err_pnp = rotation_components_deg(R_quer_global_pnp,  R_quer_global_true)

    #     # For plotting
    #     pred_cam3_gps_xy_aru.append(t_quer_global_arun[0:2])
    #     pred_cam3_gps_xy_pnp.append(t_quer_global_pnp[0:2])

    #     errors_aru.append({
    #         "trans": trans_err_aru,
    #         "rot":   rot_err_aru,
    #     })

    #     errors_pnp.append({
    #         "trans": trans_err_pnp,
    #         "rot":   rot_err_pnp,
    #     })
    if len(inlier_pts_cam2) > MIN_NUM_INLIERS:
        inlier_pts_cam2 = np.asarray(inlier_pts_cam2)
        inlier_pts_cam3 = np.asarray(inlier_pts_cam3)
        inlier_pxs_cam3 = np.asarray(inlier_pxs_cam3)

        cam_retr = cameras[0]  # cam2
        cam_quer = cameras[1]  # cam3

        # --- Ground truth poses ---
        H_retr_global_true = get_gt_H(image_to_gt_path(cam_retr.image_path))
        H_quer_global_true = get_gt_H(image_to_gt_path(cam_quer.image_path))

        # ============================================================
        # Arun: relative pose (retr → quer)
        # ============================================================
        R_rq_arun, t_rq_arun = arun(
            inlier_pts_cam2.T,
            inlier_pts_cam3.T
        )
        H_rq_arun = build_homogeneous(R_rq_arun, t_rq_arun)

        # Query pose prediction
        H_quer_global_pred_arun = H_retr_global_true @ invert_H(H_rq_arun)
        # H_quer_global_pred_arun = invert_H(H_rq_arun) @ H_retr_global_true
        # H_quer_global_pred_arun = H_rq_arun @ H_retr_global_true
        # H_retr_global_pred_arun = H_quer_global_true @ H_rq_arun

        # ============================================================
        # PnP: relative pose (retr → quer)
        # ============================================================
        ret, rvec, tvec = cv2.solvePnP(
            inlier_pts_cam2,
            inlier_pxs_cam3,
            cam_quer.K,
            cam_quer.dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        R_rq_pnp, _ = cv2.Rodrigues(rvec)
        H_rq_pnp = build_homogeneous(R_rq_pnp, tvec)

        # Query pose prediction
        H_quer_global_pred_pnp = invert_H(H_rq_pnp) @ H_retr_global_true

        # ============================================================
        # Translation error (component-wise)
        # ============================================================
        t_true = H_quer_global_true[:3, 3]
        t_arun = H_quer_global_pred_arun[:3, 3]
        t_pnp  = H_quer_global_pred_pnp[:3, 3]

        trans_err_arun = t_true - t_arun
        trans_err_pnp  = t_true - t_pnp 

        print("\nArun translational error (m):", trans_err_arun)

        # ============================================================
        # Rotation error (component-wise, degrees)
        # ============================================================
        R_true = H_quer_global_true[:3, :3]
        R_arun = H_quer_global_pred_arun[:3, :3]
        R_pnp  = H_quer_global_pred_pnp[:3, :3]

        rot_err_arun = rotation_components_deg(R_arun, R_true)
        rot_err_pnp  = rotation_components_deg(R_pnp,  R_true)

        # ============================================================
        # Save for plotting / evaluation
        # ============================================================
        pred_cam3_gps_xy_aru.append(t_arun[:2])
        pred_cam3_gps_xy_pnp.append(t_pnp[:2])

        errors_aru.append({
            "trans": trans_err_arun,   # (dx, dy, dz)
            "rot":   rot_err_arun,     # (roll, pitch, yaw) in deg
        })

        errors_pnp.append({
            "trans": trans_err_pnp,
            "rot":   rot_err_pnp,
        })

    else:
        print("\nNot enough inliers after CLIPPER")


    # ======================
    # Plot CLIPPER results
    # ======================

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
    # # Pointcloud matches (open before plt, interactive)
    # o3d.visualization.draw_geometries([
    #     model_all,
    #     data_all,
    #     model,
    #     data,
    #     corr,
    # ])


    # # Pointcloud matches (open at same time as plt, not interactive)
    # vis_o3d = o3d.visualization.Visualizer()
    # vis_o3d.create_window(window_name="Open3D", width=960, height=720)

    # for g in [model_all, data_all, model, data, corr]:
    #     vis_o3d.add_geometry(g)

    # vis_o3d.poll_events()
    # vis_o3d.update_renderer()

    # Inlier LightGlue matches
    # vis.visualize_inliers(
    #     cameras[1].image_path,
    #     cameras[0].image_path,
    #     np.array(inlier_pxs_cam3),
    #     np.array(inlier_pxs_cam2)
    # )
    
    # plt.show()

# Make gps paths list
true_cam2_gps_xy = []    # True cam2        positions
true_cam3_gps_xy = []    # True cam3 (quer) positions
for t_path, q_path, r_path in zip(path_true_all, path_quer_all, path_retr_all):
    true_cam2_gps_xy.append(get_gt_H(image_to_gt_path(t_path)))
    true_cam3_gps_xy.append(get_gt_H(image_to_gt_path(q_path)))

viz.plot_all_gps_paths(
    true_cam2_gps_xy,
    true_cam3_gps_xy,
    pred_cam3_gps_xy_aru,
    pred_cam3_gps_xy_pnp
)

# Error histograms
trans_arun = np.array([e["trans"] for e in errors_aru])
rot_arun   = np.array([e["rot"] for e in errors_aru])

trans_pnp  = np.array([e["trans"] for e in errors_pnp])
rot_pnp    = np.array([e["rot"] for e in errors_pnp])

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
labels = ["x", "y", "z"]

for i in range(3):
    axes[i].hist(trans_arun[:, i], bins=40, alpha=0.6, label="Arun")
    # axes[i].hist(trans_pnp[:,  i], bins=40, alpha=0.6, label="PnP")
    axes[i].set_xlabel(f"Translation error {labels[i]} (m)", fontsize=16)
    axes[i].grid(True)

axes[0].set_ylabel("Count", fontsize=16)
axes[1].set_title("Translation Component Error Histograms", fontsize=24)
axes[2].legend()

plt.tight_layout()
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
labels = ["roll", "pitch", "yaw"]

for i in range(3):
    axes[i].hist(rot_arun[:, i], bins=40, alpha=0.6, label="Arun")
    # axes[i].hist(rot_pnp[:,  i], bins=40, alpha=0.6, label="PnP")
    axes[i].set_xlabel(f"{labels[i]} error (deg)", fontsize=16)
    axes[i].grid(True)

axes[0].set_ylabel("Count", fontsize=16)
axes[1].set_title("Rotation Component Error Histograms", fontsize=24)
axes[2].legend()

plt.tight_layout()


plt.show()