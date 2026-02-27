import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import cv2
import time
import open3d as o3d
import json
from scipy.spatial.transform import Rotation as R
from typing import List
from dataclasses import dataclass

import clipperpy

from utils import viz
from utils import camera_geom as cg
from utils import file_loader as loader
from utils.clipper_temp import cluster_info_to_clipper_list, fuse_matrix_geometric, fuse_matrix_weighted, build_C_from_M
from utils.CameraLidar import CameraView
from utils.KeypointNN import LidarKeypointNeighbors
from utils.lightglue_loader import LightGlueVisualizer
from utils.PoseEstimation import arun, arun_weighted
from utils import Clusters
from utils.icp_tools import register_with_initial_guess

import time

START_IDX = 195

# For viz
PLOT_FLAG = True

# For ICP
ARUN_ICP = False

# For LightGlue
NUM_LG_KEYPOINTS = 2048
MAX_KEYPOINTS = 100

# For clustering
PIXEL_WINDOW = 20
BIN_WIDTH = 0.5         # meters
SIGNIF_THRESHOLD = 5    # min points in a depth bin
SIGNIF_BINS = 5

# For CLIPPER
iparams = clipperpy.invariants.ROMANParams()
iparams.point_dim = 3
iparams.fusion_method = clipperpy.invariants.ROMAN.GEOMETRIC_MEAN
iparams.ratio_feature_dim = 0   # using pca, volume, etc
iparams.cos_feature_dim = 0     # size of feature descriptors (ex. LightGlue is 256)
iparams.sigma = 0.8             # decay speed
iparams.epsilon = 1.0           # distance cutoff
iparams.gravity_guided = False

invariant = clipperpy.invariants.ROMAN(iparams)
params = clipperpy.Params()
clipper = clipperpy.CLIPPERPairwiseAndSingle(invariant, params)

DIST_DESC_FLAG = False

# For relative pose estimation
MIN_NUM_INLIERS = 10

# Imu orientation frame
H_imu_lidar = np.array([[-1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, 1, -0.05013],
                            [0, 0, 0, 1]])

# ============================================================
# ICP Helper Functions
# ============================================================
def run_icp(source_points, target_points, init_T=None):

    # Convert to Open3D format
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()

    source.points = o3d.utility.Vector3dVector(source_points)
    target.points = o3d.utility.Vector3dVector(target_points)

    # Initial transform
    if init_T is None:
        init_T = np.eye(4)

    threshold = 10.0  # distance threshold (meters)

    reg = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        init_T,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    return reg.transformation, reg.fitness, reg.inlier_rmse

def estimate_relative_pose(kps1, kps2, K):

    # Convert to float32
    pts1 = kps1.astype(np.float32)
    pts2 = kps2.astype(np.float32)

    # Essential matrix with RANSAC
    E, mask = cv2.findEssentialMat(
        pts1,
        pts2,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    # Recover pose
    _, R_temp, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

    return R_temp, t, mask_pose

def build_init_T(R_temp, t):
    T = np.eye(4)
    T[:3, :3] = R_temp
    T[:3, 3] = t.squeeze()  # t is unit direction
    return T

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

gps_ref_path = "data/makalii_point/processed_lidar_cam_gps/cam2/bag_camera_2_2025_08_13-01_35_58_5/nav/gt/0.json"

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
    gps_path = Path(gps_path)
    if not gps_path.exists():
        raise FileNotFoundError(f"GPS JSON file not found: {gps_path}")

    with open(gps_path, "r") as f:
        data = json.load(f)

    lat = data["position"]["latitude"]
    lon = data["position"]["longitude"]
    alt = data["position"]["altitude"]

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

def get_gt_H(gt_path: Path) -> np.ndarray:
    """
    Load ground-truth pose from JSON and return 4x4 homogeneous transform
    from frame_id -> child_frame_id.
    """
    gt_path = Path(gt_path)

    with open(gt_path, "r") as f:
        gt = json.load(f)

    # Translation
    t = gps_path_to_xyz(gt_path, lat0, lon0, alt0)


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


def relative_cam2_to_cam3(H_imu_retr2quer, T_ic_cam2, T_ci_cam3):
    """
    H_imu_retr2quer: 4x4 numpy, imu_retr -> imu_quer
    T_ic_cam2: 4x4 cam2 -> imu0
    T_ci_cam3: 4x4 imu0 -> cam3
    """
    H_cam2_cam3 = T_ci_cam3 @ H_imu_retr2quer @ T_ic_cam2
    return H_cam2_cam3

# ============================================================
# Get clusters
# ============================================================

@dataclass
class PointCloudClustered:
    
    kvalid: np.ndarray           # (N, ) Indicates if that keypoint had any valid lidar clusters(Number of original keypoints)
    pts_per_kp: List[np.ndarray] 
    pxs_per_kp: List[np.ndarray] 
    
    kidxs: np.ndarray            # (M,)  Associated keypoint index of each cluster
    pts: np.ndarray              # (M,3) 3D points of the cluster centroids
    pxs: np.ndarray              # (M,2) 2D pixels coords of the cluster centroids

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
def image_to_meta_path(img_path: Path) -> Path:
    img_path = Path(img_path)
    cam_imgnum = img_path.stem
    bag_dir = img_path.parents[2].name
    cam_root = img_path.parents[3]

    return (
        cam_root /
        bag_dir /
        "metadata" /
        f"timestamps.json"
    )

with open("matched_img_paths_mega.json", "r") as f:
    data = json.load(f)

# ============================================================
# Begin Processing
# ============================================================

path_true_all = data["true"]
path_quer_all = data["queries"]
path_retr_all = data["retrieved"]
path_desc_all = data["desc_info"]

pred_cam3_gps_xy_aru = []    # Retr cam3 (retr) predicted positions
pred_cam3_gps_xy_icp = []
pred_cam3_gps_xy_pnp = []    # Retr cam3 (retr) predicted positions

errors_aru = []
errors_pnp = []
errors_ess = []

lg = LightGlueVisualizer(max_keypoints=NUM_LG_KEYPOINTS)

time_icp = []
time_clipper_arun_tot = []
time_build_clipper = []
time_solve_clipper = []
time_collect_neighbors = []
time_build_clusters = []
time_inliers = []
time_solve_arun = []

for path_idx in range(START_IDX, len(path_quer_all)):

    quer_cam_path = path_quer_all[path_idx]
    retr_cam_path = path_retr_all[path_idx]  # top K=1
    desc_info_path = path_desc_all[path_idx]

    if retr_cam_path == "None" or len(retr_cam_path) == 0:
        continue


    # print(image_to_meta_path(quer_cam_path))
    # with open(image_to_meta_path(quer_cam_path), "r") as f:
    #     quer_ts = json.load(f)
    # with open(image_to_meta_path(retr_cam_path), "r") as f:
    #     retr_ts = json.load(f)
    # print("timestamp quer", quer_ts)
    # print("timestamp retr", retr_ts)

    # exit()
    
    cameras = [
        CameraView(
            name="cam2",
            image_path=Path(retr_cam_path),
            lidar_pcd_path=Path(image_to_lidar_path(Path(retr_cam_path))),
            calib_path=Path("data/calib_leftcam.json"),
            T_lidar_cam=np.array(
                                [[-1, 0, 0,  0.02],
                                [0, 0, -1, -0.07],
                                [0, -1, 0,  -0.32],
                                [0, 0, 0, 1]]
                            )
            # T_lidar_cam=np.load("data/makalii_point/cam2/T_lidar_to_cam.npy")
        #     T_lidar_cam=np.array([
        #     [-9.99925370e-01, -1.22168334e-02,  6.39677745e-05,
        #     2.00000000e-02],
        # [-1.22298005e-16, -5.23596383e-03, -9.99986292e-01,
        #     -1.00000000e-01],
        # [ 1.22170008e-02, -9.99911663e-01,  5.23557307e-03,
        #     -1.70000000e-01],
        # [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #     1.00000000e+00]
        # ]) # left (cam2) - manual_undist_ssp_nocrop
        ),
        CameraView(
            name="cam3",
            image_path=Path(quer_cam_path),
            lidar_pcd_path=Path(image_to_lidar_path(Path(quer_cam_path))),
            calib_path=Path("data/calib_rightcam.json"),
            T_lidar_cam=np.array(
                    [[1, 0, 0, -0.02],
                    [0, 0, -1, -0.07],
                    [0, 1, 0, -0.33],
                    [0, 0, 0, 1]]
                )
            # T_lidar_cam=np.load("data/makalii_point/cam3/T_lidar_to_cam.npy")
            # T_lidar_cam=np.array([
            # [ 0.99992385,  0.01225848, -0.00142493, -0.03      ],
            # [-0.0017452 ,  0.02615559, -0.99965636, -0.11      ],
            # [-0.012217  ,  0.99958272,  0.02617499, -0.12      ],
            # [ 0.        ,  0.        ,  0.        ,  1.        ]
        # ]) # right (cam3) - manual_undist_ssp_nocrop,
        ),
    ]

    for cam in cameras:
        cam.load_image_and_calib()
        cam.project_lidar()

    print("\n---------------Idx", path_idx, "---------------")
    print("quer", quer_cam_path)
    print("retr", retr_cam_path)
    print(cameras[0].lidar_pcd_path)
    print(cameras[1].lidar_pcd_path)
    print("")
    # --- Ground truth poses ---
    cam_retr = cameras[0]  # cam2
    cam_quer = cameras[1]  # cam3

    H_imur_global_gt = get_gt_H(image_to_gt_path(cam_retr.image_path))
    H_imuq_global_gt = get_gt_H(image_to_gt_path(cam_quer.image_path))
    # ============================================================
    # LightGlue matches
    # ============================================================
    # m_kpxs_cam3, m_kpxs_cam2, m_descs_cam3, m_descs_cam2, m_scores = lg.get_matched_keypoints_and_descriptors(
    #     cameras[1].image_path,
    #     cameras[0].image_path,
    # )
    desc_data = np.load(desc_info_path)

    m_kpxs_cam2 = desc_data["kps_lg_r"]
    m_kpxs_cam3 = desc_data["kps_lg_q"]
    m_descs_cam2 = desc_data["descs_lg_r"]
    m_descs_cam3 = desc_data["descs_lg_q"]
    m_scores = desc_data["m_lg_scores"]
    descs_dino_cam2 = desc_data["descs_dino_r"]
    descs_dino_cam3 = desc_data["descs_dino_q"]

    print("Num LightGlue Matches", len(m_kpxs_cam2))
    keypoints = {
        "cam2": m_kpxs_cam2,
        "cam3": m_kpxs_cam3,
    }
    descriptors = {
        "cam2": m_descs_cam2,
        "cam3": m_descs_cam3,
    }
    # Make dict to go from keypoint -> descriptor (TODO: Better class to handle this info)
    def build_kp_descriptor_dict(keypoints, descriptors, round_to=6):
        """
        keypoints: (N, 2) numpy array
        descriptors: (N, D) numpy array
        round_to: decimals for rounding to avoid float precision issues
        """

        kp_dict = {}

        for kp, desc in zip(keypoints, descriptors):
            key = tuple(np.round(kp, round_to))
            kp_dict[key] = desc

        return kp_dict
    
    kp_desc_dict_cam3 = build_kp_descriptor_dict(m_kpxs_cam3, m_descs_cam3)
    kp_desc_dict_cam2 = build_kp_descriptor_dict(m_kpxs_cam2, m_descs_cam2)


    # ============================================================
    # ICP: relative pose (retr → quer)
    # ============================================================

    icp_start = time.time()
    # Init transform from essential matrix 2D-2D lightglue matches
    K = cameras[0].K  # or however you store intrinsics
    R_retr_quer, t_retr_querr, _  = estimate_relative_pose(
        m_kpxs_cam2,   # retrieved 
        m_kpxs_cam3,   # query
        K
    )
    # t_lg = np.zeros_like(t_lg)
    T_camr_camq = build_init_T(R_retr_quer, t_retr_querr)

    # Transform init transform to lidar frame
    T_lidr_camr = cameras[0].T_lidar_cam
    T_lidq_camq = cameras[1].T_lidar_cam
    T_camq_lidq = np.linalg.inv(T_lidq_camq)

    T_lidr_lidq = T_camq_lidq @ T_camr_camq @ T_lidr_camr

    # True init lidar transform
    H_imur_imuq_gt = np.linalg.inv(H_imuq_global_gt) @ \
            H_imur_global_gt
    T_lidr_lidq = H_imu_lidar @ H_imur_imuq_gt @ np.linalg.inv(H_imu_lidar)

    print("T lidar")

    # ICP reference (source) to query (target). 
    # Points are in lidar frame, and init T is transformation of the lidar reference to query pose
    H_rq_icp, fitness_icp, rmse_icp = run_icp(cameras[0].pts_all, cameras[1].pts_all, T_lidr_lidq)

    # # Refine ICP
    # source = o3d.geometry.PointCloud()
    # source.points = o3d.utility.Vector3dVector(cameras[0].pts_all)
    # target = o3d.geometry.PointCloud()
    # target.points = o3d.utility.Vector3dVector(cameras[1].pts_all)
    # H_rq_icp = register_with_initial_guess(
    #     source,
    #     target,
    #     H_rq_icp)
    
    # Get transform
    H_quer_global_pred_icp = H_imur_global_gt @ invert_H(H_rq_icp)
    t_icp  = H_quer_global_pred_icp[:3, 3]
    pred_cam3_gps_xy_icp.append(t_icp[:2])
    time_icp.append(time.time() - icp_start)
    # ============================================================
    # Keypoint nearest neighbors
    # ============================================================
    arun_start = time.time()
    clusters_start = time.time()

    nn = {}

    for cam in cameras:

        collect_start = time.time()
        nn[cam.name] = Clusters.collect_lidar_neighbors_per_keypoint(
            keypoints[cam.name],
            descriptors[cam.name],
            cam.u,
            cam.v,
            cam.pts_cam,
            cam.pts_infront,
            pixel_radius=PIXEL_WINDOW
        )
        time_collect_neighbors.append(time.time() - collect_start)
        # nn[cam.name] = Clusters.collect_lidar_neighbors_per_keypoint_knn_xybox(
        #     keypoints[cam.name],
        #     descriptors[cam.name],
        #     cam.u,
        #     cam.v,
        #     cam.pts_cam,
        #     cam.pts_infront,
        #     # min_pixel_dist=PIXEL_WINDOW
        # )

        num_with_neighbors = sum([len(d) > 0 for d in nn[cam.name].pxs_per_kp])
        print(f"\n{cam.name} Num keypoints with neighbors:", num_with_neighbors)
        print(f"{cam.name} Num neighbors total:", len(nn[cam.name].pxs_all))

    # ---------------------------------------------------
    # Get lidar foreground clusters around each keypoint 
    # ---------------------------------------------------
    pc_clustered = {}

    for cam in cameras:
        nn_info = nn[cam.name]

        kvalid_list = []
        centroid_pts_pkp = []
        centroid_pxs_pkp = []

        centroid_kidxs = []
        centroid_pts = []
        centroid_pxs = []

        for k_idx in range(nn_info.num_kps):

            kp     = nn_info.keypoints[k_idx]
            desc   = nn_info.descriptors[k_idx]
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

                kvalid_list.append(False)
                centroid_pts_pkp.append(np.zeros(3))
                centroid_pxs_pkp.append(np.zeros(2)) 
                continue

            # -----------------------
            # Depth histogram
            # -----------------------
            bins = np.arange(depths.min(), depths.max() + BIN_WIDTH, BIN_WIDTH)
            hist_counts, bin_edges = np.histogram(depths, bins=bins)

            # Find nearest significant depth bin
            signif_bins = []
            get_bin_flag = True # Only resets to true when there is a "gap" between signif bins
            for hist_idx in range(len(hist_counts)):

                hist_count = hist_counts[hist_idx]

                # # Get each bin in order (no gaps)
                # if hist_count >= SIGNIF_THRESHOLD:
                #     signif_bins.append(hist_idx)

                # Get bins for each layer (must have gap between)
                if get_bin_flag and hist_count >= SIGNIF_THRESHOLD:
                    signif_bins.append(hist_idx)
                    get_bin_flag = False
                if hist_count < SIGNIF_THRESHOLD:
                    get_bin_flag = True
            # print("\nnew hist")
            # print(hist_counts)
            # print(signif_bins)
            

            if len(signif_bins) == 0:
                centroid_kidxs.append(-1)
                centroid_pts.append(np.zeros(3))
                centroid_pxs.append(np.zeros(2))

                kvalid_list.append(False)
                centroid_pts_pkp.append(np.zeros(3))
                centroid_pxs_pkp.append(np.zeros(2)) 
                continue

            centroid_pts_temp = []
            centroid_pxs_temp = []

            # print("signif", signif_bins.shape)
            for bin_idx in range(min(SIGNIF_BINS, len(signif_bins))): # choose closest 2 signif bins:
                b = signif_bins[bin_idx]               # nearest (smallest depth)
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

                    kvalid_list.append(False)
                    centroid_pts_pkp.append(np.zeros(3))
                    centroid_pxs_pkp.append(np.zeros(2)) 
                    continue
                # # -----------------------
                # # Option 1: Centroid
                # # -----------------------
                # centroid_pt = pts_keep.mean(axis=0)

                # # Only for visualization
                # centroid_px = cg.project_point_to_image_plane(
                #     centroid_pt,
                #     cam.T_lidar_cam,
                #     cam.K_proj
                # )

                # -----------------------
                # Option 2: Compute hybrid centroid
                # -----------------------

                # Find closest pixel in this depth bin to the keypoint
                dists_px = np.sum((pxs_keep - kp[:2])**2, axis=1)
                closest_idx = np.argmin(dists_px)

                closest_pt = pts_keep[closest_idx]

                # Use its X,Y
                cx = closest_pt[0]
                cy = closest_pt[1]

                # Use mean Z of cluster
                mean_z = pts_keep[:, 2].mean()

                centroid_pt = np.array([cx, cy, mean_z])
                centroid_px = cg.project_point_to_image_plane(
                    centroid_pt,
                    cam.T_lidar_cam,
                    cam.K_proj
                )

                # -----------------------
                # Save
                # -----------------------

                centroid_kidxs.append(k_idx)
                centroid_pts.append(centroid_pt)
                centroid_pxs.append(centroid_px)

                centroid_pts_temp.append(centroid_pt)
                centroid_pxs_temp.append(centroid_px)

            kvalid_list.append(True)
            centroid_pts_pkp.append(np.array(centroid_pts_temp))
            centroid_pxs_pkp.append(np.array(centroid_pxs_temp))

        pc_clustered[cam.name] = PointCloudClustered(
            kvalid=np.asarray(kvalid_list),
            pts_per_kp=centroid_pts_pkp,
            pxs_per_kp=centroid_pxs_pkp,
            kidxs=np.asarray(centroid_kidxs),
            pts=np.asarray(centroid_pts),
            pxs=np.asarray(centroid_pxs)
        )

    # ------------------------
    # Get valid cluster pairs 
    # ------------------------
    kvalid_filtered = {"cam2":[], "cam3":[]}
    pts_per_kp_filtered = {"cam2":[], "cam3":[]}
    pxs_per_kp_filtered = {"cam2":[], "cam3":[]}

    centroid_kidxs_cam2 = []
    centroid_kidxs_cam3 = []

    centroid_pts_cam2 = []
    centroid_pts_cam3 = []

    centroid_pxs_cam2 = []
    centroid_pxs_cam3 = []

    num_valid = 0

    # Iterate through all keypoints
    for k_idx in range(len(pc_clustered["cam2"].kvalid)):

        # Check if keypoint has valid lidar cluster
        cam2_valid = pc_clustered["cam2"].kvalid[k_idx]
        cam3_valid = pc_clustered["cam3"].kvalid[k_idx]
        # print("\nvalid?", k_idx, cam2_valid, cam3_valid)

        if cam2_valid and cam3_valid:
            kvalid_filtered["cam2"].append(1)
            pts_per_kp_filtered["cam2"].append(pc_clustered["cam2"].pts_per_kp[k_idx])
            pxs_per_kp_filtered["cam2"].append(pc_clustered["cam2"].pxs_per_kp[k_idx])

            kvalid_filtered["cam3"].append(1)
            pts_per_kp_filtered["cam3"].append(pc_clustered["cam3"].pts_per_kp[k_idx])
            pxs_per_kp_filtered["cam3"].append(pc_clustered["cam3"].pxs_per_kp[k_idx])

            # Append all centroids for both keypoints (might have different num signif depths for each cam)


            # print("\nkp", k_idx, len(kvalid_list))
            # print("len", len(pc_clustered["cam2"].pts_per_kp[k_idx]))
            for p_idx in range(len(pc_clustered["cam2"].pts_per_kp[k_idx])):
                # print("\ntest", p_idx)
                # print(pc_clustered["cam2"].pts_per_kp[k_idx])
                # print(pc_clustered["cam2"].pxs_per_kp[k_idx])
                centroid_pts_cam2.append(pc_clustered["cam2"].pts_per_kp[k_idx][p_idx])
                centroid_pxs_cam2.append(pc_clustered["cam2"].pxs_per_kp[k_idx][p_idx])
                centroid_kidxs_cam2.append(k_idx)


            for p_idx in range(len(pc_clustered["cam3"].pts_per_kp[k_idx])):
                centroid_pts_cam3.append(pc_clustered["cam3"].pts_per_kp[k_idx][p_idx])
                centroid_pxs_cam3.append(pc_clustered["cam3"].pxs_per_kp[k_idx][p_idx])
                centroid_kidxs_cam3.append(k_idx)

            num_valid += 1
            if num_valid >= MAX_KEYPOINTS:
                break

    pc_clustered_filtered = {}

    # These keypoints should now be matched putative associations
    pc_clustered_filtered["cam2"] = PointCloudClustered(
                                                    kvalid=kvalid_filtered["cam2"],               
                                                    pts_per_kp=pts_per_kp_filtered["cam2"],       
                                                    pxs_per_kp=pxs_per_kp_filtered["cam2"],        
                                                    kidxs=np.array(centroid_kidxs_cam2),
                                                    pts=np.array(centroid_pts_cam2),
                                                    pxs=np.array(centroid_pxs_cam2)
                                                )
    pc_clustered_filtered["cam3"] = PointCloudClustered(
                                                    kvalid=kvalid_filtered["cam3"],               
                                                    pts_per_kp=pts_per_kp_filtered["cam3"],       
                                                    pxs_per_kp=pxs_per_kp_filtered["cam3"],       
                                                    kidxs=np.array(centroid_kidxs_cam3),
                                                    pts=np.array(centroid_pts_cam3),
                                                    pxs=np.array(centroid_pxs_cam3)
                                                )
    time_build_clusters.append(time.time() - clusters_start)

    # viz.show_points_2d(cameras, keypoints, pc_clustered_filtered)
    # viz.show_points_3d(cameras, pc_clustered_filtered)

    # ============================================================
    # Build consistency matrix
    # ============================================================
    tbuild0 = time.time()

    # CLIPPER associations
    pc1 = []
    pc2 = []
    A = [] 
    kp_idxs_D1 = []    # original keypoint indices
    kp_idxs_D2 = []    # original keypoint indices
    pt_idxs_D1 = []    # centroid indices for each keypoint
    pt_idxs_D2 = []    # centroid indices for each keypoint
    for k_idx in range(len(pc_clustered_filtered["cam2"].pts_per_kp)): 
        # Get lidar neighbors around keypoint 
        pts_i = pc_clustered_filtered["cam2"].pts_per_kp[k_idx] 
        pts_j = pc_clustered_filtered["cam3"].pts_per_kp[k_idx] 

        pxs_i = pc_clustered_filtered["cam2"].pxs_per_kp[k_idx] 
        pxs_j = pc_clustered_filtered["cam3"].pxs_per_kp[k_idx] 
        # -------------------------------
        # Make all putative associations 
        # -------------------------------
        for idx_i in range(len(pts_i)): 
            for idx_j in range(len(pts_j)): 
                A.append([idx_i+len(pc1), idx_j+len(pc2)])

        # Gather points
        for idx_i in range(len(pts_i)):
            pc1.append(pts_i[idx_i])
            kp_idxs_D1.append(k_idx)
            pt_idxs_D1.append([idx_i])
        for idx_j in range(len(pts_j)):
            pc2.append(pts_j[idx_j])
            kp_idxs_D2.append(k_idx)
            pt_idxs_D2.append([idx_j])

    pc1 = np.array(pc1)
    pc2 = np.array(pc2)

    if len(pc1) == 0 or len(pc2) == 0:
        continue

    A = np.array(A)
    kp_idxs_D1 = np.array(kp_idxs_D1)
    pt_idxs_D1 = np.array(pt_idxs_D1)
    kp_idxs_D2 = np.array(kp_idxs_D2)
    pt_idxs_D2 = np.array(pt_idxs_D2)

    pts_D1 = np.ascontiguousarray(
        np.vstack(pc1),   # stack FIRST, then transpose
        dtype=np.float64
    )

    pts_D2 = np.ascontiguousarray(
        np.vstack(pc2),
        dtype=np.float64
    )


    # Distance + descriptor
    if DIST_DESC_FLAG:
        # Descriptors of keypoints. For now assume one to one matches.
        descs_D1 = []
        descs_D2 = []
        scores_D = []
        for k_idx in pc_clustered_filtered["cam2"].kidxs:
            descs_D1.append(descriptors["cam2"][k_idx])
            descs_D2.append(descriptors["cam3"][k_idx])
            scores_D.append(m_scores[k_idx])
        descs_D1 = np.array(descs_D1, dtype=np.float64)
        descs_D2 = np.array(descs_D2, dtype=np.float64)
        scores_D = np.array(scores_D, dtype=np.float64)

        D1 = np.array([cluster_info_to_clipper_list(pt, descriptor=desc) for pt, desc in zip(pts_D1, descs_D1)]).T
        D2 = np.array([cluster_info_to_clipper_list(pt, descriptor=desc) for pt, desc in zip(pts_D2, descs_D2)]).T

    # Distance only
    else:
        D1 = pts_D1.T   # 3 x N
        D2 = pts_D2.T   # 3 x N

    A = np.ascontiguousarray(
        np.array(A, dtype=np.int32)
    )


    print("\nNum keypoints clipper", len(pc_clustered_filtered["cam2"].pts_per_kp), D1.shape)

    # ======================
    # Run CLIPPER
    # ======================

    # Make consistency matrix
    clipper.score_pairwise_and_single_consistency(D1, D2, A)
    tbuild1 = time.time()

    # # Manually fuse score and set matrix
    # old_M = clipper.get_affinity_matrix()
    # # new_M = fuse_matrix_geometric(old_M, scores_D)
    # new_M = fuse_matrix_weighted(old_M, scores_D, beta=0.5)
    # new_C = build_C_from_M(new_M)
    # clipper.set_matrix_data(new_M, new_C)
    # Use replicator dynamics

    # Solve clipper
    tsolve0 = time.time()
    clipper.solve()
    Ain = clipper.get_selected_associations()
    tsolve1 = time.time()

    print(f"CLIPPER selected {Ain.shape[0]} inliers in {tsolve1-tsolve0:.6f} s")

    # --------------------------------------------------
    # Select inlier points
    # --------------------------------------------------
    inliers_start = time.time()
    # Ain: (K, 2) int32 indices
    idxs1 = Ain[:, 0]
    idxs2 = Ain[:, 1]

    inlier_kis_cam2 = []
    inlier_kis_cam3 = []
    inlier_pts_cam2 = []
    inlier_pts_cam3 = []
    inlier_pxs_cam2 = []
    inlier_pxs_cam3 = []
    inlier_m_kpxs_cam2 = []
    inlier_m_kpxs_cam3 = []
    for idx1, idx2 in zip(idxs1, idxs2):
        inlier_kis_cam2.append(kp_idxs_D1[idx1])
        inlier_kis_cam3.append(kp_idxs_D2[idx2])
        inlier_pts_cam2.append(pts_D1[idx1])
        inlier_pts_cam3.append(pts_D2[idx2])
        inlier_pxs_cam2.append(pc_clustered_filtered["cam2"].pxs[idx1])
        inlier_pxs_cam3.append(pc_clustered_filtered["cam3"].pxs[idx2])
        inlier_m_kpxs_cam2.append(kp_idxs_D1[idx1])
        inlier_m_kpxs_cam3.append(kp_idxs_D2[idx2])

    inlier_m_kpxs_cam2 = np.array(inlier_m_kpxs_cam2)
    inlier_m_kpxs_cam3 = np.array(inlier_m_kpxs_cam3)
    inlier_m_kpxs = {"cam2":inlier_m_kpxs_cam2, "cam3":inlier_m_kpxs_cam3}

    pc_clustered_clipper = {}
    pc_clustered_clipper["cam2"] = PointCloudClustered(
                                                    kvalid=pc_clustered["cam3"].kvalid,                # TODO: delete later
                                                    pts_per_kp=pc_clustered["cam3"].pts_per_kp,        # TODO: delete later
                                                    pxs_per_kp=pc_clustered["cam3"].pxs_per_kp,        # TODO: delete later
                                                    kidxs=np.array(inlier_kis_cam3),
                                                    pts=np.array(inlier_pts_cam2),
                                                    pxs=np.array(inlier_pxs_cam2)
                                                )
    pc_clustered_clipper["cam3"] = PointCloudClustered(
                                                    kvalid=pc_clustered["cam3"].kvalid,                # TODO: delete later
                                                    pts_per_kp=pc_clustered["cam3"].pts_per_kp,        # TODO: delete later
                                                    pxs_per_kp=pc_clustered["cam3"].pxs_per_kp,        # TODO: delete later
                                                    kidxs=np.array(inlier_kis_cam3),
                                                    pts=np.array(inlier_pts_cam3),
                                                    pxs=np.array(inlier_pxs_cam3)
                                                )
    # ======================
    # Get relative pose estimate
    # ======================

    if len(inlier_pts_cam2) > MIN_NUM_INLIERS:


        time_build_clipper.append(tbuild1 - tbuild0)
        time_solve_clipper.append(tsolve1 - tsolve0)

        inlier_pts_cam2 = np.asarray(inlier_pts_cam2)
        inlier_pts_cam3 = np.asarray(inlier_pts_cam3)
        inlier_pxs_cam3 = np.asarray(inlier_pxs_cam3)

        # ============================================================
        # Arun: relative pose (retr → quer)
        # ============================================================
        arun_solve_start = time.time()

        R_rq_arun, t_rq_arun = arun(
            inlier_pts_cam2.T,
            inlier_pts_cam3.T
        )

        H_rq_arun = build_homogeneous(R_rq_arun, t_rq_arun)
        time_solve_arun.append(time.time() - arun_solve_start)

        if ARUN_ICP:
            H_rq_arun, fitness_icp, rmse_icp = run_icp(cameras[0].pts_all, cameras[1].pts_all, H_rq_arun)

        # Query pose prediction
        H_quer_global_pred_arun = H_imur_global_gt @ invert_H(H_rq_arun)
        time_clipper_arun_tot.append(time.time() - arun_start)



        # # ============================================================
        # # PnP: relative pose (retr → quer)
        # # ============================================================
        # ret, rvec, tvec = cv2.solvePnP(
        #     inlier_pts_cam2,
        #     inlier_pxs_cam3,
        #     cam_quer.K,
        #     cam_quer.dist,
        #     flags=cv2.SOLVEPNP_ITERATIVE
        # )

        # R_rq_pnp, _ = cv2.Rodrigues(rvec)
        # H_rq_pnp = build_homogeneous(R_rq_pnp, tvec)

        # # Query pose prediction
        # # H_quer_global_pred_pnp = invert_H(H_rq_pnp) @ H_imur_global_gt
        # H_quer_global_pred_pnp = H_imur_global_gt @ invert_H(H_rq_pnp)

        # t_pnp  = H_quer_global_pred_pnp[:3, 3]
        # trans_err_pnp  = t_true - t_pnp 
        # R_pnp  = H_quer_global_pred_pnp[:3, :3]
        # rot_err_pnp  = rotation_components_deg(R_pnp,  R_true)
        # pred_cam3_gps_xy_pnp.append(t_pnp[:2])
        # errors_pnp.append({
        #     "trans": trans_err_pnp,
        #     "rot":   rot_err_pnp,
        # })

        # ============================================================
        # Translation error (component-wise)
        # ============================================================
        t_true = H_imuq_global_gt[:3, 3]
        t_arun = H_quer_global_pred_arun[:3, 3]

        # trans_err_arun = t_true - t_arun

        # print("\nArun translational error (m):", trans_err_arun)

        # # ============================================================
        # # Rotation error (component-wise, degrees)
        # # ============================================================
        # R_true = H_imuq_global_gt[:3, :3]
        # R_arun = H_quer_global_pred_arun[:3, :3]

        # rot_err_arun = rotation_components_deg(R_arun, R_true)


        # # ============================================================
        # # Save for plotting / evaluation
        # # ============================================================
        # pred_cam3_gps_xy_aru.append(t_arun[:2])

        # errors_aru.append({
        #     "trans": trans_err_arun,   # (dx, dy, dz)
        #     "rot":   rot_err_arun,     # (roll, pitch, yaw) in deg
        # })



    else:
        print("\nNot enough inliers after CLIPPER")

    time_inliers.append(time.time() - inliers_start)

    # ------------------------
    # Visualize
    # ------------------------

    if PLOT_FLAG:
        # plt.show()
        # if True:
        if len(inlier_pts_cam2) > MIN_NUM_INLIERS:
        # # if len(inlier_pts_cam2) > MIN_NUM_INLIERS and abs(trans_err_arun[0]) > 50:
        # # if len(inlier_pts_cam2) > MIN_NUM_INLIERS and abs(trans_err_arun[0]) < 3  and abs(trans_err_arun[1]) < 3:           
            

            
            # H_imu_lidar = np.array([[0, -1, 0, 0],
            #                          [1, 0, 0, 0],
            #                          [0, 0, 1, -0.05013],
            #                          [0, 0, 0, 1]])

            # Compute ground truth transformation retrieved -> query
            # H_lidar_cam2 = cameras[0].T_lidar_cam
            # H_lidar_cam3 = cameras[1].T_lidar_cam
            # H_rq_gt = H_lidar_cam3 @ H_imu_lidar @ np.linalg.inv(H_imuq_global_gt) @ \
            #             H_imur_global_gt @ np.linalg.inv(H_imu_lidar) @ np.linalg.inv(H_lidar_cam2)

            # H_rq_gt = H_lidar_cam2 @ H_imu_lidar @ np.linalg.inv(H_imur_global_gt) @ \
            #             H_imur_global_gt @ np.linalg.inv(H_imu_lidar) @ np.linalg.inv(H_lidar_cam2)
            # print("\nArun translational error (m):", trans_err_arun)
            # input("Press Enter to continue...")

            lg.visualize_matches(cameras[1].image_path, cameras[0].image_path)
            # viz.show_lidar_2d(cameras) # all lidar points
            # viz.show_lidar_neighbors_2d(cameras, keypoints, nn)
            # viz.show_points_2d(cameras, keypoints, pc_clustered_filtered, title=" Matched Centroids 2D")
            # viz.show_points_2d(cameras, keypoints, pc_clustered_clipper, kp_size=35, title=" Centroids CLIPPER 2D")
            # viz.show_points_3d(cameras, pc_clustered_filtered, centroid_size=25, title=" Matched Centroids 3D")
            # viz.show_points_3d(cameras, pc_clustered_clipper, centroid_size=25, title=" Centroids CLIPPER 3D")
            # viz.show_points_3d_same_plot(cameras, pc_clustered_clipper, centroid_size=25, title=" Centroids After CLIPPER")
            # viz.show_points_3d_same_plot_query(cameras, pc_clustered_clipper, H_rq_icp, centroid_size=25, title=" Predicted Alignment in Query Cam Frame (ICP)")
            # viz.show_points_3d_same_plot_query(cameras, pc_clustered_clipper, H_rq_arun, centroid_size=25, title=" Predicted Alignment in Query Cam Frame (Mine)")
        
            # viz.show_points_3d_same_plot_query(cameras, pc_clustered_clipper, H_rq_gt, centroid_size=25, title=" Ground Truth Alignment in Query Cam Frame")
            # viz.show_points_3d_same_plot_global(cameras, pc_clustered_clipper, [H_imur_global_gt, H_imuq_global_gt], H_imu_lidar, centroid_size=25, title=" Ground Truth Alignment in World Frame")

            plt.show()

print("\nNum poses Arun", len(pred_cam3_gps_xy_aru))
print("Time Collect Neighbors", np.mean(time_collect_neighbors), np.std(time_collect_neighbors), "Max", np.max(time_collect_neighbors))
print("Time Make Clusters", np.mean(time_build_clusters), np.std(time_build_clusters), "Max", np.max(time_build_clusters))
print("Time CLIPPER Build Graph", np.mean(time_build_clipper), np.std(time_build_clipper), "Max", np.max(time_build_clipper))
print("Time CLIPPER Solve Graph", np.mean(time_solve_clipper), np.std(time_solve_clipper), "Max", np.max(time_solve_clipper))
print("Time Get Inliers", np.mean(time_inliers), np.std(time_inliers), "Max", np.max(time_inliers))
print("Time Arun", np.mean(time_solve_arun), np.std(time_solve_arun), "Max", np.max(time_solve_arun))
print("Time CLIPPER + Arun Total", np.mean(time_clipper_arun_tot), np.std(time_clipper_arun_tot), "Max", np.max(time_clipper_arun_tot))
print("Time ICP", np.mean(time_icp), np.std(time_icp), "Max", np.max(time_icp))
if len(pred_cam3_gps_xy_aru) > 0:
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
        pred_cam3_gps_xy_icp
    )
    plt.show()
else:
    print("No valid poses")