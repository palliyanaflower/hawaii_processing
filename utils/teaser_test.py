import copy
import time
import numpy as np
import open3d as o3d
import teaserpp_python

import json
from CameraLidar import CameraView
from pathlib import Path


NOISE_BOUND = 0.05
N_OUTLIERS = 1700
OUTLIER_TRANSLATION_LB = 5
OUTLIER_TRANSLATION_UB = 10

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

def get_angular_error(R_exp, R_est):
    """
    Calculate angular error
    """
    return abs(np.arccos(min(max(((np.matmul(R_exp.T, R_est)).trace() - 1) / 2, -1.0), 1.0)));


if __name__ == "__main__":
    START_IDX = 0

    with open("matched_img_paths.json", "r") as f:
        data = json.load(f)

    # ============================================================
    # Begin Processing
    # ============================================================

    path_true_all = data["true"]
    path_quer_all = data["queries"]
    path_retr_all = data["retrieved"]

    pred_cam3_gps_xy_aru = []    # Retr cam3 (retr) predicted positions
    pred_cam3_gps_xy_pnp = []    # Retr cam3 (retr) predicted positions

    errors_aru = []
    errors_pnp = []
    errors_ess = []

    path_idx = 0

    cam_path_q = path_quer_all[path_idx]
    cam_path_r = path_retr_all[path_idx][0]  # top K=1  

    lidar_path_q = image_to_lidar_path(Path(cam_path_q))
    lidar_path_r = image_to_lidar_path(Path(cam_path_r))

    print("==================================================")
    print("        TEASER++ Python registration example      ")
    print("==================================================")

    # Load bunny ply file
    src_cloud = o3d.io.read_point_cloud(lidar_path_q)
    src = np.transpose(np.asarray(src_cloud.points))
    N = src.shape[1]

    dst_cloud = o3d.io.read_point_cloud(lidar_path_r)
    dst = np.transpose(np.asarray(dst_cloud.points))

    # Populating the parameters
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = NOISE_BOUND
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12

    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    start = time.time()
    solver.solve(src, dst)
    end = time.time()

    solution = solver.getSolution()

    print("=====================================")
    print("          TEASER++ Results           ")
    print("=====================================")

    print("Expected rotation: ")
    print(T[:3, :3])
    print("Estimated rotation: ")
    print(solution.rotation)
    print("Error (rad): ")
    # print(get_angular_error(T[:3,:3], solution.rotation))

    print("Expected translation: ")
    # print(T[:3, 3])
    print("Estimated translation: ")
    print(solution.translation)
    print("Error (m): ")
    # print(np.linalg.norm(T[:3, 3] - solution.translation))

    print("Number of correspondences: ", N)
    print("Number of outliers: ", N_OUTLIERS)
    print("Time taken (s): ", end - start)

