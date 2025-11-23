import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

# --- Paths ---
img_dir = "data/haleiwa_neighborhood/processed_data/camera"
pc_dir = "data/haleiwa_neighborhood/processed_data/lidar/pcd_merged"

# --- Camera intrinsics ---
K = np.array([
    [453.9426, 0, 388.0409],
    [0, 453.8471, 247.8123],
    [0, 0, 1]
])

# --- Distortion coefficients ---
D = np.array([
    4.479070663452148, 0.39237383008003235, -4.801498289452866e-05,
    0.0002160959702450782, 2.7756459712982178, 4.403016090393066,
    0.37853071093559265, 2.9023659229278564,
    0.0, 0.0, 0.0, 0.0,
    -0.003902334487065673, 0.0001621022674953565
])

# --- Extrinsics (lidar → cam) ---
T_lidar_to_cam_init = np.array([
    [0.0, -1.0, 0.0, -0.2],
    [-0.174, 0.0, -0.985, -0.08],
    [0.985, 0.0, -0.174, -0.06],
    [0.0, 0.0, 0.0, 1.0]
])

# --- Path to the .txt file ---
txt_path = "/home/kalliyanlay/Documents/BYU/research/camera_lidar_calibration/livox_camera_calib/data/chad_data/extrinsic.txt"

# --- Load matrix ---
T_lidar_to_cam_refine = np.loadtxt(txt_path, delimiter=",")
if T_lidar_to_cam_refine.shape != (4, 4):
    raise ValueError(f"Expected a 4x4 matrix, but got shape {T_lidar_to_cam_refine.shape}")

print("Loaded T_lidar_to_cam_refine:")
print(T_lidar_to_cam_refine) 
# T_lidar_to_cam_refine = np.array([
#     [-0.0222968,-0.999577,-0.0186846,-0.154742],
#     [-0.104604,0.0209192,-0.994294,-0.212315],
#     [0.994264,-0.020215,-0.105027,-0.0215597],
#     [0,0,0,1]
# ])
# --- Function to project LiDAR to image plane ---
def project_lidar_to_image(points, T_lidar_to_cam, K, img):
    h, w, _ = img.shape
    points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
    points_cam = (T_lidar_to_cam @ points_hom.T).T
    points_cam = points_cam[points_cam[:, 2] > 0]  # keep points in front of camera

    proj = (K @ points_cam[:, :3].T).T
    proj[:, 0] /= proj[:, 2]
    proj[:, 1] /= proj[:, 2]
    u, v = proj[:, 0].astype(int), proj[:, 1].astype(int)

    mask_valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, points_cam = u[mask_valid], v[mask_valid], points_cam[mask_valid]

    distances = np.linalg.norm(points_cam[:, :3], axis=1)
    dist_norm = (distances - distances.min()) / (distances.max() - distances.min())
    cmap = cm.get_cmap("viridis")
    colors = (cmap(dist_norm)[:, :3] * 255).astype(np.uint8)

    overlay = img.copy()
    for (x, y, color) in zip(u, v, colors):
        cv2.circle(overlay, (x, y), 1, color.tolist(), -1)
    return overlay

# --- Process frames 0000 → 0004 ---
frame_ids = [f"{i:01d}" for i in range(1)]
rows = len(frame_ids)

fig, axes = plt.subplots(rows, 2, figsize=(8, 4 * rows))
if rows == 1:
    axes = np.expand_dims(axes, axis=0)  # handle single-row case

for i, frame_id in enumerate(frame_ids):
    img_path = os.path.join(img_dir, f"{frame_id}.png")
    pc_path = os.path.join(pc_dir, f"{frame_id}.pcd")
    # img_path = "data/haleiwa_neighborhood/processed_data/camera/0.png"
    # pc_path = "data/haleiwa_neighborhood/processed_data/lidar/pcd_merged/static_scene.pcd"
    img_path = "data/cb/processed_data/images/0000.png"
    pc_path = "data/cb/processed_data/lidar/pcd/0000.pcd"
    # img_path = "/home/kalliyanlay/Documents/BYU/research/camera_lidar_calibration/data/multicam_lidar_calib_data/images_forwardcam/0003.png"
    # pc_path = "/home/kalliyanlay/Documents/BYU/research/camera_lidar_calibration/data/multicam_lidar_calib_data/lidar/pcd/0003.pcd"

    if not os.path.exists(img_path) or not os.path.exists(pc_path):
        print(f"Skipping {frame_id}: missing image or point cloud")
        continue

    # Load image + point cloud
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pcd = o3d.io.read_point_cloud(pc_path)
    points = np.asarray(pcd.points)

    # Undistort image
    h, w = img.shape[:2]

    # --- Undistort ---
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0)
    img_undist = cv2.undistort(img, K, D, None, new_K)

    # Project both extrinsics
    overlay_init = project_lidar_to_image(points, T_lidar_to_cam_init, new_K, img_undist)
    overlay_refine = project_lidar_to_image(points, T_lidar_to_cam_refine, new_K, img_undist)

    # Plot pair for this frame
    axes[i, 0].imshow(overlay_init)
    axes[i, 0].set_title(f"{frame_id} - Init Extrinsics")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(overlay_refine)
    axes[i, 1].set_title(f"{frame_id} - Refined Extrinsics")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.show()
