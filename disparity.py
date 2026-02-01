import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
left_img_path = (
    "data/makalii_point/processed_lidar_cam_gps/cam2/"
    "bag_camera_2_2025_08_13-01_35_58_5/camera/left_cam/2.png"
)

right_img_path = (
    "data/makalii_point/processed_lidar_cam_gps/cam2/"
    "bag_camera_2_2025_08_13-01_35_58_5/camera/right_cam/2.png"
)

calib_path = "data/calib_leftcam.json"

# ------------------------------------------------------------
# USER MUST SET THIS
# ------------------------------------------------------------
BASELINE_METERS = 0.15  # <-- CHANGE THIS to your true stereo baseline

# ------------------------------------------------------------
# Load calibration
# ------------------------------------------------------------
with open(calib_path, "r") as f:
    calib = json.load(f)

K = np.array(calib["cam_K"]["data"], dtype=np.float32)

fx = K[0, 0]
fy = K[1, 1]
cx = K[0, 2]
cy = K[1, 2]

# ------------------------------------------------------------
# Load images
# ------------------------------------------------------------
left_img = cv2.imread(left_img_path, cv2.IMREAD_COLOR)
right_img = cv2.imread(right_img_path, cv2.IMREAD_COLOR)

if left_img is None or right_img is None:
    raise RuntimeError("Failed to load stereo images")

left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

# Rectify 
# ------------------------------------------------------------
# StereoSGBM configuration
# ------------------------------------------------------------
num_disparities = 128  # must be divisible by 16
block_size = 5

stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,
    blockSize=7,
    P1=24 * 7 * 7,
    P2=96 * 7 * 7,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=50,
    speckleRange=2,
    preFilterCap=31,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
)

# ------------------------------------------------------------
# Compute disparity
# ------------------------------------------------------------
disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

# Mask invalid disparity
disp_valid = disparity > 1.0

# ------------------------------------------------------------
# Depth from disparity
# Z = f * b / d
# ------------------------------------------------------------
depth = np.zeros_like(disparity, dtype=np.float32)
depth[disp_valid] = fx * BASELINE_METERS / disparity[disp_valid]

# Optional: clip depth for visualization
MAX_DEPTH = 300.0  # meters
depth_vis = depth.copy()
# depth_vis[depth_vis > MAX_DEPTH] = MAX_DEPTH
# depth_vis[~disp_valid] = np.nan

# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------
plt.figure(figsize=(16, 10))

# ------------------------------------------------------------
# Left image
# ------------------------------------------------------------
plt.subplot(2, 2, 1)
plt.title("Left image")
plt.imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
plt.axis("off")

# ------------------------------------------------------------
# Right image
# ------------------------------------------------------------
plt.subplot(2, 2, 2)
plt.title("Right image")
plt.imshow(cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB))
plt.axis("off")

# ------------------------------------------------------------
# Disparity
# ------------------------------------------------------------
plt.subplot(2, 2, 3)
plt.title("Disparity (pixels)")
disp_plot = disparity.copy()
disp_plot[~disp_valid] = np.nan
plt.imshow(disp_plot, cmap="inferno")
plt.colorbar(label="pixels", fraction=0.046, pad=0.04)
plt.axis("off")

# ------------------------------------------------------------
# Depth
# ------------------------------------------------------------
plt.subplot(2, 2, 4)
plt.title("Depth (meters)")
plt.imshow(depth_vis, cmap="viridis")
plt.colorbar(label="meters", fraction=0.046, pad=0.04)
plt.axis("off")

plt.tight_layout()
plt.show()

