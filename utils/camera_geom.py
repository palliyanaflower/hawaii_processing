# Project lidar into camera
import numpy as np
import cv2

# ============================================================
# Utility Functions
# ============================================================
# Z axis in and out of camera
#     Y (down)
#     ↓
#     |
#     |
#     o----→ X (right)
#    /
#   /
#  Z (forward, into scene)

def euler_to_matrix(rx, ry, rz):
    """Convert roll-pitch-yaw (radians) to rotation matrix."""
    cx, cy, cz = np.cos([rx, ry, rz])
    sx, sy, sz = np.sin([rx, ry, rz])
    R_x = np.array([[1, 0, 0],
                    [0, cx, -sx],
                    [0, sx, cx]])
    R_y = np.array([[cy, 0, sy],
                    [0, 1, 0],
                    [-sy, 0, cy]])
    R_z = np.array([[cz, -sz, 0],
                    [sz, cz, 0],
                    [0, 0, 1]])
    return R_z @ R_y @ R_x


def build_transform(tx, ty, tz, rx, ry, rz):
    rx, ry, rz = np.radians([rx, ry, rz])
    T = np.eye(4)
    T[:3, :3] = euler_to_matrix(rx, ry, rz)
    T[:3, 3] = [tx, ty, tz]
    return T


def decompose_transform(T):
    """Extract tx,ty,tz (m) and roll,pitch,yaw (deg) from 4x4 matrix (Rz@Ry@Rx convention)."""
    tx, ty, tz = T[:3, 3]
    R = T[:3, :3]
    if abs(R[2, 0]) < 1.0:
        ry = -np.arcsin(R[2, 0])
        rx = np.arctan2(R[2, 1] / np.cos(ry), R[2, 2] / np.cos(ry))
        rz = np.arctan2(R[1, 0] / np.cos(ry), R[0, 0] / np.cos(ry))
    else:
        ry = np.pi / 2 if R[2, 0] <= -1 else -np.pi / 2
        rx = np.arctan2(-R[0, 1], -R[0, 2])
        rz = 0.0
    rx, ry, rz = np.degrees([rx, ry, rz])
    return tx, ty, tz, rx, ry, rz

def filter_pts_to_image_plane(points, colors, T, K, w, h, img):
    N = points.shape[0]

    # Homogeneous LiDAR points
    pts_h = np.hstack([points, np.ones((N, 1))])

    # Transform LiDAR → camera
    pts_h = np.hstack([points, np.ones((points.shape[0], 1))])  # homogeneous
    pts_transformed = (T @ pts_h.T).T[:, :3]

    # --- Step 1: Keep only points in front of the camera ---
    mask_infront = pts_transformed[:, 2] > 0
    pts_cam = pts_transformed[mask_infront]
    pts_infront = points[mask_infront]
    colors = colors[mask_infront]

    # --- Step 2: Project points ---
    x = pts_cam[:, 0] / pts_cam[:, 2]
    y = pts_cam[:, 1] / pts_cam[:, 2]

    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]

    # --- Step 3: Keep only points inside the image bounds ---
    mask_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)

    # Apply mask to everything so correspondence is maintained
    pts_cam = pts_cam[mask_img]
    pts_infront = pts_infront[mask_img]
    colors = colors[mask_img]
    u = u[mask_img].astype(int)
    v = v[mask_img].astype(int)


    vis_img = img.copy()
    overlay = vis_img.copy()

    alpha = 0.7     # opacity (0.2–0.4 works well)
    radius = 1      # smaller dots
    thickness = -1  # filled

    for ui, vi, c in zip(u, v, colors):
        cv2.circle(
            overlay,
            (ui, vi),
            radius,
            (int(255 * c[2]), int(255 * c[1]), int(255 * c[0])),
            thickness
        )

    # Blend once (important for performance)
    vis_img = cv2.addWeighted(
        overlay, alpha,
        vis_img, 1 - alpha,
        0
    )

    return u, v, pts_infront, pts_cam, vis_img

