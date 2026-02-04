# Project lidar into camera
import numpy as np
import cv2
from KeypointNN import LidarKeypointNeighbors

# ============================================================
# Utility Functions
# ============================================================
# Z axis in and out of camera
#     Y (down)
#     ↓
#     |
#     |
#     x----→ X (right)
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

def project_point_to_image_plane(point, T, K):
    """
    point : (3,) LiDAR point [X, Y, Z]
    T     : (4,4) LiDAR -> camera transform
    K     : (3,3) camera intrinsics
    w, h  : image width/height (optional, for bounds checking)
    """

    # 1. Homogeneous point
    p_h = np.array([point[0], point[1], point[2], 1.0])

    # 2. Transform to camera frame
    p_cam = T @ p_h
    X, Y, Z = p_cam[:3]

    # Behind camera or too close
    if Z <= 0:
        return None

    # 3. Perspective division
    x = X / Z
    y = Y / Z

    # 4. Apply intrinsics
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]

    return np.array([u, v])

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

    # Blend lidar points with image
    vis_img = img.copy()
    overlay = vis_img.copy()

    alpha = 0.7     # opacity (0.2–0.4 works well)
    radius = 1      # smaller dots
    thickness = -1  # filled
    colors_bgr = colors[:, ::-1]  # RGB → BGR for opencv

    for ui, vi, c in zip(u, v, colors_bgr):

        cv2.circle(
            overlay,
            (ui, vi),
            radius,
            (int(255 * c[2]), int(255 * c[1]), int(255 * c[0])),
            thickness
        )

    vis_img = cv2.addWeighted(
        overlay, alpha,
        vis_img, 1 - alpha,
        0
    )

    return u, v, pts_cam, vis_img, mask_infront, mask_img

def collect_lidar_neighbors_per_keypoint(
    m_kpts,
    u,
    v,
    pts_cam,
    pts_infront,
    pixel_radius=5,
    remove_zero_points=True,
) -> LidarKeypointNeighbors:
    """
    Collect LiDAR neighbors around each image keypoint in pixel space.

    Parameters
    ----------
    m_kpts : (K, 2) array-like
        Keypoints in pixel coordinates (x, y), can be float.
    u, v : (N,) ndarray
        Projected LiDAR pixel coordinates.
    pts_cam : (N, 3) ndarray
        LiDAR points in camera frame (x, y, z).
    pts_infront : (N, 3) ndarray
        LiDAR points that are in front of the camera (camera frame).
    pixel_radius : int, optional
        Pixel window radius for neighbor search.
    remove_zero_points : bool, optional
        Whether to remove all-zero LiDAR points.

    Returns
    -------
    keypoint_lidar_depths : list of (Mi,) ndarrays
        Depth values (z) per keypoint.
    lidar_nn_per_kp_px : list of (Mi, 2) ndarrays
        Pixel coordinates of neighbors per keypoint.
    lidar_nn_per_kp_pt : list of (Mi, 3) ndarrays
        3D LiDAR neighbors per keypoint (camera frame).
    nn_pts_all : (M, 3) ndarray
        All neighbor points concatenated.
    nn_pxs_all : (M, 2) ndarray
        All neighbor pixel coordinates concatenated.
    group_ids : (M,) ndarray
        Index of keypoint each neighbor belongs to.
    """

    keypoint_lidar_depths = []
    lidar_nn_per_kp_px = []
    lidar_nn_per_kp_pt = []

    # ------------------------------------------------------------
    # Per-keypoint neighbor collection
    # ------------------------------------------------------------
    for kp in m_kpts:
        x, y = kp[0], kp[1]  # keep float precision

        # Pixel window for nn
        mask = (np.abs(u - x) <= pixel_radius) & \
               (np.abs(v - y) <= pixel_radius)

        # Get any neighbors that fall within pixel window
        if np.any(mask):
            depths = pts_cam[mask, 2]
            lidar_nn_per_kp_px.append(np.stack([u[mask], v[mask]], axis=1))
            lidar_nn_per_kp_pt.append(pts_infront[mask, :])
        else:
            depths = np.array([])
            lidar_nn_per_kp_px.append(np.empty((0, 2)))
            lidar_nn_per_kp_pt.append(np.empty((0, 3)))

        keypoint_lidar_depths.append(depths)

    # ------------------------------------------------------------
    # Flatten neighbors across all keypoints
    # ------------------------------------------------------------
    nn_pts_list = []
    nn_pxs_list = []
    group_ids = []

    for i, (nn_pts, nn_pxs) in enumerate(
        zip(lidar_nn_per_kp_pt, lidar_nn_per_kp_px)
    ):
        if nn_pts is None or len(nn_pts) == 0:
            continue

        nn_pts = np.asarray(nn_pts)
        nn_pxs = np.asarray(nn_pxs)

        if remove_zero_points:
            mask = ~np.all(np.isclose(nn_pts, 0.0), axis=1)
            nn_pts = nn_pts[mask]
            nn_pxs = nn_pxs[mask]

        if nn_pts.shape[0] > 0:
            nn_pts_list.append(nn_pts)
            nn_pxs_list.append(nn_pxs)
            group_ids.append(np.full(nn_pts.shape[0], i))

    if len(nn_pts_list) > 0:
        nn_pts_all = np.concatenate(nn_pts_list, axis=0)
        nn_pxs_all = np.concatenate(nn_pxs_list, axis=0)
        group_ids = np.concatenate(group_ids, axis=0)
    else:
        nn_pts_all = np.empty((0, 3))
        nn_pxs_all = np.empty((0, 2))
        group_ids = np.empty((0,), dtype=int)
        
        
    return LidarKeypointNeighbors(
        depths_per_kp=keypoint_lidar_depths,
        pxs_per_kp=lidar_nn_per_kp_px,
        pts_per_kp=lidar_nn_per_kp_pt,
        pts_all=nn_pts_all,
        pxs_all=nn_pxs_all,
        group_ids=group_ids,

        num_kps=len(lidar_nn_per_kp_pt),
        num_pts_all=len(nn_pts_all)
    )
