from dataclasses import dataclass
import numpy as np
from typing import List
from scipy.spatial import cKDTree

@dataclass
class ClusterInfoAll:

    keypoints: np.ndarray               # (N,2)     All keypoint pixel coordinates
    descriptors: np.ndarray             # (N, 256)  All keypoint descriptors (256 from LightGlue)
    
    depths_per_kp: List[np.ndarray]     # [(Mi,), ...]
    pxs_per_kp: List[np.ndarray]        # [(Mi,2), ...] Track lidar neighbors for each keypoint
    pts_per_kp: List[np.ndarray]        # [(Mi,3), ...] Track lidar neighbors for each keypoint

    pts_all: np.ndarray                 # (M,3)
    pxs_all: np.ndarray                 # (M,2)
    group_ids: np.ndarray               # (M,)

    num_kps: int                        # N
    num_pts_all: int

def collect_lidar_neighbors_per_keypoint(
    m_kpts,
    m_descs,
    u,
    v,
    pts_cam,
    pts_infront,
    pixel_radius=5,
    remove_zero_points=True,
) -> ClusterInfoAll:
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
        
        
    return ClusterInfoAll(
        keypoints=m_kpts,
        descriptors=m_descs,
        depths_per_kp=keypoint_lidar_depths,
        pxs_per_kp=lidar_nn_per_kp_px,
        pts_per_kp=lidar_nn_per_kp_pt,
        pts_all=nn_pts_all,
        pxs_all=nn_pxs_all,
        group_ids=group_ids,

        num_kps=len(lidar_nn_per_kp_pt),
        num_pts_all=len(nn_pts_all)
    )


def collect_lidar_neighbors_per_keypoint_knn_xybox(
    m_kpts,
    m_descs,
    u,
    v,
    pts_cam,
    pts_infront,
    xy_radius=3.0,
    min_pixel_dist=3.0,
    remove_zero_points=True,
) -> ClusterInfoAll:
    """
    For each keypoint:
        1. Find nearest projected LiDAR point using KDTree.
        2. Reject if pixel distance > min_pixel_dist.
        3. Collect LiDAR neighbors within XY box (±xy_radius).
    """

    keypoint_lidar_depths = []
    lidar_nn_per_kp_px = []
    lidar_nn_per_kp_pt = []

    # --------------------------------------------
    # Build KDTree in pixel space
    # --------------------------------------------
    lidar_pixels = np.stack([u, v], axis=1)  # (N, 2)
    if lidar_pixels.shape[0] == 0:
        return ClusterInfoAll.empty()  # or handle gracefully

    tree = cKDTree(lidar_pixels)

    # --------------------------------------------
    # Per-keypoint processing
    # --------------------------------------------
    for kp in m_kpts:
        x_kp, y_kp = kp[0], kp[1]

        # Query nearest projected lidar point
        dist, nearest_idx = tree.query([x_kp, y_kp], k=1)

        # Reject if too far in pixel space
        if dist > min_pixel_dist:
            keypoint_lidar_depths.append(np.array([]))
            lidar_nn_per_kp_px.append(np.empty((0, 2)))
            lidar_nn_per_kp_pt.append(np.empty((0, 3)))
            continue

        center_pt = pts_infront[nearest_idx]
        cx, cy = center_pt[0], center_pt[1]

        # --------------------------------------------
        # 3D XY box neighborhood
        # --------------------------------------------
        # mask = (
        #     (np.abs(pts_infront[:, 0] - cx) <= xy_radius) &
        #     (np.abs(pts_infront[:, 1] - cy) <= xy_radius)
        # )

        # if np.any(mask):
        #     nn_pts = pts_infront[mask]
        #     nn_pxs = lidar_pixels[mask]
        #     depths = nn_pts[:, 2]
        # else:
        #     nn_pts = np.empty((0, 3))
        #     nn_pxs = np.empty((0, 2))
        #     depths = np.array([])

        tree3d = cKDTree(pts_infront)
        idxs = tree3d.query_ball_point(center_pt, r=xy_radius)

        nn_pts = pts_infront[idxs]
        nn_pxs = lidar_pixels[idxs]
        depths = nn_pts[:, 2]


        keypoint_lidar_depths.append(depths)
        lidar_nn_per_kp_pt.append(nn_pts)
        lidar_nn_per_kp_px.append(nn_pxs)

    # --------------------------------------------
    # Flatten all neighbors
    # --------------------------------------------
    nn_pts_list = []
    nn_pxs_list = []
    group_ids = []

    for i, (nn_pts, nn_pxs) in enumerate(
        zip(lidar_nn_per_kp_pt, lidar_nn_per_kp_px)
    ):
        if len(nn_pts) == 0:
            continue

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

    return ClusterInfoAll(
        keypoints=m_kpts,
        descriptors=m_descs,
        depths_per_kp=keypoint_lidar_depths,
        pxs_per_kp=lidar_nn_per_kp_px,
        pts_per_kp=lidar_nn_per_kp_pt,
        pts_all=nn_pts_all,
        pxs_all=nn_pxs_all,
        group_ids=group_ids,
        num_kps=len(lidar_nn_per_kp_pt),
        num_pts_all=len(nn_pts_all),
    )