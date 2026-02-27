import numpy as np
import torch
import cv2
import json
from pathlib import Path
import matplotlib.pyplot as plt
import open3d as o3d

from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

from utils.lightglue_loader import LightGlueVisualizer
from utils.file_loader import lidar_pcd

def create_multichannel_lidar_image(
    points,
    intensity=None,
    img_h=128,
    img_w=2048,
    v_fov=(-22.5, 22.5),
    h_fov=(-180, 180),
    max_range=120.0,
):
    """
    Create multi-channel LiDAR image from point cloud.

    Returns:
        lidar_img : (H,W,3)
            [depth, intensity, normal_magnitude]
        xyz_img : (H,W,3)
            backprojection map
    """

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # --------------------------------------------------
    # spherical projection
    # --------------------------------------------------
    r = np.sqrt(x**2 + y**2 + z**2)

    azimuth = np.arctan2(y, x)
    elevation = np.arcsin(z / (r + 1e-8))

    h_min, h_max = np.deg2rad(h_fov)
    v_min, v_max = np.deg2rad(v_fov)

    u = (azimuth - h_min) / (h_max - h_min)
    v = (elevation - v_min) / (v_max - v_min)

    u = (u * img_w).astype(np.int32)
    v = ((1 - v) * img_h).astype(np.int32)

    valid = (
        (u >= 0) & (u < img_w) &
        (v >= 0) & (v < img_h) &
        (r > 0) & (r < max_range)
    )

    u, v, r = u[valid], v[valid], r[valid]
    pts = points[valid]

    if intensity is not None:
        intensity = intensity[valid]

    # --------------------------------------------------
    # initialize images
    # --------------------------------------------------
    depth_img = np.full((img_h, img_w), np.inf)
    xyz_img = np.zeros((img_h, img_w, 3))
    intensity_img = np.zeros((img_h, img_w))

    # --------------------------------------------------
    # z-buffer projection
    # --------------------------------------------------
    for i in range(len(r)):
        ui, vi = u[i], v[i]

        if r[i] < depth_img[vi, ui]:
            depth_img[vi, ui] = r[i]
            xyz_img[vi, ui] = pts[i]
            if intensity is not None:
                intensity_img[vi, ui] = intensity[i]

    depth_img[depth_img == np.inf] = 0

    # --------------------------------------------------
    # depth normalization (log)
    # --------------------------------------------------
    depth_norm = np.log1p(depth_img)
    depth_norm /= depth_norm.max() + 1e-8

    # --------------------------------------------------
    # intensity normalization
    # --------------------------------------------------
    if intensity is not None:
        p1, p99 = np.percentile(intensity_img, (1, 99))
        intensity_img = np.clip(intensity_img, p1, p99)
        intensity_norm = (intensity_img - p1) / (
            p99 - p1 + 1e-8
        )
    else:
        intensity_norm = np.zeros_like(depth_norm)

    # --------------------------------------------------
    # FAST RANGE-IMAGE NORMALS
    # --------------------------------------------------
    dx = np.roll(xyz_img, -1, axis=1) - xyz_img
    dy = np.roll(xyz_img, -1, axis=0) - xyz_img

    normals = np.cross(dx, dy)

    norm = np.linalg.norm(normals, axis=2, keepdims=True) + 1e-8
    normals /= norm

    normal_mag = np.linalg.norm(normals, axis=2)
    normal_mag /= normal_mag.max() + 1e-8

    # --------------------------------------------------
    # stack channels
    # --------------------------------------------------
    lidar_img = np.stack(
        [depth_norm, intensity_norm, normal_mag],
        axis=-1
    )

    return lidar_img, xyz_img

import numpy as np


def create_bev_lidar_image(
    points,
    intensity=None,
    img_h=512,
    img_w=512,
    xlim=(-60, 60),
    ylim=(-60, 60),
    max_height=5.0,
):
    """
    Bird's-eye-view LiDAR image.

    Returns
    -------
    lidar_img : (H,W,3)
        [range, intensity, normal_magnitude]

    xyz_img : (H,W,3)
        representative 3D point per pixel
    """

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # --------------------------------------------------
    # BEV projection
    # --------------------------------------------------
    x_min, x_max = xlim
    y_min, y_max = ylim

    u = (x - x_min) / (x_max - x_min)
    v = (y - y_min) / (y_max - y_min)

    u = (u * img_w).astype(np.int32)
    v = ((1 - v) * img_h).astype(np.int32)

    valid = (
        (u >= 0) & (u < img_w) &
        (v >= 0) & (v < img_h)
    )

    u, v = u[valid], v[valid]
    pts = points[valid]

    if intensity is not None:
        intensity = intensity[valid]

    # --------------------------------------------------
    # initialize images
    # --------------------------------------------------
    height_img = np.full((img_h, img_w), -np.inf)
    xyz_img = np.zeros((img_h, img_w, 3))
    intensity_img = np.zeros((img_h, img_w))

    # --------------------------------------------------
    # top-surface z-buffer
    # keep highest point (typical BEV choice)
    # --------------------------------------------------
    for i in range(len(u)):
        ui, vi = u[i], v[i]

        if pts[i, 2] > height_img[vi, ui]:
            height_img[vi, ui] = pts[i, 2]
            xyz_img[vi, ui] = pts[i]

            if intensity is not None:
                intensity_img[vi, ui] = intensity[i]

    height_img[height_img == -np.inf] = 0

    # --------------------------------------------------
    # range channel
    # --------------------------------------------------
    range_img = np.linalg.norm(xyz_img, axis=2)

    range_norm = np.log1p(range_img)
    range_norm /= range_norm.max() + 1e-8

    # --------------------------------------------------
    # intensity normalization
    # --------------------------------------------------
    if intensity is not None:
        p1, p99 = np.percentile(intensity_img, (1, 99))
        intensity_img = np.clip(intensity_img, p1, p99)

        intensity_norm = (
            intensity_img - p1
        ) / (p99 - p1 + 1e-8)
    else:
        intensity_norm = np.zeros_like(range_norm)

    # --------------------------------------------------
    # BEV normals (surface structure)
    # --------------------------------------------------
    dx = np.roll(xyz_img, -1, axis=1) - xyz_img
    dy = np.roll(xyz_img, -1, axis=0) - xyz_img

    normals = np.cross(dx, dy)

    norm = np.linalg.norm(normals, axis=2, keepdims=True) + 1e-8
    normals /= norm

    normal_mag = np.linalg.norm(normals, axis=2)
    normal_mag /= normal_mag.max() + 1e-8

    # --------------------------------------------------
    # stack channels
    # --------------------------------------------------
    lidar_img = np.stack(
        [range_norm, intensity_norm, normal_mag],
        axis=-1
    )

    return lidar_img, xyz_img

def image_from_lidar_path(pcd_path):

    pcd = o3d.io.read_point_cloud(str(pcd_path))
    points = np.asarray(pcd.points)

    lidar_img, xyz_img = create_bev_lidar_image(points)

    return lidar_img, xyz_img

# ============================================================
# Convert lidar image → tensor
# ============================================================
def lidar_img_to_tensor(img):
    """
    img: H x W x 3 numpy float image
    returns: torch tensor [1,3,H,W]
    """

    img = img.astype(np.float32).copy()

    # normalize per-channel safely
    for c in range(img.shape[2]):
        ch = img[..., c]
        m = ch.max()
        if m > 0:
            img[..., c] = ch / m

    tensor = torch.from_numpy(img)          # H,W,3
    tensor = tensor.permute(2, 0, 1)        # 3,H,W
    tensor = tensor.unsqueeze(0)            # 1,3,H,W
    tensor = tensor.contiguous()

    return tensor


# ============================================================
# Feature extraction + matching
# ============================================================
def match_lidar_images(imgA, imgB, device="cuda"):

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    extractor = SuperPoint(
        max_num_keypoints=4096
    ).eval().to(device)

    matcher = LightGlue(
        features="superpoint"
    ).eval().to(device)

    # --------------------------------------------------
    # tensors
    # --------------------------------------------------
    tA = lidar_img_to_tensor(imgA).to(device)
    tB = lidar_img_to_tensor(imgB).to(device)

    # --------------------------------------------------
    # feature extraction
    # --------------------------------------------------
    feats0 = extractor.extract(tA)
    feats1 = extractor.extract(tB)

    # --------------------------------------------------
    # matching
    # --------------------------------------------------
    matches01 = matcher({
        "image0": feats0,
        "image1": feats1
    })

    # remove batch dimension
    feats0 = rbd(feats0)
    feats1 = rbd(feats1)
    matches01 = rbd(matches01)

    matches = matches01["matches"]

    kptsA = feats0["keypoints"].cpu().numpy()
    kptsB = feats1["keypoints"].cpu().numpy()

    return kptsA, kptsB, matches.cpu().numpy()


# ============================================================
# Human-friendly visualization
# ============================================================
def visualize_matches(imgA, imgB, kptsA, kptsB, matches,
                      max_draw=500):

    imgA = cv2.normalize(imgA, None, 0,255,
                         cv2.NORM_MINMAX).astype(np.uint8)
    imgB = cv2.normalize(imgB, None, 0,255,
                         cv2.NORM_MINMAX).astype(np.uint8)

    h1,w1,_ = imgA.shape
    h2,w2,_ = imgB.shape

    canvas = np.zeros(
        (max(h1,h2), w1+w2, 3),
        dtype=np.uint8
    )

    canvas[:h1,:w1] = imgA
    canvas[:h2,w1:] = imgB

    plt.figure(figsize=(18,8))
    plt.imshow(canvas)

    if len(matches) > max_draw:
        idx = np.random.choice(
            len(matches), max_draw, replace=False)
        matches = matches[idx]

    for m in matches:
        x1,y1 = kptsA[m[0]]
        x2,y2 = kptsB[m[1]]

        plt.plot(
            [x1, x2+w1],
            [y1, y2],
            linewidth=0.5
        )

    plt.title(f"{len(matches)} matches")
    plt.axis("off")

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

def image_to_lidarimg_path(img_path: Path, img_type: str) -> Path:
    cam_imgnum = img_path.stem
    bag_dir = img_path.parents[2].name
    cam_root = img_path.parents[3]

    return (
        cam_root /
        bag_dir /
        "lidar/" /
        Path(img_type) /
        f"{cam_imgnum}.png"
    )

def ensure_single_channel(img):
    if img.ndim == 3:
        img = img[..., 0]
    return img

def normalize_channel(x):
    x = x.astype(np.float32)
    x -= x.min()
    if x.max() > 0:
        x /= x.max()
    return x

def inpaint_channel(img, mask):
    mask = mask.astype(np.uint8) * 255
    return cv2.inpaint(
        img.astype(np.float32),
        mask,
        3,
        cv2.INPAINT_NS
    )

def img_path_to_multichannel_lidar_image(rgb_img_path):

    range_path  = image_to_lidarimg_path(rgb_img_path, "range")
    signal_path = image_to_lidarimg_path(rgb_img_path, "signal")
    reflec_path = image_to_lidarimg_path(rgb_img_path, "reflec")

    # ---- load WITHOUT conversion ----
    range_img  = cv2.imread(str(range_path),  cv2.IMREAD_UNCHANGED)
    signal_img = cv2.imread(str(signal_path), cv2.IMREAD_UNCHANGED)
    reflec_img = cv2.imread(str(reflec_path), cv2.IMREAD_UNCHANGED)

    # ---- force single channel ----
    def single(x):
        return x[...,0] if x.ndim == 3 else x

    range_img  = single(range_img)
    signal_img = single(signal_img)
    reflec_img = single(reflec_img)

    # ---- normalize ----
    range_img  = normalize_channel(range_img)
    signal_img = normalize_channel(signal_img)
    reflec_img = normalize_channel(reflec_img)


    # ---- get rid of weird stripes / artifacts ----
    invalid_mask = (
        # (range_img == 0) &
        (signal_img <= 1.0) & (signal_img >= 0.2) 
        # (reflec_img == 0)
    )

    print("Invalid count:", np.sum(invalid_mask))
    print("Total pixels:", invalid_mask.size)    

    # range_img[invalid_mask]  = 0
    signal_img[invalid_mask] = 0
    # reflec_img[invalid_mask] = 0

    # # avoid strong gradients after zeroing bad pixels
    # range_img  = inpaint_channel(range_img, invalid_mask)
    # signal_img = inpaint_channel(signal_img, invalid_mask)
    # reflec_img = inpaint_channel(reflec_img, invalid_mask)

    # ---- stack ----
    lidar_img = np.stack(
        [range_img, np.zeros_like(signal_img), reflec_img],
        axis=-1
    ).astype(np.float32)

    return lidar_img

def detect_iss_fpfh(points,
                    voxel_size=0.5,
                    visualize=True):
    """
    points: (N,3) numpy array

    Returns:
        pcd_down      : downsampled cloud
        keypoints     : ISS keypoints (Open3D point cloud)
        fpfh          : FPFH descriptor
    """

    # --------------------------------------------------
    # 1. Convert numpy → Open3D
    # --------------------------------------------------
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # --------------------------------------------------
    # 2. Downsample 
    # --------------------------------------------------
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # --------------------------------------------------
    # 3. Estimate normals
    # --------------------------------------------------
    radius_normal = voxel_size * 2

    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal,
            max_nn=30,
        )
    )

    # --------------------------------------------------
    # 4. ISS Keypoints
    # --------------------------------------------------
    keypoints = o3d.geometry.keypoint.compute_iss_keypoints(
        pcd_down,
        salient_radius=voxel_size * 8,
        non_max_radius=voxel_size * 4,
        gamma_21=0.9,
        gamma_32=0.9,
        min_neighbors=8,
    )

    print(f"# keypoints: {len(keypoints.points)}")

    # --------------------------------------------------
    # 5. Compute FPFH descriptors
    # --------------------------------------------------
    radius_feature = voxel_size * 15

    fpfh_full = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature,
            max_nn=100,
        ),
    )

    pcd_tree = o3d.geometry.KDTreeFlann(pcd_down)

    kp_indices = []

    for kp in keypoints.points:
        _, idx, _ = pcd_tree.search_knn_vector_3d(kp, 1)
        kp_indices.append(idx[0])

    kp_indices = np.array(kp_indices)

    fpfh_keypoints = o3d.pipelines.registration.Feature()
    fpfh_keypoints.data = fpfh_full.data[:, kp_indices]
    # --------------------------------------------------
    # 6. Visualization
    # --------------------------------------------------
    if visualize:

        # lidar cloud = gray
        pcd_down.paint_uniform_color([0.7, 0.7, 0.7])

        # keypoints = red
        keypoints.paint_uniform_color([1, 0, 0])

        o3d.visualization.draw_geometries(
            [pcd_down, keypoints],
            window_name="ISS Keypoints"
        )

    return pcd_down, keypoints, fpfh_keypoints

def ransac_feature_matching(
    keypointsq,
    keypointsr,
    fpfhq,
    fpfhr,
    voxel_size=0.3,
):

    distance_threshold = voxel_size * 4.0

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source=keypointsq,
        target=keypointsr,
        source_feature=fpfhq,
        target_feature=fpfhr,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,

        estimation_method=
        o3d.pipelines.registration.
        TransformationEstimationPointToPoint(False),

        ransac_n=3,

        checkers=[
            o3d.pipelines.registration.
            CorrespondenceCheckerBasedOnEdgeLength(0.9),

            o3d.pipelines.registration.
            CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],

        criteria=o3d.pipelines.registration.
        RANSACConvergenceCriteria(
            100000,
            0.999
        ),
    )

    return result

def visualize_matches_3D(keypointsq, keypointsr, result):

    # copy so colors don't overwrite originals
    src = keypointsq.transform(
        result.transformation.copy()
    )

    tgt = keypointsr

    src.paint_uniform_color([1, 0, 0])  # red
    tgt.paint_uniform_color([0, 1, 0])  # green

    o3d.visualization.draw_geometries(
        [src, tgt],
        window_name="RANSAC Alignment"
    )

def visualize_correspondences3D(
    keypointsq,
    keypointsr,
    result
):

    src = keypointsq.transform(
        result.transformation.copy()
    )

    o3d.visualization.draw_geometries(
        [
            src,
            keypointsr,
        ],
        window_name="Aligned Keypoints"
    )

    o3d.visualization.draw_geometries_with_editing(
        [src, keypointsr]
    )

def visualize_inlier_matches_3D(
    pcd_downq,
    pcd_downr,
    keypointsq,
    keypointsr,
    result,
):

    # ----------------------------------------
    # Transform query into retrieved frame
    # ----------------------------------------
    pcd_q = pcd_downq.transform(
        result.transformation.copy()
    )

    kp_q = keypointsq.transform(
        result.transformation.copy()
    )

    kp_r = keypointsr

    # ----------------------------------------
    # Color point clouds
    # ----------------------------------------
    pcd_q.paint_uniform_color([0.7, 0.7, 0.7])
    pcd_downr.paint_uniform_color([0.5, 0.5, 0.5])

    # ----------------------------------------
    # Extract inlier correspondences
    # ----------------------------------------
    corres = np.asarray(result.correspondence_set)

    src_idx = corres[:, 0]
    tgt_idx = corres[:, 1]

    src_pts = np.asarray(kp_q.points)[src_idx]
    tgt_pts = np.asarray(kp_r.points)[tgt_idx]

    # ----------------------------------------
    # Create red keypoint clouds
    # ----------------------------------------
    src_inliers = o3d.geometry.PointCloud()
    src_inliers.points = o3d.utility.Vector3dVector(src_pts)
    src_inliers.paint_uniform_color([1, 0, 0])

    tgt_inliers = o3d.geometry.PointCloud()
    tgt_inliers.points = o3d.utility.Vector3dVector(tgt_pts)
    tgt_inliers.paint_uniform_color([1, 0, 0])

    # ----------------------------------------
    # Build correspondence lines
    # ----------------------------------------
    all_points = np.vstack((src_pts, tgt_pts))

    lines = [
        [i, i + len(src_pts)]
        for i in range(len(src_pts))
    ]

    colors = [[0, 1, 0] for _ in lines]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(all_points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # ----------------------------------------
    # Visualize
    # ----------------------------------------
    o3d.visualization.draw_geometries(
        [
            pcd_q,
            pcd_downr,
            src_inliers,
            tgt_inliers,
            line_set,
        ],
        window_name="RANSAC Inlier Matches",
    )

def visualize_inlier_matches_offset(
    pcd_downq,
    pcd_downr,
    keypointsq,
    keypointsr,
    result,
    offset=[0, 100, 0],   # shift query cloud
):

    offset = np.array(offset)

    # ----------------------------------------
    # Transform query using RANSAC pose
    # ----------------------------------------
    pcd_q = pcd_downq.transform(
        result.transformation.copy()
    )
    kp_q = keypointsq.transform(
        result.transformation.copy()
    )

    pcd_r = pcd_downr
    kp_r = keypointsr

    # ----------------------------------------
    # Apply visualization offset
    # ----------------------------------------
    pcd_q.translate(offset)
    kp_q.translate(offset)

    # ----------------------------------------
    # Color clouds
    # ----------------------------------------
    pcd_q.paint_uniform_color([0.7, 0.7, 0.7])
    pcd_r.paint_uniform_color([0.4, 0.4, 0.4])

    # ----------------------------------------
    # Extract inliers
    # ----------------------------------------
    corres = np.asarray(result.correspondence_set)

    src_idx = corres[:, 0]
    tgt_idx = corres[:, 1]

    src_pts = np.asarray(kp_q.points)[src_idx]
    tgt_pts = np.asarray(kp_r.points)[tgt_idx]

    # ----------------------------------------
    # Red inlier keypoints
    # ----------------------------------------
    src_inliers = o3d.geometry.PointCloud()
    src_inliers.points = o3d.utility.Vector3dVector(src_pts)
    src_inliers.paint_uniform_color([1, 0, 0])

    tgt_inliers = o3d.geometry.PointCloud()
    tgt_inliers.points = o3d.utility.Vector3dVector(tgt_pts)
    tgt_inliers.paint_uniform_color([1, 0, 0])

    # ----------------------------------------
    # Build match lines
    # ----------------------------------------
    all_points = np.vstack((src_pts, tgt_pts))

    lines = [
        [i, i + len(src_pts)]
        for i in range(len(src_pts))
    ]

    colors = [[0, 1, 0] for _ in lines]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(all_points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # ----------------------------------------
    # Visualize
    # ----------------------------------------
    o3d.visualization.draw_geometries(
        [
            pcd_q,
            pcd_r,
            src_inliers,
            tgt_inliers,
            line_set,
        ],
        window_name="Offset Inlier Correspondences",
    )
# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    START_IDX = 0
    lg = LightGlueVisualizer(max_keypoints=2048)


    # Load image pairs
    with open("matched_img_paths_mega.json", "r") as f:
        data = json.load(f)
    path_imgq_all = data["queries"]
    path_imgr_all = data["retrieved"]

    for path_idx in range(START_IDX, len(path_imgq_all)):

        path_imgq = path_imgq_all[path_idx]
        path_imgr = path_imgr_all[path_idx]  # top K=1

        lid_imgq = img_path_to_multichannel_lidar_image(Path(path_imgq))
        lid_imgr = img_path_to_multichannel_lidar_image(Path(path_imgr))
        ptsq, _, _  = lidar_pcd(image_to_lidar_path(Path(path_imgq)))
        ptsr, _, _  = lidar_pcd(image_to_lidar_path(Path(path_imgr)))

        print(lid_imgq.shape)
        print(ptsq.shape)

        pcd_downq, keypointsq, fpfhq = detect_iss_fpfh(ptsq, visualize=False)
        pcd_downr, keypointsr, fpfhr = detect_iss_fpfh(ptsr, visualize=False)


        result = ransac_feature_matching(
            keypointsq,
            keypointsr,
            fpfhq,
            fpfhr
        )

        print(result)
        print("# inliers:", len(result.correspondence_set))

        visualize_inlier_matches_offset(
            pcd_downq,
            pcd_downr,
            keypointsq,
            keypointsr,
            result,
        )
        # kA, kB, matches = match_lidar_images(lid_imgq, lid_imgr)
        
        # visualize_matches(lid_imgq, lid_imgr, kA, kB, matches)

        lg.visualize_matches(path_imgq, path_imgr)
        plt.show()
