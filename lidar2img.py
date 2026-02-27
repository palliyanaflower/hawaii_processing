# import numpy as np
# import open3d as o3d
# import matplotlib.pyplot as plt


# def pointcloud_to_range_image(points, H, W,
#                               fov_up, fov_down):

#     x, y, z = points.T
#     depth = np.linalg.norm(points, axis=1)

#     yaw = np.arctan2(y, x)
#     pitch = np.arcsin(z / depth)

#     fov = fov_up - fov_down

#     u = 0.5 * (1 - yaw / np.pi) * W
#     v = (fov_up - pitch) / fov * H

#     u = np.clip(u.astype(np.int32), 0, W-1)
#     v = np.clip(v.astype(np.int32), 0, H-1)

#     img = np.full((H, W), np.inf)

#     for i in range(len(depth)):
#         if depth[i] < img[v[i], u[i]]:
#             img[v[i], u[i]] = depth[i]

#     img[img == np.inf] = 0
#     return img

# pcd_path = "data/makalii_point/processed_lidar_cam_gps/cam2/bag_camera_2_2025_08_13-01_35_58_19/lidar/pcd/0.pcd"
# pcd = o3d.io.read_point_cloud(str(pcd_path))
# points = np.asarray(pcd.points)

# xyz = points

# x = xyz[:, 0]
# y = xyz[:, 1]
# z = xyz[:, 2]

# depth = np.linalg.norm(xyz, axis=1)

# yaw = np.arctan2(y, x)          # horizontal angle
# pitch = np.arcsin(z / depth)    # vertical angle

# H = 64        # vertical resolution
# W = 2048      # horizontal resolution

# fov_up   = np.deg2rad(15)
# fov_down = np.deg2rad(-25)
# print("Estimated width:", W)


# range_img = pointcloud_to_range_image(
#     points, H=H, W=W, fov_up=fov_up, fov_down=fov_down
# )

# plt.figure(figsize=(12,4))
# plt.imshow(range_img, cmap="jet", aspect='auto')
# plt.colorbar()
# plt.show()


# # import numpy as np
# # import open3d as o3d
# # import matplotlib.pyplot as plt


# def lidar_to_range_image(
#     points,
#     v_fov=(-24.9, 2.0),   # degrees (Velodyne HDL-64 style)
#     h_fov=(-180, 180),
#     img_h=64,
#     img_w=1024,
#     max_range=100.0,
# ):
#     """
#     Convert LiDAR point cloud to front-view range image.

#     Returns:
#         range_image  (H,W)
#         xyz_image    (H,W,3) backprojection helper
#     """

#     x, y, z = points[:, 0], points[:, 1], points[:, 2]

#     # --------------------------------------------------
#     # spherical coordinates
#     # --------------------------------------------------
#     r = np.sqrt(x**2 + y**2 + z**2)

#     azimuth = np.arctan2(y, x)
#     elevation = np.arcsin(z / r)

#     # radians
#     h_min, h_max = np.deg2rad(h_fov)
#     v_min, v_max = np.deg2rad(v_fov)

#     # --------------------------------------------------
#     # project to image coords
#     # --------------------------------------------------
#     u = (azimuth - h_min) / (h_max - h_min)
#     v = (elevation - v_min) / (v_max - v_min)

#     u = (u * img_w).astype(np.int32)
#     v = ((1.0 - v) * img_h).astype(np.int32)

#     # valid mask
#     valid = (
#         (u >= 0) & (u < img_w) &
#         (v >= 0) & (v < img_h) &
#         (r > 0) &
#         (r < max_range)
#     )

#     u = u[valid]
#     v = v[valid]
#     r = r[valid]
#     pts = points[valid]

#     # --------------------------------------------------
#     # initialize images
#     # --------------------------------------------------
#     range_image = np.full((img_h, img_w), np.inf)
#     xyz_image = np.zeros((img_h, img_w, 3))

#     # --------------------------------------------------
#     # z-buffering (keep closest point)
#     # --------------------------------------------------
#     for i in range(len(r)):
#         ui, vi = u[i], v[i]

#         if r[i] < range_image[vi, ui]:
#             range_image[vi, ui] = r[i]
#             xyz_image[vi, ui] = pts[i]

#     range_image[range_image == np.inf] = 0

#     return range_image, xyz_image


# # =====================================================
# # LOAD POINT CLOUD
# # =====================================================
# pcd_path = "data/makalii_point/processed_lidar_cam_gps/cam2/bag_camera_2_2025_08_13-01_35_58_19/lidar/pcd/0.pcd"

# pcd = o3d.io.read_point_cloud(str(pcd_path))
# points = np.asarray(pcd.points)

# range_img, xyz_img = lidar_to_range_image(points)

# range_img = np.log1p(range_img)
# range_img /= range_img.max()
# # =====================================================
# # VISUALIZE
# # =====================================================
# plt.figure(figsize=(12, 4))
# plt.imshow(range_img, cmap="viridis")
# plt.title("Front-view LiDAR Range Image")
# plt.colorbar(label="Range (m)")
# plt.tight_layout()
# plt.show()


import numpy as np
import open3d as o3d


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

pcd_path = "data/makalii_point/processed_lidar_cam_gps/cam2/bag_camera_2_2025_08_13-01_35_58_19/lidar/pcd/0.pcd"

pcd = o3d.io.read_point_cloud(pcd_path)

points = np.asarray(pcd.points)

# Ouster intensity usually stored as colors
intensity = None
if len(pcd.colors) > 0:
    intensity = np.asarray(pcd.colors)[:, 0]

lidar_img, xyz_img = create_multichannel_lidar_image(
    points,
    intensity=intensity
)

import matplotlib.pyplot as plt

plt.figure(figsize=(15,5))

plt.subplot(131)
plt.title("Depth")
plt.imshow(lidar_img[:,:,0], cmap="inferno")

plt.subplot(132)
plt.title("Intensity")
plt.imshow(lidar_img[:,:,1], cmap="gray")

plt.subplot(133)
plt.title("Geometry")
plt.imshow(lidar_img[:,:,2], cmap="viridis")

plt.show()

import torch

img = lidar_img.astype("float32")
img = np.clip(img, 0, 1)

tensor = torch.from_numpy(img).permute(2,0,1)[None]

from lightglue import SuperPoint, LightGlue
from lightglue.utils import rbd

device = "cuda" if torch.cuda.is_available() else "cpu"

extractor = SuperPoint(
    max_num_keypoints=4096
).eval().to(device)

matcher = LightGlue(
    features="superpoint"
).eval().to(device)


featsA = extractor.extract(tensorA.to(device))
featsB = extractor.extract(tensorB.to(device))

matches01 = matcher({
    "image0": featsA,
    "image1": featsB
})

featsA, featsB, matches01 = [
    rbd(x) for x in [featsA, featsB, matches01]
]

matches = matches01["matches"]