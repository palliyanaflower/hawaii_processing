from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2

import camera_geom as cg
import file_loader as loader

@dataclass
class CameraView:
    name: str
    image_path: Path
    lidar_pcd_path: Path
    calib_path: Path
    T_lidar_cam: np.ndarray

    # Filled later
    img: np.ndarray = None
    K_proj: np.ndarray = None
    u: np.ndarray = None
    v: np.ndarray = None
    pts_all: np.ndarray = None
    pts_cam: np.ndarray = None
    pts_infront: np.ndarray = None
    pts_colors_infront: np.ndarray = None

    def load_image_and_calib(self):
        img = cv2.imread(str(self.image_path))
        h, w, _ = img.shape

        K, dist = loader.intrinsics_and_distortion(self.calib_path)
        K = K[:, :3]

        K_proj, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
        img = cv2.undistort(img, K, dist, None, K_proj)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.img = img
        self.K_proj = K_proj

    def project_lidar(self):
        points, _, colors = loader.lidar_pcd(self.lidar_pcd_path)

        (
            self.u,
            self.v,
            self.pts_cam,
            _,
            mask_infront,
            mask_img,
        ) = cg.filter_pts_to_image_plane(
            points,
            colors,
            self.T_lidar_cam,
            self.K_proj,
            self.img.shape[1],
            self.img.shape[0],
            self.img,
        )

        self.pts_infront = points[mask_infront][mask_img]
        self.pts_colors_infront = colors[mask_infront][mask_img]
        self.pts_all = points