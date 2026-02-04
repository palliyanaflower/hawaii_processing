from dataclasses import dataclass
import numpy as np
from typing import List

@dataclass
class LidarKeypointNeighbors:
    
    depths_per_kp: List[np.ndarray]     # [(Mi,), ...]
    pxs_per_kp: List[np.ndarray]        # [(Mi,2), ...] Track lidar neighbors for each keypoint
    pts_per_kp: List[np.ndarray]        # [(Mi,3), ...] Track lidar neighbors for each keypoint

    pts_all: np.ndarray                 # (M,3)
    pxs_all: np.ndarray                 # (M,2)
    group_ids: np.ndarray               # (M,)

    num_kps: int
    num_pts_all: int
