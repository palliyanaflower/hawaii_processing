import os
import torch
import numpy as np
from dinov2_class import DinoDescriptor  # <-- DINOv2-only wrapper

# -------------------------
#  Load Model
# -------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DinoDescriptor(device=device, model_type="vits14")  # use vitb14 for better performance

# -------------------------
#  Descriptor Extraction
# -------------------------
def extract_dinov2_descriptor(img_path, pooling="avg"):
    """
    Extract global descriptor for a single image using DINOv2-only.
    """
    desc = model.extract_descriptor(img_path, pooling=pooling)
    # L2 normalize (already normalized in class, but safe to double-check)
    desc = desc / np.linalg.norm(desc)
    return desc.astype(np.float32)

extract_dinov2_descriptor("../data/makalii_point/processed_lidar_cam_gps/cam2/bag_camera_2_2025_08_13-01_35_58_5/camera/rgb/0.png")