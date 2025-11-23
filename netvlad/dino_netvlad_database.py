import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.neighbors import NearestNeighbors

from dinov2_netvlad import DinoNetVLAD  # <-- your DINOv2 + NetVLAD wrapper

# -------------------------
#  Load Model
# -------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DinoNetVLAD(device=device, num_clusters=64)
model.netvlad.eval()
model.dinov2_model.eval()

# -------------------------
#  Descriptor Extraction
# -------------------------
def extract_dinov2_netvlad_descriptor(img_path):
    """
    Extract global descriptor for a single image using DINOv2 + NetVLAD.
    """
    desc = model.extract_descriptor(img_path)
    # L2 normalize final descriptor
    desc = desc / np.linalg.norm(desc)
    return desc.astype(np.float32)


# -------------------------
#  Build Database (cam2)
# -------------------------
def build_database(folder, output_file="dinov2_netvlad_db_cam2.npz"):
    descs = []
    paths = []

    img_files = sorted([f for f in os.listdir(folder) if f.endswith(('.png', '.jpg'))])
    for idx, f in enumerate(img_files):
        p = os.path.join(folder, f)
        paths.append(p)
        descs.append(extract_dinov2_netvlad_descriptor(p))

        if idx % 50 == 0:
            print(f"Processed {idx}/{len(img_files)}")

    descs = np.vstack(descs).astype(np.float32)
    paths = np.array(paths)

    print(f"Saving database to: {output_file}")
    np.savez(output_file, descs=descs, paths=paths)

    print("Done.")
    return descs, paths


# -------------------------
#  Run database building
# -------------------------
db_descs, db_paths = build_database("../data/makalii_point/processed_data/cam2")
print(f"Loaded database with {len(db_paths)} images.")
