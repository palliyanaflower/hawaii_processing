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


# -------------------------
#  Build Database (cam2)
# -------------------------
def build_database(folder, output_file="dinov2_db_cam2.npz", pooling="avg"):
    descs = []
    paths = []

    img_files = sorted([f for f in os.listdir(folder) if f.endswith(('.png', '.jpg'))])
    for idx, f in enumerate(img_files):
        p = os.path.join(folder, f)
        paths.append(p)
        descs.append(extract_dinov2_descriptor(p, pooling=pooling))

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
db_descs, db_paths = build_database("../data/makalii_point/processed_data/cam2", pooling="avg")
print(f"Loaded database with {len(db_paths)} images.")
