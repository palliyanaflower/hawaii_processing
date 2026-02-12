import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path

# -------------------------
#  Load MegaLoc
# -------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load("gmberton/MegaLoc", "get_trained_model").to(device)
model.eval()

# image preprocessing (same as before)
preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])


# -------------------------
#  Descriptor Extraction
# -------------------------
def extract_megaloc_descriptor(img_path):
    img = Image.open(img_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        desc = model(tensor).cpu().numpy().squeeze()  # shape (8448,)

    # L2 normalize
    desc = desc / np.linalg.norm(desc)
    return desc.astype(np.float32)

def build_database(root_folder, output_file="results/megaloc_db_cam2.npz", ds_val = 1):
    descs = []
    paths = []

    root = Path(root_folder)

    # -------------------------
    #  Iterate over each bag folder
    # -------------------------
    bag_folders = sorted([p for p in root.iterdir() if p.is_dir()])

    total_images = 0
    for bag in bag_folders:
        cam_dir = bag / "camera/rgb"
        if not cam_dir.exists():
            continue

        img_files = sorted([
            f for f in cam_dir.glob("*")
            if f.suffix.lower() in (".png", ".jpg", ".jpeg")
        ])

        print(f"[Bag] {bag.name} — {len(img_files)} images")
        total_images += len(img_files)

        # -------------------------
        # extract descriptors
        # -------------------------
        for idx, img_path in enumerate(img_files):
            desc = extract_megaloc_descriptor(img_path)

            descs.append(desc)
            paths.append(str(img_path.resolve()))

            # Downsample images
            if idx % ds_val == 0:
                print(f"  processed {idx}/{len(img_files)} in {bag.name}")

    # -------------------------
    #  Convert to arrays
    # -------------------------
    descs = np.vstack(descs).astype(np.float32)
    paths = np.array(paths)

    # ensure directory exists
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving → {output_file}")
    np.savez(output_file, descs=descs, paths=paths)

    print(f"\nDone. Total images processed: {total_images}")
    print("Desc shape", descs.shape)
    return descs, paths

# # -------------------------
# #  Run it
# # -------------------------
# db_descs, db_paths = build_database("../data/makalii_point/processed_data_imggps/cam2/camera")
# print(f"Database built with {len(db_paths)} images — descriptor dim = {db_descs.shape[1]}")


db_descs, db_paths = build_database(
    "../data/makalii_point/processed_lidar_cam_gps/cam2",
    output_file="results/megaloc_db_lcn_cam2.npz"
)

print(f"Database built with {len(db_paths)} images — descriptor dim = {db_descs.shape[1]}")
