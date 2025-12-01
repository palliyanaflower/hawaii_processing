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


# -------------------------
#  Build Database
# -------------------------
def build_database(folder, output_file="results/megaloc_db_cam2.npz"):
    descs = []
    paths = []

    img_files = sorted([f for f in os.listdir(folder) if f.lower().endswith((".png",".jpg",".jpeg"))])
    for idx, f in enumerate(img_files):
        path = os.path.join(folder, f)
        paths.append(path)

        desc = extract_megaloc_descriptor(path)
        descs.append(desc)

        if idx % 50 == 0:
            print(f"[MegaLoc] processed {idx}/{len(img_files)}")

    descs = np.vstack(descs).astype(np.float32)  # (N, 8448)
    paths = np.array(paths)

    output_file = Path(output_file)              # convert to Path if it's a string
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving → {output_file}")    
    np.savez(output_file, descs=descs, paths=paths)
    print("Done.\n")

    return descs, paths


# -------------------------
#  Run it
# -------------------------
db_descs, db_paths = build_database("../data/makalii_point/processed_data_imggps/cam2/camera")
print(f"Database built with {len(db_paths)} images — descriptor dim = {db_descs.shape[1]}")
