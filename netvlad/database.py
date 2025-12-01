import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.neighbors import NearestNeighbors

from network import get_netvlad_backbone

# -------------------------
#  Load Model
# -------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = get_netvlad_backbone(num_clusters=64).to(device)
# model = DinoV2NetVLAD(num_clusters=64).to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((480, 640)),
])

# -------------------------
#  Descriptor Extraction
# -------------------------
def extract_netvlad_descriptor(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        desc = model(tensor)
    return torch.nn.functional.normalize(desc, p=2, dim=1).squeeze().cpu().numpy()


# -------------------------
#  Build Database (cam2)
# -------------------------
def build_database(folder, output_file="results/netvlad_db_cam2.npz"):
    descs = []
    paths = []

    idx = 0
    for f in sorted(os.listdir(folder)):
        if f.endswith('.png') or f.endswith('.jpg'):
            p = os.path.join(folder, f)
            paths.append(p)
            descs.append(extract_netvlad_descriptor(p))

        if idx % 50 == 0:
            print(f"Processed {idx}/{len(os.listdir(folder))}")
        idx += 1

    descs = np.vstack(descs).astype(np.float32)
    paths = np.array(paths)

    print(f"Saving database to: {output_file}")
    np.savez(output_file, descs=descs, paths=paths)

    print("Done.")
    return np.vstack(descs), paths


db_descs, db_paths = build_database("../data/makalii_point/processed_data/cam2")

print(f"Loaded database with {len(db_paths)} images.")