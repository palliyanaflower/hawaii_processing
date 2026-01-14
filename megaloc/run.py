# ---------------MEGALOC RETRIEVAL---------------

import os
import re
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from PIL import Image
from torchvision import transforms


# -------------------------
# Load MegaLoc Model
# -------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load("gmberton/MegaLoc", "get_trained_model").to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])


# -------------------------
# Descriptor Extraction (MEGALOC)
# -------------------------
def extract_megaloc_descriptor(img_path):
    img = Image.open(img_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        desc = model(tensor).cpu().numpy().squeeze()   # shape = (8448,)

    desc = desc / np.linalg.norm(desc)                 # normalize
    return desc.astype(np.float32)


# -------------------------
# Load DB built earlier
# -------------------------
db_data = np.load("results/megaloc_db_cam2.npz", allow_pickle=True)
db_descs = db_data["descs"]   # (N, 8448)
db_paths = db_data["paths"]

print(f"Loaded DB with {len(db_paths)} images, dim={db_descs.shape[1]}")


# -------------------------
# Choose Query from cam3
# -------------------------
def numeric_sort_key(path):
    fname = os.path.basename(path)
    nums = re.findall(r'\d+', fname)
    return int(nums[0]) if nums else 0

query_idx = 0
bag_num = 55
query_folder = f"../data/makalii_point/processed_data_imggps/cam3/bag_camera_3_2025_08_13-01_35_58_{bag_num}/camera"  # ← CHANGE as correct

query_path = sorted(
    [os.path.join(query_folder, f)
     for f in os.listdir(query_folder)
     if f.lower().endswith(('.png','.jpg'))],
    key=numeric_sort_key
)[query_idx]

print(f"\nUsing query {query_idx}: {query_path}")

query_desc = extract_megaloc_descriptor(query_path).reshape(1,-1)


# -------------------------
# kNN Retrieval
# -------------------------
nn = NearestNeighbors(n_neighbors=3, metric='cosine')
nn.fit(db_descs)
dists, idxs = nn.kneighbors(query_desc)

top3_idxs = idxs[0]
top3_dists = dists[0]

print("\n=== Top 3 Matches (MegaLoc) ===")
for rank,(i,d) in enumerate(zip(top3_idxs,top3_dists),1):
    print(f"#{rank}: {db_paths[i]}  (dist={d:.4f})")

# TODO: Threshold to completely reject if not within certain distance / similarity?

# -------------------------
# Visualize results
# -------------------------
query_img  = cv2.cvtColor(cv2.imread(query_path), cv2.COLOR_BGR2RGB)
match_imgs = [cv2.cvtColor(cv2.imread(db_paths[i]), cv2.COLOR_BGR2RGB) for i in top3_idxs]

plt.figure(figsize=(15,5))

plt.subplot(1,4,1)
plt.imshow(query_img)
plt.title(f"Bag {bag_num}, Img {query_idx}")
plt.axis("off")

for j in range(3):
    plt.subplot(1,4,j+2)
    plt.imshow(match_imgs[j])
    plt.title(f"Match #{j+1}\nd={top3_dists[j]:.4f}")
    plt.axis("off")

plt.tight_layout()
plt.show()
