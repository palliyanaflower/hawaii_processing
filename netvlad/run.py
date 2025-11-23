# --------NETVLAD ONLY------------
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.neighbors import NearestNeighbors

from netvlad_model import get_netvlad_backbone


# -------------------------
#  Load Model
# -------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = get_netvlad_backbone(num_clusters=64).to(device)
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

data = np.load("netvlad_db_cam2.npz", allow_pickle=True)
db_descs = data["descs"]      # shape (N, D)
db_paths = data["paths"]      # list of image paths

print(f"Loaded database with {len(db_paths)} images.")


# -------------------------
#  Choose a Single Query Image (cam3)
# -------------------------
query_folder = "../data/makalii_point/processed_data/cam3"
query_idx = 700  # you can change this
query_path = os.path.join(query_folder, str(query_idx) + ".png")

print(f"\nUsing query index {query_idx}:")
print(query_path)

query_desc = extract_netvlad_descriptor(query_path).reshape(1, -1)


# -------------------------
#  Retrieve Top-3 Matches
# -------------------------
nn = NearestNeighbors(n_neighbors=3, metric='cosine')
nn.fit(db_descs)
dists, idxs = nn.kneighbors(query_desc)

top3_idxs = idxs[0]
top3_dists = dists[0]

print("\n=== Top 3 Matches ===")
for rank, (i, d) in enumerate(zip(top3_idxs, top3_dists), 1):
    print(f"#{rank}: {db_paths[i]}  (distance={d:.4f})")


# -------------------------
#  Visualization
# -------------------------
query_img = cv2.cvtColor(cv2.imread(query_path), cv2.COLOR_BGR2RGB)
match_imgs = [
    cv2.cvtColor(cv2.imread(db_paths[i]), cv2.COLOR_BGR2RGB)
    for i in top3_idxs
]

plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.imshow(query_img)
plt.title("Query " + str(query_idx))
plt.axis("off")

for j in range(3):
    plt.subplot(1, 4, j+2)
    plt.imshow(match_imgs[j])
    plt.title(f"Match #{j+1} (distance={top3_dists[j]:.4f})")
    plt.axis("off")

plt.tight_layout()
plt.show()


# # ---------------DINO ONLY---------------

# import os
# import re
# import cv2
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.neighbors import NearestNeighbors

# from dinov2_class import DinoDescriptor  # <-- DINO-only wrapper


# # -------------------------
# # Load DINOv2 Model
# # -------------------------
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = DinoDescriptor(device=device, model_type="vits14")
# model.dinov2_model.eval()


# # -------------------------
# # Descriptor Extraction
# # -------------------------
# def extract_dino_descriptor(img_path, pooling="avg"):
#     desc = model.extract_descriptor(img_path, pooling=pooling)
#     desc = desc / np.linalg.norm(desc)  # normalize again (safe)
#     return desc.astype(np.float32)


# # -------------------------
# # Load Database
# # -------------------------
# db_data = np.load("dinov2_db_cam2.npz", allow_pickle=True)
# db_descs = db_data["descs"]          # (N, D)
# db_paths = db_data["paths"]

# print(f"Loaded database with {len(db_paths)} images.")


# # -------------------------
# # Choose Query Image
# # -------------------------
# def numeric_sort_key(path):
#     """Extract numeric part of filename for proper sorting."""
#     fname = os.path.basename(path)
#     nums = re.findall(r'\d+', fname)
#     return int(nums[0]) if nums else 0

# query_folder = "../data/makalii_point/processed_data/cam3"
# query_idx = 0
# query_path = sorted(
#     [os.path.join(query_folder, f)
#      for f in os.listdir(query_folder)
#      if f.endswith(".png") or f.endswith(".jpg")],
#     key=numeric_sort_key
# )[query_idx]

# print(f"\nUsing query index {query_idx}: {query_path}")

# query_desc = extract_dino_descriptor(query_path).reshape(1, -1)


# # -------------------------
# # Nearest Neighbor Retrieval
# # -------------------------
# nn = NearestNeighbors(n_neighbors=3, metric='cosine')
# nn.fit(db_descs)
# dists, idxs = nn.kneighbors(query_desc)

# top3_idxs = idxs[0]
# top3_dists = dists[0]

# print("\n=== Top 3 Matches (DINO-only) ===")
# for rank, (i, d) in enumerate(zip(top3_idxs, top3_dists), 1):
#     print(f"#{rank}: {db_paths[i]}   (distance={d:.4f})")


# # -------------------------
# # Visualization
# # -------------------------
# query_img = cv2.cvtColor(cv2.imread(query_path), cv2.COLOR_BGR2RGB)
# match_imgs = [
#     cv2.cvtColor(cv2.imread(db_paths[i]), cv2.COLOR_BGR2RGB)
#     for i in top3_idxs
# ]

# plt.figure(figsize=(15, 5))

# plt.subplot(1, 4, 1)
# plt.imshow(query_img)
# plt.title("Query " + str(query_idx))
# plt.axis("off")

# for j in range(3):
#     plt.subplot(1, 4, j+2)
#     plt.imshow(match_imgs[j])
#     plt.title(f"Match #{j+1}\nd={top3_dists[j]:.4f}")
#     plt.axis("off")

# plt.tight_layout()
# plt.show()

