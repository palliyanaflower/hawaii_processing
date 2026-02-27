import json
import yaml
import math
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm
import torch
from sklearn.neighbors import NearestNeighbors
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from utils.lightglue_loader import LightGlueVisualizer
import utils.viz as viz
from dino.dinov2_class import DinoDescriptor



# ============================================================
# CONFIG
# ============================================================
CAM2_ROOT = Path("data/makalii_point/processed_lidar_cam_gps/cam2")
CAM3_ROOT = Path("data/makalii_point/processed_lidar_cam_gps/cam3")
DB_FILE   = Path("megaloc/results/megaloc_db_lcn_cam2.npz")

TOP_K = 5
THRESH = 1.0
MIN_NUM_MATCHES = 50

# ============================================================
# Save Folder
# ============================================================
desc_root = Path("desc_info/descriptors")
desc_root.mkdir(parents=True, exist_ok=True)


# ============================================================
# UTILS
# ============================================================

def bag_number_from_path(p: Path):
    """Extract trailing bag number from bag_camera_..._N"""
    for parent in p.parents:
        if parent.name.startswith("bag_camera"):
            return int(parent.name.split("_")[-1])
    return -1


def read_gps_json(p: Path):
    with open(p, "r") as f:
        data = json.load(f)
    return (
        float(data["position"]["latitude"]),
        float(data["position"]["longitude"]),
    )


# ---------------------------
# GPS → meters
# ---------------------------

def latlon_to_xy_m(lat, lon, lat0, lon0):
    lat = np.asarray(lat)
    lon = np.asarray(lon)

    R = 6378137.0
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    lat0_rad = np.radians(lat0)

    x = R * dlon * np.cos(lat0_rad)
    y = R * dlat
    return x, y


def gps_array_to_meters(query_lat, query_lon, db_gps):
    x, y = latlon_to_xy_m(db_gps[:,0], db_gps[:,1], query_lat, query_lon)
    return np.sqrt(x**2 + y**2)


# ============================================================
# LOAD MEGALOC
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = torch.hub.load("gmberton/MegaLoc", "get_trained_model").to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

def extract_megaloc_descriptor(img_path):
    img = Image.open(img_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        desc = model(tensor).cpu().numpy().squeeze()  # shape (8448,)

    # L2 normalize
    desc = desc / np.linalg.norm(desc)
    return desc.astype(np.float32)


# ============================================================
# LOAD DATABASE
# ============================================================

if not DB_FILE.exists():
    raise FileNotFoundError(DB_FILE)

db = np.load(DB_FILE, allow_pickle=True)
db_descs = db["descs"]

# Sort according to bag order
db_paths = np.array([Path(p) for p in db["paths"]])

pairs = list(zip(db_paths, db_descs))
pairs.sort(key=lambda x: bag_number_from_path(x[0]))

db_paths, db_descs = zip(*pairs)
db_paths = np.array(db_paths)
db_descs = np.vstack(db_descs)

print(f"Loaded DB: {len(db_paths)} images")


# ============================================================
# BUILD CAM2 GPS MAP
# ============================================================

print("Building cam2 GPS mapping...")
cam2_gps = {}

for bag in sorted(CAM2_ROOT.iterdir(), key=lambda p: int(p.name.split("_")[-1])):
    cam_folder = bag / "camera/rgb"
    gps_folder = bag / "nav/gt"
    if not cam_folder.exists() or not gps_folder.exists():
        continue

    for img in sorted(cam_folder.iterdir(), key=lambda p: int(p.stem)):
        gps_file = gps_folder / img.with_suffix(".json").name
        if gps_file.exists():
            cam2_gps[str(img.resolve())] = read_gps_json(gps_file)


# Align DB → GPS
db_index_to_gps = []
valid_mask = []

for p in db_paths:
    g = cam2_gps.get(str(p.resolve()), (np.nan, np.nan))
    db_index_to_gps.append(g)
    valid_mask.append(not np.isnan(g[0]))

db_index_to_gps = np.array(db_index_to_gps)
valid_indices = np.where(valid_mask)[0]

db_gps_valid = db_index_to_gps[valid_indices]
db_descs_valid = db_descs[valid_indices]

print("Valid DB entries with GPS:", len(valid_indices))


# Precompute fast lookup: full_idx → local_valid_idx
full_to_valid = {full: i for i, full in enumerate(valid_indices)}


# ============================================================
# NEAREST NEIGHBOR INDEX
# ============================================================

nn_full = NearestNeighbors(n_neighbors=TOP_K, metric="cosine")
nn_full.fit(db_descs)


# ============================================================
# LOAD CAM3 QUERIES
# ============================================================

print("Collecting cam3 queries...")
queries = []

for bag in sorted(CAM3_ROOT.iterdir(), key=lambda p: int(p.name.split("_")[-1])):
    cam_folder = bag / "camera/rgb"
    gps_folder = bag / "nav/gt"
    if not cam_folder.exists():
        continue

    for img in sorted(cam_folder.iterdir(), key=lambda p: int(p.stem)):
        gps_file = gps_folder / img.with_suffix(".json").name
        if gps_file.exists():
            lat, lon = read_gps_json(gps_file)
            queries.append((img, lat, lon))

print("Total queries:", len(queries))


# ============================================================
# RETRIEVAL LOOP
# ============================================================

ref_lat, ref_lon = db_gps_valid[0]

true_cam2_xy_all = np.array(
    latlon_to_xy_m(
        db_gps_valid[:,0],
        db_gps_valid[:,1],
        ref_lat,
        ref_lon
    )
).T

query_xy_list = []
retrieved_xy_list = []
path_quer = []
path_retr = []
path_true = []
path_desc = []

print("Running retrieval...")

lg = LightGlueVisualizer(max_keypoints=2048)
dd = DinoDescriptor()
idx = 0
for img_path, qlat, qlon in tqdm(queries):

    # --- True nearest by GPS ---
    gps_dists = gps_array_to_meters(qlat, qlon, db_gps_valid)
    true_local_idx = np.argmin(gps_dists)
    true_full_idx  = valid_indices[true_local_idx]

    # --- Convert query to XY ---
    qx, qy = latlon_to_xy_m(qlat, qlon, ref_lat, ref_lon)
    query_xy_list.append((qx, qy))

    # --- Descriptor retrieval ---
    qdesc = extract_megaloc_descriptor(img_path).reshape(1, -1)
    dists, ids = nn_full.kneighbors(qdesc)

    best_match_xy = [(np.nan, np.nan)]
    best_num_matches = 0
    best_path = None
    kr_best = None
    kq_best = None
    dr_best = None
    dq_best = None
    m_scores_best = None

    for rank in range(TOP_K):

        full_idx = ids[0][rank]
        rpth = db_paths[full_idx]

        # kq, kr = lg.get_matched_keypoints(img_path, rpth)
        kq, kr, dq, dr, m_scores = lg.get_matched_keypoints_and_descriptors(
                                    img_path,
                                    rpth,
                                )
        num_matches = len(kr)

        # lg.visualize_matches(img_path, rpth, title=True)
        # plt.show()
        if num_matches > best_num_matches and full_idx in full_to_valid:

            best_num_matches = num_matches

            # Keep top retrieved
            if num_matches > MIN_NUM_MATCHES:
                local_idx = full_to_valid[full_idx]
                lat, lon = db_gps_valid[local_idx]
                rx, ry = latlon_to_xy_m(lat, lon, ref_lat, ref_lon)
                best_match_xy = [(rx, ry)]
                best_path = rpth

                kr_best = kr
                kq_best = kq
                dr_best = dr
                dq_best = dq
                m_scores_best = m_scores

    # If not enough keypoints, don't save
    if best_num_matches > MIN_NUM_MATCHES:
        # Get dino descriptors for query and best retrieved 
        descs_dino_cam2 = dd.extract_keypoint_descriptors(best_path, kr)
        descs_dino_cam3 = dd.extract_keypoint_descriptors(img_path, kq)

        # Save descriptors (LightGlue and Dino)
        file_name = f"{idx:06d}.npz"
        file_path = desc_root / file_name
        path_desc.append(file_path)
        idx += 1

        np.savez_compressed(file_path, 
                kps_lg_r=kr_best,
                kps_lg_q=kq_best,
                descs_lg_r=dr_best,
                descs_lg_q=dq_best,
                m_lg_scores=m_scores_best,
                descs_dino_r=descs_dino_cam2,
                descs_dino_q=descs_dino_cam3,
                best_path=best_path,
                best_match_xy=np.array(best_match_xy)
        )

        retrieved_xy_list.append(best_match_xy)
        path_quer.append(img_path)
        path_retr.append(best_path)
        path_true.append(db_paths[true_full_idx])

# ============================================================
# SAVE MATCHES
# ============================================================

def path_to_str(p):
    if isinstance(p, list):
        return [str(x) for x in p]
    return str(p)

json.dump(
    {
        "queries": path_to_str(path_quer),
        "retrieved": path_to_str(path_retr),
        "true": path_to_str(path_true),
        "desc_info": path_to_str(path_desc)
    },
    open("matched_img_paths_mega.json", "w"),
    indent=2
)


# ============================================================
# VISUALIZE
# ============================================================
# viz.plot_retrieval_paths(
#     true_cam2_xy_list_all,
#     query_xy_list,
#     true_cam2_xy_list,
#     retrieved_cam2_xy_list,
#     path_quer,
#     path_retr,
#     path_true,
#     num_matched_list
# )

viz.plot_all_retrieval_paths(
    true_cam2_xy_all,
    query_xy_list,
    retrieved_xy_list,
)