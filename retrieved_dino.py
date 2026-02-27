import json
from pathlib import Path

import numpy as np
from tqdm import tqdm
import torch
from sklearn.neighbors import NearestNeighbors
from PIL import Image
from torchvision import transforms

from utils.lightglue_loader import LightGlueVisualizer
import utils.viz as viz
import matplotlib.pyplot as plt
from dino.dinov2_class import DinoDescriptor  # <-- DINOv2-only wrapper

# ============================================================
# CONFIG
# ============================================================
CAM2_ROOT = Path("data/makalii_point/processed_lidar_cam_gps/cam2")
CAM3_ROOT = Path("data/makalii_point/processed_lidar_cam_gps/cam3")
DB_FILE   = Path("dino/results/dino_db_lcn_cam2.npz")

TOP_K = 3
THRESH = 1.0
MIN_NUM_MATCHES = 30


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
# LOAD DINO
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = DinoDescriptor(device=device, model_type="vits14")  # use vits14 faster, vitb14 for better performance

def extract_dinov2_descriptor(img_path, pooling="avg"):
    """
    Extract global descriptor for a single image using DINOv2-only.
    """
    desc = model.extract_descriptor(img_path, pooling=pooling)
    # L2 normalize (already normalized in class, but safe to double-check)
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

print("Running retrieval...")

for img_path, qlat, qlon in tqdm(queries):

    # --- True nearest by GPS ---
    gps_dists = gps_array_to_meters(qlat, qlon, db_gps_valid)
    true_local_idx = np.argmin(gps_dists)
    true_full_idx  = valid_indices[true_local_idx]

    # --- Convert query to XY ---
    qx, qy = latlon_to_xy_m(qlat, qlon, ref_lat, ref_lon)
    query_xy_list.append((qx, qy))

    # --- Descriptor retrieval ---
    qdesc = extract_dinov2_descriptor(img_path).reshape(1, -1)
    dists, ids = nn_full.kneighbors(qdesc)

    best_match_xy = [(np.nan, np.nan)]
    best_num_matches = 0
    best_path = None

    for rank in range(TOP_K):

        full_idx = ids[0][rank]
        rpth = db_paths[full_idx]

        vis = LightGlueVisualizer()
        kq, kr = vis.get_matched_keypoints(img_path, rpth)
        num_matches = len(kr)

        # vis.visualize_matches(img_path, rpth, title=True)
        # plt.show()
        if num_matches > best_num_matches and full_idx in full_to_valid:

            best_num_matches = num_matches

            if num_matches > MIN_NUM_MATCHES:
                local_idx = full_to_valid[full_idx]
                lat, lon = db_gps_valid[local_idx]
                rx, ry = latlon_to_xy_m(lat, lon, ref_lat, ref_lon)
                best_match_xy = [(rx, ry)]
                best_path = rpth

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
    },
    open("matched_img_paths_dino.json", "w"),
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