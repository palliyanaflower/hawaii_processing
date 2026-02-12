import json
import yaml
import math
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import json

import torch
from sklearn.neighbors import NearestNeighbors
from PIL import Image
from torchvision import transforms

from utils.lightglue_loader import LightGlueVisualizer
import utils.viz as viz

# Description: GUI for clicking through each single query match (after MegaLoc + LightGlue)

# =========================
# Config
# =========================
CAM2_ROOT = Path("data/makalii_point/processed_lidar_cam_gps/cam2")
CAM3_ROOT = Path("data/makalii_point/processed_lidar_cam_gps/cam3")
DB_FILE = Path("megaloc/results/megaloc_db_lcn_cam2.npz")

TOP_K = 5       # number MegaLoc guesses
THRESH = 1.0    # neighbor distance for accepting match
MIN_NUM_MATCHES = 100

# =========================
# Utilities: GPS <-> meters
# =========================
def gps_deg_to_meters(lat1, lon1, lat2, lon2):
    """
    Approximate conversion of lat/lon differences to meters (local equirectangular).
    Accurate enough for up to several kilometers.
    """
    # meters per degree latitude ~111320
    meters_per_deg_lat = 111320.0
    # meters per degree longitude depends on latitude
    meters_per_deg_lon = 111320.0 * math.cos(math.radians(lat1))
    dlat_m = (lat2 - lat1) * meters_per_deg_lat
    dlon_m = (lon2 - lon1) * meters_per_deg_lon
    return math.hypot(dlat_m, dlon_m)

def latlon_to_xy_m(lat, lon, lat0, lon0):
    """
    Convert lat/lon to local x,y in meters relative to reference (lat0, lon0).
    Can handle scalar or arrays.
    lat, lon: scalar or (N,) array
    Returns x, y (scalar or array)
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    lat0 = float(lat0)
    lon0 = float(lon0)

    R = 6378137.0  # WGS84 Earth radius in meters

    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    lat0_rad = np.radians(lat0)

    x = R * dlon * np.cos(lat0_rad)
    y = R * dlat

    return x, y

def gps_array_to_meters(query_lat, query_lon, db_gps_array):

    """
    Vectorized distance from one lat/lon to an array of lat/lon in meters.
    db_gps_array: (N,2) with columns [lat, lon]
    """
    x, y = latlon_to_xy_m(db_gps_array[:,0], db_gps_array[:,1], query_lat, query_lon)
    return np.sqrt(x**2 + y**2)

# =========================
# Load MegaLoc model
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
print("Loading MegaLoc model (this may take a moment)...")
model = torch.hub.load("gmberton/MegaLoc", "get_trained_model").to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def extract_descriptor(img_path):
    img = Image.open(img_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        desc = model(tensor).cpu().numpy().squeeze()
    desc = desc.astype(np.float32)
    # L2 normalize (defensive)
    norm = np.linalg.norm(desc)
    if norm > 0:
        desc /= norm
    return desc

# =========================
# Load DB descriptors + paths
# =========================
if not DB_FILE.exists():
    raise FileNotFoundError(f"DB file not found: {DB_FILE}")

db = np.load(DB_FILE, allow_pickle=True)
db_descs = db["descs"]   # (N, D)
db_paths = np.array([Path(p) for p in db["paths"]])  # array of Paths

print(f"\nLoaded DB: {len(db_paths)} images, descriptor dim = {db_descs.shape[1]}")

# =========================
# Load cam2 GPS mapping
# =========================

# Helper to read gps yaml file (YAML format has keys 'lat' and 'lon')
def read_gps_yaml(p: Path):
    with open(p, "r") as f:
        data = yaml.safe_load(f)
    return float(data["lat"]), float(data["lon"])

def read_gps_json(p: Path):
    with open(p, "r") as f:
        data = json.load(f)

    try:
        lat = float(data["lat"])
        lon = float(data["lon"])
    except KeyError as e:
        raise KeyError(f"Missing key {e} in GPS file: {p}")

    return lat, lon

# For speed, build a mapping from path string -> gps (lat,lon) by scanning cam2 bags
print("Scanning cam2 bags to build GPS mapping...")
cam2_gps_map = {}  # mapping: cam_file_abs_path -> (lat, lon)

for bag in sorted(d for d in CAM2_ROOT.iterdir() if d.is_dir()):
    gps_folder = bag / "nav/gps"
    cam_folder = bag / "camera/rgb"

    if not gps_folder.exists() or not cam_folder.exists():
        print("Folder does not exist")
        continue

    for cam_file in sorted(cam_folder.iterdir()):
        if not cam_file.is_file():
            print("Cam file does not exist")
            continue

        # assume same filename, already synced
        gps_file = gps_folder / cam_file.with_suffix(".json").name

        if not gps_file.exists():
            print("Gps file does not exist")
            continue

        cam2_gps_map[str(cam_file.resolve())] = read_gps_json(gps_file)

# Now align gps to db_paths order
# We'll build: db_index -> (lat, lon) and also a numpy array aligned with db_descs
missing_gps = 0
db_index_to_gps = [None] * len(db_paths)
for i, p in enumerate(db_paths):
    p_abs = str(p.resolve())
    if p_abs in cam2_gps_map:
        db_index_to_gps[i] = cam2_gps_map[p_abs]
    else:
        # Sometimes DB paths might be relative vs absolute; try matching by suffix (bag/.../camera/NN.png)
        found = False
        for k in cam2_gps_map.keys():
            if str(p) in k or k.endswith(str(p)):
                db_index_to_gps[i] = cam2_gps_map[k]
                found = True
                break
        if not found:
            missing_gps += 1
            db_index_to_gps[i] = (np.nan, np.nan)

if missing_gps > 0:
    print(f"\nWarning: {missing_gps} DB entries have no GPS mapping (they'll be ignored in GPS-based computations).")

# Make numpy array with only the DB entries that have valid GPS
valid_indices = [i for i, g in enumerate(db_index_to_gps) if not (math.isnan(g[0]) or math.isnan(g[1]))]
db_gps_arr = np.array([db_index_to_gps[i] for i in valid_indices], dtype=np.float64)  # shape (M,2)
db_descs_valid = db_descs[valid_indices]  # descriptors of entries that have GPS

# Build NN on full DB descriptors (for retrieval)
nn_full = NearestNeighbors(n_neighbors=TOP_K, metric="cosine")
nn_full.fit(db_descs)

# Also build NN on GPS positions to find true nearest in cam2 efficiently
# We'll use db_gps_arr for GPS nearest lookup (kdtree-like via brute-force vectorized search)
print("DB GPS array shape:", db_gps_arr.shape)

# =========================
# Load cam3 queries (all bags)
# =========================
print("\nCollecting cam3 queries (images + gps)...")
queries = []  # list of dicts: {img_path, lat, lon}
for bag in sorted([d for d in CAM3_ROOT.iterdir() if d.is_dir()]):
    cam_folder = bag / "camera/rgb"
    gps_folder = bag / "nav/gps"

    if not gps_folder.exists() or not cam_folder.exists():
        continue

    for cam_file in sorted(cam_folder.iterdir()):
        if not cam_file.is_file():
            continue

        # assume same filename, possibly different extension
        gps_file = gps_folder / cam_file.with_suffix(".json").name

        lat, lon = read_gps_json(gps_file)
        queries.append({"img": cam_file, "lat": lat, "lon": lon, "bag": bag.name})

print(f"Total cam3 queries found: {len(queries)}")
if len(queries) == 0:
    raise RuntimeError("No cam3 queries found. Check CAM3_ROOT path and expected structure.")

# ---------------------------
# Get true GPS pose for cam2
# ---------------------------
# Pick a reference lat lon at the first cam3 query GPS
ref_lat = float(queries[0]["lat"])
ref_lon = float(queries[0]["lon"])

# Convert true cam2 gps → xy meters
true_cam2_xy_list_all = []
for true_lat, true_lon in db_gps_arr:
    tx, ty = latlon_to_xy_m(true_lat, true_lon, ref_lat, ref_lon)
    true_cam2_xy_list_all.append((tx, ty))
true_cam2_xy_list_all = np.array(true_cam2_xy_list_all)


# =========================
# Get LightGlue Best Match
# =========================
query_xy_list = []
true_cam2_xy_list = []
retrieved_cam2_xy_list = []
num_matched_list = []

path_quer = []
path_true = []
path_retr = []

print("Running retrieval for each query (this may take time if many queries and GPU is used for descriptors)...")
for q in tqdm(queries): # progress bar
# for i in range(5):
    # q = queries[i]
    qpth = q["img"]
    qlat = q["lat"]
    qlon = q["lon"]

    # -----------------------------
    # Get true poses / correspondences
    # -----------------------------
    # compute true nearest cam2 by GPS (meters) using db_gps_arr which corresponds to valid_indices
    # first compute distances to valid entries
    gps_dists_m = gps_array_to_meters(qlat, qlon, db_gps_arr)  # shape (M,)
    true_local_idx = int(np.argmin(gps_dists_m))
    true_db_index = valid_indices[true_local_idx]  # index into full db_paths
    true_distance_m = float(gps_dists_m[true_local_idx])
    true_lat, true_lon = db_gps_arr[true_local_idx]

    # -----------------------------
    # Get GPS info
    # -----------------------------
    # convert cam3 query gps → xy meters
    qx, qy = latlon_to_xy_m(qlat, qlon, ref_lat, ref_lon)
    query_xy_list.append((qx, qy))

    # convert true cam2 gps → xy meters
    tx, ty = latlon_to_xy_m(true_lat, true_lon, ref_lat, ref_lon)
    true_cam2_xy_list.append((tx, ty))

    # -----------------------------
    # Retrieve top-K by MegaLoc descriptor
    # -----------------------------
    qdesc = extract_descriptor(qpth).reshape(1, -1)
    dists, ids = nn_full.kneighbors(qdesc, n_neighbors=TOP_K)
    topk_ids = ids[0].tolist()  # indices into db_paths (full DB)
    topk_dists = dists[0].tolist()

    # ---Get top k matches for each query---
    # compute retrieval error in meters: distance between retrieved top-1 gps and true gps
    # but first get gps for top-1 (if gps exists)
    retrieved_k = [(np.nan, np.nan)]
    path_retr_temp = [""]
    highest_num_matched = 0
    for i in range(TOP_K):
        # -------------------------
        # Get LightGlue Matches
        # -------------------------
        rpth =  db_paths[topk_ids[i]]   # Path to retrieved image
        vis = LightGlueVisualizer()
        m_kpxs_cam3, m_kpxs_cam2 = vis.get_matched_keypoints(
            qpth,
            rpth,
        )

        num_matched = len(m_kpxs_cam2)

        # print("num matched points", len(m_kpxs_cam2), len(m_kpxs_cam3))

        # -------------------------
        # Save info
        # -------------------------
        retrieved_idx = topk_ids[i]
        retrieved_dist = topk_dists[i]
        if retrieved_idx in valid_indices and retrieved_dist < THRESH:
            # map retrieved_idx to position in db_gps_arr
            retrieved_local_pos = valid_indices.index(retrieved_idx)
            retrieved_lat, retrieved_lon = db_gps_arr[retrieved_local_pos]
            retrieval_error_m = gps_deg_to_meters(retrieved_lat, retrieved_lon, qlat, qlon)
            # true error between retrieved and true location:
            true_match_lat, true_match_lon = db_gps_arr[true_local_idx]
            retrieved_vs_true_error_m = gps_deg_to_meters(retrieved_lat, retrieved_lon, true_match_lat, true_match_lon)
            # distance between gps retrieved index and true index
            x_r, y_r = latlon_to_xy_m(retrieved_lat, retrieved_lon, ref_lat, ref_lon)
            x_q, y_q = latlon_to_xy_m(true_lat, true_lon, ref_lat, ref_lon)
            dist_true_retr = np.sqrt((x_q - x_r)**2 + (y_q - y_r)**2)
        else:
            # retrieved has no GPS; set NaNs
            retrieval_error_m = float("nan")
            retrieved_vs_true_error_m = float("nan")

        # ----------------------------------------------
        # Keep reference with highest LightGlue matches
        # ----------------------------------------------
        if num_matched > highest_num_matched:
            highest_num_matched = num_matched

            if num_matched > MIN_NUM_MATCHES:
                rx, ry = latlon_to_xy_m(retrieved_lat, retrieved_lon, ref_lat, ref_lon)
                retrieved_k = [(rx, ry)]
                path_retr_temp = [rpth]


    retrieved_cam2_xy_list.append(retrieved_k)
    num_matched_list.append(highest_num_matched)

    # Get image paths
    path_quer.append(qpth)
    path_retr.append(path_retr_temp)
    path_true.append(db_paths[true_local_idx])

def path_to_str(p):
    if isinstance(p, (list, tuple)):
        return [path_to_str(x) for x in p]
    return str(p)

data = {
    "queries": path_to_str(path_quer),
    "retrieved": path_to_str(path_retr),
    "true": path_to_str(path_true),
}
with open("matched_img_paths.json", "w") as f:
    json.dump(data, f, indent=2)
# =========================
# Visualize Results
# =========================
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
    true_cam2_xy_list_all,
    query_xy_list,
    retrieved_cam2_xy_list,
)