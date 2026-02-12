# #!/usr/bin/env python3
"""
evaluate_megaloc_retrieval.py

Evaluates MegaLoc retrieval: cam3 queries -> cam2 database.

Outputs:
 - results/retrieval_eval_cam3_vs_cam2.csv
 - prints Recall@1, Recall@5, mean/median error, percentiles
 - plots: histogram + CDF
 - sample good/bad visualizations (matplotlib)
"""

import os
import json
import yaml
import math
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from sklearn.neighbors import NearestNeighbors
from PIL import Image
from torchvision import transforms

# Description: A single static image showing the query and reference matches

# -------------------------
# Config
# -------------------------
CAM2_ROOT = Path("../data/makalii_point/processed_data_imggps/cam2")
CAM3_ROOT = Path("../data/makalii_point/processed_data_imggps/cam3")
DB_FILE = Path("results/megaloc_db_cam2.npz")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 5   # compute Recall@1 and Recall@5
EXAMPLE_SHOW = 6  # number of good/bad example triplets to show

# -------------------------
# Plot helper
# -------------------------
def plot_retrieval_paths(
    true_cam2_gps_xy_all,
    query_gps_xy,         # list of (x,y)
    true_cam2_gps_xy,     # list of (x,y)
    retrieved_cam2_gps_xy # list of (x,y)
):
    """
    Plot XY paths for:
    - cam3 true
    - cam2 true
    - cam2 retrieved (from MegaLoc)
    Draw lines from each retrieved cam2 point to the true cam2 point.
    All arguments must be lists of (x,y) in meters.
    """

    query_gps_xy      = np.array(query_gps_xy)
    true_cam2_gps_xy  = np.array(true_cam2_gps_xy)
    retrieved_cam2_xy = np.array(retrieved_cam2_gps_xy)

    plt.figure(figsize=(10,10))

    # Plot true cam3 path
    plt.scatter(
        query_gps_xy[:,0], query_gps_xy[:,1],
        label="cam3 (query) true path",
        s=40,
    )

    # Plot true cam2 path
    plt.scatter(
        true_cam2_gps_xy_all[:,0], true_cam2_gps_xy_all[:,1],
        label="cam2 true path",
        s=40,
        marker='*'
    )

    # Plot retrieved cam2 path (sparse)
    plt.scatter(
        retrieved_cam2_xy[:,0], retrieved_cam2_xy[:,1],
        label="cam2 retrieved path",
        s=40,
        alpha=0.5
        # color="orange"
    )

    # Draw lines between retrieved and true cam2 points
    for (rx, ry), (tx, ty) in zip(retrieved_cam2_xy, query_gps_xy):
        # skip NaN points
        if np.isnan(rx) or np.isnan(ry):
            continue
        plt.plot([rx, tx], [ry, ty], color='red', linestyle='--', linewidth=1)
    for (rx, ry), (tx, ty) in zip(true_cam2_gps_xy, query_gps_xy):
        # skip NaN points
        if np.isnan(rx) or np.isnan(ry):
            continue
        plt.plot([rx, tx], [ry, ty], color='red', linestyle='--', linewidth=1, alpha=0.3)

    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.title("Retrieval Evaluation — XY Paths in Meters")
    plt.legend()
    # plt.axis("equal")
    plt.xlim(-300,200)
    plt.grid(True)
    plt.show()


# -------------------------
# Utilities: GPS <-> meters
# -------------------------
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

# -------------------------
# Load MegaLoc model
# -------------------------
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

# -------------------------
# Load DB descriptors + paths
# -------------------------
if not DB_FILE.exists():
    raise FileNotFoundError(f"DB file not found: {DB_FILE}")

db = np.load(DB_FILE, allow_pickle=True)
db_descs = db["descs"]   # (N, D)
db_paths = np.array([Path(p) for p in db["paths"]])  # array of Paths

print(f"\nLoaded DB: {len(db_paths)} images, descriptor dim = {db_descs.shape[1]}")

# -------------------------
# Load cam2 GPS mapping (from matches.json/gps/*.yaml)
# -------------------------
# We'll build: db_index -> (lat, lon) and also a numpy array aligned with db_descs
db_index_to_gps = [None] * len(db_paths)

# Helper to read gps yaml file (YAML format has keys 'lat' and 'lon')
def read_gps_yaml(p: Path):
    with open(p, "r") as f:
        data = yaml.safe_load(f)
    return float(data["lat"]), float(data["lon"])

# For speed, build a mapping from path string -> gps (lat,lon) by scanning cam2 bags
print("Scanning cam2 bags to build GPS mapping...")
cam2_gps_map = {}   # mapping of absolute path str -> (lat,lon)
for bag in sorted([d for d in CAM2_ROOT.iterdir() if d.is_dir()]):
    matches_json = bag / "matches.json"
    gps_folder = bag / "gps"
    cam_folder = bag / "camera"
    # load matches.json if exists (helps if you want to use those)
    if matches_json.exists():
        with open(matches_json, "r") as f:
            matches = json.load(f)
        for m in matches:
            cam_file = cam_folder / m["camera_file"]
            gps_file = gps_folder / m["gps_file"]
            if cam_file.exists() and gps_file.exists():
                cam2_gps_map[str(cam_file.resolve())] = read_gps_yaml(gps_file)

    else:
        # fallback: try pairing by index 0.yaml -> camera/0.png etc.
        if gps_folder.exists() and cam_folder.exists():
            for g in sorted([p for p in gps_folder.iterdir() if p.suffix in (".yaml", ".yml")]):
                try:
                    idx = g.stem
                    cam_file = cam_folder / f"{idx}.png"
                    if cam_file.exists():
                        cam2_gps_map[str(cam_file.resolve())] = read_gps_yaml(g)
                except Exception:
                    continue

# Now align gps to db_paths order
missing_gps = 0
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
    print(f"Warning: {missing_gps} DB entries have no GPS mapping (they'll be ignored in GPS-based computations).")

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

# -------------------------
# Load cam3 queries (all bags)
# -------------------------
print("Collecting cam3 queries (images + gps)...")
queries = []  # list of dicts: {img_path, lat, lon}

# bags = [Path("../data/makalii_point/processed_data_imggps/cam3/bag_camera_3_2025_08_13-01_35_58_46"),
#         Path("../data/makalii_point/processed_data_imggps/cam3/bag_camera_3_2025_08_13-01_35_58_47")]
# for bag in bags:
for bag in sorted([d for d in CAM3_ROOT.iterdir() if d.is_dir()]):
    matches_json = bag / "matches.json"
    cam_folder = bag / "camera"
    gps_folder = bag / "gps"

    if matches_json.exists():
        with open(matches_json, "r") as f:
            matches = json.load(f)
        for m in matches:
            img_p = cam_folder / m["camera_file"]
            gps_p = gps_folder / m["gps_file"]
            if img_p.exists() and gps_p.exists():
                lat, lon = read_gps_yaml(gps_p)
                queries.append({"img": img_p, "lat": lat, "lon": lon, "bag": bag.name})
    else:
        # fallback: pair by index
        if cam_folder.exists() and gps_folder.exists():
            for img_p in sorted([p for p in cam_folder.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")]):
                idx = img_p.stem
                gps_p = gps_folder / f"{idx}.yaml"
                if gps_p.exists():
                    lat, lon = read_gps_yaml(gps_p)
                    queries.append({"img": img_p, "lat": lat, "lon": lon, "bag": bag.name})

print(f"Total cam3 queries found: {len(queries)}")
if len(queries) == 0:
    raise RuntimeError("No cam3 queries found. Check CAM3_ROOT path and expected structure.")

# Pick a reference lat lon
# Reference origin at the first cam3 query GPS
ref_lat = float(queries[0]["lat"])
ref_lon = float(queries[0]["lon"])

# convert true cam2 gps → xy meters
true_cam2_xy_list_all = []
for true_lat, true_lon in db_gps_arr:
    tx, ty = latlon_to_xy_m(true_lat, true_lon, ref_lat, ref_lon)
    true_cam2_xy_list_all.append((tx, ty))
true_cam2_xy_list_all = np.array(true_cam2_xy_list_all)

# -------------------------
# Evaluate retrievals
# -------------------------
# For eval metrics
results = []
recall1_count = 0
recall5_count = 0
in_xm_count = 0
errors_m = []
X_METERS = 30

# For plotting gps
query_xy_list = []         # cam3 true gps
true_cam2_xy_list = []
retrieved_cam2_xy_list = []# MegaLoc retrieved cam2 gps

print("Running retrieval for each query (this may take time if many queries and GPU is used for descriptors)...")
for q in tqdm(queries):
    qimg = q["img"]
    qlat = q["lat"]
    qlon = q["lon"]

    # descriptor
    qdesc = extract_descriptor(qimg).reshape(1, -1)

    # retrieve top-K by descriptor
    dists, ids = nn_full.kneighbors(qdesc, n_neighbors=TOP_K)
    topk_ids = ids[0].tolist()  # indices into db_paths (full DB)
    topk_dists = dists[0].tolist()

    # compute true nearest cam2 by GPS (meters) using db_gps_arr which corresponds to valid_indices
    # first compute distances to valid entries
    gps_dists_m = gps_array_to_meters(qlat, qlon, db_gps_arr)  # shape (M,)
    true_local_idx = int(np.argmin(gps_dists_m))
    true_db_index = valid_indices[true_local_idx]  # index into full db_paths
    true_distance_m = float(gps_dists_m[true_local_idx])

    true_lat, true_lon = db_gps_arr[true_local_idx]


    # check if true_db_index is in top-k results
    in_topk = true_db_index in topk_ids
    in_top1 = (topk_ids[0] == true_db_index)

    if in_top1:
        recall1_count += 1
    if in_topk:
        recall5_count += 1

    # compute retrieval error in meters: distance between retrieved top-1 gps and true gps
    # but first get gps for top-1 (if gps exists)
    retrieved_idx = topk_ids[0]
    retrieved_gps = None
    if retrieved_idx in valid_indices:
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
        in_xm = (dist_true_retr <= X_METERS)
        if in_xm:
            in_xm_count += 1
    else:
        # retrieved has no GPS; set NaNs
        retrieval_error_m = float("nan")
        retrieved_vs_true_error_m = float("nan")
        print("Not valid gps")
        exit()



    errors_m.append(retrieved_vs_true_error_m if not math.isnan(retrieved_vs_true_error_m) else 1e9)

    results.append({
        "query_img": str(qimg),
        "query_bag": q["bag"],
        "query_lat": float(qlat),
        "query_lon": float(qlon),
        "true_db_index": int(true_db_index),
        "true_db_path": str(db_paths[true_db_index]),
        "true_db_distance_m": float(true_distance_m),
        "retrieved_index_top1": int(retrieved_idx),
        "retrieved_path_top1": str(db_paths[retrieved_idx]),
        "retrieval_rank_of_true": int(topk_ids.index(true_db_index)) if true_db_index in topk_ids else -1,
        "retrieved_vs_true_error_m": float(retrieved_vs_true_error_m),
        "topk_dists": topk_dists,
        "in_xm": bool(in_xm),
        "X_METERS": float(X_METERS),

    })

    # convert cam3 query gps → xy meters
    qx, qy = latlon_to_xy_m(qlat, qlon, ref_lat, ref_lon)
    query_xy_list.append((qx, qy))

    # convert true cam2 gps → xy meters
    tx, ty = latlon_to_xy_m(true_lat, true_lon, ref_lat, ref_lon)
    true_cam2_xy_list.append((tx, ty))

    # convert retrieved cam2 gps → xy meters
    if not math.isnan(retrieved_vs_true_error_m):
        rx, ry = latlon_to_xy_m(retrieved_lat, retrieved_lon, ref_lat, ref_lon)
        retrieved_cam2_xy_list.append((rx, ry))
    else:
        retrieved_cam2_xy_list.append((np.nan, np.nan))


# -------------------------
# Compute metrics
# -------------------------
Nq = len(results)
recall1 = recall1_count / Nq
recall5 = recall5_count / Nq
recallXm = in_xm_count / Nq

errors_m = np.array([r["retrieved_vs_true_error_m"] for r in results])
# filter out extreme NaNs (we set missing to 1e9) and ignore those if very large
valid_errors = errors_m[np.isfinite(errors_m) & (errors_m < 1e8)]

mean_err = float(np.mean(valid_errors))
median_err = float(np.median(valid_errors))
percentiles = np.percentile(valid_errors, [25,50,75,90,95]).tolist()

print("\n=========== Retrieval Evaluation Summary ===========")
print(f"Queries: {Nq}")
print(f"Recall@1: {recall1_count}/{Nq} = {recall1*100:.2f}%")
print(f"Recall@{TOP_K}: {recall5_count}/{Nq} = {recall5*100:.2f}%")
print(f"Within {X_METERS}m: {in_xm_count}/{Nq} = {recallXm*100:.2f}%")
print(f"Mean retrieved_vs_true error: {mean_err:.2f} m")
print(f"Median error: {median_err:.2f} m")
print("Percentiles (25,50,75,90,95):", [f"{p:.2f}" for p in percentiles])
print("=====================================================\n")

# Save CSV of results
csv_file = OUTPUT_DIR / "retrieval_eval_cam3_vs_cam2.csv"
with open(csv_file, "w", newline="") as f:
    fieldnames = list(results[0].keys())
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        writer.writerow(r)
print(f"Saved per-query results CSV: {csv_file}")


# -------------------------
# Plots: histogram + CDF
# -------------------------
plot_retrieval_paths(
    true_cam2_xy_list_all,
    query_xy_list,
    true_cam2_xy_list,
    retrieved_cam2_xy_list
)

exit()

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(valid_errors, bins=50)
plt.title("Histogram of retrieved_vs_true error (m)")
plt.xlabel("meters")
plt.ylabel("count")

plt.subplot(1,2,2)
vals = np.sort(valid_errors)
cdf = np.arange(1, len(vals)+1) / len(vals)
plt.plot(vals, cdf)
plt.title("CDF of retrieved_vs_true error")
plt.xlabel("meters")
plt.ylabel("CDF")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "retrieval_error_hist_cdf.png")
print(f"Saved plot: {OUTPUT_DIR/'retrieval_error_hist_cdf.png'}")
plt.show()

# -------------------------
# Show some example triplets: good vs bad
# -------------------------
# sort results by retrieved_vs_true_error_m
sorted_results = sorted(results, key=lambda r: r["retrieved_vs_true_error_m"] if r["retrieved_vs_true_error_m"] >= 0 else 1e9)
good = [r for r in sorted_results if r["retrieved_vs_true_error_m"] < 5][:EXAMPLE_SHOW]   # < 5m
bad = [r for r in sorted_results if r["retrieved_vs_true_error_m"] > 20][:EXAMPLE_SHOW]   # > 20m

def show_triplets(example_list, title):
    n = len(example_list)
    if n == 0:
        print(f"No {title} examples to show.")
        return
    plt.figure(figsize=(9, 3*n))
    for i, r in enumerate(example_list):
        qimg = cv2.imread(r["query_img"])[:,:,::-1]
        retimg = cv2.imread(r["retrieved_path_top1"])[:,:,::-1]
        trueimg = cv2.imread(r["true_db_path"])[:,:,::-1]
        # Row: query | retrieved | true
        ax_q = plt.subplot(n, 3, 3*i+1)
        ax_q.imshow(qimg); ax_q.axis("off"); ax_q.set_title("Query")
        ax_r = plt.subplot(n, 3, 3*i+2)
        ax_r.imshow(retimg); ax_r.axis("off"); ax_r.set_title(f"Retrieved\nerr={r['retrieved_vs_true_error_m']:.1f}m")
        ax_t = plt.subplot(n, 3, 3*i+3)
        ax_t.imshow(trueimg); ax_t.axis("off"); ax_t.set_title("True")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

try:
    import cv2
    show_triplets(good, "Example GOOD retrievals (retrieved close to true <5m)")
    show_triplets(bad, "Example BAD retrievals (retrieved far from true >20m)")
except Exception as e:
    print("Could not show example images (cv2 error).", e)

print("Done.")
