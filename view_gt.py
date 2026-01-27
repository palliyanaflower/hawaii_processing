import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import re

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def quaternion_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return np.arctan2(siny_cosp, cosy_cosp)


def plot_gt_pixels_with_heading_and_cameras(
    gt_dir,
    map_image_path,
    ref_pixel,
    meters_per_pixel,
    heading_length_m=1.0,
    camera_length_m=0.6,
    point_size=8,
    arrow_width=0.002,
    figsize=(10, 10),
):
    """
    Plot GT (x,y) poses and heading / left / right camera directions
    converted to pixels and overlaid on a satellite image.
    """
    gt_dir = Path(gt_dir)
    files = sorted(gt_dir.glob("*.json"))

    if not files:
        raise RuntimeError(f"No .json files found in {gt_dir}")

    xs_px, ys_px = [], []
    hx_px, hy_px = [], []
    lx_px, ly_px = [], []
    rx_px, ry_px = [], []

    for f in files:
        with open(f, "r") as jf:
            d = json.load(jf)

        # --- Position in meters (ENU) ---
        x_m = d["position"]["x"]
        y_m = d["position"]["y"]

        # meters → pixels
        x_px = x_m / meters_per_pixel + ref_pixel[0]
        y_px = -y_m / meters_per_pixel + ref_pixel[1]

        # --- Orientation ---
        q = d["orientation"]
        yaw = quaternion_to_yaw(
            q["qx"], q["qy"], q["qz"], q["qw"]
        )

        # Heading (ENU)
        hdx = np.cos(yaw)
        hdy = np.sin(yaw)

        # Left / Right (±90°)
        ldx = -np.sin(yaw)
        ldy =  np.cos(yaw)

        rdx =  np.sin(yaw)
        rdy = -np.cos(yaw)

        # Convert direction vectors to pixel space
        hx_px.append( hdx * heading_length_m / meters_per_pixel)
        hy_px.append(-hdy * heading_length_m / meters_per_pixel)

        lx_px.append( ldx * camera_length_m / meters_per_pixel)
        ly_px.append(-ldy * camera_length_m / meters_per_pixel)

        rx_px.append( rdx * camera_length_m / meters_per_pixel)
        ry_px.append(-rdy * camera_length_m / meters_per_pixel)

        xs_px.append(x_px)
        ys_px.append(y_px)

    xs_px = np.array(xs_px)
    ys_px = np.array(ys_px)

    # --- Plot ---
    # img = Image.open(map_image_path)
    # plt.figure(figsize=figsize)
    # plt.imshow(img)

    # Positions
    plt.scatter(xs_px, ys_px, s=point_size, color="black", label="GT position")

    # Heading arrows
    plt.quiver(
        xs_px, ys_px,
        hx_px, hy_px,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=arrow_width,
        color="red",
        label="Heading",
    )

    # Left camera arrows
    plt.quiver(
        xs_px, ys_px,
        lx_px, ly_px,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=arrow_width,
        color="blue",
        label="Left camera",
    )

    # Right camera arrows
    plt.quiver(
        xs_px, ys_px,
        rx_px, ry_px,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=arrow_width,
        color="green",
        label="Right camera",
    )


def plot_points_with_heading_and_cameras(
    gt_dir,
    heading_length=10.0,
    camera_length=5.0,
    point_size=20,
    arrow_width=0.002,
    figsize=(8, 8),
):
    """
    Plot each GT pose as:
      • point at (x, y)
      • heading arrow
      • left (+90°) and right (-90°) camera direction arrows
    """
    gt_dir = Path(gt_dir)
    files = sorted(gt_dir.glob("*.json"))

    if not files:
        raise RuntimeError(f"No .json files found in {gt_dir}")

    xs, ys = [], []
    hx, hy = [], []
    lx, ly = [], []
    rx, ry = [], []

    for f in files:
        with open(f, "r") as jf:
            d = json.load(jf)

        x = d["position"]["x"]
        y = d["position"]["y"]

        q = d["orientation"]
        yaw = quaternion_to_yaw(
            q["qx"], q["qy"], q["qz"], q["qw"]
        )

        # Heading
        hdx = np.cos(yaw)
        hdy = np.sin(yaw)

        # Left / Right (±90 deg)
        ldx = -np.sin(yaw)
        ldy =  np.cos(yaw)

        rdx =  np.sin(yaw)
        rdy = -np.cos(yaw)

        xs.append(x)
        ys.append(y)

        hx.append(hdx)
        hy.append(hdy)

        lx.append(ldx)
        ly.append(ldy)

        rx.append(rdx)
        ry.append(rdy)

    xs = np.array(xs)
    ys = np.array(ys)

    # --- Plot ---
    # plt.figure(figsize=figsize)

    # Positions
    plt.scatter(xs, ys, s=point_size, color="black", label="Position")

    # Heading arrows
    plt.quiver(
        xs, ys, hx, hy,
        angles="xy",
        scale_units="xy",
        scale=1.0 / heading_length,
        width=arrow_width,
        color="red",
        label="Heading",
    )

    # Left camera arrows
    plt.quiver(
        xs, ys, lx, ly,
        angles="xy",
        scale_units="xy",
        scale=1.0 / camera_length,
        width=arrow_width,
        color="blue",
        label="Left camera",
    )

    # Right camera arrows
    plt.quiver(
        xs, ys, rx, ry,
        angles="xy",
        scale_units="xy",
        scale=1.0 / camera_length,
        width=arrow_width,
        color="green",
        label="Right camera",
    )

    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("X [m] (East)")
    plt.ylabel("Y [m] (North)")
    plt.title("GT Position with Heading and Camera Directions")
    # plt.legend()
    plt.tight_layout()
    plt.show()


# --- Helper: natural sort key (extracts numeric index from filename) ---
def extract_index(path):
    # Extract the last number in the filename (e.g., _12_ from ..._12_gps.npy)
    match = re.search(r"_(\d+)_gps", path.stem)
    return int(match.group(1)) if match else -1


def latlon2pixels_segmented(gps_arrays_dir, map_image_path, ref_latlon, ref_pixel, meters_per_pixel=0.32):
    """
    Plot segmented GPS (lat, lon) trajectories on a map image, converting them to pixels.

    Parameters
    ----------
    gps_arrays_dir : str or Path
        Directory containing .npy files with GPS arrays (each Nx3: lat, lon, alt).
    map_image_path : str
        Path to the background map image.
    ref_latlon : tuple
        Reference latitude/longitude (lat, lon) corresponding to ref_pixel.
    ref_pixel : tuple
        Pixel coordinates (x, y) corresponding to ref_latlon on the map.
    meters_per_pixel : float
        Conversion scale from meters to pixels (default: 0.32).
    """
    gps_arrays_dir = Path(gps_arrays_dir)

    # --- Load and sort .npy files numerically ---
    def extract_index(path):
        import re
        match = re.search(r"_(\d+)_gps", path.stem)
        return int(match.group(1)) if match else -1

    files = sorted(gps_arrays_dir.glob("*.npy"), key=extract_index)
    if not files:
        raise RuntimeError(f"No .npy files found in {gps_arrays_dir}")

    # --- Conversion constants ---
    R = 6378137.0  # Earth radius (m)
    lat0 = np.deg2rad(ref_latlon[0])

    def gps_to_meters(latlon):
        """Convert lat/lon to local Cartesian coordinates (meters)."""
        lat = np.deg2rad(latlon[:, 0])
        lon = np.deg2rad(latlon[:, 1])
        dlat = (lat - np.deg2rad(ref_latlon[0])) * R
        dlon = (lon - np.deg2rad(ref_latlon[1])) * R * np.cos(lat0)
        return np.column_stack((dlon, dlat))

    # --- Prepare color map ---
    colors = plt.cm.tab20(np.arange(len(files)) % 20 / 20)

    # --- Load map ---
    img = Image.open(map_image_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)

    # --- Plot each segment ---
    print("Index to Bag")
    for i, f in enumerate(files):
        data = np.load(f)
        if data.shape[1] < 2:
            continue

        robot_latlon = data[:, :2]
        robot_xy = gps_to_meters(robot_latlon)
        robot_offsets = robot_xy / meters_per_pixel
        robot_offsets[:, 1] = -robot_offsets[:, 1]
        robot_pixels = robot_offsets + np.array(ref_pixel)

        color = colors[i]
        plt.plot(robot_pixels[:, 0], robot_pixels[:, 1], '-', color=color, linewidth=2)
        plt.text(robot_pixels[0, 0], robot_pixels[0, 1], str(i), fontsize=10, color=color, weight='bold')

        print(f"{i}: {f.stem}")

    # --- Mark reference pixel ---
    plt.scatter(*ref_pixel, color='blue', s=80, marker="*", label="Start")
    plt.scatter(*robot_pixels[-1], color='blue', s=60, label="End")

    # plt.legend(fontsize=8)
    plt.axis('off')
    # plt.title("Segmented GPS Trajectories")
    # plt.savefig(gps_arrays_dir.parent / "gps_segments_pixels.png", dpi=300)
    # plt.savefig(gps_arrays_dir.parent / "gps_segments_pixels.pdf")
    # plt.show()

# --- User settings ---
base = "data/makalii_point"
gps_arrays_dir = Path(base + "/gps_arrays")
map_image_path = base + "/satellite.png"
save_fig = True  # set False to skip saving figure

params = {}
with open(base + "/plot_satellite_params.txt", "r") as f:
    for line in f:
        key, value = line.strip().split("=")
        params[key] = float(value)

ref_pixel = (params["x_px"], params["y_px"])
ref_latlon = np.array([params["ref_lat"], params["ref_lon"]])
meters_per_pixel = params["meters_per_pixel"]

# print(ref_pixel, ref_latlon, meters_per_pixel)

latlon2pixels_segmented(gps_arrays_dir, map_image_path, ref_latlon, ref_pixel, meters_per_pixel)

for cam_num in [2, 3]:
    data_root = Path("data/makalii_point/processed_lidar_cam_gps/cam" + str(cam_num))
    # Iterate over all camera bags in processed folder
    for camera_bag_dir in data_root.iterdir():
        if not camera_bag_dir.is_dir():
            continue

        # Ground truth folder inside nav/gt
        gt_dir = camera_bag_dir / "nav" / "gt"
        if not gt_dir.exists():
            print(f"No ground truth for {camera_bag_dir.name}, skipping.")
            continue

        print(f"Plotting GT for {camera_bag_dir.name}")
        plot_gt_pixels_with_heading_and_cameras(
            gt_dir=gt_dir,
            map_image_path=map_image_path,
            ref_pixel=ref_pixel,
            meters_per_pixel=meters_per_pixel,
            heading_length_m=100.0,
            camera_length_m=50,
            point_size=30,
        )

plt.axis("off")
plt.title("GT Pose + Heading + Camera Directions (Pixel Space)")
plt.tight_layout()
plt.show()