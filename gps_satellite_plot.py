import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import re

# --- Helper: natural sort key (extracts numeric index from filename) ---
def extract_index(path):
    # Extract the last number in the filename (e.g., _12_ from ..._12_gps.npy)
    match = re.search(r"_(\d+)_gps", path.stem)
    return int(match.group(1)) if match else -1

def m2pixels(robot_xy, map_image_path, robot_start_pixel, meters_per_pixel=0.32):
    """
    Convert robot and modem coordinates (in meters) to pixel positions on a map image and plot them.

    Parameters
    ----------
    robot_xy : np.ndarray
        Nx2 array of robot trajectory positions in meters (x, y).
    map_image_path : str
        Path to the background map image.
    robot_start_pixel : tuple
        Pixel coordinates (x, y) of the robot's starting position on the image.
    meters_per_pixel : float
        Conversion scale from meters to pixels (default: 0.32).
    """

    # --- Convert robot trajectory ---
    pixel_offsets = robot_xy / meters_per_pixel
    pixel_offsets[:, 1] = -pixel_offsets[:, 1]  # invert y axis (image origin is top-left)
    trajectory_pixels = pixel_offsets + np.array(robot_start_pixel)

    # --- Plot ---
    img = Image.open(map_image_path)
    # plt.figure(figsize=(10, 10))
    plt.imshow(img)

    # Plot trajectory
    plt.plot(trajectory_pixels[:, 0], trajectory_pixels[:, 1], 'k-', linewidth=2, label="Trajectory", alpha=0.8)

    # Plot start point
    plt.scatter(*robot_start_pixel, color='blue', s=120, linewidth=1.5, zorder=5, label='Start')

    plt.legend()
    plt.axis('off')
    plt.savefig('icra_main.pdf')
    plt.savefig('icra_main.png')
    plt.show()

def latlon2pixels(robot_latlon, map_image_path, ref_latlon, ref_pixel, meters_per_pixel=0.32):
    """
    Convert GPS (lat, lon) coordinates to pixel positions on a map image.

    Parameters
    ----------
    robot_latlon : np.ndarray
        Nx2 array of robot GPS coordinates (lat, lon).
    map_image_path : str
        Path to the map image.
    ref_latlon : tuple
        Reference latitude/longitude (lat, lon) that corresponds to ref_pixel.
    ref_pixel : tuple
        Pixel coordinates (x, y) corresponding to ref_latlon on the map.
    meters_per_pixel : float
        Scale for conversion from meters to pixels.
    """

    # --- Conversion constants ---
    R = 6378137.0  # Earth radius in meters
    lat0 = np.deg2rad(ref_latlon[0])

    # --- Convert GPS to local metric coordinates ---
    def gps_to_meters(latlon):
        lat, lon = np.deg2rad(latlon[:, 0]), np.deg2rad(latlon[:, 1])
        dlat = (lat - np.deg2rad(ref_latlon[0])) * R
        dlon = (lon - np.deg2rad(ref_latlon[1])) * R * np.cos(lat0)
        return np.column_stack((dlon, dlat))  # (x_meters, y_meters)

    robot_xy = gps_to_meters(robot_latlon)

    # --- Convert meters to pixels ---
    robot_offsets = robot_xy / meters_per_pixel
    robot_offsets[:, 1] = -robot_offsets[:, 1]
    robot_pixels = robot_offsets + np.array(ref_pixel)

    # --- Plot ---
    img = Image.open(map_image_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)

    plt.plot(robot_pixels[:, 0], robot_pixels[:, 1], 'r-', linewidth=2, label="Trajectory", alpha=0.8)
    plt.scatter(*ref_pixel, color='blue', s=60, marker="*", label="Start", zorder=5)
    plt.scatter(*robot_pixels[-1], color='blue', s=60, label="End", zorder=5)


    plt.legend()
    plt.axis('off')
    plt.savefig('icra_main_latlon.pdf')
    plt.savefig('icra_main_latlon.png')
    plt.show()

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
        plt.plot(robot_pixels[:, 0], robot_pixels[:, 1], '-', color=color, linewidth=2, label=f"Segment {i}")
        plt.text(robot_pixels[0, 0], robot_pixels[0, 1], str(i), fontsize=10, color=color, weight='bold')

        print(f"{i}: {f.stem}")

    # --- Mark reference pixel ---
    plt.scatter(*ref_pixel, color='blue', s=80, marker="*", zorder=5, label="Start")
    plt.scatter(*robot_pixels[-1], color='blue', s=60, label="End", zorder=5)

    # plt.legend(fontsize=8)
    plt.axis('off')
    plt.title("Segmented GPS Trajectories")
    plt.savefig(gps_arrays_dir.parent / "gps_segments_pixels.png", dpi=300)
    plt.savefig(gps_arrays_dir.parent / "gps_segments_pixels.pdf")
    plt.show()

# --- User settings ---
base = "makalii_point"
gps_arrays_dir = Path(base + "/gps_arrays")
map_image_path = base + "/satellite.png"
save_fig = True  # set False to skip saving figure
img = Image.open(map_image_path)
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.show()

params = {}
with open(base + "/plot_satellite_params.txt", "r") as f:
    for line in f:
        key, value = line.strip().split("=")
        params[key] = float(value)

ref_pixel = (params["x_px"], params["y_px"])
ref_latlon = np.array([params["ref_lat"], params["ref_lon"]])
meters_per_pixel = params["meters_per_pixel"]

print(ref_pixel, ref_latlon, meters_per_pixel)

latlon2pixels_segmented(gps_arrays_dir, map_image_path, ref_latlon, ref_pixel, meters_per_pixel)
