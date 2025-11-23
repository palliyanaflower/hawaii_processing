import open3d as o3d
from pathlib import Path
import argparse
import numpy as np

def merge_pcd_range(pcd_dir, start_idx, end_idx, output_path, voxel_size=None):
    pcd_dir = Path(pcd_dir)
    files = sorted(pcd_dir.glob("*.pcd"), key=lambda f: int(f.stem))
    if not files:
        raise RuntimeError(f"No .pcd files found in {pcd_dir}")

    # Clamp range to available files
    start_idx = max(0, start_idx)
    end_idx = min(len(files), end_idx)

    print(f"📁 Found {len(files)} .pcd files")
    print(f"🔢 Combining indices [{start_idx}:{end_idx}]")

    merged_points = []

    for i, f in enumerate(files[start_idx:end_idx]):
        pcd = o3d.io.read_point_cloud(str(f))
        pts = np.asarray(pcd.points)
        if len(pts) == 0:
            print(f"⚠️ Skipping empty file: {f.name}")
            continue
        merged_points.append(pts)
        print(f"✅ Added {f.name}: {len(pts)} points")

    if not merged_points:
        raise RuntimeError("No valid points found in selected range.")

    merged_points = np.vstack(merged_points)
    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(merged_points)

    print(f"\nTotal merged points: {len(merged_points)}")

    # Optional downsampling
    if voxel_size and voxel_size > 0:
        print(f"Applying voxel downsampling with voxel size = {voxel_size} m")
        merged_pcd = merged_pcd.voxel_down_sample(voxel_size)
        print(f"Downsampled points: {len(merged_pcd.points)}")

    # Save
    o3d.io.write_point_cloud(str(output_path), merged_pcd)
    print(f"\nSaved merged PCD to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge a range of PCD files into one.")
    parser.add_argument("--pcd_dir", type=str, required=True, help="Path to folder containing .pcd files")
    parser.add_argument("--start", type=int, required=True, help="Start index (inclusive)")
    parser.add_argument("--end", type=int, required=True, help="End index (exclusive)")
    parser.add_argument("--output", type=str, default="merged.pcd", help="Output PCD file path")
    parser.add_argument("--voxel_size", type=float, default=None, help="Optional voxel size (e.g., 0.01)")
    args = parser.parse_args()

    merge_pcd_range(args.pcd_dir, args.start, args.end, args.output, args.voxel_size)

# python merge_pcd.py \
#   --pcd_dir data/haleiwa_neighborhood/processed_data/lidar/pcd \
#   --start 0 \
#   --end 250 \
#   --output data/haleiwa_neighborhood/processed_data/lidar/pcd_merged/static_scene.pcd \
#   --voxel_size 0.02

# python merge_pcd.py \
#   --pcd_dir data/cb/processed_data/lidar/pcd \
#   --start 0 \
#   --end 250 \
#   --output data/cb/processed_data/lidar/pcd_merged/static_scene.pcd
