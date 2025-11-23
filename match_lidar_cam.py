import pandas as pd
import json
from pathlib import Path

def match_camera_lidar_pandas(
    timestamps_path: Path,
    max_time_diff_ms: float = 50.0,
    save_path: Path | None = None
):
    # --- Load timestamps.json ---
    data = json.loads(timestamps_path.read_text())
    cam_df = pd.DataFrame(data["camera"])
    lid_df = pd.DataFrame(data["lidar"])

    # --- Convert from ns → s ---
    cam_df["t_camera"] = cam_df["t"] * 1e-9
    lid_df["t_lidar"] = lid_df["t"] * 1e-9
    cam_df = cam_df.drop(columns=["t"])
    lid_df = lid_df.drop(columns=["t"])

    # --- Sort for merge_asof (required) ---
    cam_df = cam_df.sort_values("t_camera")
    lid_df = lid_df.sort_values("t_lidar")

    # --- Merge based on nearest timestamp ---
    matched = pd.merge_asof(
        cam_df,
        lid_df,
        left_on="t_camera",
        right_on="t_lidar",
        direction="nearest",
        tolerance=max_time_diff_ms / 1000.0,
        suffixes=("_camera", "_lidar")
    )

    # --- Drop unmatched rows ---
    matched = matched.dropna(subset=["file_lidar"])

    # --- Compute delta (ms) ---
    matched["dt_ms"] = (matched["t_lidar"] - matched["t_camera"]) * 1000.0

    # --- Keep relevant columns only ---
    matched = matched[[
        "t_camera", "file_camera",
        "t_lidar", "file_lidar",
        "dt_ms"
    ]]

    print(f"Matched {len(matched)} / {len(cam_df)} camera frames within {max_time_diff_ms} ms")

    # --- Optional save ---
    if save_path:
        matched.to_json(save_path, orient="records", indent=2)
        print(f"Saved matches to {save_path}")

    return matched


# Example usage
if __name__ == "__main__":
    env = "cb"
    matched_df = match_camera_lidar_pandas(
        timestamps_path=Path("data/" + env +"/processed_data/metadata/timestamps.json"),
        max_time_diff_ms=50.0,
        save_path=Path("data/" + env +"/processed_data/metadata/matched_pairs.json")
    )
    print(matched_df.head())
