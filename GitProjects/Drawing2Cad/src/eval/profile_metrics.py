# src/eval/profile_metrics.py

from dataclasses import dataclass     # For simple structured containers
import json, csv                      # To read axis metadata (json) and golden profiles (csv)
import numpy as np                    # For array math
from typing import Tuple, Dict        # Type hints for clarity


# Define a simple container for a profile: z-coordinates and r-values
@dataclass
class Profile:
    z: np.ndarray   # axis direction (height if vertical, x-position if horizontal)
    r: np.ndarray   # radius = distance from axis at each z


# Load a golden profile from CSV (created by make_golden_profile.py)
def load_golden_profile(csv_path: str) -> Profile:
    zs, rs = [], []
    with open(csv_path, "r") as f:
        rdr = csv.DictReader(f)         # Read CSV with headers
        for row in rdr:                 # Each row has z and r
            zs.append(float(row["z"]))  # Convert z string to float
            rs.append(float(row["r"]))  # Convert r string to float
    z = np.asarray(zs, dtype=float)
    r = np.asarray(rs, dtype=float)
    order = np.argsort(z)               # Ensure z is sorted ascending
    return Profile(z=z[order], r=r[order])


# Load axis truth from JSON (saved alongside golden profile)
def load_axis_truth(json_path: str) -> Tuple[str, float, int, int]:
    with open(json_path, "r") as f:
        data = json.load(f)
    # Returns orientation, position, and image size
    return data["axis_type"], float(data["axis_pos"]), int(data["image_width"]), int(data["image_height"])


# Resample a profile onto a uniform z-grid so we can compare with another profile
def resample_profile(profile: Profile, z_min: float, z_max: float, n: int = 200) -> Profile:
    z_grid = np.linspace(z_min, z_max, n)      # Create evenly spaced z-values
    order = np.argsort(profile.z)              # Ensure input z is sorted
    r_grid = np.interp(z_grid, profile.z[order], profile.r[order])  # Interpolate radii onto new z-grid
    return Profile(z=z_grid, r=r_grid)


# Compute errors between predicted and golden profiles
def profile_errors(pred: Profile, gold: Profile, n_eval: int = 200) -> Dict[str, float]:
    # Use overlapping z-range only, so we compare fair regions
    zmin = max(float(pred.z.min()), float(gold.z.min()))
    zmax = min(float(pred.z.max()), float(gold.z.max()))
    if zmax <= zmin:
        # No overlap → can't compare
        return {"mae_r": float("nan"), "rmse_r": float("nan")}

    # Resample both to the same z grid
    pr = resample_profile(pred, zmin, zmax, n_eval).r
    gr = resample_profile(gold, zmin, zmax, n_eval).r

    diff = pr - gr
    # Mean absolute error (average size of errors)
    mae = float(np.mean(np.abs(diff)))
    # Root mean square error (penalizes bigger errors more)
    rmse = float(np.sqrt(np.mean(diff*diff)))

    return {"mae_r": mae, "rmse_r": rmse}


# Compute axis error in pixels
def axis_error(pred_axis_val_fullimg: float, axis_pos_fullimg: float) -> float:
    # Just absolute distance between predicted axis and golden axis
    return float(abs(pred_axis_val_fullimg - axis_pos_fullimg))
