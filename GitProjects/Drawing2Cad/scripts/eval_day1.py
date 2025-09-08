# scripts/eval_day1.py
"""
Evaluate Day 1 and dump overlays + axis diagnostics.

Overlays:
  Pred axis (RED), Golden axis (CYAN), Pred silhouette (GREEN), Gold (YELLOW), Crop (MAGENTA)

Also dumps:
  - per-scanline debug (extractor)
  - axis candidates with call/none/err breadcrumbs
  - artifacts from voters (holes, ridge, mirror_curve, core, edges, solid)
"""

import sys, os
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import argparse
import numpy as np
import cv2

from src.eval.profile_metrics import (
    Profile, load_golden_profile, load_axis_truth, profile_errors, axis_error
)
from src.pipeline.view_detect import locate_section_view, detect_axis
from src.pipeline.profile_extract import extract_profile

def draw_axes(overlay, axis_type, pred_axis_full, gold_axis_full,
              color_pred=(0, 0, 255), color_gold=(255, 255, 0)):
    h, w = overlay.shape[:2]
    if axis_type == "vertical":
        x_pred = int(np.clip(round(pred_axis_full), 0, w - 1))
        x_gold = int(np.clip(round(gold_axis_full), 0, w - 1))
        cv2.line(overlay, (x_pred, 0), (x_pred, h - 1), color_pred, 2)
        cv2.line(overlay, (x_gold, 0), (x_gold, h - 1), color_gold, 2)
    else:
        y_pred = int(np.clip(round(pred_axis_full), 0, h - 1))
        y_gold = int(np.clip(round(gold_axis_full), 0, h - 1))
        cv2.line(overlay, (0, y_pred), (w - 1, y_pred), color_pred, 2)
        cv2.line(overlay, (0, y_gold), (w - 1, y_gold), color_gold, 2)
    return overlay

def draw_crop_rect(overlay, crop_box, color=(255, 0, 255)):
    x0, y0, x1, y1 = crop_box
    cv2.rectangle(overlay, (int(x0), int(y0)), (int(x1), int(y1)), color, 2)
    return overlay

def pred_profile_polyline_full(section_shape, crop_box, axis_type, axis_in_section, rs, zs):
    x0, y0, x1, y1 = crop_box
    Hs, Ws = section_shape[:2]
    pts = []
    for r, z in zip(rs, zs):
        if axis_type == "vertical":
            y_full = y0 + (Hs - z)
            x_full = x0 + axis_in_section + r
        else:
            x_full = x0 + z
            y_full = y0 + axis_in_section + r
        pts.append((int(round(x_full)), int(round(y_full))))
    return np.array(pts, dtype=np.int32)

def gold_profile_polyline_full(axis_type, axis_pos_full, gold_profile: Profile, full_h):
    pts = []
    for z, r in zip(gold_profile.z, gold_profile.r):
        if axis_type == "vertical":
            y_full = full_h - z
            x_full = axis_pos_full + r
        else:
            y_full = axis_pos_full + r
            x_full = z
        pts.append((int(round(x_full)), int(round(y_full))))
    return np.array(pts, dtype=np.int32)

def draw_polyline(overlay, pts, color, thickness=2):
    if len(pts) >= 2:
        cv2.polylines(overlay, [pts], isClosed=False, color=color, thickness=thickness)
    return overlay

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--use-gold-axis", dest="use_gold_axis", action="store_true")
    parser.add_argument("--dump-threshold", type=float, default=50.0)
    args = parser.parse_args()

    # Print which detector file actually loaded
    import src.pipeline.view_detect as vd
    print("[DEBUG] using view_detect.py at:", getattr(vd, "__file__", "<unknown>"))

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Cannot read image {args.image}")
    H, W = img.shape[:2]

    axis_json = args.gold.replace("/profiles/", "/axis/").replace(".csv", ".json")
    if not os.path.exists(axis_json):
        raise SystemExit("Missing axis JSON; run make_golden_profile.py first.")
    axis_type, axis_pos_full, W_truth, H_truth = load_axis_truth(axis_json)
    gold = load_golden_profile(args.gold)

    section, crop_box = locate_section_view(img)
    print("[DEBUG] section shape:", getattr(section, "shape", None),
          "dtype:", getattr(section, "dtype", None),
          "crop_box:", crop_box)

    dbg_dir = os.path.join(os.path.dirname(args.gold), "..", "debug")
    os.makedirs(dbg_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.image))[0]
    cv2.imwrite(os.path.join(dbg_dir, f"{base}_section.png"), section)

    debug = {}
    if args.use_gold_axis:
        x0, y0, x1, y1 = crop_box
        axis_in_section = int(np.clip(
            round(axis_pos_full - (x0 if axis_type == "vertical" else y0)),
            0,
            (section.shape[1] - 1 if axis_type == "vertical" else section.shape[0] - 1)
        ))
    else:
        print("[DEBUG] detect_axis called!")
        axis_in_section = detect_axis(section, axis_type, debug=debug)

    pred_axis_full = (crop_box[0] + axis_in_section) if axis_type == "vertical" else (crop_box[1] + axis_in_section)

    # ---------- dump axis debug ----------
    cand_txt = os.path.join(dbg_dir, f"{base}_axis_candidates.txt")
    with open(cand_txt, "w") as f:
        f.write(f"axis_type={axis_type}\n")
        for name, pos, q in debug.get("candidates", []):
            try:
                f.write(f"{name:16s}  pos={int(pos):4d}  q={float(q):.3f}\n")
            except Exception:
                f.write(f"{name}  pos={pos}  q={q}\n")
        if "best" in debug:
            f.write(f"BEST: {debug['best']}\n")
        if debug.get("errors"):
            f.write("\n-- errors --\n")
            for line in debug["errors"]:
                f.write(line + "\n")

    for k, im in debug.get("artifacts", {}).items():
        outp = os.path.join(dbg_dir, f"{base}_art_{k}.png")
        if im.ndim == 2:
            cv2.imwrite(outp, im)
        else:
            cv2.imwrite(outp, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

    dbg_scan_path = os.path.join(dbg_dir, f"{base}_scan.png")
    rs, zs = extract_profile(section, axis_in_section, axis_type, debug_save=dbg_scan_path)
    if rs is None:
        print("[!] Profile extraction failed.")
        print(f"[overlay] scan  → {dbg_scan_path}")
        out_full_axes = os.path.join(dbg_dir, f"{base}_axes_only.png")
        overlay_full = img.copy()
        overlay_full = draw_axes(overlay_full, axis_type, pred_axis_full, axis_pos_full)
        overlay_full = draw_crop_rect(overlay_full, crop_box)
        cv2.imwrite(out_full_axes, overlay_full)
        print(f"[overlay] axes-only → {out_full_axes}")
        raise SystemExit("Profile extraction failed; see scan/axes overlays.")

    zs_full = (np.asarray(zs, float) + float(crop_box[0])) if axis_type == "horizontal" else np.asarray(zs, float)
    pred = Profile(z=zs_full, r=np.asarray(rs, float))

    perf = profile_errors(pred, gold, n_eval=200)
    ax_err = axis_error(pred_axis_full, axis_pos_full)

    print("=== Day 1 Evaluation ===")
    print(f"Image: {args.image}")
    print(f"Axis type: {axis_type}")
    print(f"Axis error (px): {ax_err:.3f}")
    print(f"Profile MAE_r (px): {perf['mae_r']:.3f}")
    print(f"Profile RMSE_r (px): {perf['rmse_r']:.3f}")
    print(f"[overlay] scan  → {dbg_scan_path}")

    out_full = os.path.join(dbg_dir, f"{base}_full.png")
    out_crop = os.path.join(dbg_dir, f"{base}_crop.png")

    overlay_full = img.copy()
    overlay_full = draw_axes(overlay_full, axis_type, pred_axis_full, axis_pos_full)
    overlay_full = draw_crop_rect(overlay_full, crop_box)
    pred_pts_full = pred_profile_polyline_full(section.shape, crop_box, axis_type, axis_in_section, rs, zs)
    gold_pts_full = gold_profile_polyline_full(axis_type, axis_pos_full, gold, full_h=H)
    overlay_full = draw_polyline(overlay_full, pred_pts_full, (0, 255, 0), 2)
    overlay_full = draw_polyline(overlay_full, gold_pts_full, (0, 255, 255), 2)
    cv2.imwrite(out_full, overlay_full)

    overlay_crop = section.copy()
    if axis_type == "vertical":
        cv2.line(overlay_crop, (int(axis_in_section), 0),
                 (int(axis_in_section), overlay_crop.shape[0] - 1), (0, 0, 255), 2)
    else:
        cv2.line(overlay_crop, (0, int(axis_in_section)),
                 (overlay_crop.shape[1] - 1, int(axis_in_section)), (0, 0, 255), 2)

    pred_pts_crop = []
    for r, z in zip(rs, zs):
        if axis_type == "vertical":
            y = int(round(section.shape[0] - z))
            x = int(round(axis_in_section + r))
        else:
            x = int(round(z))
            y = int(round(axis_in_section + r))
        pred_pts_crop.append((x, y))
    pred_pts_crop = np.array(pred_pts_crop, dtype=np.int32)
    overlay_crop = draw_polyline(overlay_crop, pred_pts_crop, (0, 255, 0), 2)

    gold_pts_crop = []
    H_full = H
    x0, y0, x1, y1 = crop_box
    for z, r in zip(gold.z, gold.r):
        if axis_type == "vertical":
            y = int(round((H_full - z) - y0))
            x = int(round((axis_pos_full - x0) + r))
        else:
            x = int(round(z - x0))
            y = int(round((axis_pos_full - y0) + r))
        gold_pts_crop.append((x, y))
    gold_pts_crop = np.array(gold_pts_crop, dtype=np.int32)
    overlay_crop = draw_polyline(overlay_crop, gold_pts_crop, (0, 255, 255), 2)

    cv2.imwrite(out_crop, overlay_crop)

    print(f"[overlay] full  → {out_full}")
    print(f"[overlay] crop  → {out_crop}")
    print(f"[axis dbg] candidates → {cand_txt}")

if __name__ == "__main__":
    main()
