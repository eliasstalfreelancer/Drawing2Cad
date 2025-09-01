# src/pipeline/profile_extract.py
# Robust profile extractor (v2.1 tuned):
# - builds a clean, FILLED part mask via largest-contour fill
# - uses a distance transform to pick a stable OUTER silhouette point per scanline
# - adaptively chooses the side with material
# - trims outliers (MAD) and smooths the profile
# - always writes a scan overlay when debug_save is provided

from typing import Optional, Tuple
import numpy as np
import cv2


# ----------------- helpers to make a solid part mask -----------------

def _binarize_inv(gray: np.ndarray) -> np.ndarray:
    """Binary image with ink as 255 (white)."""
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

def _strip_page_border(bw: np.ndarray, margin_ratio: float = 0.02) -> np.ndarray:
    """Remove a thin border frame around the page."""
    h, w = bw.shape
    m = int(round(min(h, w) * margin_ratio))
    bw[:m, :] = 0; bw[-m:, :] = 0; bw[:, :m] = 0; bw[:, -m:] = 0
    return bw

def _remove_border_touching(mask: np.ndarray, margin: int = 1) -> np.ndarray:
    """Drop any component touching the image border (likely page frame)."""
    h, w = mask.shape
    num, labels = cv2.connectedComponents(mask)
    if num <= 1:
        return mask
    keep = np.zeros_like(mask)
    for i in range(1, num):
        ys, xs = np.where(labels == i)
        if ys.size == 0:
            continue
        if (ys.min() <= margin or ys.max() >= h-1-margin or
            xs.min() <= margin or xs.max() >= w-1-margin):
            continue  # skip border-touching
        keep[labels == i] = 255
    return keep

def _filled_part_mask(section_bgr: np.ndarray) -> np.ndarray:
    """
    Create a solid (filled) mask of the part region in the section crop:
      - binarize, strip page border
      - close gaps, open to remove thin dims/text
      - suppress 45° hatch by diagonal opening
      - remove border-touching blobs
      - keep LARGEST contour and fill it
    Returns a mask with 255 inside the part, 0 elsewhere.
    """
    g = cv2.cvtColor(section_bgr, cv2.COLOR_BGR2GRAY)
    bw = _binarize_inv(g)
    bw = _strip_page_border(bw)

    # stronger cleanup (tuned)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 3)   # was 2
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,  np.ones((1,4), np.uint8), 2)   # was (1,3),1

    # suppress 45° hatching
    hatch45 = np.array([[1,0],[0,1]], dtype=np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hatch45, 2)

    # remove page frame & keep interior components
    bw = _remove_border_touching(bw, margin=2)

    # Fill the largest contour — more stable than flood fill for complex shapes
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid = np.zeros_like(bw)
    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
        cv2.drawContours(solid, [cnt], -1, 255, thickness=cv2.FILLED)
        # final cleanup
        solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 1)
    return solid  # 255 inside solid part, 0 outside


# ----------------- outlier trimming & smoothing -----------------

def _trim_outliers_mad(values: np.ndarray, k: float = 3.0) -> np.ndarray:
    """
    Mask outliers by Median Absolute Deviation.
    Returns a boolean mask (True = keep).  (k tightened from 3.5 → 3.0)
    """
    v = values.astype(np.float32)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) + 1e-6
    score = np.abs(v - med) / (1.4826 * mad)  # ~z-score
    return score < k

def _smooth_1d(values: np.ndarray, win: int = 9) -> np.ndarray:
    """Box smoothing (win widened to 9); keeps endpoints reasonable."""
    win = max(3, win | 1)  # odd
    pad = win // 2
    v = values.astype(np.float32)
    vpad = np.pad(v, (pad, pad), mode="edge")
    ker = np.ones(win, dtype=np.float32) / win
    sm = np.convolve(vpad, ker, mode="valid")
    return sm.astype(np.float32)


# ----------------- main API -----------------

def extract_profile(section_bgr: np.ndarray,
                    axis_pos: int,
                    axis_type: str,
                    n_samples: int = 360,
                    debug_save: Optional[str] = None
                    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract the OUTER silhouette as (rs, zs) from the section crop using a distance transform.

    Returns
    -------
    rs : np.ndarray (float32)
        Radial distances (pixels) from the axis to the outer contour.
        Preferred "positive" side:
          - vertical axis: right of axis (if material exists), else left
          - horizontal axis: below axis (if material exists), else above
    zs : np.ndarray (float32)
        Axial coordinate used for comparison:
          - vertical axis: z = 0 at bottom, increases upward (z = Hs - y)
          - horizontal axis: z = x (left → right)
    """
    Hs, Ws = section_bgr.shape[:2]
    axis_type = axis_type.lower().strip()
    if axis_type not in ("vertical", "horizontal"):
        raise ValueError("axis_type must be 'vertical' or 'horizontal'")

    solid = _filled_part_mask(section_bgr)  # 255 inside part
    inside = (solid > 0).astype(np.uint8)

    # Distance transform on the inside (distance to background).
    dist = cv2.distanceTransform(inside, distanceType=cv2.DIST_L2, maskSize=3)

    # Debug overlay
    overlay = cv2.cvtColor(section_bgr, cv2.COLOR_BGR2RGB)
    if axis_type == "vertical":
        ax = int(np.clip(axis_pos, 0, Ws-1))
        cv2.line(overlay, (ax, 0), (ax, Hs-1), (255, 0, 0), 2)
    else:
        ay = int(np.clip(axis_pos, 0, Hs-1))
        cv2.line(overlay, (0, ay), (Ws-1, ay), (255, 0, 0), 2)

    rs_list, zs_list = [], []

    if axis_type == "vertical":
        ys = np.linspace(0, Hs - 1, num=min(n_samples, Hs)).astype(int)
        for y in ys:
            # Prefer RIGHT side; refine with ±4 px window (tuned)
            xR0 = int(np.clip(axis_pos + 1, 0, Ws - 1))
            row_inside = inside[y, xR0:Ws]
            if row_inside.any():
                xs = np.where(row_inside > 0)[0]
                x_right = xR0 + xs.max()
                xw0 = max(xR0, x_right - 4)
                xw1 = min(Ws - 1, x_right + 4)
                win = dist[y, xw0:xw1 + 1]
                x_star = xw0 + int(np.argmax(win))
                r = float(x_star - axis_pos)
                x_hit = int(x_star)
            else:
                # Fallback LEFT
                xL0 = int(np.clip(axis_pos - 1, 0, Ws - 1))
                row_inside_L = inside[y, 0:xL0 + 1]
                if not row_inside_L.any():
                    continue
                xs = np.where(row_inside_L > 0)[0]
                x_left = xs.min()
                xw0 = max(0, x_left - 4)
                xw1 = min(xL0, x_left + 4)
                win = dist[y, xw0:xw1 + 1]
                x_star = xw0 + int(np.argmax(win))
                r = float(axis_pos - x_star)
                x_hit = int(x_star)

            z = float(Hs - 1 - y)  # bottom->top
            rs_list.append(r); zs_list.append(z)
            cv2.circle(overlay, (int(x_hit), int(y)), 2, (0, 255, 0), -1)

    else:
        xs = np.linspace(0, Ws - 1, num=min(n_samples, Ws)).astype(int)
        for x in xs:
            # Prefer DOWN; refine with ±4 px window (tuned)
            yD0 = int(np.clip(axis_pos + 1, 0, Hs - 1))
            col_inside = inside[yD0:Hs, x]
            if col_inside.any():
                ys_idx = np.where(col_inside > 0)[0]
                y_down = yD0 + ys_idx.max()
                yw0 = max(yD0, y_down - 4)
                yw1 = min(Hs - 1, y_down + 4)
                win = dist[yw0:yw1 + 1, x]
                y_star = yw0 + int(np.argmax(win))
                r = float(y_star - axis_pos)
                y_hit = int(y_star)
            else:
                # Fallback UP
                yU0 = int(np.clip(axis_pos - 1, 0, Hs - 1))
                col_inside_U = inside[0:yU0 + 1, x]
                if not col_inside_U.any():
                    continue
                ys_idx = np.where(col_inside_U > 0)[0]
                y_up = ys_idx.min()
                yw0 = max(0, y_up - 4)
                yw1 = min(yU0, y_up + 4)
                win = dist[yw0:yw1 + 1, x]
                y_star = yw0 + int(np.argmax(win))
                r = float(axis_pos - y_star)
                y_hit = int(y_star)

            z = float(x)
            rs_list.append(r); zs_list.append(z)
            cv2.circle(overlay, (int(x), int(y_hit)), 2, (0, 255, 0), -1)

    # Convert to arrays
    if len(rs_list) == 0:
        if debug_save is not None:
            cv2.imwrite(debug_save, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        return None, None

    rs = np.asarray(rs_list, dtype=np.float32)
    zs = np.asarray(zs_list, dtype=np.float32)

    # Trim outliers (MAD k=3.0) and smooth (win=9)
    keep = _trim_outliers_mad(rs, k=3.0)
    if keep.sum() >= max(10, int(0.05 * keep.size)):
        rs = rs[keep]; zs = zs[keep]
    if rs.size >= 7:
        order = np.argsort(zs)
        zs = zs[order]; rs = rs[order]
        rs = _smooth_1d(rs, win=9)

    if debug_save is not None:
        cv2.imwrite(debug_save, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return rs, zs
