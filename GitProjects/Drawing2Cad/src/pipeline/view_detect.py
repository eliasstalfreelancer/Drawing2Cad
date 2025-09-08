# src/pipeline/view_detect.py
# Axis detection ensemble + debug artifacts

from __future__ import annotations
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any

# ----------------- binarization & utilities -----------------

def _binarize_inv(gray: np.ndarray) -> np.ndarray:
    # Ink (dark) -> 255
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

def _strip_page_border(bw: np.ndarray, margin_ratio: float = 0.02) -> np.ndarray:
    h, w = bw.shape
    m = int(round(min(h, w) * margin_ratio))
    bw[:m, :] = 0; bw[-m:, :] = 0; bw[:, :m] = 0; bw[:, -m:] = 0
    return bw

def _remove_border_touching(mask: np.ndarray, margin: int = 1) -> np.ndarray:
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
            continue
        keep[labels == i] = 255
    return keep

def _mad_keep(values: np.ndarray, k: float = 3.0) -> np.ndarray:
    v = values.astype(np.float32)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) + 1e-6
    z = np.abs(v - med) / (1.4826 * mad)
    return z < k

# ----------------- section view localization -----------------

def locate_section_view(img: np.ndarray) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
    """Prefer the band with section hatching; return crop and (x0,y0,x1,y1)."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = _binarize_inv(gray)
    bw = _strip_page_border(bw)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((1,3), np.uint8), 1)

    thirds = np.array_split(bw, 3, axis=1)
    sums_fg = [int(t.sum()) for t in thirds]

    gx = cv2.Sobel(bw, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(bw, cv2.CV_32F, 0, 1, ksize=3)
    diag45  = np.maximum(0, gx + gy)
    diag135 = np.maximum(0, gx - gy)
    hatch_energy = cv2.GaussianBlur(diag45 + diag135, (0,0), 1.0)
    thirds_h = np.array_split(hatch_energy, 3, axis=1)
    sums_h = [float(t.sum()) for t in thirds_h]

    scores = [0.4*s_fg + 0.6*s_h for s_fg, s_h in zip(sums_fg, sums_h)]
    best = int(np.argmax(scores))
    s_sorted = sorted(scores, reverse=True)
    decisive = (s_sorted[0] > 1.2 * s_sorted[1]) if len(s_sorted) > 1 else True

    if decisive:
        x0 = [0, w//3, 2*w//3][best]
        x1 = [w//3, 2*w//3, w][best]
        return img[:, x0:x1], (x0, 0, x1, h)
    else:
        return img, (0, 0, w, h)

# ----------------- filled part mask -----------------

def _filled_part_mask(section_bgr: np.ndarray, debug: Optional[Dict[str,Any]]=None) -> np.ndarray:
    """Make a solid part mask (255 inside)."""
    g = cv2.cvtColor(section_bgr, cv2.COLOR_BGR2GRAY)
    bw = _binarize_inv(g)
    bw = _strip_page_border(bw)

    # strong cleanup for dims/hatching
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 3)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,  np.ones((1,4), np.uint8), 2)
    hatch45 = np.array([[1,0],[0,1]], dtype=np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hatch45, 2)

    bw = _remove_border_touching(bw, margin=2)

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid = np.zeros_like(bw)
    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
        cv2.drawContours(solid, [cnt], -1, 255, thickness=cv2.FILLED)
        solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 1)

    if debug is not None:
        debug.setdefault("artifacts", {})["solid"] = solid.copy()
    return solid

# ----------------- candidate 0: mirror-correlation -----------------

def _axis_mirror_correlation(section_bgr: np.ndarray, axis_type: str,
                             debug: Optional[Dict[str,Any]]=None
                             ) -> Tuple[Optional[int], float]:
    solid = _filled_part_mask(section_bgr, debug)
    M = (solid > 0).astype(np.uint8)
    Hs, Ws = M.shape

    if axis_type == "vertical":
        x_min = int(0.10 * Ws); x_max = int(0.90 * Ws)
        if x_max <= x_min: return None, 0.0
        scores = []
        for x in range(x_min, x_max):
            L = M[:, :x]; R = M[:, x:Ws]
            w = min(L.shape[1], R.shape[1])
            if w <= 2: scores.append(0.0); continue
            Lc = L[:, x-w:x]; Rc = R[:, :w]
            Lm = np.fliplr(Lc)
            inter = np.logical_and(Lm, Rc).sum()
            uni   = np.logical_or (Lm, Rc).sum() + 1e-6
            scores.append(float(inter/uni))
        s = np.array(scores, np.float32)
        if s.size == 0: return None, 0.0
        if s.size >= 7:
            k = np.ones(7, np.float32)/7.0
            s = np.convolve(s, k, mode="same")
        idx = int(np.argmax(s)); x_best = x_min + idx
        if debug is not None:
            chart = np.zeros((80, Ws, 3), dtype=np.uint8)
            vals = np.zeros(Ws, np.float32); vals[x_min:x_max] = s
            vmax = max(1e-6, float(vals.max()))
            for x in range(Ws):
                h = int(76 * (vals[x]/vmax))
                cv2.line(chart, (x, 79), (x, 79-h), (0,255,255), 1)
            cv2.line(chart, (x_best, 0), (x_best, 79), (0,0,255), 1)
            debug.setdefault("artifacts", {})["mirror_curve"] = chart
        return int(x_best), float(np.clip(s[idx], 0.0, 1.0))

    else:
        y_min = int(0.10 * Hs); y_max = int(0.90 * Hs)
        if y_max <= y_min: return None, 0.0
        scores = []
        for y in range(y_min, y_max):
            T = M[:y, :]; B = M[y:Hs, :]
            h = min(T.shape[0], B.shape[0])
            if h <= 2: scores.append(0.0); continue
            Tc = T[y-h:y, :]; Bc = B[:h, :]
            Tm = np.flipud(Tc)
            inter = np.logical_and(Tm, Bc).sum()
            uni   = np.logical_or (Tm, Bc).sum() + 1e-6
            scores.append(float(inter/uni))
        s = np.array(scores, np.float32)
        if s.size == 0: return None, 0.0
        if s.size >= 7:
            k = np.ones(7, np.float32)/7.0
            s = np.convolve(s, k, mode="same")
        idx = int(np.argmax(s)); y_best = y_min + idx
        if debug is not None:
            chart = np.zeros((80, Hs, 3), dtype=np.uint8)
            vals = np.zeros(Hs, np.float32); vals[y_min:y_max] = s
            vmax = max(1e-6, float(vals.max()))
            for y in range(Hs):
                h2 = int(76 * (vals[y]/vmax))
                cv2.line(chart, (y, 79), (y, 79-h2), (0,255,255), 1)
            cv2.line(chart, (y_best, 0), (y_best, 79), (0,0,255), 1)
            debug.setdefault("artifacts", {})["mirror_curve"] = chart
        return int(y_best), float(np.clip(s[idx], 0.0, 1.0))

# ----------------- candidate 1: parallel-pair -----------------

def _axis_parallel_pair(section_bgr: np.ndarray, axis_type: str,
                        debug: Optional[Dict[str,Any]]=None
                        ) -> Tuple[Optional[int], float]:
    if axis_type != "vertical":
        return None, 0.0

    solid = _filled_part_mask(section_bgr, debug)
    inside = (solid > 0).astype(np.uint8)

    kernel = np.ones((7,7), np.uint8)
    core = cv2.erode(inside, kernel, iterations=2)
    if debug is not None:
        debug.setdefault("artifacts", {})["core"] = (core*255).astype(np.uint8)

    g = cv2.cvtColor(section_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(g, (5,5), 0), 60, 140)
    edges = cv2.bitwise_and(edges, edges, mask=core)
    if debug is not None:
        debug["artifacts"]["edges"] = edges.copy()

    Hs, Ws = core.shape
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                            minLineLength=max(0.6*Hs, 60),
                            maxLineGap=8)
    if lines is None or len(lines) < 2:
        return None, 0.0

    xs, ls = [], []
    xL, xR = int(0.25*Ws), int(0.75*Ws)
    for x1,y1,x2,y2 in lines[:,0,:]:
        if abs(x1-x2) > 3: continue
        length = abs(y2 - y1)
        xmid = 0.5*(x1+x2)
        if not (xL <= xmid <= xR): continue
        xs.append(xmid); ls.append(length)
    if len(xs) < 2:
        return None, 0.0

    best = None; best_score = -1
    for i in range(len(xs)):
        for j in range(i+1, len(xs)):
            gap = abs(xs[i] - xs[j])
            if gap < 0.06*Ws or gap > 0.45*Ws:  # inner gap sanity
                continue
            score = ls[i] + ls[j] - 0.1*abs(ls[i]-ls[j])
            if score > best_score:
                best_score = score
                best = (xs[i], xs[j], ls[i], ls[j], gap)
    if best is None:
        return None, 0.0

    x1, x2, l1, l2, gap = best
    axis_pos = int(round(0.5*(x1 + x2)))
    center_pen = 1.0 - abs(axis_pos - 0.5*Ws)/(0.5*Ws)
    cov = min(1.0, (l1 + l2)/max(1.0, Hs))
    q = float(np.clip(0.5*cov + 0.5*center_pen, 0.0, 1.0))
    return axis_pos, q

# ----------------- candidate 2: symmetry midpoints -----------------

def _axis_symmetry(section_bgr: np.ndarray, axis_type: str
                   ) -> Tuple[Optional[int], float]:
    solid = _filled_part_mask(section_bgr)
    inside = (solid > 0).astype(np.uint8)
    Hs, Ws = inside.shape
    mids, rel_diffs = [], []

    if axis_type == "vertical":
        for y in range(Hs):
            xs = np.where(inside[y] > 0)[0]
            if xs.size < 2: continue
            L, R = int(xs.min()), int(xs.max())
            span = R - L
            if span < 8:  continue
            mid = 0.5*(L+R)
            rL = mid - L; rR = R - mid
            rel = abs(rL - rR) / max(rL, rR, 1e-6)
            mids.append(mid); rel_diffs.append(rel)
    else:
        for x in range(Ws):
            ys = np.where(inside[:, x] > 0)[0]
            if ys.size < 2: continue
            T, B = int(ys.min()), int(ys.max())
            span = B - T
            if span < 8: continue
            mid = 0.5*(T+B)
            rT = mid - T; rB = B - mid
            rel = abs(rT - rB) / max(rT, rB, 1e-6)
            mids.append(mid); rel_diffs.append(rel)

    if len(mids) < max(20, 0.05*(Hs if axis_type=="vertical" else Ws)):
        return None, 0.0

    mids = np.asarray(mids, np.float32)
    rel_diffs = np.asarray(rel_diffs, np.float32)
    keep = np.logical_and(_mad_keep(mids, 3.0), _mad_keep(rel_diffs, 3.0))
    if keep.any():
        mids = mids[keep]; rel_diffs = rel_diffs[keep]

    axis_pos = int(round(float(np.median(mids))))
    med_rel = float(np.median(rel_diffs))
    q = float(np.clip(1.0 - med_rel, 0.0, 1.0))
    return axis_pos, q

# ----------------- candidate 3: medial-axis line fit -----------------

def _axis_medial(section_bgr: np.ndarray, axis_type: str,
                 debug: Optional[Dict[str,Any]]=None
                 ) -> Tuple[Optional[int], float]:
    solid = _filled_part_mask(section_bgr, debug)
    inside = (solid > 0).astype(np.uint8)
    Hs, Ws = inside.shape
    dist = cv2.distanceTransform(inside, cv2.DIST_L2, 3)
    ridge = (dist > (0.6*dist.max())).astype(np.uint8) * 255
    ridge = cv2.morphologyEx(ridge, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
    if debug is not None:
        debug.setdefault("artifacts", {})["ridge"] = ridge.copy()

    ys, xs = np.where(ridge > 0)
    if xs.size < 20: return None, 0.0

    if axis_type == "vertical":
        axis_pos = int(round(float(np.median(xs))))
        mad = float(np.median(np.abs(xs - np.median(xs))) + 1e-6)
        q_straight = float(np.clip(1.0 - mad / max(1.0, 0.12 * Ws), 0.0, 1.0))
        cov = (ys.max() - ys.min()) / max(1.0, (Hs-1))
        q = float(np.clip(0.7*q_straight + 0.3*cov, 0.0, 1.0))
        return axis_pos, q
    else:
        A = np.vstack([xs.astype(np.float32), np.ones_like(xs, np.float32)]).T
        sol, _, _, _ = np.linalg.lstsq(A, ys.astype(np.float32), rcond=None)
        b = float(sol[1])
        y_pred = A @ sol
        ss_res = float(np.sum((ys - y_pred)**2))
        ss_tot = float(np.sum((ys - ys.mean())**2)) + 1e-6
        r2 = max(0.0, 1.0 - ss_res / ss_tot)
        cov = (xs.max() - xs.min()) / max(1.0, (Ws-1))
        quality = float(np.clip(0.5*r2 + 0.5*cov, 0.0, 1.0))
        return int(round(b)), quality

# ----------------- candidate 4: bbox center -----------------

def _axis_bbox_center(section_bgr: np.ndarray, axis_type: str) -> Tuple[int, float]:
    g  = cv2.cvtColor(section_bgr, cv2.COLOR_BGR2GRAY)
    bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    num, labels = cv2.connectedComponents(bw)
    Hs, Ws = bw.shape
    if num <= 1:
        return (Ws//2 if axis_type=="vertical" else Hs//2), 0.1
    sizes = [(labels == i).sum() for i in range(1, num)]
    keep_id = 1 + int(np.argmax(sizes))
    inside = (labels == keep_id).astype(np.uint8)

    ys, xs = np.where(inside > 0)
    if xs.size == 0 or ys.size == 0:
        return (Ws//2 if axis_type=="vertical" else Hs//2), 0.1
    if axis_type == "vertical":
        return int(round((xs.min() + xs.max())/2.0)), 0.3
    else:
        return int(round((ys.min() + ys.max())/2.0)), 0.3

# ----------------- helpful extra voters (lenient) -----------------

def _axis_width_midpoint(section_bgr: np.ndarray, axis_type: str) -> Tuple[Optional[int], float]:
    if axis_type != "vertical":
        return None, 0.0

    g = cv2.cvtColor(section_bgr, cv2.COLOR_BGR2GRAY)
    bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    Hs, Ws = bw.shape

    # “inside” largest blob
    num, labels = cv2.connectedComponents(bw)
    if num <= 1:
        return None, 0.0
    sizes = [(labels == i).sum() for i in range(1, num)]
    keep_id = 1 + int(np.argmax(sizes))
    inside = (labels == keep_id).astype(np.uint8) * 255

    dist = cv2.distanceTransform((inside > 0).astype(np.uint8), cv2.DIST_L2, 3)
    y0 = int(0.15 * Hs); y1 = int(0.95 * Hs)
    mids = []
    for y in range(y0, y1):
        xs = np.where(inside[y] > 0)[0]
        if xs.size < 2: continue
        L, R = int(xs.min()), int(xs.max())
        if dist[y, L] < 1.5 or dist[y, R] < 1.5:  # suppress thin noise
            continue
        mids.append(0.5*(L+R))
    if len(mids) < max(20, int(0.05*(y1-y0))):
        return None, 0.0
    mids = np.array(mids, np.float32)
    med = float(np.median(mids))
    spread = float(np.median(np.abs(mids - med)) + 1e-6)
    q = float(np.clip(1.0 - spread / max(1.0, 0.15*Ws), 0.0, 1.0))
    return int(round(med)), q

def _axis_inner_gap(section_bgr: np.ndarray, axis_type: str) -> Tuple[Optional[int], float]:
    """Row-wise SOLID–VOID–SOLID smallest inner gap voter."""
    if axis_type != "vertical":
        return None, 0.0
    g = cv2.cvtColor(section_bgr, cv2.COLOR_BGR2GRAY)
    bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    num, labels = cv2.connectedComponents(bw)
    if num <= 1:
        return None, 0.0
    sizes = [(labels == i).sum() for i in range(1, num)]
    keep_id = 1 + int(np.argmax(sizes))
    inside = (labels == keep_id).astype(np.uint8)

    Hs, Ws = inside.shape
    y0 = int(0.15*Hs); y1 = int(0.95*Hs)
    mids, gaps = [], []
    for y in range(y0, y1):
        row = inside[y]
        xs = np.where(row > 0)[0]
        if xs.size < 4: continue
        runs = []
        start = None
        for x in range(Ws):
            if row[x] and start is None: start = x
            elif (not row[x]) and start is not None:
                runs.append((start, x-1)); start = None
        if start is not None: runs.append((start, Ws-1))
        if len(runs) < 2: continue
        best = None
        for i in range(len(runs)-1):
            L0, L1 = runs[i]; R0, R1 = runs[i+1]
            gap = R0 - L1 - 1
            if gap <= 0: continue
            if (L1-L0+1) < 4 or (R1-R0+1) < 4: continue
            if best is None or gap < best[0]:
                best = (gap, L1, R0)
        if best is None: continue
        gap, left_edge, right_edge = best
        mid = 0.5*(left_edge + right_edge)
        mids.append(mid); gaps.append(gap)

    if len(mids) < 25:
        return None, 0.0
    mids = np.asarray(mids, np.float32)
    gaps = np.asarray(gaps, np.float32)
    med_mid = float(np.median(mids))
    mad_mid = float(np.median(np.abs(mids - med_mid)) + 1e-6)
    med_gap = float(np.median(gaps))
    q_consistency = float(np.clip(1.0 - mad_mid / max(1.0, 0.06*Ws), 0.0, 1.0))
    q_narrow = float(np.clip(1.0 - med_gap / max(6.0, 0.25*Ws), 0.0, 1.0))
    q = 0.6*q_consistency + 0.4*q_narrow
    return int(round(med_mid)), q

# --- NEW: bore / inner-void voter via contour hierarchy -------------

def _axis_bore_hole_center(section_bgr: np.ndarray,
                           axis_type: str,
                           debug: Optional[Dict[str, Any]] = None) -> Tuple[Optional[int], float]:
    if axis_type != "vertical":
        return None, 0.0

    g  = cv2.cvtColor(section_bgr, cv2.COLOR_BGR2GRAY)
    ink = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # very light cleanup (don’t erase a narrow bore)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1,3), np.uint8), 1)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8), 1)

    # contour hierarchy: child contours of the main external contour = “holes”
    contours, hier = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    Hs, Ws = ink.shape
    if not contours or hier is None:
        if debug is not None:
            debug.setdefault("artifacts", {})["holes_dbg"] = np.zeros_like(ink)
        return None, 0.0
    hier = hier[0]
    ext_ids = [i for i,(_,_,_,pa) in enumerate(hier) if pa == -1]
    if not ext_ids:
        if debug is not None:
            debug.setdefault("artifacts", {})["holes_dbg"] = np.zeros_like(ink)
        return None, 0.0
    main_id = max(ext_ids, key=lambda i: cv2.contourArea(contours[i]))

    holes = np.zeros_like(ink)
    for i,(_,_,_,pa) in enumerate(hier):
        if pa == main_id:
            cv2.drawContours(holes, contours, i, 255, thickness=cv2.FILLED)

    if debug is not None:
        debug.setdefault("artifacts", {})["holes_dbg"] = holes.copy()

    # work in a generous vertical band to avoid end effects
    y0, y1 = int(0.06*Hs), int(0.96*Hs)
    band = (holes[y0:y1] > 0).astype(np.uint8)
    if band.sum() < 10:
        return None, 0.0

    # column-wise vertical coverage (0..1)
    col_cov = band.sum(axis=0).astype(np.float32)
    Hband = float(max(1, y1 - y0))
    col_cov /= Hband

    # keep only columns that persist across many rows → through-bore
    sel = None
    for thr in (0.60, 0.50, 0.40, 0.30):
        idx = np.where(col_cov >= thr)[0]
        if idx.size:
            sel = idx; break
    if sel is None:
        # fallback: top-K columns by coverage
        k = max(5, int(0.02*Ws))
        sel = np.argsort(-col_cov)[:k]
        sel.sort()

    # split selected indices into contiguous blocks; choose block with max total coverage
    blocks = []
    s = int(sel[0]); p = int(sel[0])
    for x in sel[1:]:
        x = int(x)
        if x == p + 1:
            p = x
        else:
            blocks.append((s, p)); s = p = x
    blocks.append((s, p))
    bestL, bestR = max(blocks, key=lambda ab: float(col_cov[ab[0]:ab[1]+1].sum()))

    # weighted centroid of the best block (coverage as weights)
    xs = np.arange(bestL, bestR + 1, dtype=np.float32)
    w  = col_cov[bestL:bestR + 1]
    med_x = float(np.sum(xs * w) / max(1e-6, np.sum(w)))

    # quality: consistency (narrow block), coverage, and border safety
    width = float(bestR - bestL + 1)
    q_width  = float(np.clip(1.0 - width / max(4.0, 0.15*Ws), 0.0, 1.0))
    q_cov    = float(np.clip(col_cov[bestL:bestR + 1].max(), 0.0, 1.0))
    q_border = min(med_x, (Ws - 1) - med_x) / max(1.0, Ws / 2.0)
    q = 0.45*q_width + 0.45*q_cov + 0.10*q_border

    # small coverage plot for debugging
    if debug is not None:
        hdbg = 80
        chart = np.zeros((hdbg, Ws, 3), dtype=np.uint8)
        vals = (col_cov * (hdbg - 4)).astype(np.int32)
        for x in range(Ws):
            cv2.line(chart, (x, hdbg-1), (x, hdbg-1 - int(vals[x])), (255,255,0), 1)
        cv2.line(chart, (int(round(med_x)), 0), (int(round(med_x)), hdbg-1), (0,0,255), 1)
        cv2.rectangle(chart, (bestL, 0), (bestR, hdbg-1), (0,255,0), 1)
        debug.setdefault("artifacts", {})["holes_cov"] = chart

    return int(round(med_x)), q

# ----------------- public API (ensemble + debug) -----------------

def detect_axis(section_bgr: np.ndarray, axis_type: str,
                debug: Optional[Dict[str,Any]] = None) -> int:
    """
    Try multiple candidates and pick best by weighted fusion.
    Records detailed diagnostics in 'debug' if provided.
    """
    if debug is not None:
        debug.setdefault("candidates", [])
    Hs, Ws = section_bgr.shape[:2]
    cands: list[tuple[str, int, float]] = []

    def _log(name, pos=None, q=None, err: Exception=None):
        if debug is None: return
        if err is not None:
            debug["candidates"].append((f"{name}_err", -1, 0.0))
        elif pos is None:
            debug["candidates"].append((f"{name}_none", -1, 0.0))
        else:
            debug["candidates"].append((name, int(pos), float(q)))

    # 0) mirror-correlation
    try:
        _log("mirror_corr_call")
        pos, q = _axis_mirror_correlation(section_bgr, axis_type, debug)
        if pos is not None: cands.append(("mirror_corr", pos, q)); _log("mirror_corr", pos, q)
        else: _log("mirror_corr")
    except Exception as e:
        _log("mirror_corr", err=e)

    # 1) parallel-pair
    try:
        _log("parallel_pair_call")
        pos, q = _axis_parallel_pair(section_bgr, axis_type, debug)
        if pos is not None: cands.append(("parallel_pair", pos, q)); _log("parallel_pair", pos, q)
        else: _log("parallel_pair")
    except Exception as e:
        _log("parallel_pair", err=e)

    # 2) symmetry
    try:
        _log("symmetry_call")
        pos, q = _axis_symmetry(section_bgr, axis_type)
        if pos is not None: cands.append(("symmetry", pos, q)); _log("symmetry", pos, q)
        else: _log("symmetry")
    except Exception as e:
        _log("symmetry", err=e)

    # 2.5) width-midpoint
    try:
        _log("width_mid_call")
        pos, q = _axis_width_midpoint(section_bgr, axis_type)
        if pos is not None: cands.append(("width_mid", pos, q)); _log("width_mid", pos, q)
        else: _log("width_mid")
    except Exception as e:
        _log("width_mid", err=e)

    # 2.75) inner-gap
    try:
        _log("inner_gap_call")
        pos, q = _axis_inner_gap(section_bgr, axis_type)
        if pos is not None: cands.append(("inner_gap", pos, q)); _log("inner_gap", pos, q)
        else: _log("inner_gap")
    except Exception as e:
        _log("inner_gap", err=e)

    # 2.8) bore hole center (hierarchy)
    try:
        _log("bore_hole_call")
        pos, q = _axis_bore_hole_center(section_bgr, axis_type, debug)
        if pos is not None: cands.append(("bore_hole", pos, q)); _log("bore_hole", pos, q)
        else: _log("bore_hole")
    except Exception as e:
        _log("bore_hole", err=e)

    # 3) medial
    try:
        _log("medial_call")
        pos, q = _axis_medial(section_bgr, axis_type, debug)
        if pos is not None: cands.append(("medial", pos, q)); _log("medial", pos, q)
        else: _log("medial")
    except Exception as e:
        _log("medial", err=e)

    # 4) bbox
    try:
        _log("bbox_call")
        pos, q = _axis_bbox_center(section_bgr, axis_type)
        cands.append(("bbox", pos, q)); _log("bbox", pos, q)
    except Exception as e:
        _log("bbox", err=e)

    # Fallback if nothing
    if not cands:
        best_pos = Ws//2 if axis_type == "vertical" else Hs//2
        if debug is not None:
            debug["best"] = ("fallback_center", best_pos, 0.0)
        return best_pos

    # --- SAFETY GUARD: drop near-border candidates (10% margin) ---
    margin = 0.10 * (Ws if axis_type == "vertical" else Hs)
    filtered = []
    for name, pos, q in cands:
        if axis_type == "vertical":
            if pos < margin or pos > Ws - margin: continue
        else:
            if pos < margin or pos > Hs - margin: continue
        filtered.append((name, pos, q))
    if filtered:
        cands = filtered

    # --- SHORT-CIRCUIT: trust bore center if decent ---
    # --- SHORT-CIRCUIT: prefer a visible bore even if q is modest ---
    for name, pos, q in cands:
        if name == "bore_hole" and q >= 0.25:   # was 0.35
            safe = (min(pos, (Ws - 1) - pos) / max(1.0, Ws / 2.0)) > 0.15 if axis_type == "vertical" \
                else (min(pos, (Hs - 1) - pos) / max(1.0, Hs / 2.0)) > 0.15
            if safe:
                if debug is not None:
                    debug["best"] = (name, int(pos), float(q))
                return int(np.clip(pos, 0, (Ws - 1 if axis_type == "vertical" else Hs - 1)))


    # --- Weighted fusion ---
    def border_safety(pos):
        if axis_type == "vertical":
            return min(pos, Ws - 1 - pos) / max(1.0, Ws / 2.0)
        else:
            return min(pos, Hs - 1 - pos) / max(1.0, Hs / 2.0)

    pos_arr, w_arr = [], []
    for name, pos, q in cands:
        base = float(np.clip(q, 0.0, 1.0)) * float(np.clip(border_safety(pos), 0.0, 1.0))
        if name == "bore_hole":
            base *= 1.5  # extra pull for inner-void center
        w = base
        if w <= 0: 
            continue
        pos_arr.append(int(pos)); w_arr.append(w)


    if not pos_arr:
        name, pos, q = max(cands, key=lambda t: t[2])
        if debug is not None:
            debug["best"] = (name, pos, q)
        return int(np.clip(pos, 0, (Ws - 1 if axis_type == "vertical" else Hs - 1)))

    pos_arr = np.array(pos_arr, np.float32)
    w_arr   = np.array(w_arr,   np.float32)

    # reject outliers: keep within 25 px of median
    med = float(np.median(pos_arr))
    keep = np.abs(pos_arr - med) <= 25.0
    if keep.sum() >= 2:
        pos_arr = pos_arr[keep]; w_arr = w_arr[keep]

    fused = int(round(float(np.average(pos_arr, weights=w_arr))))
    if debug is not None:
        debug["best"] = ("fused", fused, float(w_arr.max()))
    return int(np.clip(fused, 0, (Ws - 1 if axis_type == "vertical" else Hs - 1)))
