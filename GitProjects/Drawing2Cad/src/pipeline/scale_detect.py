# -*- coding: utf-8 -*-
"""
Scale detection — OCR + horizontal & vertical dimension pairing + consensus.

- Multi-pass OCR (gray/bin + 2×, PSM 6 & 11) with IOU de-dup.
- Horizontal & vertical span candidates (Hough + merge + mild filters).
- "Stitch across number" to bridge a dim line split by the text.
- Arrowhead evidence sampled slightly outside each span end (soft bonus).
- Leader-line crossing penalty (straight path to the span crosses another span).
- Consensus clustering on px/mm (widened prior so hi-res pages aren't punished).
- Debug contains: all numbers, all candidate pairs, all arrow evidences for
  candidate spans, picked pair, consensus cluster size, and picked arrow metrics.
"""

from __future__ import annotations
import os, re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# --- Tesseract ---------------------------------------------------------------
try:
    import pytesseract
    from pytesseract import Output as TessOutput
except Exception:
    pytesseract = None
    TessOutput = None  # type: ignore

if pytesseract is not None and os.name == "nt":
    default_win_path = r"C:\Users\elias\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
    if os.path.exists(default_win_path):
        try:
            pytesseract.pytesseract.tesseract_cmd = default_win_path
        except Exception:
            pass


# --- result container --------------------------------------------------------
@dataclass
class ScaleResult:
    px_per_mm: float = 1.0
    unit: str = "mm"
    confidence: float = 0.0
    method: str = "unknown"
    debug: Dict[str, Any] = None  # type: ignore
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --- OCR helpers -------------------------------------------------------------
_NUM_RE = re.compile(r"^\d{1,4}$")  # digits only

def _iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    return inter / max(1.0, area_a + area_b - inter)

def _merge_hits(hits: List[Dict[str, Any]], iou_thr: float = 0.5) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for h in sorted(hits, key=lambda d: d["conf"], reverse=True):
        keep = True
        for m in merged:
            if h["text"] == m["text"] and _iou(h["bbox"], m["bbox"]) >= iou_thr:
                keep = False; break
        if keep:
            merged.append(h)
    merged.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))
    return merged

def _tesseract_numbers(bgr: np.ndarray) -> List[Dict[str, Any]]:
    """Multi-pass OCR (gray/bin, 2×, PSM 6 & 11) with de-duplication."""
    if pytesseract is None:
        return []
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    binv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    bpos = 255 - binv  # black text on white

    def up2(img): return cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    imgs = [("gray", gray, 1.0), ("bin", bpos, 1.0), ("gray2x", up2(gray), 2.0), ("bin2x", up2(bpos), 2.0)]
    psms = ["6", "11"]
    cfg_tpl = "--oem 1 --psm {psm} -c tessedit_char_whitelist=0123456789"

    hits: List[Dict[str, Any]] = []
    for _, img, scale in imgs:
        for psm in psms:
            cfg = cfg_tpl.format(psm=psm)
            data = pytesseract.image_to_data(img, output_type=TessOutput.DICT, config=cfg)
            N = len(data["text"])
            for i in range(N):
                txt = (data["text"][i] or "").strip()
                if not txt or not _NUM_RE.match(txt): continue
                conf_s = str(data["conf"][i])
                if not conf_s.replace('.', '', 1).isdigit(): continue
                conf = float(conf_s) / 100.0
                if conf <= 0: continue
                x, y, w, h = map(int, (data["left"][i], data["top"][i], data["width"][i], data["height"][i]))
                if scale != 1.0:
                    x = int(round(x/scale)); y = int(round(y/scale))
                    w = int(round(w/scale)); h = int(round(h/scale))
                hits.append({"text": txt, "value": int(txt), "conf": conf, "bbox": (x, y, x+w, y+h)})
    return _merge_hits(hits, iou_thr=0.5)


# --- span finding (horizontal & vertical) ------------------------------------
def _canny_for_lines(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (3,3), 0)
    v = np.median(g)
    lo = int(max(0, 0.66 * v))
    hi = int(min(255, 1.33 * v))
    return cv2.Canny(g, lo, hi)

def _find_span_candidates_oriented(bgr: np.ndarray, orient: str) -> List[Tuple[int,int,int,int]]:
    """
    Find spans in given orientation ('h' or 'v') via Hough, merge collinear, mild filters.
    Returns thin boxes (x0,y0,x1,y1).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    e = _canny_for_lines(gray)
    lines = cv2.HoughLinesP(
        e, 1, np.pi/180, threshold=100,
        minLineLength=max(25, int(0.02*max(bgr.shape[:2]))),
        maxLineGap=12
    )
    spans: List[Tuple[int,int,int,int]] = []
    if lines is None: return spans
    H, W = gray.shape

    segs: List[Tuple[int,int,int,int]] = []  # (a0,a1,row_or_col,axisFlag)
    for x1,y1,x2,y2 in lines[:,0,:]:
        dx, dy = x2-x1, y2-y1
        if orient == "h":
            if abs(dy) > 3: continue
            x0, x3 = (x1, x2) if x1 <= x2 else (x2, x1)
            y = int(round(0.5*(y1+y2)))
            segs.append((x0, x3, y, 0))
        else:
            if abs(dx) > 3: continue
            y0, y3 = (y1, y2) if y1 <= y2 else (y2, y1)
            x = int(round(0.5*(x1+x2)))
            segs.append((y0, y3, x, 1))

    # merge collinear
    segs.sort(key=lambda s: (s[2], s[0]))
    merged: List[Tuple[int,int,int,int]] = []
    for a0, a1, r, axis in segs:
        if not merged:
            merged.append((a0, a1, r, axis)); continue
        ma0, ma1, mr, maxis = merged[-1]
        if axis == maxis and abs(mr - r) <= 4 and (a0 <= ma1 + 10):
            merged[-1] = (min(ma0,a0), max(ma1,a1), int(round(0.5*(mr+r))), axis)
        else:
            merged.append((a0,a1,r,axis))

    margin_h, margin_w = int(0.02*H), int(0.02*W)
    for a0,a1,r,axis in merged:
        if orient == "h":
            span = a1 - a0
            if span < 25: continue
            if a0 <= margin_w or a1 >= (W - margin_w): continue
            if span >= 0.90 * W: continue
            y0,y1 = r-3, r+3
            spans.append((a0, y0, a1, y1))
        else:
            span = a1 - a0
            if span < 25: continue
            if a0 <= margin_h or a1 >= (H - margin_h): continue
            if span >= 0.90 * H: continue
            x0,x1 = r-3, r+3
            spans.append((x0, a0, x1, a1))
    return spans


# --- arrowhead evidence (orientation-aware) ----------------------------------
def _arrow_evidence(gray: np.ndarray, span_box: Tuple[int,int,int,int], orient: str) -> Dict[str, Any]:
    """
    Return orientation-agnostic evidence:
      endpoints a_pt / b_pt (left/right for 'h', top/bottom for 'v'),
      densities d_a, d_m, d_b, and booleans a_ok, b_ok.
    Samples a bit *outside* the span ends too.
    """
    (x0, y0, x1, y1) = span_box
    H, W = gray.shape
    # binary + small close to solidify triangles
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    bw = (bw > 0).astype(np.uint8)

    def dens(cx, cy, half):
        x0_, x1_ = max(0, cx-half), min(W, cx+half)
        y0_, y1_ = max(0, cy-half), min(H, cy+half)
        roi = bw[y0_:y1_, x0_:x1_]
        return float(roi.mean()) if roi.size else 0.0

    half = max(8, min(18, int(0.012 * max(H, W))))
    off  = max(6, int(0.9 * half))

    if orient == "h":
        cy = int(round(0.5*(y0+y1)))
        ax, ay = int(x0), cy
        bx, by = int(x1), cy
        d_a = max(dens(ax, ay, half), dens(ax - off, ay, half))
        d_b = max(dens(bx, by, half), dens(bx + off, by, half))
        d_m = dens(int(0.5*(x0+x1)), cy, half)
    else:
        cx = int(round(0.5*(x0+x1)))
        ax, ay = cx, int(y0)
        bx, by = cx, int(y1)
        d_a = max(dens(ax, ay, half), dens(ax, ay - off, half))
        d_b = max(dens(bx, by, half), dens(bx, by + off, half))
        d_m = dens(cx, int(0.5*(y0+y1)), half)

    delta, abs_min = 0.10, 0.16
    a_ok = (d_a >= max(abs_min, d_m + delta))
    b_ok = (d_b >= max(abs_min, d_m + delta))

    return {
        "span_bbox": (int(x0), int(y0), int(x1), int(y1)),
        "orient": orient,
        "a_pt": (int(ax), int(ay)),
        "b_pt": (int(bx), int(by)),
        "d_a": float(d_a),
        "d_m": float(d_m),
        "d_b": float(d_b),
        "a_ok": bool(a_ok),
        "b_ok": bool(b_ok),
    }

def _arrow_mult(gray: np.ndarray, span_box: Tuple[int,int,int,int], orient: str) -> float:
    info = _arrow_evidence(gray, span_box, orient)
    if info["a_ok"] and info["b_ok"]: return 1.15
    if info["a_ok"] or info["b_ok"]:  return 0.85
    return 0.60


# --- leader-crossing (perpendicular path) ------------------------------------
def _leader_crosses(num_cx: float, num_cy: float,
                    span_box: Tuple[int,int,int,int],
                    all_spans_h: List[Tuple[int,int,int,int]],
                    all_spans_v: List[Tuple[int,int,int,int]],
                    orient: str,
                    tol: int = 4) -> bool:
    """
    For 'h' span: leader is vertical from number to span.
    For 'v' span: leader is horizontal from number to span.
    Crossing if another span's centerline intersects that path.
    """
    x0, y0, x1, y1 = span_box
    if orient == "h":
        s_cy = 0.5*(y0+y1)
        lead_x = min(max(num_cx, x0), x1)
        ylo, yhi = (num_cy, s_cy) if num_cy <= s_cy else (s_cy, num_cy)
        for sb in all_spans_h + all_spans_v:
            if sb == span_box: continue
            bx0, by0, bx1, by1 = sb
            b_cx, b_cy = 0.5*(bx0+bx1), 0.5*(by0+by1)
            if abs(b_cy - s_cy) <= tol and (abs(by1-by0) <= 6):
                continue
            if (bx0 - tol) <= lead_x <= (bx1 + tol) and (ylo + tol) <= b_cy <= (yhi - tol):
                return True
        return False
    else:
        s_cx = 0.5*(x0+x1)
        lead_y = min(max(num_cy, y0), y1)
        xlo, xhi = (num_cx, s_cx) if num_cx <= s_cx else (s_cx, num_cx)
        for sb in all_spans_h + all_spans_v:
            if sb == span_box: continue
            bx0, by0, bx1, by1 = sb
            b_cx, b_cy = 0.5*(bx0+bx1), 0.5*(by0+by1)
            if abs(b_cx - s_cx) <= tol and (abs(bx1-bx0) <= 6):
                continue
            if (by0 - tol) <= lead_y <= (by1 + tol) and (xlo + tol) <= b_cx <= (xhi - tol):
                return True
        return False


# --- scoring (orientation-aware) ---------------------------------------------
def _score_pair(num: Dict[str,Any],
                span_box: Tuple[int,int,int,int],
                W: int, H: int,
                spans_h: List[Tuple[int,int,int,int]],
                spans_v: List[Tuple[int,int,int,int]],
                gray: np.ndarray,
                orient: str) -> Tuple[float,int,bool]:
    """
    Score a (number, span) pairing.
    Returns (score, span_px, leader_cross_flag).
    """
    nx0, ny0, nx1, ny1 = num["bbox"]
    x0, y0, x1, y1 = span_box
    n_cx = 0.5*(nx0 + nx1)
    n_cy = 0.5*(ny0 + ny1)

    if orient == "h":
        s_cy = 0.5*(y0 + y1)
        span_len = abs(x1 - x0)
        y_align = max(0.0, 1.0 - abs(n_cy - s_cy)/max(1.0, 0.06*H))
        bracket = 1.0 if (x0 - 10) <= n_cx <= (x1 + 10) else 0.0
        margin_pen = 0.75 if (s_cy < 0.04*H or s_cy > 0.96*H) else 1.0
        full_pen = max(0.0, 1.0 - (span_len / max(1.0, 0.85*W)))
        span_norm = min(1.0, span_len / max(1.0, 0.25*W))
        align = y_align
    else:
        s_cx = 0.5*(x0 + x1)
        span_len = abs(y1 - y0)
        x_align = max(0.0, 1.0 - abs(n_cx - s_cx)/max(1.0, 0.06*W))
        bracket = 1.0 if (y0 - 10) <= n_cy <= (y1 + 10) else 0.0
        margin_pen = 0.75 if (s_cx < 0.04*W or s_cx > 0.96*W) else 1.0
        full_pen = max(0.0, 1.0 - (span_len / max(1.0, 0.85*H)))
        span_norm = min(1.0, span_len / max(1.0, 0.25*H))
        align = x_align

    val_wt = min(1.0, num["value"] / 80.0)
    sc = (0.32 * num["conf"]) + (0.25 * align) + (0.15 * bracket) \
         + (0.10 * span_norm) + (0.10 * val_wt) + (0.08 * full_pen)
    sc *= margin_pen

    sc *= _arrow_mult(gray, span_box, orient)

    crossed = _leader_crosses(n_cx, n_cy, span_box, spans_h, spans_v, orient)
    if crossed:
        sc *= 0.30

    return float(sc), int(span_len), bool(crossed)


# --- consensus clustering -----------------------------------------------------
def _cluster_pxmm(cands: List[Dict[str, Any]], rel_tol: float = 0.07) -> Tuple[float, List[int], float]:
    if not cands:
        return 1.0, [], 0.0
    ratios = [c["span_px"] / float(max(1, c["value_mm"])) for c in cands]
    weights = [float(c["score"]) for c in cands]
    order = np.argsort(ratios)
    ratios = [ratios[i] for i in order]
    weights = [weights[i] for i in order]

    best_w = -1.0
    best_idx: List[int] = []
    N = len(ratios); j = 0
    for i in range(N):
        r = ratios[i]; lo, hi = r*(1.0-rel_tol), r*(1.0+rel_tol)
        while j < N and ratios[j] <= hi: j += 1
        w = sum(weights[k] for k in range(i, j) if ratios[k] >= lo)
        if w > best_w:
            best_w = w; best_idx = list(range(i, j))
    if best_w <= 0:
        return 1.0, [], 0.0

    wsum = sum(weights[k] for k in best_idx)
    pxmm = sum(ratios[k]*weights[k] for k in best_idx) / max(1e-6, wsum)

    # generous prior so hi-res pages don't get punished
    prior = 1.0
    if pxmm < 3.0 or pxmm > 90.0:
        prior = 0.75
    elif pxmm < 4.0 or pxmm > 70.0:
        prior = 0.90

    total_w = sum(weights)
    consensus = prior * (best_w / max(1e-6, total_w))
    return float(pxmm), [order[k] for k in best_idx], float(consensus)


# --- main API -----------------------------------------------------------------
def detect_scale(page_bgr: np.ndarray, section_bgr: Optional[np.ndarray] = None) -> ScaleResult:
    H, W = page_bgr.shape[:2]
    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)

    result = ScaleResult(px_per_mm=1.0, unit="mm", confidence=0.0, method="unknown", debug={})
    debug: Dict[str, Any] = {
        "method": "unknown",
        "numbers": [],
        "arrow_span_candidates": [],
        "picked": None,
        "methods_tried": [],
        "arrows": [],
        "cluster_size": 0,
    }
    result.debug = debug

    # OCR
    nums = _tesseract_numbers(page_bgr)
    debug["numbers"] = [{"text": n["text"], "conf": n["conf"], "bbox": n["bbox"]} for n in nums]

    # Spans (both orientations)
    spans_h = _find_span_candidates_oriented(page_bgr, "h")
    spans_v = _find_span_candidates_oriented(page_bgr, "v")

    debug["methods_tried"].append({"name":"ocr-arrow-span", "confidence":0.0,
                                   "why": f"nums={len(nums)} spans_h={len(spans_h)} spans_v={len(spans_v)}"})

    if not nums or (not spans_h and not spans_v):
        return result

    has_big_values = any(n["value"] >= 10 for n in nums)

    def stitch_across_number(num, spans, orient) -> List[Tuple[int,int,int,int]]:
        """Create synthetic span that bridges across the number gap."""
        nx0, ny0, nx1, ny1 = num["bbox"]
        n_cx = 0.5*(nx0+nx1); n_cy = 0.5*(ny0+ny1)
        out = []
        if orient == "h":
            rows = [(sb, 0.5*(sb[1]+sb[3])) for sb in spans]
            rows = [ (sb, cy) for (sb,cy) in rows if abs(cy - n_cy) <= 6 ]
            left  = [sb for (sb,_) in rows if sb[2] <= nx0 - 2]
            right = [sb for (sb,_) in rows if sb[0] >= nx1 + 2]
            if left and right:
                L = max(left, key=lambda s: s[2])
                R = min(right, key=lambda s: s[0])
                y0 = int(round(0.5*(L[1]+L[3])))-3
                y1 = y0+6
                out.append((L[0], y0, R[2], y1))
        else:
            cols = [(sb, 0.5*(sb[0]+sb[2])) for sb in spans]
            cols = [ (sb, cx) for (sb,cx) in cols if abs(cx - n_cx) <= 6 ]
            top =  [sb for (sb,_) in cols if sb[3] <= ny0 - 2]
            bot =  [sb for (sb,_) in cols if sb[1] >= ny1 + 2]
            if top and bot:
                T = max(top, key=lambda s: s[3])
                B = min(bot, key=lambda s: s[1])
                x0 = int(round(0.5*(T[0]+T[2])))-3
                x1 = x0+6
                out.append((x0, T[1], x1, B[3]))
        return out

    def build_candidates(min_val: int) -> List[Dict[str, Any]]:
        cands: List[Dict[str, Any]] = []
        TOP_K = 2
        MIN_FRAC = 0.65

        for num in nums:
            if num["value"] < min_val:
                continue

            for orient, spans in (("h", spans_h), ("v", spans_v)):
                scored = []
                for sb in spans:
                    nx0, ny0, nx1, ny1 = num["bbox"]
                    if orient == "h":
                        if abs(0.5*(ny0+ny1) - 0.5*(sb[1]+sb[3])) > 0.12*H: continue
                        if (sb[2] < nx0 - 150) or (sb[0] > nx1 + 150): continue
                    else:
                        if abs(0.5*(nx0+nx1) - 0.5*(sb[0]+sb[2])) > 0.12*W: continue
                        if (sb[3] < ny0 - 150) or (sb[1] > ny1 + 150): continue
                    sc, span_px, crossed = _score_pair(num, sb, W, H, spans_h, spans_v, gray, orient)
                    scored.append((sc, span_px, crossed, sb))

                # stitch across number
                for sb in stitch_across_number(num, spans, orient):
                    sc, span_px, crossed = _score_pair(num, sb, W, H, spans_h, spans_v, gray, orient)
                    scored.append((sc, span_px, crossed, sb))

                if not scored:
                    continue
                scored.sort(key=lambda t: t[0], reverse=True)
                best_sc = scored[0][0]
                kept = 0
                for sc, span_px, crossed, sb in scored:
                    if sc < best_sc * MIN_FRAC:
                        continue
                    pxmm = span_px / float(max(1, num["value"]))
                    cands.append({
                        "value_mm": int(num["value"]),
                        "score": float(sc),
                        "ocr_conf": float(num["conf"]),
                        "span_px": int(span_px),
                        "num_bbox": tuple(map(int, num["bbox"])),
                        "span_bbox": tuple(map(int, sb)),
                        "px_per_mm": float(pxmm),
                        "leader_cross": bool(crossed),
                        "orient": orient,
                    })
                    kept += 1
                    if kept >= TOP_K:
                        break
        return cands

    has_big = any(n["value"] >= 10 for n in nums)
    candidates = build_candidates(min_val=10 if has_big else 1)

    # collect arrow evidences for candidate spans (unique)
    seen = set(); arrows = []
    for c in candidates:
        sb = c["span_bbox"]; ori = c["orient"]
        key = (sb, ori)
        if key in seen: continue
        seen.add(key)
        arrows.append(_arrow_evidence(gray, sb, ori))
    debug["arrows"] = arrows

    debug["arrow_span_candidates"] = [{
        "score": c["score"], "ocr_conf": c["ocr_conf"], "span_px": c["span_px"],
        "number_bbox": c["num_bbox"], "arrow_span_bbox": c["span_bbox"],
        "value": c["value_mm"], "px_per_mm": c["px_per_mm"], "leader_cross": c["leader_cross"],
        "orient": c["orient"],
    } for c in candidates]

    if not candidates:
        return result  # unknown

    pxmm, idxs, consensus = _cluster_pxmm(candidates, rel_tol=0.07)

    # tiny-singleton fail-safe
    if idxs and len(idxs) == 1:
        only = candidates[idxs[0]]
        if only["value_mm"] < 10:
            big_only = [c for c in candidates if c["value_mm"] >= 10]
            if big_only:
                pxmm_big, idxs_big, consensus_big = _cluster_pxmm(big_only, rel_tol=0.07)
                if idxs_big:
                    cluster = [big_only[i] for i in idxs_big]
                    cluster.sort(key=lambda c: c["score"], reverse=True)
                    picked = cluster[0]
                    result.px_per_mm = float(pxmm_big)
                    result.confidence = float(min(1.0, consensus_big * (0.6 + 0.4*picked["score"])))
                    result.method = "ocr-arrow-span-consensus"
                    debug["method"] = result.method
                    debug["cluster_size"] = len(cluster)
                    debug["picked"] = {
                        "value": picked["value_mm"], "ocr_conf": picked["ocr_conf"], "score": picked["score"],
                        "span_px": picked["span_px"], "number_bbox": picked["num_bbox"],
                        "arrow_span_bbox": picked["span_bbox"], "leader_cross": picked.get("leader_cross", False),
                        "orient": picked["orient"],
                    }
                    # attach arrow metrics for picked
                    ae = _arrow_evidence(gray, picked["span_bbox"], picked["orient"])
                    debug["picked"]["arrow"] = {
                        "a_ok": ae["a_ok"], "b_ok": ae["b_ok"],
                        "d_a": ae["d_a"], "d_m": ae["d_m"], "d_b": ae["d_b"],
                    }
                    debug["methods_tried"].append({"name":"ocr-arrow-span-consensus",
                                                   "confidence":result.confidence,
                                                   "px_per_mm":result.px_per_mm,
                                                   "span_px":picked["span_px"],
                                                   "why": f"re-picked; tiny-singleton was value={only['value_mm']}"})
                    return result
            return result  # unknown

    if not idxs:
        best = max(candidates, key=lambda c: c["score"])
        if best["value_mm"] < 10:
            return result
        result.px_per_mm = best["span_px"] / float(best["value_mm"])
        result.confidence = float(best["score"]) * 0.6
        result.method = "ocr-arrow-span"
        debug["method"] = result.method
        debug["cluster_size"] = 1
        debug["picked"] = {
            "value": best["value_mm"], "ocr_conf": best["ocr_conf"], "score": best["score"],
            "span_px": best["span_px"], "number_bbox": best["num_bbox"],
            "arrow_span_bbox": best["span_bbox"], "leader_cross": best.get("leader_cross", False),
            "orient": best["orient"],
        }
        ae = _arrow_evidence(gray, best["span_bbox"], best["orient"])
        debug["picked"]["arrow"] = {
            "a_ok": ae["a_ok"], "b_ok": ae["b_ok"],
            "d_a": ae["d_a"], "d_m": ae["d_m"], "d_b": ae["d_b"],
        }
        debug["methods_tried"].append({"name":"ocr-arrow-span","confidence":result.confidence,
                                       "px_per_mm":result.px_per_mm,"span_px":best["span_px"],
                                       "why":"fallback single-best"})
        return result

    # inside winning cluster pick highest scoring
    cluster = [candidates[i] for i in idxs]
    cluster.sort(key=lambda c: c["score"], reverse=True)
    picked = cluster[0]

    result.px_per_mm = float(pxmm)
    result.confidence = float(min(1.0, consensus * (0.6 + 0.4*picked["score"])))
    result.method = "ocr-arrow-span-consensus"
    debug["method"] = result.method
    debug["cluster_size"] = len(cluster)
    debug["picked"] = {
        "value": picked["value_mm"], "ocr_conf": picked["ocr_conf"], "score": picked["score"],
        "span_px": picked["span_px"], "number_bbox": picked["num_bbox"],
        "arrow_span_bbox": picked["span_bbox"], "leader_cross": picked.get("leader_cross", False),
        "orient": picked["orient"],
    }
    ae = _arrow_evidence(gray, picked["span_bbox"], picked["orient"])
    debug["picked"]["arrow"] = {
        "a_ok": ae["a_ok"], "b_ok": ae["b_ok"],
        "d_a": ae["d_a"], "d_m": ae["d_m"], "d_b": ae["d_b"],
    }
    debug["methods_tried"].append({
        "name":"ocr-arrow-span-consensus","confidence":result.confidence,
        "px_per_mm":result.px_per_mm,"span_px":picked["span_px"],
        "why": f"cluster_size={len(cluster)}"
    })
    return result
