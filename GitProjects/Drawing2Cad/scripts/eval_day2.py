from __future__ import annotations
import argparse, json, os, sys
from dataclasses import asdict, is_dataclass
import cv2
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.pipeline.view_detect import locate_section_view
from src.pipeline.scale_detect import detect_scale

def _to_dict(res):
    if res is None:
        return {"px_per_mm":1.0,"unit":"mm","confidence":0.0,"method":"unknown","debug":{}}
    if is_dataclass(res):
        return asdict(res)
    if isinstance(res, dict):
        if "debug" not in res: res["debug"] = {}
        return res
    base = {}
    for k in ("px_per_mm","unit","confidence","method","debug"):
        base[k] = getattr(res, k, {} if k=="debug" else None)
    if base["debug"] is None: base["debug"] = {}
    return base

def put_label(img, text, org, scale=0.6, color=(255,255,255), bg=(0,0,0), thick=1):
    x, y = int(org[0]), int(org[1])
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    pad = 4
    cv2.rectangle(img, (x- pad, y- th- pad), (x+ tw+ pad, y+ pad), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def draw_ocr_overlay(page_bgr, dbg, out_path):
    img = page_bgr.copy()
    for n in dbg.get("numbers", []):
        x0,y0,x1,y1 = map(int, n.get("bbox", (0,0,0,0)))
        conf = float(n.get("conf", 0.0))
        text = str(n.get("text","?"))
        cv2.rectangle(img, (x0,y0), (x1,y1), (255,0,0), 2)
        put_label(img, f"{text}  conf={conf:.2f}", (x0, max(15, y0-6)),
                  color=(0,0,0), bg=(255,255,0))
    cv2.imwrite(out_path, img)

def draw_scale_debug(page_bgr, res_dict, out_path):
    img = page_bgr.copy()
    dbg = res_dict.get("debug", {}) or {}
    picked = dbg.get("picked") or {}

    # --- draw ALL arrow evidences (orientation-aware) -------------------------
    for a in dbg.get("arrows", []) or []:
        sb = a.get("span_bbox")
        if not sb: continue
        x0, y0, x1, y1 = map(int, sb)
        orient = a.get("orient", "h")
        A = a.get("a_pt", (x0, (y0+y1)//2))
        B = a.get("b_pt", (x1, (y0+y1)//2))

        if orient == "h":
            cy = int(round(0.5 * (y0 + y1)))
            cv2.line(img, (x0, cy), (x1, cy), (180, 220, 255), 1, cv2.LINE_AA)
        else:
            cx = int(round(0.5 * (x0 + x1)))
            cv2.line(img, (cx, y0), (cx, y1), (180, 220, 255), 1, cv2.LINE_AA)

        a_ok = bool(a.get("a_ok", False))
        b_ok = bool(a.get("b_ok", False))
        cv2.circle(img, (int(A[0]), int(A[1])), 5, (0, 200, 0) if a_ok else (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (int(B[0]), int(B[1])), 5, (0, 200, 0) if b_ok else (0, 0, 255), -1, cv2.LINE_AA)

        dA = a.get("d_a", 0.0); dM = a.get("d_m", 0.0); dB = a.get("d_b", 0.0)
        put_label(img, f"A={dA:.2f} M={dM:.2f} B={dB:.2f}",
                  (x0, max(15, y0-6)), scale=0.5, color=(0,0,0), bg=(180,220,255))

    # --- draw candidates -------------------------------------------------------
    for c in dbg.get("arrow_span_candidates", []) or []:
        ab = c.get("arrow_span_bbox") or c.get("span_bbox")
        nb = c.get("number_bbox") or c.get("num_bbox")
        score = float(c.get("score", 0.0))
        conf  = float(c.get("ocr_conf", c.get("conf", 0.0)))
        span  = int(c.get("span_px", 0))
        if ab:
            x0,y0,x1,y1 = map(int, ab)
            cv2.rectangle(img, (x0,y0), (x1,y1), (255,0,0), 2)
            val = c.get("value", c.get("value_mm", "?"))
            put_label(img, f"val={val}  sc={score:.2f} conf={conf:.2f} span={span}",
                      (x0, max(15, y0-6)), color=(255,255,255), bg=(60,60,255))
        if nb:
            nx0,ny0,nx1,ny1 = map(int, nb)
            cv2.rectangle(img, (nx0,ny0), (nx1,ny1), (180,180,255), 1)

    # --- highlight picked pair -------------------------------------------------
    if picked:
        ab = picked.get("arrow_span_bbox") or picked.get("span_bbox")
        nb = picked.get("number_bbox") or picked.get("num_bbox")
        val = picked.get("value", "?")
        nconf = float(picked.get("ocr_conf", 0.0))
        if ab:
            x0,y0,x1,y1 = map(int, ab)
            cv2.rectangle(img, (x0,y0), (x1,y1), (0,255,255), 3)  # yellow span
        if nb:
            nx0,ny0,nx1,ny1 = map(int, nb)
            cv2.rectangle(img, (nx0,ny0), (nx1,ny1), (0,0,255), 3)  # red number
            put_label(img, f"[PICKED {val}] conf={nconf:.2f}",
                      (nx0, max(15, ny0-8)), color=(255,255,255), bg=(0,0,255))

    # --- banner ----------------------------------------------------------------
    pxmm = float(res_dict.get("px_per_mm", 1.0))
    conf = float(res_dict.get("confidence", 0.0))
    meth = str(res_dict.get("method", "unknown"))
    num_info = ""
    if picked:
        num_info = f"  num={picked.get('value','?')}  num_conf={float(picked.get('ocr_conf',0.0)):.2f}"
    put_label(img, f"px_per_mm={pxmm:.4f}  conf={conf:.2f}  method={meth}{num_info}",
              (8, 22), scale=0.7, color=(0,0,0), bg=(255,255,255), thick=2)

    cv2.imwrite(out_path, img)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="input page image")
    ap.add_argument("--outdir", default=None, help="where to write debug overlays")
    args = ap.parse_args()

    page = cv2.imread(args.image)
    if page is None:
        print(f"ERROR: cannot read image: {args.image}")
        sys.exit(1)

    section, _ = locate_section_view(page)
    res = detect_scale(page, section_bgr=section)
    rd = _to_dict(res)

    print("=== Day 2: scale detection ===")
    printable = {k:v for k,v in rd.items() if k != "debug"}
    print(json.dumps(printable, indent=2))

    dbg = rd.get("debug", {})
    tried = dbg.get("methods_tried", None)
    print("\n[methods tried]")
    if not tried:
        print("  (none)")
    else:
        for m in tried:
            name = m.get("name", "?")
            conf = float(m.get("confidence", m.get("score", 0.0)))
            pxmm = m.get("px_per_mm", None)
            extra = []
            if pxmm is not None:
                extra.append(f"px_per_mm={pxmm:.3f}" if isinstance(pxmm,(int,float)) else f"px_per_mm={pxmm}")
            span = m.get("span_px", None)
            if span is not None:
                extra.append(f"span={span}")
            why = m.get("why")
            if why: extra.append(why)
            line = f"  - {name:14s}: conf={conf:.2f}"
            if extra: line += "  " + "  ".join(extra)
            print(line)

    picked = dbg.get("picked") or {}
    if picked:
        val   = picked.get("value", "?")
        nconf = float(picked.get("ocr_conf", 0.0))
        span  = picked.get("span_px", "?")
        orient = picked.get("orient", "?")
        try:
            pxmm_from_pair = float(span) / float(val)
        except Exception:
            pxmm_from_pair = None
        print("\n[picked]")
        line = f"  number={val}  ocr_conf={nconf:.2f}  span_px={span}  orient={orient}"
        if pxmm_from_pair is not None:
            line += f"  px_per_mm≈{pxmm_from_pair:.3f}"
        print(line)
    else:
        print("\n[picked]\n  (none)")

    # --- alerts / quality gates ---------------------------------------------
    WARN_CONF      = 0.25   # overall scale confidence
    WARN_OCR_CONF  = 0.70   # OCR confidence for picked number
    WARN_CLUSTER_N = 2      # min cluster size for consensus

    alerts = []
    overall_conf = float(rd.get("confidence", 0.0))
    if overall_conf < WARN_CONF:
        alerts.append(f"LOW_SCALE_CONF ({overall_conf:.2f} < {WARN_CONF})")

    cluster_sz = int(dbg.get("cluster_size") or 0)
    if cluster_sz < WARN_CLUSTER_N:
        alerts.append(f"WEAK_CONSENSUS (cluster_size={cluster_sz})")

    if picked:
        nconf = float(picked.get("ocr_conf", 0.0))
        if nconf < WARN_OCR_CONF:
            alerts.append(f"LOW_OCR_FOR_PICKED_NUM ({nconf:.2f} < {WARN_OCR_CONF})")
        arrow = picked.get("arrow") or {}
        a_ok = bool(arrow.get("a_ok", False))
        b_ok = bool(arrow.get("b_ok", False))
        if not (a_ok and b_ok):
            alerts.append(f"WEAK_ARROWHEAD_EVIDENCE (a_ok={a_ok}, b_ok={b_ok})")
        if picked.get("leader_cross", False):
            alerts.append("LEADER_LINE_CROSSES_ANOTHER_SPAN")

    print("\n[alerts]")
    if alerts:
        for a in alerts:
            print(f"  - {a}")
    else:
        print("  (none)")

    outdir = args.outdir or os.path.join(os.path.dirname(args.image), "..", "golden", "debug")
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]
    ocr_path   = os.path.join(outdir, f"{stem}_ocr_overlay.png")
    scale_path = os.path.join(outdir, f"{stem}_scale_overlay.png")

    draw_ocr_overlay(page, dbg, ocr_path)
    draw_scale_debug(page, rd, scale_path)

    print(f"\n[overlay] ocr   → {ocr_path}")
    print(f"[overlay] scale → {scale_path}")

if __name__ == "__main__":
    main()
