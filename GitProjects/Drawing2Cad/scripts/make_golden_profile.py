"""
Make a golden (z, r) profile by clicking on the section view.
USAGE:
  python GitProjects/Drawing2Cad/scripts/make_golden_profile.py --image GitProjects/Drawing2Cad/data/samples/example1.png --out GitProjects/Drawing2Cad/data/golden/profiles/example1.csv

What you do:
  1) The script shows the image.
  2) You click the section view's axis first (red vertical helper).
  3) Then you click along the outer silhouette from bottom to top.
  4) Close the window to save.

Why (z, r)?
  - r = horizontal distance from axis to your clicked point (radius)
  - z = vertical height measured upward from the bottom of the section images
"""
import argparse, os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",required=True)
    ap.add_argument("--out",required=True)
    args = ap.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise SyntaxError(f"Cannot read image {args.image}")
    rgb = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    print("Step 1: Click TWO points along the axis of revolution (so we know its orientation).")
    plt.figure(figsize=(10,6))
    plt.imshow(rgb)
    axis_clicks = plt.ginput(2, timeout=0)  # two clicks define the axis
    plt.close()
    if len(axis_clicks) < 2:
        raise SystemExit("Need two clicks to define axis")
    (x1, y1), (x2, y2) = axis_clicks
    dx, dy = (x2 - x1), (y2 - y1)
    if abs(dx) < abs(dy):
        axis_type = "vertical"
        axis_pos = (x1 + x2) / 2.0
    else:
        axis_type = "horizontal"
        axis_pos = (y1 + y2) / 2.0

    print(f"Axis defined as {axis_type} at pos={axis_pos:.2f}")

    
    overlay = rgb.copy()
    if axis_type == "vertical":
        x_int = int(round(axis_pos))
        overlay[:, x_int-1:x_int+1, :] = [255, 0, 0]  # red vertical stripe
    else:
        y_int = int(round(axis_pos))
        overlay[y_int-1:y_int+1, :, :] = [255, 0, 0]  # red horizontal stripe

    print("Click along the OUTER silhouette from bottom to top.\n"
          "When you're done, close the window (or press 'q').")
    plt.figure(figsize=(10,6))
    plt.imshow(overlay)
    pts = plt.ginput(n=-1,timeout=0)
    plt.close()

    pts = np.array(pts, dtype=np.float32)
    x = pts[:, 0]
    y = pts[:, 1]

    if axis_type == "vertical":
        r = np.abs(x - axis_pos)
        z = (h - y)   # measure height from bottom
    else:  # horizontal
        r = np.abs(y - axis_pos)
        z = x         # measure along x

    if len(pts) < 2:
        raise SystemExit("Need at least 2 clicks for a polyline.")
   

    # 5) Save CSV with header
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["z", "r"])
        for zi, ri in zip(z, r):
            wcsv.writerow([f"{zi:.3f}", f"{ri:.3f}"])
    print(f"Saved golden profile CSV: {args.out}")

    # 6) Save true axis (for axis error metric)
    axis_json = args.out.replace("/profiles/", "/axis/").replace(".csv", ".json")
    os.makedirs(os.path.dirname(axis_json), exist_ok=True)
    with open(axis_json, "w") as jf:
        jf.write(f'{{"axis_type": "{axis_type}", "axis_pos": {axis_pos:.3f}, '
                f'"image_width": {w}, "image_height": {h}}}\n')
if __name__ == "__main__":
    main()