import argparse
import os
import tempfile
import numpy as np
import pandas as pd
import cv2
from scipy.interpolate import splprep, splev


def image_to_csv(image_path: str, csv_path: str) -> None:
    # Lazy import to avoid heavy deps if only CSV is used
    from kolam2csv import image_to_kolam_csv

    image_to_kolam_csv(image_path, csv_path)


def load_all_points(csv_path: str):
    df = pd.read_csv(csv_path)
    strokes = []
    for col in df.columns:
        if col.startswith("x-kolam"):
            idx = col.split()[-1]
            x = df[f"x-kolam {idx}"].to_numpy()
            y = df[f"y-kolam {idx}"].to_numpy()
            P = np.c_[x, y]
            P = P[np.isfinite(P).all(axis=1)]
            if len(P) >= 2:
                strokes.append(P)
    return strokes


def normalize_strokes(strokes):
    all_pts = np.vstack(strokes)
    all_pts = all_pts - all_pts.mean(axis=0)
    span = max(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1]))
    span = span if span > 0 else 1.0
    all_pts = all_pts / span * 300.0
    result = []
    offset = 0
    for P in strokes:
        n = len(P)
        result.append(all_pts[offset:offset + n])
        offset += n
    return result


def fit_spline(P: np.ndarray, samples: int = 1200, smooth: float = 10.0) -> np.ndarray:
    # Remove duplicates that break splines
    if len(P) < 2:
        return P
    diff = np.diff(P, axis=0)
    keep = np.ones(len(P), dtype=bool)
    keep[1:] = (np.linalg.norm(diff, axis=1) > 1e-6)
    P = P[keep]

    # Parameterize by cumulative distance for stability
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    if d[-1] == 0:
        return P
    u = d / d[-1]

    try:
        tck, _ = splprep([P[:, 0], P[:, 1]], u=u, s=smooth)
        u_new = np.linspace(0, 1, samples)
        x_new, y_new = splev(u_new, tck)
        return np.column_stack([x_new, y_new])
    except Exception:
        # Fallback to polyline
        return P


def render_curve(C: np.ndarray, width: int, height: int, thickness: int, bg: str) -> np.ndarray:
    def hex_to_bgr(hex_color: str):
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return (b, g, r)
        return (59, 95, 46)

    canvas = np.full((height, width, 3), hex_to_bgr(bg), dtype=np.uint8)

    pad = 0.05
    xmin, ymin = C.min(axis=0)
    xmax, ymax = C.max(axis=0)
    span_x = max(xmax - xmin, 1e-6)
    span_y = max(ymax - ymin, 1e-6)
    scale = min((width * (1 - 2 * pad)) / span_x, (height * (1 - 2 * pad)) / span_y)

    def to_px(pt):
        x = int((pt[0] - xmin) * scale + width * pad)
        y = int((pt[1] - ymin) * scale + height * pad)
        return (x, height - y)

    for i in range(1, len(C)):
        cv2.line(canvas, to_px(C[i - 1]), to_px(C[i]), (255, 255, 255), thickness, lineType=cv2.LINE_AA)

    return canvas


def main():
    ap = argparse.ArgumentParser(description="Render smooth kolam curve from image or CSV (no UI)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--img", help="Path to kolam image (JPG/PNG)")
    g.add_argument("--csv", help="Path to kolam CSV (x-kolam i / y-kolam i)")
    ap.add_argument("--out", default="curve.png", help="Output PNG file")
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--samples", type=int, default=1200)
    ap.add_argument("--smooth", type=float, default=10.0)
    ap.add_argument("--thickness", type=int, default=3)
    ap.add_argument("--bg", default="#2e5f3b")
    ap.add_argument("--maxsize", type=int, default=1100, help="Downscale image longest side before tracing (0=disable)")
    ap.add_argument("--maxpts", type=int, default=6000, help="Cap polyline points before spline fit (0=disable)")
    args = ap.parse_args()

    # Prepare CSV
    if args.img:
        with tempfile.TemporaryDirectory() as td:
            tmp_csv = os.path.join(td, "tmp.csv")
            src_path = args.img
            # Optional downscale large images to speed up skeletonization
            if args.maxsize and args.maxsize > 0:
                img = cv2.imread(args.img, cv2.IMREAD_COLOR)
                if img is None:
                    raise SystemExit(f"Failed to read image: {args.img}")
                h, w = img.shape[:2]
                m = max(h, w)
                if m > args.maxsize:
                    scale = args.maxsize / float(m)
                    resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                    tmp_img = os.path.join(td, "resized.jpg")
                    cv2.imwrite(tmp_img, resized)
                    src_path = tmp_img
            image_to_csv(src_path, tmp_csv)
            strokes = load_all_points(tmp_csv)
            strokes = normalize_strokes(strokes)
            P = np.vstack(strokes)
            if args.maxpts and len(P) > args.maxpts:
                idx = np.linspace(0, len(P) - 1, args.maxpts, dtype=int)
                P = P[idx]
            C = fit_spline(P, samples=args.samples, smooth=args.smooth)
            img = render_curve(C, args.width, args.height, args.thickness, args.bg)
            cv2.imwrite(args.out, img)
    else:
        strokes = load_all_points(args.csv)
        if not strokes:
            raise SystemExit("No strokes found in CSV")
        strokes = normalize_strokes(strokes)
        P = np.vstack(strokes)
        if args.maxpts and len(P) > args.maxpts:
            idx = np.linspace(0, len(P) - 1, args.maxpts, dtype=int)
            P = P[idx]
        C = fit_spline(P, samples=args.samples, smooth=args.smooth)
        img = render_curve(C, args.width, args.height, args.thickness, args.bg)
        cv2.imwrite(args.out, img)

    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()


