import argparse
import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize

def skeleton_to_paths(skel):
    skel = (skel > 0).astype(np.uint8)
    coords = np.argwhere(skel == 1)
    visited = np.zeros_like(skel, dtype=bool)

    def neighbors(y, x):
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y+dy, x+dx
                if 0 <= ny < skel.shape[0] and 0 <= nx < skel.shape[1]:
                    if skel[ny, nx] == 1 and not visited[ny, nx]:
                        yield ny, nx

    paths = []
    for (y, x) in coords:
        if visited[y, x]:
            continue
        path = [(x, -y)]  # flip y for turtle coordinates
        visited[y, x] = True
        stack = [(y, x)]
        while stack:
            cy, cx = stack.pop()
            for ny, nx in neighbors(cy, cx):
                visited[ny, nx] = True
                path.append((nx, -ny))
                stack.append((ny, nx))
        if len(path) > 1:
            paths.append(np.array(path, float))
    return paths

def image_to_kolam_csv(image_path, csv_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    skel = skeletonize(binary > 0)

    paths = skeleton_to_paths(skel)

    # normalize all together
    all_pts = np.vstack(paths)
    all_pts -= all_pts.mean(axis=0)
    span = max(np.ptp(all_pts[:,0]), np.ptp(all_pts[:,1]))
    all_pts = all_pts/span*300

    # split back
    out = {}
    offset = 0
    max_len = max(len(p) for p in paths)
    for i, path in enumerate(paths, 1):
        n = len(path)
        out[f"x-kolam {i}"] = np.full(max_len, np.nan)
        out[f"y-kolam {i}"] = np.full(max_len, np.nan)
        out[f"x-kolam {i}"][:n] = all_pts[offset:offset+n,0]
        out[f"y-kolam {i}"][:n] = all_pts[offset:offset+n,1]
        offset += n
    pd.DataFrame(out).to_csv(csv_path, index=False)
    print(f"✅ Saved {csv_path} with {len(paths)} strokes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    image_to_kolam_csv(args.img, args.csv)
