import argparse
import os
import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
import networkx as nx

from preprocess import preprocess_kolam

def skeleton_to_edges(skel):
    skel = (skel > 0).astype(np.uint8)
    coords = np.argwhere(skel == 1)

    def neighbors(y, x):
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y+dy, x+dx   # rename local variable
                if 0 <= ny < skel.shape[0] and 0 <= nx_ < skel.shape[1]:
                    if skel[ny, nx_] == 1:
                        yield ny, nx_

    G = nx.Graph()
    for (y, x) in coords:
        u = (x, -y)  # flip y for turtle coordinates
        for ny, nx_ in neighbors(y, x):
            v = (nx_, -ny)
            G.add_edge(u, v)
    return G

def make_eulerian(G):
    # Take the largest connected component only
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    # Use built-in eulerize to pair odd-degree nodes
    G = nx.eulerize(G)
    return G

def image_to_kolam_csv(image_path, csv_path):
    # PreProcess Image & Replace Path
    image_path = preprocess_kolam(image_path)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    skel = skeletonize(binary > 0)

    ### Build graph from skeleton
    G = skeleton_to_edges(skel)

    ### Force Eulerian
    G = make_eulerian(G)

    ### Compute Eulerian circuit
    path = list(nx.eulerian_circuit(G))

    # Flatten into a single stroke of points
    points = [path[0][0]] + [v for (_, v) in path]

    points = np.array(points, float)

    # Normalize
    points -= points.mean(axis=0)
    span = max(np.ptp(points[:,0]), np.ptp(points[:,1]))
    points = points / span * 300

    # Save as 1 stroke
    out = {
        "x-kolam 1": np.full(len(points), np.nan),
        "y-kolam 1": np.full(len(points), np.nan),
    }
    out["x-kolam 1"][:] = points[:,0]
    out["y-kolam 1"][:] = points[:,1]

    pd.DataFrame(out).to_csv(csv_path, index=False)
    print(f"✅ Saved {csv_path} as Eulerian single stroke ({len(points)} points)")

    # Delete the File
    os.remove(image_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    image_to_kolam_csv(args.img, args.csv)
