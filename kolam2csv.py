import argparse
import cv2
import numpy as np
import pandas as pd
import igraph as ig
from skimage.morphology import skeletonize

def skeleton_to_edges_igraph(skel):
    skel = (skel > 0).astype(np.uint8)
    coords = np.argwhere(skel)
    coord_dict = {tuple(c): i for i, c in enumerate(coords)}
    edges = []

    for y, x in coords:
        u = coord_dict[(y, x)]
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if (ny, nx_) in coord_dict:
                    v = coord_dict[(ny, nx_)]
                    if u < v:
                        edges.append((u, v))

    coords_list = [(x, -y) for y, x in coords]  # flip y for turtle coords
    G = ig.Graph(edges=edges, directed=False)
    G.vs['coords'] = coords_list

    # Keep only largest connected component
    G = G.clusters().giant()
    return G

def make_eulerian(G):
    """
    Make graph Eulerian by pairing odd-degree vertices.
    """
    odd_vertices = [v.index for v in G.vs if G.degree(v) % 2 == 1]
    while odd_vertices:
        u = odd_vertices.pop()
        # Find closest odd vertex
        distances = [np.linalg.norm(np.array(G.vs[u]['coords']) - np.array(G.vs[v]['coords'])) for v in odd_vertices]
        idx = np.argmin(distances)
        v = odd_vertices.pop(idx)
        G.add_edge(u, v)
    return G

def is_eulerian(G):
    return all(d % 2 == 0 for d in G.degree()) and G.is_connected()

def eulerian_circuit(G, start=0):
    """
    Compute Eulerian circuit using Hierholzer's algorithm.
    Returns list of vertex indices.
    """
    # Make a copy of adjacency list
    edges = {v.index: set(G.neighbors(v)) for v in G.vs}
    circuit = []
    stack = [start]

    while stack:
        v = stack[-1]
        if edges[v]:
            u = edges[v].pop()
            edges[u].remove(v)
            stack.append(u)
        else:
            circuit.append(stack.pop())

    circuit.reverse()
    return circuit

def image_to_kolam_csv(image_path, csv_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    skel = skeletonize(binary > 0)

    G = skeleton_to_edges_igraph(skel)

    if not is_eulerian(G):
        G = make_eulerian(G)

    path = eulerian_circuit(G)
    points = [G.vs[v]['coords'] for v in path]

    points = np.array(points, float)
    points -= points.mean(axis=0)
    span = max(np.ptp(points[:,0]), np.ptp(points[:,1]))
    points = points / span * 300

    out = {
        "x-kolam 1": points[:,0],
        "y-kolam 1": points[:,1]
    }
    pd.DataFrame(out).to_csv(csv_path, index=False)
    print(f"✅ Saved {csv_path} as Eulerian single stroke ({len(points)} points)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    image_to_kolam_csv(args.img, args.csv)
